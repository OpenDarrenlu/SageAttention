"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl

# Mapping from human-readable P precision names to integer constexpr codes.
# The kernel is specialized on the code so each combination is compiled once
# and cached by Triton.
P_DTYPE_MAP = {
    "fp16": 0,
    "bf16": 1,
    "int8": 2,
    "int4": 3,
    "mxfp8": 4,
    "mxfp4": 5,
    "nvfp4": 6,
    "mxint4": 7,
}

# Mapping from compute dtype to integer code.  The PV matmul is performed in
# this dtype after dequantizing the low-bit P and V tiles.
COMPUTE_DTYPE_MAP = {
    torch.float16: 0,
    torch.bfloat16: 1,
}


@triton.jit
def _uniform_quant_tile(tile, max_val, scale_max):
    """
    Uniform unsigned quantization helper for a non-negative tile.
    ``scale_max`` is the positive maximum of the target grid.
    """
    eps = 1e-9
    max_val = tl.maximum(max_val, eps)
    scale = max_val / scale_max
    q = tile / scale[:, None]
    q = q + 0.5
    q_int = q.to(tl.int32)
    q_int = tl.where(q_int > scale_max, scale_max, tl.where(q_int < 0, 0, q_int))
    return q_int.to(tl.float32) * scale[:, None]


@triton.jit
def _quantize_p_per_token(p, P_DTYPE_CODE: tl.constexpr, P_BLOCK_N: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Simulate per-token / per-block quantization of the attention weights ``p``.

    ``p`` is always non-negative because it comes from softmax, so unsigned
    quantization grids are used for the integer-style formats to avoid wasting
    one bit on the sign.  For the float-style formats we use a per-block scale
    that maps the block's max value to the format's positive maximum; this is a
    software emulation of block-scaled low-bit float formats.

    All paths return a float32 tile that has been through quantize/dequantize,
    so the downstream PV matmul sees only the precision loss induced by the
    chosen format.

    Parameters
    ----------
    P_BLOCK_N : tl.constexpr
        Granularity of the P scale along the KV dimension.  ``BLOCK_N`` means
        one scale per query token per KV tile (the coarsest practical
        granularity in a streaming kernel).  Smaller values (e.g. 32, 16) mean
        block scaling within the tile.
    """
    if P_DTYPE_CODE == 0:
        # fp16: native round-trip through fp16.
        return p.to(tl.float16).to(tl.float32)
    if P_DTYPE_CODE == 1:
        # bf16: native round-trip through bf16.
        return p.to(tl.bfloat16).to(tl.float32)

    # scale_max for each low-bit format (positive maximum of the unsigned grid)
    scale_max = tl.zeros([], dtype=tl.float32)
    if P_DTYPE_CODE == 2:
        scale_max = 255.0
    elif P_DTYPE_CODE == 3 or P_DTYPE_CODE == 7:
        scale_max = 15.0
    elif P_DTYPE_CODE == 4:
        scale_max = 448.0
    elif P_DTYPE_CODE == 5:
        scale_max = 6.0
    elif P_DTYPE_CODE == 6:
        scale_max = 28.0

    if P_BLOCK_N == BLOCK_N:
        # Per-KV-tile scale (coarsest, one scale per query token per tile).
        max_val = tl.max(p, axis=1)
        return _uniform_quant_tile(p, max_val, scale_max)

    # Finer block scaling: split the BLOCK_N tile into P_BLOCK_N sub-blocks.
    out = tl.zeros([p.shape[0], p.shape[1]], dtype=tl.float32)
    n_idx = tl.arange(0, BLOCK_N)
    for n_start in range(0, BLOCK_N, P_BLOCK_N):
        n_mask = (n_idx >= n_start) & (n_idx < n_start + P_BLOCK_N)
        # p is non-negative, so zero-padding gives the correct block max.
        block_p = tl.where(n_mask[None, :], p, 0.0)
        max_val = tl.max(block_p, axis=1)
        deq_block = _uniform_quant_tile(block_p, max_val, scale_max)
        out = tl.where(n_mask[None, :], deq_block, out)
    return out


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, stride_kn, stride_vn,
                    start_m, mask_ptrs, stride_maskn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    P_DTYPE_CODE: tl.constexpr, COMPUTE_DTYPE_CODE: tl.constexpr,
                    P_BLOCK_N: tl.constexpr, V_BLOCK_SIZE: tl.constexpr,
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=False)
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=-1.0e6)
        if not skip:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask)
            k_scale = tl.load(K_scale_ptr)

            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            else:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]

            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))

            # Load the correct V scale for this KV tile.
            if V_BLOCK_SIZE == 0:
                v_block_idx = 0
            else:
                v_block_idx = start_n // V_BLOCK_SIZE
            v_scale = tl.load(V_scale_ptr + v_block_idx * HEAD_DIM)

            # Dequantize V to the compute dtype (fp16/bf16) before the dot.
            if COMPUTE_DTYPE_CODE == 0:
                v_compute = v.to(tl.float16) * v_scale.to(tl.float16)
            else:
                v_compute = v.to(tl.bfloat16) * v_scale.to(tl.bfloat16)

            # Quantize/dequantize P per-token / per-block to emulate the requested low-bit P
            # precision, then perform the PV matmul in the compute dtype.
            p_sim = _quantize_p_per_token(p, P_DTYPE_CODE, P_BLOCK_N, BLOCK_N)
            if COMPUTE_DTYPE_CODE == 0:
                p_compute = p_sim.to(tl.float16)
            else:
                p_compute = p_sim.to(tl.bfloat16)

            acc += tl.dot(p_compute, v_compute).to(tl.float32)

            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, V_scale, VM, Out, mask, Lse,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,
              stride_oz, stride_oh, stride_on,
              stride_maskz, stride_maskh, stride_maskm, stride_maskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              num_v_blocks: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              P_DTYPE_CODE: tl.constexpr,
              COMPUTE_DTYPE_CODE: tl.constexpr,
              P_BLOCK_N: tl.constexpr,
              V_BLOCK_SIZE: tl.constexpr,
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)
    if V_BLOCK_SIZE == 0:
        v_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * HEAD_DIM
    else:
        v_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * num_v_blocks * HEAD_DIM
    vm_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * HEAD_DIM

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    V_scale_ptr = V_scale + v_scale_offset + offs_k
    VM_ptr = VM + vm_offset + offs_k
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, V_scale_ptr, stride_kn, stride_vn,
                                    start_m, mask_ptrs, stride_maskn,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n,
                                    P_DTYPE_CODE, COMPUTE_DTYPE_CODE, P_BLOCK_N, V_BLOCK_SIZE,
                                    )
    acc = acc / l_i[:, None]
    vm = tl.load(VM_ptr)
    acc = acc + vm[None, :]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def forward(q, k, v, q_scale, k_scale, v_scale, vm, tensor_layout="HND", attn_mask=None,
            output_dtype=torch.float16, return_lse=False, p_dtype="fp16", p_block_n: int = 64,
            v_block_size: int = 0):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    p_dtype_code = P_DTYPE_MAP.get(p_dtype)
    if p_dtype_code is None:
        raise ValueError(f"Unsupported p_dtype '{p_dtype}'. Supported: {list(P_DTYPE_MAP.keys())}")
    compute_dtype_code = COMPUTE_DTYPE_MAP.get(output_dtype)
    if compute_dtype_code is None:
        raise ValueError(f"Unsupported output_dtype '{output_dtype}'. Supported: fp16, bf16")

    if p_block_n <= 0:
        p_block_n = BLOCK_N
    if BLOCK_N % p_block_n != 0:
        raise ValueError(f"p_block_n ({p_block_n}) must divide BLOCK_N ({BLOCK_N})")

    # v_block_size == 0 means per-channel V scale (legacy).
    # v_block_size > 0 means per-channel-per-block scale; currently only
    # v_block_size == BLOCK_N is supported in the kernel.
    if v_block_size < 0:
        raise ValueError(f"v_block_size ({v_block_size}) must be >= 0")
    if v_block_size > 0 and v_block_size != BLOCK_N:
        raise ValueError(f"v_block_size ({v_block_size}) must be 0 or equal to BLOCK_N ({BLOCK_N})")

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    num_v_blocks = triton.cdiv(kv_len, v_block_size) if v_block_size > 0 else 1

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)

    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, v_scale, vm, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        num_v_blocks,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        STAGE=stage, RETURN_LSE=return_lse,
        P_DTYPE_CODE=p_dtype_code, COMPUTE_DTYPE_CODE=compute_dtype_code,
        P_BLOCK_N=p_block_n, V_BLOCK_SIZE=v_block_size,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4)

    return o, lse
