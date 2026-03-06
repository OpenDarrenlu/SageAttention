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

import torch
import triton
import triton.language as tl


def pack_int4(x_int8: torch.Tensor) -> torch.Tensor:
    """
    Pack two signed int4 values into one int8 lane:
    low nibble = even index, high nibble = odd index.
    """
    if x_int8.dtype != torch.int8:
        raise TypeError("pack_int4 expects torch.int8 input")
    if x_int8.size(-1) % 2 != 0:
        raise ValueError("The last dimension must be even for int4 packing")

    x = x_int8.to(torch.int16).clamp_(-8, 7)
    lo = (x[..., 0::2] & 0x0F).to(torch.int16)
    hi = (x[..., 1::2] & 0x0F).to(torch.int16)
    packed = lo | (hi << 4)
    packed = torch.where(packed >= 128, packed - 256, packed)
    return packed.to(torch.int8).contiguous()


def unpack_int4(x_packed: torch.Tensor) -> torch.Tensor:
    """Unpack int4 pairs from int8 lanes produced by pack_int4."""
    if x_packed.dtype != torch.int8:
        raise TypeError("unpack_int4 expects torch.int8 input")

    x = x_packed.to(torch.int16) & 0xFF
    lo = x & 0x0F
    hi = (x >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo).to(torch.int8)
    hi = torch.where(hi >= 8, hi - 16, hi).to(torch.int8)

    unpacked = torch.empty(
        (*x_packed.shape[:-1], x_packed.size(-1) * 2),
        dtype=torch.int8,
        device=x_packed.device,
    )
    unpacked[..., 0::2] = lo
    unpacked[..., 1::2] = hi
    return unpacked


@triton.jit
def quant_query_per_thread_int8_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 127. + 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_key_per_thread_int8_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        C: tl.constexpr, BLK: tl.constexpr):      
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    # offs_n = off_blk * BLK + tl.cat(tl.arange(0, BLK // 8) * 8, tl.arange(0, BLK // 8) * 8 + 1, True) + off_tld * 2
    # offs_k = tl.arange(0, C)

    # input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    # output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    # scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    # x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    # x = x.to(tl.float32)
    # scale = tl.max(tl.abs(x)) / 127. + 0.0000001
    # x_int8 = x / scale
    # x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    # x_int8 = x_int8.to(tl.int8)
    # tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    # tl.store(scale_ptrs, scale)

    offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
    offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
    offs_k = tl.arange(0, C)

    input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + offs_k[None, :]
    input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + offs_k[None, :]
    output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None] * stride_on + offs_k[None, :]
    output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
    x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
    x0_int8 = x0 / scale
    x1_int8 = x1 / scale
    x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
    x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
    x0_int8 = x0_int8.to(tl.int8)
    x1_int8 = x1_int8.to(tl.int8)
    tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
    tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_query_per_thread_int4_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C // 2)

    input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + (offs_k * 2)[None, :]
    input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + (offs_k * 2 + 1)[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x0 = tl.load(input_ptrs0, mask=offs_n[:, None] < L)
    x1 = tl.load(input_ptrs1, mask=offs_n[:, None] < L)
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 7. + 0.0000001
    x0_int8 = x0 / scale
    x1_int8 = x1 / scale
    x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
    x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
    x0_int8 = x0_int8.to(tl.int8).to(tl.int16)
    x1_int8 = x1_int8.to(tl.int8).to(tl.int16)
    packed = (x0_int8 & 0x0F) | ((x1_int8 & 0x0F) << 4)
    packed = packed.to(tl.int8)
    tl.store(output_ptrs, packed, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_key_per_thread_int4_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        C: tl.constexpr, BLK: tl.constexpr):      
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
    offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
    offs_k = tl.arange(0, C // 2)

    input_ptrs0_lo = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + (offs_k * 2)[None, :]
    input_ptrs0_hi = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + (offs_k * 2 + 1)[None, :]
    input_ptrs1_lo = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + (offs_k * 2)[None, :]
    input_ptrs1_hi = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + (offs_k * 2 + 1)[None, :]
    output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None] * stride_on + offs_k[None, :]
    output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    x0_lo = tl.load(input_ptrs0_lo, mask=offs_n0[:, None] < L).to(tl.float32)
    x0_hi = tl.load(input_ptrs0_hi, mask=offs_n0[:, None] < L).to(tl.float32)
    x1_lo = tl.load(input_ptrs1_lo, mask=offs_n1[:, None] < L).to(tl.float32)
    x1_hi = tl.load(input_ptrs1_hi, mask=offs_n1[:, None] < L).to(tl.float32)
    scale = max(max(tl.max(tl.abs(x0_lo)), tl.max(tl.abs(x0_hi))), max(tl.max(tl.abs(x1_lo)), tl.max(tl.abs(x1_hi)))) / 7. + 0.0000001

    x0_lo = (x0_lo / scale + 0.5 * tl.where(x0_lo >= 0, 1, -1)).to(tl.int8).to(tl.int16)
    x0_hi = (x0_hi / scale + 0.5 * tl.where(x0_hi >= 0, 1, -1)).to(tl.int8).to(tl.int16)
    x1_lo = (x1_lo / scale + 0.5 * tl.where(x1_lo >= 0, 1, -1)).to(tl.int8).to(tl.int16)
    x1_hi = (x1_hi / scale + 0.5 * tl.where(x1_hi >= 0, 1, -1)).to(tl.int8).to(tl.int16)

    packed0 = ((x0_lo & 0x0F) | ((x0_hi & 0x0F) << 4)).to(tl.int8)
    packed1 = ((x1_lo & 0x0F) | ((x1_hi & 0x0F) << 4)).to(tl.int8)
    tl.store(output_ptrs0, packed0, mask=offs_n0[:, None] < L)
    tl.store(output_ptrs1, packed1, mask=offs_n1[:, None] < L)
    tl.store(scale_ptrs, scale)

def per_thread_int8(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, sm_scale=None, tensor_layout="HND"):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    quant_key_per_thread_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK
    )

    return q_int8, q_scale, k_int8, k_scale


def per_thread_int4(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, sm_scale=None, tensor_layout="HND"):
    if q.size(-1) % 2 != 0 or k.size(-1) % 2 != 0:
        raise ValueError("head_dim must be even for int4 packing")

    q_int4 = torch.empty((*q.shape[:-1], q.size(-1) // 2), dtype=torch.int8, device=q.device)
    k_int4 = torch.empty((*k.shape[:-1], k.size(-1) // 2), dtype=torch.int8, device=k.device)


    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int4.stride(0), q_int4.stride(1), q_int4.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int4.stride(0), k_int4.stride(1), k_int4.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int4.stride(0), q_int4.stride(2), q_int4.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int4.stride(0), k_int4.stride(2), k_int4.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int4_kernel[grid](
        q, q_int4, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    quant_key_per_thread_int4_kernel[grid](
        k, k_int4, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK
    )

    return q_int4, q_scale, k_int4, k_scale
