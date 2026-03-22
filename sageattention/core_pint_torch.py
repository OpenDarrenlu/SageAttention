import torch
import torch.nn.functional as F

from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
from .triton.attn_qk_int8_pint_v_int8 import forward as attn_forward
from .triton.quant_per_channel import per_channel_int8 as per_channel_int8_triton

from typing import Any, List, Literal, Optional, Tuple, Union

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

def sageattn_pint_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """
        
    arch = get_cuda_arch_versions()[q.device.index]
    if True: # arch == "sm86":
        return sageattn_qk_int8_p_pint_vint8_torch(q, k, v, tensor_layout=tensor_layout, sm_scale=sm_scale, return_lse=return_lse, smooth_k=True)
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def sageattn_qk_int8_p_pint_vint8_torch(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    smooth_v: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 quantization for Q and K, BF16 PV with BF16 accumulation, implemented using Triton.
    The BF16 accumulator is added to a FP32 buffer immediately after each iteration.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    quantization_backend : str
        The quantization backend, either "triton" or "cuda".
        "cuda" backend offers better performance due to kernel fusion.

    attn_mask : Optional[torch.Tensor]
        The attention mask tensor, of dtype bool or float32.
        Should be able to broadcast to the shape of the matrix qk^T.
        Default: None.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype, "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
        v_int8, v_scale, vm = per_channel_int8_triton(v, tensor_layout, smooth_v=smooth_v)
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")

    if attn_mask is not None:
        if tensor_layout == "HND":
            target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        elif tensor_layout == "NHD":
            target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
        else:
            raise ValueError(f"tensor_layout {tensor_layout} not supported")
        try:
            attn_mask = attn_mask.expand(target_shape)
        except Exception:
            raise AssertionError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}")
    
    # o, lse = attn_forward(q_int8, k_int8, v_int8, q_scale, k_scale, v_scale, vm, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=attn_mask, return_lse=return_lse)
    # import ipdb; ipdb.set_trace()
    import torch.nn.functional as F
    import torchmm
    BLK_Q = 128
    BLK_K = 64
    def safe_block_matmul(q_int8, k_int8, q_scale, k_scale, BLK_Q, BLK_K):
        if tensor_layout == "NHD":
            q_int8 = q_int8.transpose(1, 2)
            k_int8 = k_int8.transpose(1, 2)
        # 1. 获取原始形状
        B, H, Sq, Dq = q_int8.shape
        B, H, Sk, Dk = k_int8.shape
        
        # 2. 计算需要填充的大小 (使其成为 BLK 的倍数)
        pad_q = (BLK_Q - Sq % BLK_Q) % BLK_Q
        pad_k = (BLK_K - Sk % BLK_K) % BLK_K
        
        # 3. 填充 q_int8 和 k_int8 (填充 0，因为 0 * scale = 0，不影响数值)
        # padding 格式为 (left, right, top, bottom, ...)，这里只填充序列维度 (dim 2)
        q_int8_padded = F.pad(q_int8, (0, 0, 0, pad_q), mode='constant', value=0)
        k_int8_padded = F.pad(k_int8, (0, 0, 0, pad_k), mode='constant', value=0)
        
        # 4. 执行 Matmul (使用填充后的张量)
        # 注意：这里使用 torch.matmul，如果是特定库的 torchmm 请替换回原函数
        S_q = torchmm.matmul(q_int8_padded.to(torch.int32), 
                        k_int8_padded.transpose(-2, -1).to(torch.int32))
        
        # 获取填充后的序列长度，用于后续的 view 操作
        Sq_padded = q_int8_padded.shape[2]
        Sk_padded = k_int8_padded.shape[2]
        
        # 5. 应用 Q 的 Scale (基于填充后的维度进行 view)
        # view 形状：[B, H, NumBlocks_Q, BLK_Q, Sk_padded]
        S_q = S_q.view(B, H, -1, BLK_Q, Sk_padded)
        
        # q_scale 通常为 [B, H, NumBlocks_Q, 1] 或 [B, H, NumBlocks_Q]
        # 需要 broadcast 到 [B, H, NumBlocks_Q, 1, 1]
        # 原代码有两个 unsqueeze，假设 q_scale 是 3 维 [B, H, Nb]，这里保持原逻辑
        S_q = S_q * q_scale.unsqueeze(-1).unsqueeze(-1)
        
        # 6. 应用 K 的 Scale (基于填充后的维度进行 view)
        # view 形状：[B, H, Sq_padded, NumBlocks_K, BLK_K]
        S_q = S_q.view(B, H, Sq_padded, -1, BLK_K)
        
        # k_scale 需要 broadcast 到 [B, H, 1, NumBlocks_K, 1]
        # 原代码 unsqueeze(-2).unsqueeze(-1)，保持原逻辑
        S_q = S_q * k_scale.unsqueeze(-2).unsqueeze(-1)
        
        # 7. 还原形状并切片回原始长度
        S_q = S_q.view(B, H, Sq_padded, Sk_padded)
        
        # 切片去除填充部分
        S_q = S_q[:, :, :Sq, :Sk]
        
        return S_q

    # --- 使用示例 ---
    # S_q = safe_block_matmul(q_int8, k_int8, q_scale, k_scale, BLK_Q, BLK_K)
    S_q = safe_block_matmul(q_int8, k_int8, q_scale, k_scale, BLK_Q, BLK_K)
    S_q -= torch.max(S_q, dim=-1, keepdim=True)[0]
    P_q = torch.softmax(S_q, dim=-1).to(torch.bfloat16)
    from .triton.quant_pint import bf16_to_fixed_point_triton, bf16_to_fixed_point_torch
    P_int32, P_scale = bf16_to_fixed_point_torch(P_q)
    # import ipdb; ipdb.set_trace()
    if tensor_layout == "NHD":
        v_int8 = v_int8.transpose(1, 2)
    O_q = torchmm.matmul(P_int32, v_int8.to(torch.int32)).to(torch.float32) * P_scale * v_scale[:,:,None,:]
    o = O_q + vm[:,:,None,:]
    if tensor_layout == "NHD":
        o = o.transpose(1, 2)
    
    # o = P_q.to(torch.float16) @ v.to(torch.float16)
    
    o = o[..., :head_dim_og].to(dtype)
    # print(f"v_int8({v_int8.shape}): {v_int8}")
    # print(f"v_scale({v_scale.shape}): {v_scale}")
    # print(f"vm({vm.shape}): {vm}")
    torch.save({"o":o}, "pint_o_int8.pt")

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
