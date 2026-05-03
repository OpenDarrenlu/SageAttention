import torch
import torch.nn.functional as F

from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
from .triton.attn_qk_int8_lut_v_int8 import forward as attn_forward
from .triton.quant_per_channel import per_channel_int8 as per_channel_int8_triton
from .triton.quant_per_channel import per_channel_int4 as per_channel_int4_triton

from typing import Any, List, Literal, Optional, Tuple, Union

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

def sageattn_lut(
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
        return sageattn_qk_int8_p_lut_vint8_triton(q, k, v, tensor_layout=tensor_layout, sm_scale=sm_scale, return_lse=return_lse, smooth_k=True)
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def sageattn_qk_int8_p_lut_vint8_triton(
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
    
    o, lse = attn_forward(q_int8, k_int8, v_int8, q_scale, k_scale, v_scale, vm, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=attn_mask, return_lse=return_lse)
    o = o[..., :head_dim_og]
    # print(f"v_int8({v_int8.shape}): {v_int8}")
    # print(f"v_scale({v_scale.shape}): {v_scale}")
    # print(f"vm({vm.shape}): {vm}")
    # torch.save({"o":o}, "lut_o_int8.pt")

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_lut_int4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """
    SageAttention LUT variant with INT4 quantization for V.
    Q/K are INT8 quantized (per-block), V is INT4 quantized (per-channel).

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query, key, value tensors. Same shape conventions as ``sageattn_lut``.
    tensor_layout : str
        "HND" or "NHD".  Default: "HND".
    sm_scale : Optional[float]
        Softmax scale.  Defaults to ``1.0 / sqrt(head_dim)``.
    return_lse : bool
        Whether to return log-sum-exp.  Default: False.

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Output tensor, and optionally LSE.
    """
    arch = get_cuda_arch_versions()[q.device.index]
    if True:
        return sageattn_qk_int8_p_lut_vint4_triton(
            q, k, v, tensor_layout=tensor_layout, sm_scale=sm_scale,
            return_lse=return_lse, smooth_k=True, **kwargs
        )
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def sageattn_qk_int8_p_lut_vint4_triton(
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
    SageAttention LUT with per-block INT8 Q/K and per-channel INT4 V.

    Same interface as ``sageattn_qk_int8_p_lut_vint8_triton`` except V is
    quantized to INT4 (stored in INT8 containers, values in [-8, 7]).

    The existing Triton attention kernel is reused because the packed
    representation is identical to INT8 — only the scale magnitude differs.
    """
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype, "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."

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

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
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
        v_int4, v_scale, vm = per_channel_int4_triton(v, tensor_layout, smooth_v=smooth_v)
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

    o, lse = attn_forward(q_int8, k_int8, v_int4, q_scale, k_scale, v_scale, vm, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=attn_mask, return_lse=return_lse)
    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
