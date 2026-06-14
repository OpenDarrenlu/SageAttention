import torch
import torch.nn.functional as F

from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
from .triton.attn_qk_int8_lut_v_int8 import forward as attn_forward
from .triton.quant_per_channel import per_channel_int8 as per_channel_int8_triton
from .triton.quant_per_channel import per_channel_int4 as per_channel_int4_triton
from .triton.quant_per_channel import per_channel_int2 as per_channel_int2_triton
from .triton.quant_per_channel import (
    per_channel_int8_block as per_channel_int8_block_triton,
    per_channel_int4_block as per_channel_int4_block_triton,
    per_channel_int2_block as per_channel_int2_block_triton,
)

from typing import Any, List, Literal, Optional, Tuple, Union

# Valid V precisions handled by the per-channel quantizer.
V_QUANT_DTYPES = {"int8", "int4", "int2"}
# Valid P precisions handled by the attention kernel (software-emulated).
P_QUANT_DTYPES = {"fp16", "bf16", "int8", "int4", "mxfp8", "mxfp4", "nvfp4", "mxint4"}


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
    p_quant_dtype: str = "fp16",
    v_quant_dtype: str = "int8",
    p_block_n: int = 64,
    v_block_size: int = 0,
    **kwargs: Any,
):
    """
    LUT-based SageAttention with configurable P and V quantization precisions.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query/key/value tensors.  Same shape conventions as the original
        ``sageattn_lut``.
    tensor_layout : str
        "HND" or "NHD".  Default: "HND".
    sm_scale : Optional[float]
        Softmax scale.  Defaults to ``1.0 / sqrt(head_dim)``.
    return_lse : bool
        Whether to return log-sum-exp.  Default: False.
    p_quant_dtype : str
        Precision used for the online per-token quantization of the attention
        weights P.  Supported: "fp16", "bf16", "int8", "int4", "mxfp8",
        "mxfp4", "nvfp4", "mxint4".  Default: "fp16".
    v_quant_dtype : str
        Precision used for the per-channel quantization of V.  Supported:
        "int8", "int4", "int2".  Default: "int8".
    p_block_n : int
        Block size for P scaling along the KV dimension.  64 means one scale
        per query token per KV tile; smaller values (32, 16) enable finer
        block scaling.  Must divide 64.  Default: 64.
    v_block_size : int
        Block size for V scaling along the KV dimension.  0 means per-channel
        (legacy); 64 enables MXINTx per-channel-per-block scaling, which
        improves INT2/INT4 V accuracy.  Default: 0.

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Output tensor, and optionally LSE.
    """
    return sageattn_qk_int8_p_lut_vq_triton(
        q, k, v,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        return_lse=return_lse,
        smooth_k=True,
        p_quant_dtype=p_quant_dtype,
        v_quant_dtype=v_quant_dtype,
        p_block_n=p_block_n,
        v_block_size=v_block_size,
        **kwargs,
    )


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

    This is kept for backward compatibility; it is equivalent to calling
    ``sageattn_lut(..., v_quant_dtype="int4")``.
    """
    return sageattn_qk_int8_p_lut_vq_triton(
        q, k, v,
        tensor_layout=tensor_layout,
        sm_scale=sm_scale,
        return_lse=return_lse,
        smooth_k=True,
        p_quant_dtype="fp16",
        v_quant_dtype="int4",
        **kwargs,
    )


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
    Backward-compatible entry point: P in fp16, V in int8.
    """
    return sageattn_qk_int8_p_lut_vq_triton(
        q, k, v,
        tensor_layout=tensor_layout,
        quantization_backend=quantization_backend,
        attn_mask=attn_mask,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        p_quant_dtype="fp16",
        v_quant_dtype="int8",
        **kwargs,
    )


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
    Backward-compatible entry point: P in fp16, V in int4.
    """
    return sageattn_qk_int8_p_lut_vq_triton(
        q, k, v,
        tensor_layout=tensor_layout,
        quantization_backend=quantization_backend,
        attn_mask=attn_mask,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        p_quant_dtype="fp16",
        v_quant_dtype="int4",
        **kwargs,
    )


def sageattn_qk_int8_p_lut_vq_triton(
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
    p_quant_dtype: str = "fp16",
    v_quant_dtype: str = "int8",
    p_block_n: int = 64,
    v_block_size: int = 0,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 Q/K and configurable per-token P /
    per-channel V quantization, implemented using Triton.

    The PV matmul is performed in the original input dtype (fp16/bf16) after
    dequantizing the low-bit P and V tiles, so the kernel only emulates the
    precision loss of the low-bit formats.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Query/key/value tensors.
    tensor_layout : str
        "HND" or "NHD".  Default: "HND".
    quantization_backend : str
        Only "triton" is supported for this variant.
    attn_mask : Optional[torch.Tensor]
        Attention mask tensor.
    sm_scale : Optional[float]
        Softmax scale.
    smooth_k : bool
        Subtract per-sequence K mean before QK matmul.  Default: True.
    smooth_v : bool
        Subtract per-channel V mean before V quantization.  Default: True.
    return_lse : bool
        Return log-sum-exp.  Default: False.
    p_quant_dtype : str
        Online P quantization precision.  Default: "fp16".
    v_quant_dtype : str
        Per-channel V quantization precision.  Default: "int8".
    p_block_n : int
        Block size for P scaling along the KV dimension.  Default: 64.
    v_block_size : int
        Block size for V scaling along the KV dimension.  0 means per-channel
        (legacy).  64 enables per-channel-per-block (MXINTx) scaling, which
        significantly improves INT2/INT4 V accuracy.  Default: 0.

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Output tensor, and optionally LSE.
    """
    if p_quant_dtype not in P_QUANT_DTYPES:
        raise ValueError(f"Unsupported p_quant_dtype '{p_quant_dtype}'. Choose from {P_QUANT_DTYPES}")
    if v_quant_dtype not in V_QUANT_DTYPES:
        raise ValueError(f"Unsupported v_quant_dtype '{v_quant_dtype}'. Choose from {V_QUANT_DTYPES}")

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
        if v_block_size == 0:
            if v_quant_dtype == "int8":
                v_quant, v_scale, vm = per_channel_int8_triton(v, tensor_layout, smooth_v=smooth_v)
            elif v_quant_dtype == "int4":
                v_quant, v_scale, vm = per_channel_int4_triton(v, tensor_layout, smooth_v=smooth_v)
            elif v_quant_dtype == "int2":
                v_quant, v_scale, vm = per_channel_int2_triton(v, tensor_layout, smooth_v=smooth_v)
            else:
                raise ValueError(f"Unsupported v_quant_dtype: {v_quant_dtype}")
        else:
            # MXINTx: per-channel-per-block scaling along the sequence dimension.
            if v_quant_dtype == "int8":
                v_quant, v_scale, vm = per_channel_int8_block_triton(v, tensor_layout, block_size=v_block_size, smooth_v=smooth_v)
            elif v_quant_dtype == "int4":
                v_quant, v_scale, vm = per_channel_int4_block_triton(v, tensor_layout, block_size=v_block_size, smooth_v=smooth_v)
            elif v_quant_dtype == "int2":
                v_quant, v_scale, vm = per_channel_int2_block_triton(v, tensor_layout, block_size=v_block_size, smooth_v=smooth_v)
            else:
                raise ValueError(f"Unsupported v_quant_dtype: {v_quant_dtype}")
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

    o, lse = attn_forward(
        q_int8, k_int8, v_quant, q_scale, k_scale, v_scale, vm,
        tensor_layout=tensor_layout,
        output_dtype=dtype,
        attn_mask=attn_mask,
        return_lse=return_lse,
        p_dtype=p_quant_dtype,
        p_block_n=p_block_n,
        v_block_size=v_block_size,
    )
    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
