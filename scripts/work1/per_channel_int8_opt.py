import torch
import triton
import triton.language as tl


_EPS = 1e-9


@triton.jit
def _stats_kernel(
    v_ptr,
    scale_ptr,
    mean_ptr,
    b,
    m,
    d,
    stride_vb,
    stride_vm,
    stride_vd,
    scale_max,
    smooth_v: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    """
    Compute per-(B,D) statistics over M:
    - mean (optional, only when smooth_v=True)
    - abs max (of centered values if smooth_v=True, raw values otherwise)
    - scale = abs_max / scale_max
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * block_d + tl.arange(0, block_d)
    mask_d = offs_d < d

    base_bd = pid_b * stride_vb + offs_d * stride_vd

    mean = tl.zeros([block_d], dtype=tl.float32)
    if smooth_v:
        sum_vals = tl.zeros([block_d], dtype=tl.float32)
        for offs_m in tl.range(0, m, block_m):
            idx_m = offs_m + tl.arange(0, block_m)
            mask_m = idx_m < m
            ptrs = v_ptr + base_bd[None, :] + idx_m[:, None] * stride_vm
            vals = tl.load(ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            sum_vals += tl.sum(vals, axis=0)
        mean = sum_vals / m

    abs_max = tl.zeros([block_d], dtype=tl.float32)
    for offs_m in tl.range(0, m, block_m):
        idx_m = offs_m + tl.arange(0, block_m)
        mask_m = idx_m < m
        ptrs = v_ptr + base_bd[None, :] + idx_m[:, None] * stride_vm
        vals = tl.load(ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        if smooth_v:
            vals = vals - mean[None, :]
        abs_max = tl.maximum(abs_max, tl.max(tl.abs(vals), axis=0))

    scale = tl.maximum(abs_max / scale_max, 1e-9)
    tl.store(scale_ptr + pid_b * d + offs_d, scale, mask=mask_d)
    if smooth_v:
        tl.store(mean_ptr + pid_b * d + offs_d, mean, mask=mask_d)


@triton.jit
def _quant_kernel(
    v_ptr,
    q_ptr,
    scale_ptr,
    mean_ptr,
    b,
    m,
    d,
    stride_vb,
    stride_vm,
    stride_vd,
    stride_qb,
    stride_qm,
    stride_qd,
    smooth_v: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    """
    Quantize v to int8 using per-(B,D) scale and optional mean smoothing.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_d = pid_d * block_d + tl.arange(0, block_d)
    mask = (offs_m[:, None] < m) & (offs_d[None, :] < d)

    v_ptrs = v_ptr + pid_b * stride_vb + offs_m[:, None] * stride_vm + offs_d[None, :] * stride_vd
    vals = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)

    scale = tl.load(scale_ptr + pid_b * d + offs_d, mask=offs_d < d, other=1.0)
    if smooth_v:
        mean = tl.load(mean_ptr + pid_b * d + offs_d, mask=offs_d < d, other=0.0)
        vals = vals - mean[None, :]

    q = vals / scale[None, :]
    q = q + 0.5 * tl.where(q >= 0, 1.0, -1.0)
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)

    q_ptrs = q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    tl.store(q_ptrs, q.to(tl.int8), mask=mask)


def _reshape_to_bmd(v: torch.Tensor, tensor_layout: str):
    # HND: [B, H, M, D] -> [B*H, M, D]
    # NHD: [B, M, H, D] -> [B*H, M, D]
    batch_size = v.shape[0]
    head_dim = v.shape[-1]
    if tensor_layout == "HND":
        num_kv_heads = v.shape[1]
        kv_len = v.shape[2]
        return v.view(batch_size * num_kv_heads, kv_len, head_dim), batch_size, num_kv_heads, kv_len
    num_kv_heads = v.shape[2]
    kv_len = v.shape[1]
    return (
        v.permute(0, 2, 1, 3).contiguous().view(batch_size * num_kv_heads, kv_len, head_dim),
        batch_size,
        num_kv_heads,
        kv_len,
    )


def _reshape_from_bmd(q: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor, layout: str, bs: int, h: int, m: int):
    if layout == "HND":
        q_out = q.view(bs, h, m, q.shape[-1])
        s_out = scale.view(bs, h, q.shape[-1])
        m_out = mean.view(bs, h, q.shape[-1]) if mean is not None else None
        return q_out, s_out, m_out
    q_out = q.view(bs, h, m, q.shape[-1]).permute(0, 2, 1, 3).contiguous()
    s_out = scale.view(bs, h, q.shape[-1])
    m_out = mean.view(bs, h, q.shape[-1]) if mean is not None else None
    return q_out, s_out, m_out


def _per_channel_int8_triton(v_bmd: torch.Tensor, scale_max: float, smooth_v: bool):
    if v_bmd.device.type != "cuda":
        raise ValueError("Triton quantization requires CUDA tensor.")
    v_bmd = v_bmd.contiguous()
    b, m, d = v_bmd.shape

    q = torch.empty_like(v_bmd, dtype=torch.int8)
    scale = torch.empty((b, d), dtype=torch.float32, device=v_bmd.device)
    mean = torch.empty((b, d), dtype=torch.float32, device=v_bmd.device) if smooth_v else None
    mean_ptr = mean if smooth_v else scale

    block_d = 128 if d >= 128 else triton.next_power_of_2(d)
    block_m_stats = 128
    grid_stats = (b, triton.cdiv(d, block_d))
    _stats_kernel[grid_stats](
        v_bmd,
        scale,
        mean_ptr,
        b,
        m,
        d,
        v_bmd.stride(0),
        v_bmd.stride(1),
        v_bmd.stride(2),
        scale_max,
        smooth_v=smooth_v,
        block_m=block_m_stats,
        block_d=block_d,
        num_warps=4,
    )

    block_m_q = 128
    grid_q = (b, triton.cdiv(m, block_m_q), triton.cdiv(d, block_d))
    _quant_kernel[grid_q](
        v_bmd,
        q,
        scale,
        mean_ptr,
        b,
        m,
        d,
        v_bmd.stride(0),
        v_bmd.stride(1),
        v_bmd.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        smooth_v=smooth_v,
        block_m=block_m_q,
        block_d=block_d,
        num_warps=4,
    )
    return q, scale, mean


def per_channel_int8_torch(
    v: torch.Tensor,
    tensor_layout: str = "HND",
    scale_max: float = 127.0,
    smooth_v: bool = True,
):
    """
    Reference PyTorch implementation with the same math as Triton path.
    """
    assert tensor_layout in ("HND", "NHD"), f"Unsupported tensor layout: {tensor_layout}"
    assert v.dtype in (torch.float16, torch.bfloat16), f"Unsupported dtype: {v.dtype}"

    v_bmd, bs, h, m = _reshape_to_bmd(v, tensor_layout)
    v_f32 = v_bmd.float()

    mean = v_f32.mean(dim=1) if smooth_v else None
    centered = v_f32 - mean[:, None, :] if smooth_v else v_f32
    abs_max = centered.abs().amax(dim=1)
    scale = torch.clamp(abs_max / scale_max, min=_EPS)

    q = centered / scale[:, None, :]
    q = q + 0.5 * torch.where(q >= 0, 1.0, -1.0)
    q = torch.clamp(q, min=-127, max=127).to(torch.int8)

    return _reshape_from_bmd(q, scale, mean, tensor_layout, bs, h, m)


def per_channel_int8(
    v: torch.Tensor,
    tensor_layout: str = "HND",
    scale_max: float = 127.0,
    smooth_v: bool = True,
):
    """
    quantize tensor `v` to int8 with per channel quantization.
    The quantization is done per channel, with the scale value and smooth factor calculated per channel.

    Parameters
    ----------
    v : torch.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    scale_max : float
        The maximum scale value for the quantization. Default is 127.0 (upper bound of INT8 data format).

    smooth_v : bool
        Whether to smooth the quantized tensor. Default is True.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        A tuple containing:
        - The quantized tensor `v_int8`. Shape:
            - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``, with `int8` dtype.
            - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``, with `int8` dtype.
        - The scale tensor of `v`. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.

    Note
    ----
    - The tensors `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - The returned mean tensor will be None if `smooth_v` is False. Otherwise it will have dtype ``torch.float32``.
    """
    assert tensor_layout in ("HND", "NHD"), f"Unsupported tensor layout: {tensor_layout}"
    assert v.dtype in (torch.float16, torch.bfloat16), f"Unsupported dtype: {v.dtype}"
    assert v.is_cuda, "Input must be CUDA tensor for Triton kernel."

    v_bmd, bs, h, m = _reshape_to_bmd(v, tensor_layout)
    q, scale, mean = _per_channel_int8_triton(v_bmd, scale_max=scale_max, smooth_v=smooth_v)
    return _reshape_from_bmd(q, scale, mean, tensor_layout, bs, h, m)


def _dequantize(q: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor, tensor_layout: str):
    if tensor_layout == "HND":
        scale_view = scale.unsqueeze(2)  # [B,H,1,D]
        mean_view = mean.unsqueeze(2) if mean is not None else 0.0
    else:
        scale_view = scale.unsqueeze(1)  # [B,1,H,D]
        mean_view = mean.unsqueeze(1) if mean is not None else 0.0
    return q.float() * scale_view + mean_view


def run_accuracy_and_perf_demo():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required.")

    device = "cuda"
    torch.manual_seed(0)

    test_cases = [
        ("HND", True, (4, 32, 2048, 128), torch.float16),
        ("HND", False, (4, 32, 2048, 128), torch.float16),
        ("NHD", True, (4, 2048, 32, 128), torch.bfloat16),
        ("NHD", False, (4, 2048, 32, 128), torch.bfloat16),
    ]

    print("=== Accuracy + Performance Comparison ===")
    for layout, smooth, shape, dtype in test_cases:
        v = torch.randn(shape, device=device, dtype=dtype) * 2.5 + 0.3

        # warmup
        for _ in range(10):
            per_channel_int8(v, tensor_layout=layout, smooth_v=smooth)
            per_channel_int8_torch(v, tensor_layout=layout, smooth_v=smooth)
        torch.cuda.synchronize()

        q_t, s_t, m_t = per_channel_int8(v, tensor_layout=layout, smooth_v=smooth)
        q_ref, s_ref, m_ref = per_channel_int8_torch(v, tensor_layout=layout, smooth_v=smooth)

        max_q_diff = (q_t.float() - q_ref.float()).abs().max().item()
        max_s_diff = (s_t - s_ref).abs().max().item()
        max_m_diff = 0.0 if (m_t is None and m_ref is None) else (m_t - m_ref).abs().max().item()

        deq_t = _dequantize(q_t, s_t, m_t, layout)
        deq_ref = _dequantize(q_ref, s_ref, m_ref, layout)
        deq_err_t = (deq_t - v.float()).abs().max().item()
        deq_err_ref = (deq_ref - v.float()).abs().max().item()

        iters = 100
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            per_channel_int8_torch(v, tensor_layout=layout, smooth_v=smooth)
        e1.record()
        torch.cuda.synchronize()
        torch_ms = e0.elapsed_time(e1) / iters

        e0.record()
        for _ in range(iters):
            per_channel_int8(v, tensor_layout=layout, smooth_v=smooth)
        e1.record()
        torch.cuda.synchronize()
        triton_ms = e0.elapsed_time(e1) / iters

        print(f"\nlayout={layout}, smooth_v={smooth}, dtype={dtype}, shape={shape}")
        print(f"  max |q_triton - q_torch|      = {max_q_diff:.6f}")
        print(f"  max |scale_triton - scale_torch| = {max_s_diff:.6e}")
        if smooth:
            print(f"  max |mean_triton - mean_torch|   = {max_m_diff:.6e}")
        print(f"  max dequant err (triton)      = {deq_err_t:.6f}")
        print(f"  max dequant err (torch)       = {deq_err_ref:.6f}")
        print(f"  time torch  : {torch_ms:.4f} ms")
        print(f"  time triton : {triton_ms:.4f} ms")
        print(f"  speedup     : {torch_ms / triton_ms:.2f}x")


if __name__ == "__main__":
    run_accuracy_and_perf_demo()