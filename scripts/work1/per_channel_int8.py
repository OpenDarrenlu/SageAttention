import torch
import triton
import triton.language as tl

def per_channel_int8(
    v: torch.Tensor,
    tensor_layout: str ="HND",
    scale_max: float = 127.0,
    smooth_v: bool = True
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
    # 验证输入参数
    assert tensor_layout in ["HND", "NHD"], f"Unsupported tensor layout: {tensor_layout}"
    assert v.dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {v.dtype}"
    
    # 保存原始形状
    orig_shape = v.shape
    batch_size = orig_shape[0]
    head_dim = orig_shape[-1]
    
    if tensor_layout == "HND":
        # HND: [batch_size, num_kv_heads, kv_len, head_dim]
        num_kv_heads = orig_shape[1]
        kv_len = orig_shape[2]
        # 重塑为 [batch_size * num_kv_heads, kv_len, head_dim] 以便按通道计算
        v_reshaped = v.view(batch_size * num_kv_heads, kv_len, head_dim)
    else:
        # NHD: [batch_size, kv_len, num_kv_heads, head_dim]
        kv_len = orig_shape[1]
        num_kv_heads = orig_shape[2]
        # 重塑为 [batch_size * num_kv_heads, kv_len, head_dim] 以便按通道计算
        v_reshaped = v.permute(0, 2, 1, 3).reshape(batch_size * num_kv_heads, kv_len, head_dim)
    
    # 计算统计信息和量化
    v_quant, scale, mean = _per_channel_int8_triton(v_reshaped, scale_max, smooth_v)
    
    # 重塑回原始形状
    if tensor_layout == "HND":
        v_quant = v_quant.view(batch_size, num_kv_heads, kv_len, head_dim)
        scale = scale.view(batch_size, num_kv_heads, head_dim)
        if smooth_v:
            mean = mean.view(batch_size, num_kv_heads, head_dim)
    else:
        v_quant = v_quant.view(batch_size, num_kv_heads, kv_len, head_dim).permute(0, 2, 1, 3)
        scale = scale.view(batch_size, num_kv_heads, head_dim)
        if smooth_v:
            mean = mean.view(batch_size, num_kv_heads, head_dim)
    
    return v_quant, scale, mean


@triton.jit
def _get_stats_kernel(
    V_ptr, Scale_ptr, Mean_ptr,
    B, M, D,
    stride_vb, stride_vm, stride_vd,
    SCALE_MAX: tl.constexpr,
    SMOOTH_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    计算每个通道的均值和最大值，以确定缩放因子
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # 计算批次和通道的偏移量
    b_offsets = pid_b
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    # 掩码，确保不超出边界
    d_mask = d_offsets < D
    
    if SMOOTH_V:
        # --- 计算 Mean ---
        sum_vals = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            m_mask = m_offsets < M
            
            # 加载数据
            v_ptrs = V_ptr + b_offsets * stride_vb + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            sum_vals += tl.sum(v, axis=0)
        
        mean = sum_vals / M
        
        # --- 中心化并计算 Max(Abs) ---
        max_abs = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            m_mask = m_offsets < M
            
            # 加载数据
            v_ptrs = V_ptr + b_offsets * stride_vb + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 中心化
            v_centered = v - mean[None, :]
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(v_centered), axis=0))
        
        # 计算缩放因子
        scale = max_abs / SCALE_MAX
    else:
        # --- 只计算 Max(Abs) ---
        max_abs = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_offsets = m + tl.arange(0, BLOCK_M)
            m_mask = m_offsets < M
            
            # 加载数据
            v_ptrs = V_ptr + b_offsets * stride_vb + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(v), axis=0))
        
        # 计算缩放因子
        scale = max_abs / SCALE_MAX
        mean = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    # 确保缩放因子不为零
    scale = tl.maximum(scale, 1e-9)
    
    # 存储结果
    scale_ptr = Scale_ptr + b_offsets * D + d_offsets
    tl.store(scale_ptr, scale, mask=d_mask)
    
    if SMOOTH_V:
        mean_ptr = Mean_ptr + b_offsets * D + d_offsets
        tl.store(mean_ptr, mean, mask=d_mask)


@triton.jit
def _apply_quant_kernel(
    V_ptr, Quant_ptr, Scale_ptr, Mean_ptr,
    B, M, D,
    stride_vb, stride_vm, stride_vd,
    stride_qb, stride_qm, stride_qd,
    SMOOTH_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    应用量化到输入张量
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    # 计算偏移量
    b_offsets = pid_b
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    
    # 掩码，确保不超出边界
    m_mask = m_offsets < M
    d_mask = d_offsets < D
    mask = m_mask[:, None] & d_mask[None, :]
    
    # 加载数据
    v_ptrs = V_ptr + b_offsets * stride_vb + m_offsets[:, None] * stride_vm + d_offsets[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=mask).to(tl.float32)
    
    # 加载缩放因子
    scale_ptr = Scale_ptr + b_offsets * D + d_offsets
    scale = tl.load(scale_ptr, mask=d_mask)
    
    # 加载均值并中心化（如果需要）
    if SMOOTH_V:
        mean_ptr = Mean_ptr + b_offsets * D + d_offsets
        mean = tl.load(mean_ptr, mask=d_mask)
        v = v - mean[None, :]
    
    # 应用量化
    q = v / scale[None, :]
    q += 0.5 * tl.where(q >= 0, 1, -1)  # 模拟四舍五入
    
    # 存储量化结果
    q_ptrs = Quant_ptr + b_offsets * stride_qb + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    tl.store(q_ptrs, q.to(tl.int8), mask=mask)


def _per_channel_int8_triton(v: torch.Tensor, scale_max: float, smooth_v: bool):
    """
    使用Triton实现的通道级int8量化
    """
    v = v.contiguous()
    B, M, D = v.shape  # B: batch_size * num_kv_heads, M: kv_len, D: head_dim
    
    # 分配输出张量
    scale = torch.empty((B, D), dtype=torch.float32, device=v.device)
    mean = torch.empty((B, D), dtype=torch.float32, device=v.device) if smooth_v else None
    v_quant = torch.empty_like(v, dtype=torch.int8)
    
    # 配置Triton内核参数
    BLOCK_M_1 = 1024
    BLOCK_D_1 = triton.next_power_of_2(D) if D <= 64 else 64
    grid_1 = (B, triton.cdiv(D, BLOCK_D_1))
    
    dummy_mean = mean if smooth_v else scale
    
    # 计算统计信息
    _get_stats_kernel[grid_1](
        v, scale, dummy_mean,
        B, M, D,
        v.stride(0), v.stride(1), v.stride(2),
        SCALE_MAX=scale_max,
        SMOOTH_V=smooth_v,
        BLOCK_M=BLOCK_M_1,
        BLOCK_D=BLOCK_D_1,
    )
    
    # 应用量化
    BLOCK_M_2 = 128
    BLOCK_D_2 = BLOCK_D_1
    grid_2 = (B, triton.cdiv(M, BLOCK_M_2), triton.cdiv(D, BLOCK_D_2))
    
    _apply_quant_kernel[grid_2](
        v, v_quant, scale, dummy_mean,
        B, M, D,
        v.stride(0), v.stride(1), v.stride(2),
        v_quant.stride(0), v_quant.stride(1), v_quant.stride(2),
        SMOOTH_V=smooth_v,
        BLOCK_M=BLOCK_M_2,
        BLOCK_D=BLOCK_D_2,
    )
    
    return v_quant, scale, mean if smooth_v else None


def per_channel_int8_pytorch(
    v: torch.Tensor,
    tensor_layout: str ="HND",
    scale_max: float = 127.0,
    smooth_v: bool = True
):
    """
    使用PyTorch实现的通道级int8量化
    与Triton实现保持数学一致性
    """
    # 验证输入参数
    assert tensor_layout in ["HND", "NHD"], f"Unsupported tensor layout: {tensor_layout}"
    assert v.dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {v.dtype}"
    
    # 保存原始形状
    orig_shape = v.shape
    batch_size = orig_shape[0]
    head_dim = orig_shape[-1]
    
    if tensor_layout == "HND":
        # HND: [batch_size, num_kv_heads, kv_len, head_dim]
        num_kv_heads = orig_shape[1]
        kv_len = orig_shape[2]
        # 重塑为 [batch_size * num_kv_heads, kv_len, head_dim] 以便按通道计算
        v_reshaped = v.view(batch_size * num_kv_heads, kv_len, head_dim)
    else:
        # NHD: [batch_size, kv_len, num_kv_heads, head_dim]
        kv_len = orig_shape[1]
        num_kv_heads = orig_shape[2]
        # 重塑为 [batch_size * num_kv_heads, kv_len, head_dim] 以便按通道计算
        v_reshaped = v.permute(0, 2, 1, 3).reshape(batch_size * num_kv_heads, kv_len, head_dim)
    
    # 计算均值（如果需要）
    mean = None
    if smooth_v:
        mean = v_reshaped.mean(dim=1, keepdim=False).float()
        # 中心化
        v_centered = v_reshaped - mean.unsqueeze(1)
    else:
        v_centered = v_reshaped
    
    # 计算每个通道的绝对值最大值
    abs_max = torch.max(torch.abs(v_centered), dim=1, keepdim=False)[0]
    
    # 计算缩放因子
    scale = abs_max / scale_max
    scale = torch.clamp(scale, min=1e-9)
    
    # 应用量化
    v_quant = v_centered / scale.unsqueeze(1)
    # 模拟四舍五入
    v_quant += 0.5 * torch.where(v_quant >= 0, 1, -1)
    v_quant = v_quant.to(torch.int8)
    
    # 重塑回原始形状
    if tensor_layout == "HND":
        v_quant = v_quant.view(batch_size, num_kv_heads, kv_len, head_dim)
        scale = scale.view(batch_size, num_kv_heads, head_dim)
        if smooth_v:
            mean = mean.view(batch_size, num_kv_heads, head_dim)
    else:
        v_quant = v_quant.view(batch_size, num_kv_heads, kv_len, head_dim).permute(0, 2, 1, 3)
        scale = scale.view(batch_size, num_kv_heads, head_dim)
        if smooth_v:
            mean = mean.view(batch_size, num_kv_heads, head_dim)
    
    return v_quant, scale, mean


if __name__ == "__main__":
    import time
    
    # 配置测试参数
    batch_size = 4
    num_kv_heads = 32
    kv_len = 2048
    head_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing on device: {device}")
    print(f"Tensor dimensions: batch_size={batch_size}, num_kv_heads={num_kv_heads}, kv_len={kv_len}, head_dim={head_dim}")
    
    # 生成测试数据
    v_hnd = torch.randn((batch_size, num_kv_heads, kv_len, head_dim), device=device, dtype=torch.float16) * 2.5 + 0.5
    v_nhd = torch.randn((batch_size, kv_len, num_kv_heads, head_dim), device=device, dtype=torch.float16) * 2.5 + 0.5
    
    # 测试不同的布局和smooth_v设置
    test_cases = [
        ("HND", True),
        ("HND", False),
        ("NHD", True),
        ("NHD", False)
    ]
    
    for layout, smooth in test_cases:
        print(f"\n=== Testing layout={layout}, smooth_v={smooth} ===")
        
        # 选择测试数据
        v = v_hnd if layout == "HND" else v_nhd
        
        # 计算PyTorch版本
        start = time.time()
        v_q_pytorch, s_pytorch, m_pytorch = per_channel_int8_pytorch(v, tensor_layout=layout, smooth_v=smooth)
        torch.cuda.synchronize() if device == "cuda" else None
        pytorch_time = (time.time() - start) * 1000
        
        # 计算Triton版本
        start = time.time()
        v_q_triton, s_triton, m_triton = per_channel_int8(v, tensor_layout=layout, smooth_v=smooth)
        torch.cuda.synchronize() if device == "cuda" else None
        triton_time = (time.time() - start) * 1000
        
        # 计算精度差异
        diff_q = torch.max(torch.abs(v_q_pytorch.float() - v_q_triton.float())).item()
        diff_s = torch.max(torch.abs(s_pytorch - s_triton)).item()
        
        print(f"PyTorch time: {pytorch_time:.3f} ms")
        print(f"Triton time: {triton_time:.3f} ms")
        print(f"Speedup: {pytorch_time / triton_time:.2f}x")
        print(f"Max quantization difference: {diff_q}")
        print(f"Max scale difference: {diff_s:.8f}")
        
        if smooth:
            diff_m = torch.max(torch.abs(m_pytorch - m_triton)).item()
            print(f"Max mean difference: {diff_m:.8f}")
        
        # 计算反量化误差
        if layout == "HND":
            # HND布局：需要在kv_len维度(维度2)添加广播维度
            s_pytorch_reshaped = s_pytorch.unsqueeze(2)
            s_triton_reshaped = s_triton.unsqueeze(2)
            if smooth:
                m_pytorch_reshaped = m_pytorch.unsqueeze(2)
                m_triton_reshaped = m_triton.unsqueeze(2)
                dequant_error_pytorch = (v_q_pytorch * s_pytorch_reshaped + m_pytorch_reshaped - v).abs().max().item()
                dequant_error_triton = (v_q_triton * s_triton_reshaped + m_triton_reshaped - v).abs().max().item()
            else:
                dequant_error_pytorch = (v_q_pytorch * s_pytorch_reshaped - v).abs().max().item()
                dequant_error_triton = (v_q_triton * s_triton_reshaped - v).abs().max().item()
        else:
            # NHD布局：需要在kv_len维度(维度1)添加广播维度
            s_pytorch_reshaped = s_pytorch.unsqueeze(1)
            s_triton_reshaped = s_triton.unsqueeze(1)
            if smooth:
                m_pytorch_reshaped = m_pytorch.unsqueeze(1)
                m_triton_reshaped = m_triton.unsqueeze(1)
                dequant_error_pytorch = (v_q_pytorch * s_pytorch_reshaped + m_pytorch_reshaped - v).abs().max().item()
                dequant_error_triton = (v_q_triton * s_triton_reshaped + m_triton_reshaped - v).abs().max().item()
            else:
                dequant_error_pytorch = (v_q_pytorch * s_pytorch_reshaped - v).abs().max().item()
                dequant_error_triton = (v_q_triton * s_triton_reshaped - v).abs().max().item()
        
        print(f"Max dequantization error (PyTorch): {dequant_error_pytorch:.8f}")
        print(f"Max dequantization error (Triton): {dequant_error_triton:.8f}")
    
    print("\nAll tests completed!")