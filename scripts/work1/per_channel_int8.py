import torch
import triton
import triton.language as tl

@triton.jit
def _per_channel_int8_kernel(
    V_ptr, Out_ptr, Scale_ptr, Mean_ptr,
    stride_v_0, stride_v_1, stride_v_d,
    stride_o_0, stride_o_1, stride_o_d,
    stride_s_0, stride_s_1,
    dim0, dim1, D,
    SMOOTH_V: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # 将前两个维度展平为 1D grid
    pid = tl.program_id(0)
    
    # 根据 pid 计算 dim0 和 dim1 的索引
    idx0 = pid // dim1
    idx1 = pid % dim1
    
    # 计算当前 batch/head 对应的起始内存偏移量
    v_offset = idx0 * stride_v_0 + idx1 * stride_v_1
    o_offset = idx0 * stride_o_0 + idx1 * stride_o_1
    s_offset = idx0 * stride_s_0 + idx1 * stride_s_1
    
    # 生成 D 维度的索引和掩码 (防止 head_dim 不是 2 的幂次)
    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D
    
    # 加载向量
    v_ptrs = V_ptr + v_offset + d_offsets * stride_v_d
    v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # 1. Smooth V (去均值)
    if SMOOTH_V:
        # 计算该通道的均值
        mean = tl.sum(v, axis=0) / D
        v = tl.where(mask, v - mean, 0.0)
        # 存储均值
        tl.store(Mean_ptr + s_offset, mean)
        
    # 2. 计算 Scale
    abs_v = tl.abs(v)
    max_val = tl.max(abs_v, axis=0)
    # 防止除以 0 的情况发生
    scale = tl.maximum(max_val / 127.0, 1e-12)
    tl.store(Scale_ptr + s_offset, scale)
    
    # 3. 量化并限制在 [-128, 127]
    v_quant = v / scale
    # 四舍五入 (Round)
    v_quant = tl.where(v_quant >= 0, v_quant + 0.5, v_quant - 0.5)
    v_quant = tl.maximum(-128.0, tl.minimum(127.0, v_quant))
    
    # 4. 转换为 int8 并写回显存
    v_quant_int8 = v_quant.to(tl.int8)
    o_ptrs = Out_ptr + o_offset + d_offsets * stride_o_d
    tl.store(o_ptrs, v_quant_int8, mask=mask)


def per_channel_int8(
    v: torch.Tensor, 
    tensor_layout: str = "HND", 
    smooth_v: bool = True
):
    """
    针对 3D 张量的最后一个维度(head_dim)进行 per-channel INT8 量化。
    """
    assert v.dim() == 3, "输入张量必须是 3D 的"
    assert v.is_cuda, "输入张量必须在 GPU 上"
    assert tensor_layout in ["HND", "NHD"], "布局参数必须为 'HND' 或 'NHD'"
    
    # 抽象出前两个维度，这样逻辑上无视 HND 或 NHD 差异
    dim0, dim1, D = v.shape
    stride_v_0, stride_v_1, stride_v_d = v.stride()
    
    # 分配输出内存
    v_quant = torch.empty_like(v, dtype=torch.int8)
    stride_o_0, stride_o_1, stride_o_d = v_quant.stride()
    
    # 分配 Scale 和 Mean 内存，保留原来的 3D 形状但最后一个维度为 1
    # 这样它们就能自然继承与 v 对应的内存排布形式
    stats_shape = (dim0, dim1, 1)
    scale = torch.empty(stats_shape, device=v.device, dtype=torch.float32)
    stride_s_0, stride_s_1, _ = scale.stride()
    
    mean = None
    if smooth_v:
        mean = torch.empty(stats_shape, device=v.device, dtype=torch.float32)

    # 寻找大于等于 D 的下一个 2 的幂次方，用于分配 Shared Memory 大小
    BLOCK_D = triton.next_power_of_2(D)
    
    # Grid 设为前两个维度的乘积 (通常对应 num_heads * seq_len)
    grid = lambda meta: (dim0 * dim1, )
    
    # 启动 Kernel
    _per_channel_int8_kernel[grid](
        v, v_quant, scale, mean,
        stride_v_0, stride_v_1, stride_v_d,
        stride_o_0, stride_o_1, stride_o_d,
        stride_s_0, stride_s_1,
        dim0, dim1, D,
        SMOOTH_V=smooth_v,
        BLOCK_D=BLOCK_D
    )
    
    if smooth_v:
        return v_quant, scale, mean
    else:
        return v_quant, scale
    
# 假设前面的 per_channel_int8 和 _per_channel_int8_kernel 已经定义在上下文中
# from your_module import per_channel_int8 

def per_channel_int8_pytorch(v: torch.Tensor, smooth_v: bool = True):
    """
    PyTorch 版本的参考实现，用于正确性对比
    """
    v_float = v.clone().float()
    
    mean = None
    if smooth_v:
        # 计算最后一个维度的均值
        mean = v_float.mean(dim=-1, keepdim=True)
        v_float = v_float - mean
        
    # 计算 Scale
    abs_v = v_float.abs()
    max_val = abs_v.max(dim=-1, keepdim=True)[0]
    scale = torch.clamp(max_val / 127.0, min=1e-12)
    
    # 量化
    v_quant = v_float / scale
    # 模拟 Triton 中使用的四舍五入逻辑
    v_quant = torch.where(v_quant >= 0, torch.floor(v_quant + 0.5), torch.ceil(v_quant - 0.5))
    v_quant = torch.clamp(v_quant, -128.0, 127.0).to(torch.int8)
    
    if smooth_v:
        return v_quant, scale, mean
    else:
        return v_quant, scale


def test_correctness():
    """
    测试 Triton Kernel 和 PyTorch 结果是否一致
    """
    print("=== 开始正确性测试 ===")
    torch.manual_seed(0)
    
    H, N, D = 32, 1024, 128
    
    for layout in ["HND", "NHD"]:
        for smooth in [True, False]:
            # 生成测试数据
            if layout == "HND":
                v = torch.randn((H, N, D), dtype=torch.float16, device='cuda')
            else: # NHD
                # 通过 permute 生成 NHD 的 stride
                v = torch.randn((N, H, D), dtype=torch.float16, device='cuda').permute(1, 0, 2)
            
            # PyTorch 结果
            pt_res = per_channel_int8_pytorch(v, smooth_v=smooth)
            # Triton 结果
            triton_res = per_channel_int8(v, tensor_layout=layout, smooth_v=smooth)
            
            v_quant_pt, scale_pt = pt_res[0], pt_res[1]
            v_quant_tr, scale_tr = triton_res[0], triton_res[1]
            
            # 验证 Scale (允许一定的浮点误差)
            scale_diff = torch.max(torch.abs(scale_pt - scale_tr)).item()
            assert scale_diff < 1e-4, f"Scale 计算不一致! 最大误差: {scale_diff}"
            
            # 验证均值 (如果启用了 smooth_v)
            if smooth:
                mean_pt, mean_tr = pt_res[2], triton_res[2]
                mean_diff = torch.max(torch.abs(mean_pt - mean_tr)).item()
                assert mean_diff < 1e-4, f"Mean 计算不一致! 最大误差: {mean_diff}"
            
            # 验证量化结果 (因为浮点精度差异，允许最大 1 的量化误差)
            quant_diff = torch.max(torch.abs(v_quant_pt.float() - v_quant_tr.float())).item()
            assert quant_diff <= 1, f"Quantization 计算不一致! 最大误差: {quant_diff}"
            
            print(f"✅ Layout: {layout}, Smooth: {smooth} 测试通过!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 用作 X 轴的变量：序列长度 seq_len
        x_vals=[128, 256, 512, 1024, 2048, 4096], # 测试不同的序列长度
        line_arg='provider', # 用作线条的变量（区分 PyTorch 和 Triton）
        line_vals=['pytorch', 'triton'], # 线条的值
        line_names=['PyTorch', 'Triton'], # 图例名称
        styles=[('blue', '-'), ('green', '-')], # 线条样式
        ylabel='Time (ms)', # Y 轴标签
        plot_name='per_channel_int8_performance', # 图表名称
        args={'H': 32, 'D': 128} # 其他固定参数：32个Head，head_dim 为 128
    )
)
def benchmark(N, H, D, provider):
    """
    性能基准测试
    """
    v = torch.randn((H, N, D), dtype=torch.float16, device='cuda')
    quantiles = [0.5, 0.2, 0.8] # 返回中位数，以及 20% 和 80% 的分位数
    
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: per_channel_int8_pytorch(v, smooth_v=True), 
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: per_channel_int8(v, tensor_layout="HND", smooth_v=True), 
            quantiles=quantiles
        )
    return ms, max_ms, min_ms

if __name__ == "__main__":
    # 1. 跑正确性测试
    test_correctness()
    print("\n")
    
    # 2. 跑性能测试
    print("=== 开始性能 Benchmark ===")
    benchmark.run(print_data=True, show_plots=False)