import torch
import triton
import triton.language as tl

@triton.jit
def _per_channel_int8_kernel(
    V_ptr, Out_ptr, Scale_ptr, Mean_ptr,
    stride_v_b, stride_v_0, stride_v_1, stride_v_d,
    stride_o_b, stride_o_0, stride_o_1, stride_o_d,
    stride_s_b, stride_s_0, stride_s_1,
    B, dim0, dim1, D,
    SMOOTH_V: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # 1D grid，覆盖 Batch, dim0 (Head/Seq) 和 dim1 (Seq/Head)
    pid = tl.program_id(0)
    
    # 根据 pid 反推 B, dim0 和 dim1 的索引
    # 假设展开顺序为 B -> dim0 -> dim1
    idx1 = pid % dim1
    pid_b_dim0 = pid // dim1
    idx0 = pid_b_dim0 % dim0
    idx_b = pid_b_dim0 // dim0
    
    # 计算当前 batch/head/seq 对应的起始内存偏移量
    v_offset = idx_b * stride_v_b + idx0 * stride_v_0 + idx1 * stride_v_1
    o_offset = idx_b * stride_o_b + idx0 * stride_o_0 + idx1 * stride_o_1
    s_offset = idx_b * stride_s_b + idx0 * stride_s_0 + idx1 * stride_s_1
    
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


def per_channel_int8_triton(
    v: torch.Tensor, 
    tensor_layout: str = "BHND", 
    smooth_v: bool = True
):
    """
    针对 4D 张量的最后一个维度(head_dim)进行 per-channel INT8 量化。
    支持格式如 'BHND' (Batch, Head, Seq, Dim) 或 'BNHD' (Batch, Seq, Head, Dim)。
    """
    assert v.dim() == 4, "输入张量必须是 4D 的 (包含 batch 维度)"
    assert v.is_cuda, "输入张量必须在 GPU 上"
    assert tensor_layout in ["BHND", "BNHD"], "布局参数应为包含 Batch 维度的 'BHND' 或 'BNHD'"
    
    # 提取 4D 形状与步长
    B, dim0, dim1, D = v.shape
    stride_v_b, stride_v_0, stride_v_1, stride_v_d = v.stride()
    
    # 分配输出内存
    v_quant = torch.empty_like(v, dtype=torch.int8)
    stride_o_b, stride_o_0, stride_o_1, stride_o_d = v_quant.stride()
    
    # 分配 Scale 和 Mean 内存，保留原来的 4D 形状但最后一个维度为 1
    stats_shape = (B, dim0, dim1, 1)
    scale = torch.empty(stats_shape, device=v.device, dtype=torch.float32)
    stride_s_b, stride_s_0, stride_s_1, _ = scale.stride()
    
    mean = None
    if smooth_v:
        mean = torch.empty(stats_shape, device=v.device, dtype=torch.float32)

    # 寻找大于等于 D 的下一个 2 的幂次方，用于分配 Shared Memory 大小
    BLOCK_D = triton.next_power_of_2(D)
    
    # Grid 设为前三个维度的乘积 (B * dim0 * dim1)
    # 不使用 3D grid 是为了避免 seq_len 超过 65535 时的 CUDA block 限制
    grid = lambda meta: (B * dim0 * dim1, )
    
    # 启动 Kernel
    _per_channel_int8_kernel[grid](
        v, v_quant, scale, mean,
        stride_v_b, stride_v_0, stride_v_1, stride_v_d,
        stride_o_b, stride_o_0, stride_o_1, stride_o_d,
        stride_s_b, stride_s_0, stride_s_1,
        B, dim0, dim1, D,
        SMOOTH_V=smooth_v,
        BLOCK_D=BLOCK_D
    )
    
    if smooth_v:
        return v_quant, scale, mean
    else:
        return v_quant, scale, None
      
def per_channel_int8_torch(v: torch.Tensor, smooth_v: bool = True):
    """
    纯 PyTorch 实现的 per-channel INT8 量化，用于精度和性能基准测试。
    """
    # 【修复1：对齐计算精度】在 Triton 中我们将数据 load 成了 FP32 进行计算
    # 所以在 PyTorch 中也先将其转换为 float32，消除 FP16 累加误差
    v_fp32 = v.float()
    
    # 保持原有的维度，在最后一个维度 (D) 上操作
    if smooth_v:
        # 计算均值并保持维度以便广播 (B, dim0, dim1, 1)
        mean = v_fp32.mean(dim=-1, keepdim=True)
        v_fp32 = v_fp32 - mean
    else:
        mean = None

    # 计算 Scale
    abs_v = v_fp32.abs()
    max_val = abs_v.max(dim=-1, keepdim=True)[0]
    scale = torch.clamp(max_val / 127.0, min=1e-12)

    # 量化
    v_quant = v_fp32 / scale
    
    # 【修复2：对齐舍入规则】
    # Triton 的逻辑是: v_quant >= 0 则 +0.5, 否则 -0.5, 然后类型转换时直接向零截断 (trunc)
    # 相当于传统的四舍五入。我们用 torch.trunc 和 torch.sign 来等价替代 torch.round
    v_quant = torch.trunc(v_quant + torch.sign(v_quant) * 0.5)
    
    # 限制在 [-128, 127]
    v_quant = torch.clamp(v_quant, -128.0, 127.0)
    
    # 转换为 int8
    v_quant_int8 = v_quant.to(torch.int8)

    # 返回时，为了匹配外部类型，你可以选择将 mean 和 scale 保持 float32，
    # 或者转回和输入一样的类型（Triton 中是以 fp32 存储出来的，所以这里直接返回 fp32 没问题）
    return v_quant_int8, scale, mean
  
def test_precision():
    print("=== 开始精度测试 ===")
    torch.manual_seed(42)
    B, H, N, D = 2, 8, 2048, 128
    
    # 初始化输入
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16) * 5.0
    
    # 运行 PyTorch 基准
    ref_quant, ref_scale, ref_mean = per_channel_int8_torch(v, smooth_v=True)
    
    # 运行 Triton 实现
    tri_quant, tri_scale, tri_mean = per_channel_int8_triton(v, tensor_layout="BHND", smooth_v=True)
    
    # 比较 Mean
    mean_diff = torch.max(torch.abs(ref_mean - tri_mean)).item()
    print(f"Mean 最大误差: {mean_diff:.6f}")
    
    # 比较 Scale
    scale_diff = torch.max(torch.abs(ref_scale - tri_scale)).item()
    print(f"Scale 最大误差: {scale_diff:.6f}")
    
    # 比较 Quantized Tensor (INT8)
    # 由于硬件浮点累加顺序差异和舍入方式差异，可能会有极少数 +/- 1 的误差
    diff_quant = torch.abs(ref_quant.float() - tri_quant.float())
    exact_match_ratio = (diff_quant == 0).float().mean().item() * 100
    max_quant_diff = diff_quant.max().item()
    
    print(f"INT8 结果完全匹配率: {exact_match_ratio:.2f}%")
    print(f"INT8 结果最大绝对误差: {max_quant_diff}")
    assert max_quant_diff <= 1.0, "量化结果误差过大，请检查逻辑！"
    print("精度测试通过！\n")


# ---------------------------------------------------------------------------
# 性能测试 (Benchmark)
# 使用 triton.testing.perf_report 绘制性能曲线
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # 将 Sequence Length 作为 X 轴
        x_vals=[256 * i for i in range(1, 17)],  # N 从 256 到 4096
        line_arg='provider',  # 使用不同的 provider 画线 (PyTorch vs Triton)
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',  # Y 轴显示内存带宽利用率
        plot_name='per-channel-int8-quantization-performance',
        args={'B': 4, 'H': 32, 'D': 128}  # 固定其他维度
    )
)
def benchmark(B, H, N, D, provider):
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    
    # 定义测试的闭包
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        fn = lambda: per_channel_int8_torch(v, smooth_v=True)
    else:
        fn = lambda: per_channel_int8_triton(v, tensor_layout="BHND", smooth_v=True)

    # 预热并运行测速
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    # 计算显存读写量 (GB)
    # 读取: v (FP16, 2 bytes)
    # 写入: v_quant (INT8, 1 byte) + scale (FP32, 4 bytes/D) + mean (FP32, 4 bytes/D)
    num_elements = B * H * N * D
    num_stats_elements = B * H * N
    
    gb = (num_elements * 2 + num_elements * 1 + num_stats_elements * 4 + num_stats_elements * 4) / 1e9
    
    # 返回吞吐量 GB/s
    return gb / (ms / 1000), gb / (max_ms / 1000), gb / (min_ms / 1000)

if __name__ == "__main__":
    # 1. 运行精度测试
    test_precision()
    
    # 2. 运行性能测试
    print("=== 开始性能测试 ===")
    benchmark.run(print_data=True, show_plots=False)