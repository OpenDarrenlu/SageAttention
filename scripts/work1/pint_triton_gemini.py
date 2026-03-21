import torch
import triton
import triton.language as tl

# ==========================================
# 1. PyTorch 参考实现
# ==========================================
def bf16_to_fixed_point_pt(x_bf16: torch.Tensor):
    """
    原始 PyTorch 实现 (作为 Reference)
    """
    if x_bf16.dtype != torch.bfloat16:
        x_bf16 = x_bf16.to(torch.bfloat16)

    x_int = x_bf16.view(torch.int16).to(torch.int32) & 0xFFFF
    raw_exponent = (x_int >> 7) & 0xFF
    mantissa = x_int & 0x7F
    m_int = mantissa | 0x80
    exponent = raw_exponent - 127
    
    exponent_tensor = exponent.to(torch.int32)
    m_int_tensor = m_int.to(torch.int32)
    shift = exponent_tensor + 9
    out_fixed = torch.zeros_like(m_int_tensor, dtype=torch.int32)

    mask_left  = (exponent_tensor >= -9) & (exponent_tensor <= -1)
    mask_right = (exponent_tensor >= -15) & (exponent_tensor <= -10)

    out_fixed[mask_left] = m_int_tensor[mask_left] << shift[mask_left]
    out_fixed[mask_right] = m_int_tensor[mask_right] >> (-shift[mask_right])

    scale = 1 / (2 ** 16)
    return out_fixed, scale
# ==========================================
# Triton Kernel 实现 (已修复)
# ==========================================
@triton.jit
def bf16_to_fixed_point(x):
    # 1. 提取 bf16 的底层位模式
    # 【修复2】使用 .to(bitcast=True) 转为 int16，再向上转换为 int32 并用掩码截断
    # 这是跨 Triton 版本最稳健的比特级别提取方式
    x_int16 = x.to(tl.int16, bitcast=True)
    x_int32 = x_int16.to(tl.int32) & 0xFFFF

    # 提取8位 exponent 和 7位 mantissa
    raw_exponent = (x_int32 >> 7) & 0xFF
    mantissa = x_int32 & 0x7F
    
    # 加上默认的 1
    m_int = mantissa | 0x80
    
    # 还原真实 exponent
    exponent = raw_exponent - 127
    
    # 2. 定点化移位操作
    shift = exponent + 9

    # 划分掩码条件
    mask_left = (exponent >= -9) & (exponent <= -1)
    mask_right = (exponent >= -15) & (exponent <= -10)

    # 安全地计算移位量
    shift_left_amt = tl.where(mask_left, shift, 0)
    shift_right_amt = tl.where(mask_right, -shift, 0)

    # 执行移位
    val_left = m_int << shift_left_amt
    val_right = m_int >> shift_right_amt

    # 初始化输出为 0，并应用掩码选择正确的分支
    out = tl.zeros_like(x_int32)
    out = tl.where(mask_left, val_left, out)
    out = tl.where(mask_right, val_right, out)
    
    # 【修复1】必须显式 return，Triton 内部函数不能依赖形参进行 inplace 修改
    return out

@triton.jit
def bf16_to_fixed_point_kernel(
    x_ptr,          # 输入 bfloat16 指针
    out_ptr,        # 输出 int32 指针
    n_elements,     # 元素总数
    BLOCK_SIZE: tl.constexpr, # 线程块大小
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载 bf16 数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # 调用定点化函数 (接收返回值)
    out = bf16_to_fixed_point(x)
    
    # 3. 写回内存
    tl.store(out_ptr + offsets, out, mask=mask)

def bf16_to_fixed_point_triton(x_bf16: torch.Tensor):
    """
    Triton Kernel 的 Python 封装
    """
    assert x_bf16.is_cuda and x_bf16.is_contiguous(), "输入 Tensor 必须在 GPU 上且内存连续"
    if x_bf16.dtype != torch.bfloat16:
        print("输入 Tensor 类型不是 bfloat16，已转换为 bfloat16")
        x_bf16 = x_bf16.to(torch.bfloat16)

    n_elements = x_bf16.numel()
    
    # 分配输出内存
    out_fixed = torch.empty_like(x_bf16, dtype=torch.int32)
    
    # 自动调优的线程块大小，通常 1024 比较合适
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    bf16_to_fixed_point_kernel[grid](
        x_bf16, out_fixed, n_elements, 
        BLOCK_SIZE=1024
    )
    
    scale = 1 / (2 ** 16)
    return out_fixed, scale

# ==========================================
# 3. 精度校验 (Correctness Test)
# ==========================================
def test_correctness():
    print("--- 精度校验开始 ---")
    torch.manual_seed(42)
    
    # 生成测试数据 [0, 1) 的分布
    # 同时也加入一些极小值和边界值来测试不同的 shift 逻辑
    sizes = [5, 10, 1024, 100000]
    
    for N in sizes:
        x = torch.rand((N), dtype=torch.bfloat16, device='cuda')
        # 强制插入一些 edge cases (例如精确的 0.5, 以及很小的数触发 right shift)
        x[0] = 0.5
        x[1] = 0.0
        x[2] = 0.0001
        
        out_tr, scale = bf16_to_fixed_point_triton(x)
        # test out_tr * scale == x
        is_close = (out_tr * scale == x.to(torch.float32))
        if not is_close.all():
            print("Triton 输出与 PyTorch 输出不一致")
            print("max diff ", torch.max(torch.abs(out_tr * scale - x.to(torch.float32))))
            
        # print("Triton 输出:", out_tr)
        assert (out_tr < 2**16).all(), "Triton 输出超出范围"
        
        out_pt, _ = bf16_to_fixed_point_pt(x)
        is_correct = torch.equal(out_pt, out_tr)
        print(f"Size {N:<8}: {'✅ Passed' if is_correct else '❌ Failed'}")
        
        if not is_correct:
            print("差异数据：")
            diff_mask = out_pt != out_tr
            print("PyTorch:", out_pt[diff_mask][:5])
            print("Triton: ", out_tr[diff_mask][:5])
            print("Inputs: ", x[diff_mask][:5])
            return

    print("--- 精度校验通过 ---\n")


# ==========================================
# 4. 性能测试 (Performance Benchmark)
# ==========================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # X轴作为基准测试的参数名
        x_vals=[2**i for i in range(12, 28, 2)],  # 改变 Tensor 大小 (4K 到 128M 元素)
        x_log=True,  # X轴使用对数刻度
        line_arg='provider',  # Y轴的不同线条
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',  # Y轴衡量吞吐量
        plot_name='bf16-to-fixed-point-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.bfloat16)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: bf16_to_fixed_point_pt(x), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: bf16_to_fixed_point_triton(x), quantiles=quantiles)
    
    # 计算吞吐量 (GB/s)
    # 输入为 bf16 (2 bytes), 输出为 int32 (4 bytes)，总计 6 bytes / element
    gbps = lambda ms: (6 * size) / (ms * 1e6)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # 1. 验证准确性
    test_correctness()
    
    # 2. 跑性能基准测试
    print("--- 性能基准测试开始 (GB/s) ---")
    benchmark.run(print_data=True, show_plots=False)
    '''
    bf16-to-fixed-point-performance:
            size   PyTorch      Triton
    0      4096.0  0.010046    4.800000
    1     16384.0  0.037008   19.200000
    2     65536.0  0.160468   63.999998
    3    262144.0  0.650572  127.999995
    4   1048576.0  1.545320  166.054047
    5   4194304.0  3.468003  178.086953
    6  16777216.0  4.550058  181.038673
    7  67108864.0  1.692875  181.960210
    '''