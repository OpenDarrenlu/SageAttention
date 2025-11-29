import torch
import time
import numpy as np

def test_tensorcore_int8():
    # 检查环境
    if not torch.cuda.is_available():
        print("CUDA is not available. Please install PyTorch with CUDA support.")
        return
    
    # 检查是否支持 INT8 Tensor Core (Ampere+ 架构)
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        print(f"Your GPU (Compute Capability {major}.{minor}) does NOT support INT8 Tensor Core.")
        print("INT8 Tensor Core requires Ampere (8.0+) or newer architecture.")
        return
    
    print(f"Testing INT8 Tensor Core on GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {major}.{minor} | Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 验证 PyTorch 版本 (需要 2.0+ 以获得稳定的 torch._int_mm)
    if not hasattr(torch, '_int_mm'):
        print("\nERROR: torch._int_mm not found. You need PyTorch 2.0+ for direct INT8 testing.")
        print("Suggestion: Upgrade PyTorch with 'pip install --upgrade torch'")
        print("Falling back to quantized linear layer test (less accurate)...\n")
        return test_quantized_linear()
    
    device = torch.device("cuda")
    torch.manual_seed(42)  # 确保可重复性
    
    # 优化参数：根据 4GB VRAM 限制调整矩阵大小
    # 目标：最大化计算密度，同时避免 OOM
    M, K, N = 8192*2, 8192*12, 8192*2  # 对于 4GB GPU，这是安全上限
    '''
    INT8 Tensor Core Performance Test Results
    ==================================================
    Matrix Size: 16384x98304 x 98304x16384
    Average Time: 4903.3910 ms | Runs: 20
    Measured Performance: 10.76 TOPS'''

    # 创建 INT8 张量 (Tensor Core 需要 torch.int8)
    a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
    
    # 预热：跳过首次运行的 JIT 编译开销
    for _ in range(10):
        torch._int_mm(a, b)
    torch.cuda.synchronize()
    
    # 正式测试：多次运行取平均
    num_runs = 20
    times = []
    
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch._int_mm(a, b)  # 关键：直接调用 INT8 Tensor Core 操作
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # 计算性能指标
    avg_time_ms = np.mean(times)
    avg_time_s = avg_time_ms / 1000.0
    
    # 计算 TOPS: 2 * M * N * K (每个乘加算 2 次操作)
    flops = 2 * M * N * K
    tops = flops / avg_time_s / 1e12
    
    # 计算理论峰值 (RTX 3050 Laptop)
    fp32_peak = 28.52  # TFLOPS (来自 GPU 规格)
    int8_peak = fp32_peak * 4  # 无稀疏性
    int8_sparse_peak = fp32_peak * 8  # 启用稀疏性
    
    # 输出结果
    print(f"\n{'='*50}")
    print(f"INT8 Tensor Core Performance Test Results")
    print(f"{'='*50}")
    print(f"Matrix Size: {M}x{K} x {K}x{N}")
    print(f"Average Time: {avg_time_ms:.4f} ms | Runs: {num_runs}")
    print(f"Measured Performance: {tops:.2f} TOPS")
    print(f"\nTheoretical Peak (RTX 3050):")
    print(f"- INT8 (no sparsity):  {int8_peak:.2f} TOPS")
    print(f"- INT8 (with sparsity): {int8_sparse_peak:.2f} TOPS")
    print(f"\nUtilization:")
    print(f"- {tops/int8_peak*100:.1f}% of non-sparse peak")
    print(f"- {tops/int8_sparse_peak*100:.1f}% of sparse peak")
    print(f"{'='*50}")
    
    # 实用建议
    if tops < int8_peak * 0.3:
        print("\n⚠️ WARNING: Performance significantly below theoretical peak!")
        print("Possible reasons:")
        print("- Memory bandwidth bottleneck (check with Nsight Compute)")
        print("- Small matrix size (try increasing M/N/K within VRAM limits)")
        print("- GPU clock throttling (common in laptops)")

if __name__ == "__main__":
    print("="*50)
    print("GPU INT8 Tensor Core Performance Tester")
    print("="*50)
    test_tensorcore_int8()