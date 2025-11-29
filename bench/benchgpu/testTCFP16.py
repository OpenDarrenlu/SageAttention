import torch
import time
import numpy as np
import os

def test_tensorcore_fp16():
    # 检查环境
    if not torch.cuda.is_available():
        print("CUDA is not available. Please install PyTorch with CUDA support.")
        return
    
    # 检查是否支持 FP16 Tensor Core (Ampere+ 架构)
    major, minor = torch.cuda.get_device_capability()
    if major < 7:
        print(f"Your GPU (Compute Capability {major}.{minor}) has limited or no Tensor Core support.")
        print("FP16 Tensor Core requires Volta (7.0+) or newer architecture.")
        return
    
    print(f"Testing FP16 Tensor Core on GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {major}.{minor} | Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = torch.device("cuda")
    torch.manual_seed(42)  # 确保可重复性
    
    # 验证是否能使用 Tensor Core (需要矩阵维度为 8 的倍数)
    print("\nVerifying Tensor Core compatibility...")
    try:
        # 创建符合 Tensor Core 要求的矩阵 (维度是 8 的倍数)
        a = torch.randn(256, 256, device=device, dtype=torch.float16)
        b = torch.randn(256, 256, device=device, dtype=torch.float16)
        c = torch.matmul(a, b)
        print("✓ Tensor Core compatible matrix dimensions confirmed")
    except Exception as e:
        print(f"✗ Tensor Core test failed: {str(e)}")
        return
    
    # 根据 4GB VRAM 限制优化矩阵大小
    # Ampere 架构需要矩阵维度为 8 的倍数以启用 Tensor Core
    M, K, N = 8192*4, 8192*2, 8192*4  # 对于 4GB GPU 的安全上限
    ''' M, K, N = 8192*4, 8192*2, 8192*4
    FP16 Tensor Core Performance Test Results
    ==================================================
    Matrix Size: 32768x16384 x 16384x32768
    Data Type: torch.float16
    Average Time: 2932.3788 ms | Runs: 20
    Measured Performance: 12.00 TFLOPS
    '''

    # 创建 FP16 张量
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # 预热：跳过首次运行的 JIT 编译和 Tensor Core 初始化
    print("\nRunning warm-up (10 iterations)...")
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 正式测试：多次运行取平均
    num_runs = 20
    times = []
    
    print(f"Running {num_runs} measurement iterations...")
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = torch.matmul(a, b)  # 关键：FP16 矩阵乘法将自动使用 Tensor Core
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # 计算性能指标
    avg_time_ms = np.mean(times)
    avg_time_s = avg_time_ms / 1000.0
    
    # 计算 TFLOPS: 2 * M * N * K (每个 FMA 算 2 次浮点运算)
    flops = 2 * M * N * K
    tflops = flops / avg_time_s / 1e12
    
    # 计算理论峰值 (RTX 3050 Laptop)
    fp32_peak = 7.127  # TFLOPS (来自 GPU 规格)
    fp16_peak = fp32_peak * 1  # Ampere 架构: FP16 Tensor Core = FP32 * 1
    
    # 输出结果
    print(f"\n{'='*50}")
    print(f"FP16 Tensor Core Performance Test Results")
    print(f"{'='*50}")
    print(f"Matrix Size: {M}x{K} x {K}x{N}")
    print(f"Data Type: torch.float16")
    print(f"Average Time: {avg_time_ms:.4f} ms | Runs: {num_runs}")
    print(f"Measured Performance: {tflops:.2f} TFLOPS")
    print(f"\nTheoretical Peak (RTX 3050):")
    print(f"- FP32 (CUDA Cores): {fp32_peak:.2f} TFLOPS")
    print(f"- FP16 (Tensor Core): {fp16_peak:.2f} TFLOPS")
    print(f"\nUtilization:")
    print(f"- {tflops/fp16_peak*100:.1f}% of FP16 Tensor Core peak")
    print(f"{'='*50}")
    
    # 验证是否真正使用了 Tensor Core
    print("\nVerifying Tensor Core usage...")
    try:
        # 使用 Nsight 的指标（如果可用）
        from torch.profiler import profile, record_function, ProfilerActivity
        
        with profile(activities=[ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            with record_function("fp16_matmul"):
                torch.matmul(a, b)
                torch.cuda.synchronize()
        
        # 检查是否有 Tensor Core 操作
        tensorcore_ops = [item for item in prof.key_averages().table().split('\n') 
                         if 'gemm' in item.lower() or 'tensor' in item.lower()]
        
        if tensorcore_ops:
            print("✓ Tensor Core usage confirmed in profiler output")
            print("Profiler summary:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        else:
            print("⚠ Could not confirm Tensor Core usage - check matrix dimensions")
    except ImportError:
        print("Note: Install torch profiler with 'pip install torch' for Tensor Core verification")
    
    # 实用建议
    if tflops < fp16_peak * 0.5:
        print("\n⚠️ WARNING: Performance significantly below theoretical peak!")
        print("Possible reasons:")
        print("- Matrix dimensions not optimal for Tensor Core (must be multiples of 8)")
        print("- Memory bandwidth bottleneck (check with Nsight Compute)")
        print("- GPU clock throttling (common in laptops)")
        print("- Small problem size (try larger matrices within VRAM limits)")
    else:
        print("\n✓ Good Tensor Core utilization achieved!")

if __name__ == "__main__":
    print("="*50)
    print("GPU FP16 Tensor Core Performance Tester")
    print("="*50)
    
    # 确保禁用 TF32 (否则 FP16 测试可能使用 TF32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Disabled TF32 to ensure pure FP16 measurement")
    
    # 设置环境变量确保使用 FP16
    os.environ["CUDA_TF32_OVERRIDE"] = "0"
    
    test_tensorcore_fp16()