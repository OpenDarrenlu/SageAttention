import torch
import sys
import os

USE_FLASH_ATTN = True # whether to use flash attention or torch's scaled_dot_product_attention

def clear_l2_cache(device_id=0):
    """
    通过分配并访问一个超大张量来污染/驱逐 GPU L2 缓存。
    """
    # 1. 计算所需大小
    # 现代高性能 GPU 的 L2 Cache 大小通常在 6MB 到 48MB 之间。
    # 我们需要一个足够大的张量来保证驱逐效果，这里使用 4GB (float32)。
    # 4 * 1024 * 1024 * 1024 字节 / 4 字节/float32 = 1073741824 元素
    size_in_elements = 1024 * 1024 * 1024 # 10亿个元素 (约 4GB)

    # 2. 指定设备
    with torch.cuda.device(device_id):
        # 3. 创建超大张量
        # 使用 torch.empty() 避免初始化的时间消耗
        dummy_data = torch.empty(size_in_elements, dtype=torch.float32, device='cuda')
        
        # 4. 执行读写操作，强制访问内存并污染缓存
        # 必须确保这是一个执行了实际工作的 kernel
        dummy_data.fill_(1.0)  # 写入操作
        
        # 5. 再次读取或求和，确保数据被访问 (可选，但更保险)
        _ = dummy_data.sum()
        
        # 6. 关键：同步操作！确保污染 kernel 执行完毕
        torch.cuda.synchronize()
        
        # 7. 释放内存
        del dummy_data
        torch.cuda.empty_cache() # 帮助释放被 PyTorch 缓存的内存

def test_precision_comparison_simple(id=0):
    """
    测试精度对比脚本的简单版本，验证基本功能
    """
    print("\n=== 测试精度对比脚本（简单版本） ===")
    
    try:
        # 导入必要的模块
        from sageattention.core import sageattn
        from flash_attn.flash_attn_interface import flash_attn_func
        from torch.nn.functional import scaled_dot_product_attention
        import numpy as np
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 简单配置
        batch_size = 1
        seq_len = 4096
        n_heads = 12
        head_dim = 128
        dtype = torch.float16
        causal = False
        
        # 生成测试数（HND布局）
        # print(f"生成测试数据: batch_size={batch_size}, seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}")
        # q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        # k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        # v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        # 从 .pt 文件加载 q, k, v
        qkv = torch.load(f'qkv_tensors_30layers/qkv_tensors_{id}.pt')
        # import ipdb; ipdb.set_trace()
        q,k,v = qkv["query"], qkv["key"], qkv["value"]
        q = q.to(dtype=dtype, device='cuda')
        k = k.to(dtype=dtype, device='cuda')
        v = v.to(dtype=dtype, device='cuda')
        from fast_hadamard_transform import hadamard_transform
        # 对 q, k, v 进行 Hadamard 变换
        import math
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        
        # 转换为NHD布局
        q_nhd = q.permute(0, 2, 1, 3)
        k_nhd = k.permute(0, 2, 1, 3)
        v_nhd = v.permute(0, 2, 1, 3)
        
        # 对 q, k, v 进行 Hadamard 变换
        # import math
        # q_nld = hadamard_transform(q_nld.float(), scale=1/math.sqrt(q_nld.shape[-1])).to(dtype)
        # k_nld = hadamard_transform(k_nld.float(), scale=1/math.sqrt(k_nld.shape[-1])).to(dtype)

        # # 预热
        # print("预热中...")
        # with torch.no_grad():
        #     for _ in range(2):
        #         sageattn(q, k, v, tensor_layout='HND', is_causal=causal)
        #         flash_attn_func(q_nld, k_nld, v_nld, causal=causal)
        #         # scaled_dot_product_attention(q, k, v, is_causal=causal)
        # torch.cuda.synchronize()

        # start_time = torch.cuda.Event(enable_timing=True)
        # end_time = torch.cuda.Event(enable_timing=True)
        
        # 运行sageattn
        print("运行SageAttention...")
        # torch.cuda.synchronize()
        with torch.no_grad():
            # 清除L2缓存
            # clear_l2_cache()
            # --- NCU Range: SageAttention ---
            # torch.cuda.nvtx.range_push("SageAttention")
            # start_time.record()
            # sage_output = sageattn(q, k, v, tensor_layout='HND', is_causal=causal)
            sage_output = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout='NHD', is_causal=causal)
            # end_time.record()
            # torch.cuda.synchronize()
            # print(f"SageAttention运行时间: {start_time.elapsed_time(end_time):.4f} ms")
            # torch.cuda.nvtx.range_pop()
        sage_output = sage_output.permute(0, 2, 1, 3)  # 转回HND布局
        print(f"SageAttention输出形状: {sage_output.shape}")
        # 运行flash attention 2
        print("运行Flash Attention 2...")
        # torch.cuda.synchronize()
        with torch.no_grad():
            # 清除L2缓存
            # clear_l2_cache()
            # --- NCU Range: FlashAttention2 ---
            # torch.cuda.nvtx.range_push("FlashAttention2")
            # start_time.record()
            if USE_FLASH_ATTN:
                flash_output_nld = flash_attn_func(q_nhd, k_nhd, v_nhd, causal=causal)
            else:
                flash_output_nld = scaled_dot_product_attention(q, k, v, is_causal=causal)
            # end_time.record()
            # torch.cuda.synchronize()
            # print(f"Flash Attention 2运行时间: {start_time.elapsed_time(end_time):.4f} ms") 
            # torch.cuda.nvtx.range_pop()
        flash_output = flash_output_nld.permute(0, 2, 1, 3)  # 转回HND布局
        print(f"Flash Attention 2输出形状: {flash_output.shape}")
        
        # 计算简单的精度指标
        max_error = (sage_output - flash_output).abs().max().item()
        rel_error = (sage_output - flash_output).abs().max() / (flash_output.abs().max() + 1e-8)

        cos_sim = torch.nn.functional.cosine_similarity(
            sage_output,
            flash_output,
        )
        
        print(f"\n简单精度指标:")
        print(f"最大绝对误差 (Max Error): {max_error:.8f}")
        print(f"最大相对误差: {rel_error.item():.8f}")
        print(f"Cosine相似度: {cos_sim.mean().item():.8f}")
        with open(f"precision_comparison.txt", "a") as f:
            f.write(f"{id}\t{max_error:.8f}\t{rel_error.item():.8f}\t{cos_sim.mean().item():.8f}\n")
        
        print("\n精度对比基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"\n精度对比测试失败!")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison_simple(seq_len: int = 4096):
    """
    测试性能对比脚本的简单版本，验证基本功能
    """
    print("\n=== 测试性能对比脚本（简单版本） ===")
    try:
        # 导入必要的模块
        from sageattention.core import sageattn
        from flash_attn.flash_attn_interface import flash_attn_func
        from torch.nn.functional import scaled_dot_product_attention
        import time
        
        # 简单配置
        batch_size = 1

        print(f"seq_len: {seq_len//1024}k")
        n_heads = 12
        head_dim = 128
        dtype = torch.float16
        causal = False
        
        # 生成测试数据
        print(f"生成测试数据: batch_size={batch_size}, seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        # calculate the memory usage
        mem_usage = q.element_size() * batch_size * n_heads * seq_len * head_dim * 3 # q, k, v
        print(f"内存占用: {mem_usage / 1024**3:.4f} GB")
        
        # 转换为NHD布局
        q_nld = q.permute(0, 2, 1, 3)
        k_nld = k.permute(0, 2, 1, 3)
        v_nld = v.permute(0, 2, 1, 3)
        
        # 预热
        WARMUP = 1
        print("预热中...")
        with torch.no_grad():
            for _ in range(WARMUP):
                sageattn(q, k, v, tensor_layout='HND', is_causal=causal)
                flash_attn_func(q_nld, k_nld, v_nld, causal=causal)
                # scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        ITERS = 2
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # 测量SageAttention性能
        print("测量SageAttention性能...")
        elapsed_times_sage = []
        with torch.no_grad():
            for _ in range(ITERS):
                clear_l2_cache()
                start.record()
                # sageattn(q, k, v, tensor_layout='HND', is_causal=causal)
                sageattn(q_nld, k_nld, v_nld, tensor_layout='NHD', is_causal=causal)
                end.record()
                torch.cuda.synchronize()
                elapsed_times_sage.append(start.elapsed_time(end))
        sage_time = sum(elapsed_times_sage) / ITERS  # 毫秒
        
        # 测量Flash Attention 2性能
        print("测量Flash Attention 2性能...")
        elapsed_times_flash = []
        with torch.no_grad():
            for _ in range(ITERS):
                clear_l2_cache()
                start.record()
                flash_attn_func(q_nld, k_nld, v_nld, causal=causal)
                end.record()
                torch.cuda.synchronize()
                elapsed_times_flash.append(start.elapsed_time(end))
        flash_time = sum(elapsed_times_flash) / ITERS  # 毫秒
        
        # # 测量scaled_dot_product_attention性能
        # print("测量scaled_dot_product_attention性能...")
        # elapsed_times_sdp = []
        # with torch.no_grad():
        #     for _ in range(ITERS):
        #         clear_l2_cache()
        #         start.record()
        #         scaled_dot_product_attention(q, k, v, is_causal=causal)
        #         end.record()
        #         torch.cuda.synchronize()
        #         elapsed_times_sdp.append(start.elapsed_time(end))
        # sdp_time = sum(elapsed_times_sdp) / ITERS  # 毫秒
        
        # 计算性能指标
        speedup = 1/(sage_time / flash_time if flash_time > 0 else float('inf'))
        sage_throughput = (batch_size * seq_len) / (sage_time / 1000)
        flash_throughput = (batch_size * seq_len) / (flash_time / 1000)
        # sdp_throughput = (batch_size * seq_len) / (sdp_time / 1000)
        
        print(f"\n简单性能指标:")
        print(f"SageAttention 平均时间: {sage_time:.2f} ms")
        print(f"Flash Attention 2 平均时间: {flash_time:.2f} ms")
        # print(f"scaled_dot_product_attention 平均时间: {sdp_time:.2f} ms")

        print(f"性能提升: {speedup:.2f}x")
        print(f"SageAttention 吞吐量: {sage_throughput:.0f} tokens/sec")
        print(f"Flash Attention 2 吞吐量: {flash_throughput:.0f} tokens/sec")
        # print(f"scaled_dot_product_attention 吞吐量: {sdp_throughput:.0f} tokens/sec")
        
        print("\n性能对比基本功能测试通过!")
        return True
        
    except Exception as e:
        print(f"\n性能对比测试失败!")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主测试函数
    """
    print("=== 开始测试对比脚本 ===")
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: 测试需要CUDA设备")
        return 1
    
    print(f"CUDA设备: {torch.cuda.get_device_name()}")
    print(f"PyTorch版本: {torch.__version__}")
    
    seq_len = 1024 # 1k
    # seq_len = seq_len << 1 # 2k
    # seq_len = seq_len << 2 # 4k
    # seq_len = seq_len << 3 # 8k
    seq_len = seq_len << 4 # 16k
    # seq_len = seq_len << 5 # 32k
    # seq_len = seq_len << 6 # 64k
    # seq_len = seq_len << 7 # 128k
    # seq_len = seq_len << 8 # 256k
    # seq_len = seq_len << 9 # 512k
    # seq_len = seq_len << 10 # 1024k

    # 运行所有测试
    tests = [
        test_precision_comparison_simple,
        # test_performance_comparison_simple,
    ]
    
    # # use torch.profiler
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True
    # ) as profiler:
    #     test_performance_comparison_simple(seq_len)
    # profiler.export_chrome_trace(f"trace{seq_len//1024}k.json")

    for i in range(31):
        test_precision_comparison_simple(id=i)

    # for test_func in tests:
    #     result = test_func()
    #     if not result:
    #         print(f"\n{test_func.__name__} 测试失败!")
    #         return 1

if __name__ == "__main__":
    sys.exit(main())
