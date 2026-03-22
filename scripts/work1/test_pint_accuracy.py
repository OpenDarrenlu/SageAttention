import torch
import sys
import os

def test_precision_comparison_simple(id=0, seq_len=40960, cmp_target="flash_attn"):
    try:
        # 导入必要的模块
        from sageattention import sageattn, sageattn_pint, sageattn_pint_torch
        from flash_attn.flash_attn_interface import flash_attn_func
        from torch.nn.functional import scaled_dot_product_attention
        import numpy as np
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 简单配置
        batch_size = 1
        # seq_len = 40960
        n_heads = 12
        head_dim = 128
        dtype = torch.bfloat16
        causal = False
        
        # 生成测试数（HND布局）
        print(f"生成测试数据: batch_size={batch_size}, seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device='cuda')
        # 从 .pt 文件加载 q, k, v
        # qkv = torch.load(f'../../qkv_tensors_30layers/qkv_tensors_{id}.pt')
        # # import ipdb; ipdb.set_trace()
        # q,k,v = qkv["query"].cuda(), qkv["key"].cuda(), qkv["value"].cuda()
        # 转换为NHD布局
        q_nhd = q.permute(0, 2, 1, 3)
        k_nhd = k.permute(0, 2, 1, 3)
        v_nhd = v.permute(0, 2, 1, 3)
        
        with torch.no_grad():
            sage_output = sageattn_pint_torch(q_nhd, k_nhd, v_nhd, tensor_layout='NHD', is_causal=causal)
            sage_output = sage_output.permute(0, 2, 1, 3)  # 转回HND布局
        
        with torch.no_grad():
            if cmp_target == "flash_attn":
                ref_output_nld = flash_attn_func(q_nhd, k_nhd, v_nhd, causal=causal)
                ref_output = ref_output_nld.permute(0, 2, 1, 3)  # 转回HND布局
            elif cmp_target == "scaled_dot_product_attention":
                ref_output = scaled_dot_product_attention(q, k, v, is_causal=causal)
            elif cmp_target == "sageattn":
                ref_output = sageattn(q, k, v, tensor_layout='HND', is_causal=causal)
            elif cmp_target == "sageattn_pint_torch":
                ref_output = sageattn_pint_torch(q, k, v, tensor_layout='HND', is_causal=causal)
            else:
                raise ValueError(f"未知对比目标: {cmp_target}")
        
        # 计算简单的精度指标
        max_error = (sage_output - ref_output).abs().max().item()
        rel_error = (sage_output - ref_output).abs().max() / (ref_output.abs().max() + 1e-8)

        cos_sim = torch.nn.functional.cosine_similarity(
            sage_output,
            ref_output,
        )
        SNR = 10 * torch.log10(ref_output.square().mean() / (sage_output - ref_output).square().mean().item())
        
        print(f"\n简单精度指标:")
        print(f"最大绝对误差 (Max Error): {max_error:.8f}")
        print(f"最大相对误差: {rel_error.item():.8f}")
        print(f"Cosine相似度: {cos_sim.mean().item():.8f}")
        print(f"SNR: {SNR:.8f}")
        with open(f"precision_comparison.txt", "a") as f:
            f.write(f"{id}\t{max_error:.8f}\t{rel_error.item():.8f}\t{cos_sim.mean().item():.8f}\t{SNR:.8f}\n")
        
        return True
        
    except Exception as e:
        print(f"\n精度对比测试失败!")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主测试函数
    """
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: 测试需要CUDA设备")
        return 1
    
    seq_len = 1024 # 1k
    seq_len = seq_len << 1 # 2k
    # seq_len = seq_len << 2 # 4k
    # seq_len = seq_len << 3 # 8k
    # seq_len = seq_len << 4 # 16k
    # seq_len = seq_len << 5 # 32k
    # seq_len = seq_len << 6 # 64k
    # seq_len = seq_len << 7 # 128k
    # seq_len = seq_len << 8 # 256k
    # seq_len = seq_len << 9 # 512k
    # seq_len = seq_len << 10 # 1024k
    
    for i in range(1):
        for cmp_target in ["flash_attn", "scaled_dot_product_attention", "sageattn", "sageattn_pint_torch"]:
        # for cmp_target in ["sageattn_pint_torch", "sageattn"]: # "sageattn", "scaled_dot_product_attention"]: # , "sageattn_pint_torch"]:
            print(f"=== 测试对比目标: {cmp_target} ===")
            test_precision_comparison_simple(id=i, seq_len=seq_len, cmp_target=cmp_target)

if __name__ == "__main__":
    sys.exit(main())
