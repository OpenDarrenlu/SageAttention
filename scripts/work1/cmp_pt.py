import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def load_pt_file(file_path):
    """加载 .pt 文件，支持字典或直接张量"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = torch.load(file_path, map_location='cpu')
    
    if isinstance(data, torch.Tensor):
        return {"tensor": data}
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError(f"不支持的文件格式: {type(data)}")

def calculate_metrics(tensor_a, tensor_b, epsilon=1e-8):
    """计算指标"""
    t1 = tensor_a.float()
    t2 = tensor_b.float()

    abs_err = torch.abs(t1 - t2)
    denom = torch.maximum(torch.abs(t1), torch.abs(t2))
    denom = torch.clamp(denom, min=epsilon)
    rel_err = abs_err / denom
    
    v1 = t1.flatten()
    v2 = t2.flatten()
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        cos_sim = torch.tensor(1.0 if norm1 == norm2 else 0.0)
    else:
        cos_sim = torch.dot(v1, v2) / (norm1 * norm2)
        
    return cos_sim, abs_err, rel_err

def visualize_errors(rel_err, key_name, top_n=20):
    """可视化相对误差热力图"""
    if rel_err.numel() == 0:
        print("警告: 无数据点。")
        return
    rel_err = rel_err.reshape(-1, rel_err.shape[-1])
    plt.figure(figsize=(10, 8))
    plt.imshow(rel_err, cmap='viridis', aspect='auto')
    plt.title(f"相对误差热力图 - {key_name}")
    plt.colorbar()
    plt.savefig(f"{key_name}_rel_err.png")

def print_top_errors(rel_err, tensor_a, tensor_b, top_n=10):
    """打印相对误差最大的前 N 个位置"""
    if rel_err.numel() == 0:
        return

    flat_rel_err = rel_err.flatten()
    top_k_indices = torch.topk(flat_rel_err, k=min(top_n, flat_rel_err.numel())).indices
    
    print(f"\n--- 前 {len(top_k_indices)} 个最大相对误差详情 ---")
    print(f"{'Index':<15} | {'Value A':<12} | {'Value B':<12} | {'Rel Error':<12}")
    print("-" * 60)
    
    original_shape = rel_err.shape
    for idx in top_k_indices:
        multi_idx = []
        remainder = idx.item()
        for dim_size in reversed(original_shape):
            multi_idx.append(remainder % dim_size)
            remainder //= dim_size
        multi_idx.reverse()
        multi_idx_tuple = tuple(multi_idx)
        
        val_a = tensor_a[multi_idx_tuple].item()
        val_b = tensor_b[multi_idx_tuple].item()
        err_val = rel_err[multi_idx_tuple].item()
        
        print(f"{str(multi_idx_tuple):<15} | {val_a:<12.6f} | {val_b:<12.6f} | {err_val:<12.6e}")

def print_first_n_comparison(tensor_a, tensor_b, n=100, epsilon=1e-8):
    """打印前N个元素的对比，包含diff和rel_diff"""
    t1 = tensor_a.float().flatten()
    t2 = tensor_b.float().flatten()
    
    n = min(n, t1.numel())
    
    print(f"\n--- 前 {n} 个元素逐点对比 (diff = |A-B|, rel_diff = diff/max(|A|,|B|)) ---")
    print(f"{'Idx':<8} | {'Value A':<16} | {'Value B':<16} | {'Diff':<16} | {'Rel_Diff':<16}")
    print("-" * 78)
    
    for i in range(n):
        val_a = t1[i].item()
        val_b = t2[i].item()
        diff = abs(val_a - val_b)
        denom = max(abs(val_a), abs(val_b), epsilon)
        rel_diff = diff / denom
        
        print(f"{i:<8} | {val_a:<16.6e} | {val_b:<16.6e} | {diff:<16.6e} | {rel_diff:<16.6e}")
    
    # 可选：统计前100个元素的误差摘要
    diffs = torch.abs(t1[:n] - t2[:n])
    rel_diffs = diffs / torch.clamp(torch.maximum(torch.abs(t1[:n]), torch.abs(t2[:n])), min=epsilon)
    print(f"\n📊 前 {n} 个元素误差统计:")
    print(f"   Mean Diff: {diffs.mean().item():.6e} | Max Diff: {diffs.max().item():.6e}")
    print(f"   Mean Rel_Diff: {rel_diffs.mean().item():.6e} | Max Rel_Diff: {rel_diffs.max().item():.6e}")

def main():
    parser = argparse.ArgumentParser(description="对比两个 .pt 文件的精度差异并可视化")
    parser.add_argument("file1", type=str, help="第一个 .pt 文件路径")
    parser.add_argument("file2", type=str, help="第二个 .pt 文件路径")
    parser.add_argument("--top_n", type=int, default=20, help="打印并标注前 N 个最大误差点")
    parser.add_argument("--plot", action="store_true", help="是否生成误差热力图")
    parser.add_argument("--print_first_n", type=int, default=100, 
                        help="打印前N个元素的逐点对比（默认100，设为0则关闭）")
    args = parser.parse_args()

    try:
        data1 = load_pt_file(args.file1)
        data2 = load_pt_file(args.file2)
    except Exception as e:
        print(f"错误: {e}")
        return

    common_keys = set(data1.keys()).intersection(data2.keys())
    if not common_keys:
        print("警告: 无共同键名。")
        return

    for key in common_keys:
        t1, t2 = data1[key], data2[key]
        print(f"\n=== 对比键: '{key}' ===")
        
        if t1.shape != t2.shape:
            print(f"❌ 形状不匹配: {t1.shape} vs {t2.shape}")
            continue
        
        cos_sim, abs_err, rel_err = calculate_metrics(t1, t2)
        
        print(f"✅ Cosine Similarity: {cos_sim.item():.8f}")
        print(f"📈 Max Relative Error: {rel_err.max().item():.8e}")
        
        # 1. 新增：打印前N个元素对比（diff + rel_diff）
        if args.print_first_n > 0:
            print_first_n_comparison(t1, t2, n=args.print_first_n)
        
        # 2. 打印文本详情（最大误差点）
        print_top_errors(rel_err, t1, t2, top_n=args.top_n)
        
        # 3. 可视化功能
        if args.plot:
            visualize_errors(rel_err, key, top_n=args.top_n)

if __name__ == "__main__":
    main()