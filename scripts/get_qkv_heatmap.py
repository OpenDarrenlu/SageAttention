import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # 添加进度条

def get_headmap(tensor_path):
    # 1️⃣ 直接在 GPU 上处理所有计算（避免 CPU-GPU 频繁传输）
    qkv = torch.load(tensor_path, map_location='cuda')  # 直接加载到 GPU
    q, k, v = qkv["query"], qkv["key"], qkv["value"]
    
    # 2️⃣ GPU 上高效 reshape（避免 detach 和 cpu）
    seq_len = q.shape[2]
    q = q.transpose(1, 2).reshape(seq_len, -1)  # [seq_len, channels]
    k = k.transpose(1, 2).reshape(seq_len, -1)
    v = v.transpose(1, 2).reshape(seq_len, -1)
    
    # 3️⃣ 在 GPU 上计算统计信息（避免传输完整数据）
    q_min, q_max = q.min().item(), q.max().item()
    k_min, k_max = k.min().item(), k.max().item()
    v_min, v_max = v.min().item(), v.max().item()
    
    # 4️⃣ 仅传输必要数据（关键优化！）
    # 选择性传输：只传输 1% 的数据点（下采样）用于可视化
    sample_ratio = 1  # 1% 采样率
    q_sample = q[::int(1/sample_ratio), ::int(1/sample_ratio)].detach().cpu().to(torch.float32).numpy()
    k_sample = k[::int(1/sample_ratio), ::int(1/sample_ratio)].detach().cpu().to(torch.float32).numpy()
    v_sample = v[::int(1/sample_ratio), ::int(1/sample_ratio)].detach().cpu().to(torch.float32).numpy()
    
    # 5️⃣ 使用更高效的 imshow 替代 heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Q', 'K', 'V']
    data_list = [
        (q_sample, q_min, q_max),
        (k_sample, k_min, k_max),
        (v_sample, k_min, k_max)
    ]
    
    for i, (ax, title, (data, vmin, vmax)) in enumerate(zip(axes, titles, data_list)):
        # ⚡ 使用 imshow 替代 heatmap（10-100x 速度提升）
        im = ax.imshow(
            data, 
            aspect='auto', 
            cmap='hot', 
            vmin=vmin, 
            vmax=vmax,
            interpolation='nearest'  # 关键：避免平滑计算
        )
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Channel (sampled)', fontsize=12)
        ax.set_ylabel('Token (sampled)', fontsize=12)
        
        # ⚡ 高效 colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, pad=0.1)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])
    
    plt.suptitle(f'(a) QKV distribution: {os.path.basename(tensor_path)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{tensor_path.split("/")[-1].split(".")[0]}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# 6️⃣ 批量处理优化
# tensor_paths = [os.path.join('../qkv_tensors_30layers', f) for f in os.listdir('../qkv_tensors_30layers') if f.endswith('.pt')]
tensor_paths = ['../qkv_tensors_30layers/qkv_tensors_0_hadamard.pt']

# 使用 tqdm 显示进度
for tensor_path in tqdm(tensor_paths, desc="Processing QKV tensors"):
    get_headmap(tensor_path)