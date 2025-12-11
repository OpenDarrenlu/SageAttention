import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Step 1: 模拟数据 ---
# 假设你从模型中获取了 Q/K/V 的张量
# shape: [batch, num_heads, seq_len, head_dim]
# q = torch.randn(batch_size, num_heads, seq_len, head_dim)
# k = torch.randn(batch_size, num_heads, seq_len, head_dim)
# v = torch.randn(batch_size, num_heads, seq_len, head_dim)
def get_headmap(tensor_path):
    qkv = torch.load(tensor_path)
    # import ipdb; ipdb.set_trace()
    q,k,v = qkv["query"], qkv["key"], qkv["value"]
    seq_len = q.shape[2]
    q = q.to(dtype=torch.float32, device='cuda').transpose(1, 2).reshape(seq_len, -1)
    k = k.to(dtype=torch.float32, device='cuda').transpose(1, 2).reshape(seq_len, -1)
    v = v.to(dtype=torch.float32, device='cuda').transpose(1, 2).reshape(seq_len, -1)

    # --- Step 2: reshape to [token, channel] ---
    # 合并 heads 和 head_dim → channels
    q = q.detach().cpu().numpy()
    k = k.detach().cpu().numpy()
    v = v.detach().cpu().numpy()

    # --- Step 3: 绘制热力图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ['Q', 'K', 'V']
    data_list = [q, k, v]

    for i, (ax, title, data) in enumerate(zip(axes, titles, data_list)):
        # 使用 seaborn 热力图
        sns.heatmap(data, ax=ax, cmap='hot', cbar=False, xticklabels=False, yticklabels=False)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Token', fontsize=12)

        # 添加 colorbar 到每个子图下方
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal', shrink=0.6, pad=0.1)
        cbar.ax.tick_params(labelsize=10)
        vmin, vmax = data.min(), data.max()
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])

    plt.suptitle('(a) QKV distribution in YourModel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{tensor_path.split("/")[-1].split(".")[0]}.png')
    plt.close()
  
import os
tensor_paths = [os.path.join('../qkv_tensors_30layers', f) for f in os.listdir('../qkv_tensors_30layers') if f.endswith('.pt')]

for tensor_path in tensor_paths[:1]:
    print(tensor_path)
    get_headmap(tensor_path)
