import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

def get_headmap(tensor_path):
    # 1️⃣ 加载到 CPU（避免 GPU 依赖）
    qkv = torch.load(tensor_path, map_location='cpu')
    q, k, v = qkv["query"], qkv["key"], qkv["value"]
    
    # 2️⃣ reshape on GPU
    seq_len = q.shape[2]
    q = q.transpose(1, 2).reshape(seq_len, -1).cuda()  # [seq_len, channels]
    k = k.transpose(1, 2).reshape(seq_len, -1).cuda()
    v = v.transpose(1, 2).reshape(seq_len, -1).cuda()
    
    with torch.no_grad():
        P = F.softmax(torch.mm(q, k.transpose(0, 1), out_dtype=torch.float32), dim=-1)

    # 🔍 判断 subnormal: 非零且绝对值 < tiny
    dtype_info = torch.finfo(P.dtype)
    tiny = dtype_info.tiny
    print(f"dtype: {P.dtype}, tiny: {tiny}")
    is_subnormal = (torch.abs(P) < tiny) & (P != 0)  # 修正：排除零值，只保留真正的 subnormal
    is_normal = (torch.abs(P) >= tiny)  # 包括正规数(不包括零)

    # cal ratio of subnormal in P
    subnormal_counts = is_subnormal.float().sum().item()
    print(f"Subnormal ratio in P: {subnormal_counts / P.numel():.6f} ({subnormal_counts}/{P.numel()})")
    print(f"Normal ratio in P: {is_normal.float().sum().item() / P.numel():.6f} ({is_normal.float().sum().item()}/{P.numel()})")

    # # 转为 numpy
    # subnormal_mask = is_subnormal.detach().cpu().numpy()
    # normal_mask = is_normal.detach().cpu().numpy()

    # # ===== 图1：全图 subnormal 分布（二值图）=====
    # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    # im1 = ax1.imshow(subnormal_mask, aspect='auto', cmap='gray', interpolation='nearest')
    # ax1.set_title('Subnormal Distribution in P (Full)', fontsize=14, fontweight='bold')
    # ax1.set_xlabel('Key Token')
    # ax1.set_ylabel('Query Token')
    # cbar1 = fig1.colorbar(im1, ax=ax1, orientation='horizontal', shrink=0.6, pad=0.1, ticks=[0, 1])
    # cbar1.ax.set_xticklabels(['Normal/Zero', 'Subnormal'])
    # plt.tight_layout()
    # save_name1 = f'P_subnormal_{os.path.basename(tensor_path).split(".")[0].split("_")[-1]}.png'
    # plt.savefig(save_name1, dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig1)

    # # ===== 新增：图2：前256x256子图的 subnormal 分布 =====
    # N = min(256, seq_len)
    # subnormal_mask_256 = subnormal_mask[:N, :N]

    # fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    # im2 = ax2.imshow(subnormal_mask_256, aspect='equal', cmap='gray', interpolation='nearest')
    # ax2.set_title(f'Subnormal Distribution in P (Top {N}×{N})', fontsize=14, fontweight='bold')
    # ax2.set_xlabel('Key Token')
    # ax2.set_ylabel('Query Token')
    # cbar2 = fig2.colorbar(im2, ax=ax2, orientation='horizontal', shrink=0.6, pad=0.1, ticks=[0, 1])
    # cbar2.ax.set_xticklabels(['Normal/Zero', 'Subnormal'])
    # plt.tight_layout()
    # save_name2 = f'P_subnormal_256x256_{os.path.basename(tensor_path).split(".")[0].split("_")[-1]}.png'
    # plt.savefig(save_name2, dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig2)
    
# 批量处理
tensor_dir = '../qkv_tensors_30layers'
TEST_ALL = True
if os.path.exists(tensor_dir) and TEST_ALL:
    tensor_paths = [os.path.join(tensor_dir, f) for f in os.listdir(tensor_dir) if f.endswith('.pt')]
else:
    tensor_paths = ['../qkv_tensors_30layers/qkv_tensors_0.pt']  # fallback

for tensor_path in tqdm(tensor_paths, desc="Processing QKV tensors"):
    if os.path.exists(tensor_path):
        get_headmap(tensor_path)
    else:
        print(f"Warning: {tensor_path} not found.")