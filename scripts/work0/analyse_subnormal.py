import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def analyze_fp16_distribution_v2(file_path):
    # 1. 加载数据并保持原始精度
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return
        
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    prob_tensor = checkpoint['prob'].detach()
    
    # 2. 提取 FP16 物理边界
    f16_info = torch.finfo(torch.float16)
    f16_tiny = f16_info.tiny  # 6.1035e-05 (最小正规数)
    
    # 3. 准备数据
    prob_np = prob_tensor.to(torch.float32).numpy().flatten()
    # 过滤掉纯0（避免log报错），并计算统计量
    non_zero_probs = prob_np[prob_np > 0]
    mu, std = norm.fit(prob_np)
    
    # 动态设定显示上限：取 99% 分位数，或者 50 倍 tiny
    p99 = np.percentile(prob_np, 99)
    display_max = max(p99 * 2, f16_tiny * 50) 
    display_max = min(display_max, prob_np.max()) # 不超过绝对最大值

    # 4. 创建画布：双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    def plot_on_ax(ax, use_log=False):
        # 绘制直方图，bins 增加到 300 以提高分辨率
        counts, bins, _ = ax.hist(prob_np, bins=300, range=(0, display_max), 
                                  density=True, alpha=0.5, color='#1f77b4', 
                                  label='Actual Prob')
        
        # 绘制拟合正态曲线
        x_plot = np.linspace(0, display_max, 1000)
        y_plot = norm.pdf(x_plot, mu, std)
        ax.plot(x_plot, y_plot, 'r--', lw=2, label=rf'Normal Fit ($\mu$={mu:.1e})')
        
        # 标注 Subnormal 边界
        ax.axvline(x=f16_tiny, color='orange', linestyle=':', lw=2, label='FP16 Tiny Boundary')
        if f16_tiny < display_max:
            ax.fill_betweenx([0, counts.max() * 1.1 if not use_log else counts.max() * 10], 
                             0, f16_tiny, color='orange', alpha=0.1, label='Subnormal Zone')
        
        if use_log:
            ax.set_yscale('log')
            ax.set_title("Micro View: Log Scale (See fine details)")
        else:
            ax.set_title("Macro View: Linear Scale")
            
        ax.set_xlabel("Probability Value")
        ax.set_ylabel("Density")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, which="both", ls="-", alpha=0.15)
        ax.set_xlim(-display_max * 0.01, display_max)

    # 执行绘图
    plot_on_ax(ax1, use_log=False)
    plot_on_ax(ax2, use_log=True)

    plt.suptitle(rf"Fine-grained Distribution Analysis: {os.path.basename(file_path)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 5. 打印详细报告
    subnormal_count = torch.sum((prob_tensor > 0) & (prob_tensor < f16_tiny)).item()
    print(f"--- 详细分析结果 ---")
    print(f"最大值 (Max): {prob_np.max():.2e}")
    print(f"99% 分位数: {p99:.2e}")
    print(f"Subnormal 占比: {subnormal_count/prob_tensor.numel()*100:.4f}%")
    if subnormal_count > 0:
        print(f"注意: 存在大量数值落入 Subnormal 区间，FP16 精度可能受损。")

    save_path = f"refined_analysis_{os.path.basename(file_path)}.png"
    plt.savefig(save_path, dpi=200)
    print(f"图表已保存至: {save_path}")
    plt.show()

# 运行分析
analyze_fp16_distribution_v2("../example/qkvp_tensors_0.pt")