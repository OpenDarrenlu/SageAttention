"""
MMA Instruction Shape Visualization (English Only)
Single figure, 2x2 layout:
  Row 0 (Single Cycle): HMMA (M=8,N=4,K=8) | LMMA (M=8,N=64,K=8)
  Row 1 (8 Cycles):     HMMA (M=16,N=8,K=16) | LMMA (M=16,N=128,K=16)

Fix: C/D matrix now fully visible with proper vertical centering.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.font_manager as fm

# ============================================================
# Font setup (English only)
# ============================================================
def setup_font():
    preferred_fonts = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    return None

setup_font()

# ============================================================
# Color scheme
# ============================================================
COLOR_BG = '#5A8F00'
COLOR_GRID = '#FFFFFF'
COLOR_A = '#9B7EBD'
COLOR_B = '#5BC0DE'
COLOR_C = '#8BC34A'
COLOR_EDGE = 'black'
ALPHA_BLOCK = 0.8


def draw_grid_bg(ax, x0, y0, w, h, step=1, color=COLOR_GRID, alpha=0.3):
    for x in np.arange(x0, x0 + w + step, step):
        ax.plot([x, x], [y0, y0 + h], color=color, linewidth=0.5, alpha=alpha, zorder=1)
    for y in np.arange(y0, y0 + h + step, step):
        ax.plot([x0, x0 + w], [y, y], color=color, linewidth=0.5, alpha=alpha, zorder=1)


def draw_block(ax, ox, oy, m, n, color, label, edgecolor=COLOR_EDGE, lw=2.5):
    rect = patches.FancyBboxPatch(
        (ox, oy), n, m, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=edgecolor, linewidth=lw, alpha=ALPHA_BLOCK, zorder=3
    )
    ax.add_patch(rect)
    for i in range(1, m):
        ax.plot([ox, ox + n], [oy + i, oy + i], color=edgecolor, lw=0.4, alpha=0.35, zorder=4)
    for j in range(1, n):
        ax.plot([ox + j, ox + j], [oy, oy + m], color=edgecolor, lw=0.4, alpha=0.35, zorder=4)
    if label:
        ax.text(ox + n/2, oy + m/2, label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='black', zorder=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.85))


def plot_panel(ax, title, A_shape, B_shape, C_shape):
    """Draw single MMA panel with proper vertical centering"""
    ax.set_facecolor(COLOR_BG)
    a_m, a_k = A_shape
    b_k, b_n = B_shape
    c_m, c_n = C_shape

    margin = 2
    gap_h = 3
    gap_v = 1.5

    # Calculate content height for both sides
    left_h = a_m
    right_h = b_k + gap_v + c_m
    content_h = max(left_h, right_h)

    total_w = (a_k + 2) + gap_h + (max(b_n, c_n) + 2) + margin * 2
    total_h = content_h + margin * 2

    # Vertical center baseline
    base_y = margin + content_h / 2

    # Left: A matrix (vertically centered)
    a_x = margin + 1
    a_y = base_y - a_m / 2
    draw_grid_bg(ax, a_x - 1, a_y - 1, a_k + 2, a_m + 2, step=1)
    draw_block(ax, a_x, a_y, a_m, a_k, COLOR_A, 'A')
    ax.text(a_x + a_k/2, a_y - 0.7, f"{a_m}x{a_k}",
            ha='center', va='top', fontsize=9, color='#FFD700', fontweight='bold')

    # Right side x position
    r_x = a_x + a_k + gap_h
    right_w = max(b_n, c_n)

    # Right background grid
    right_bg_x = r_x - 1
    right_bg_y = base_y - right_h / 2 - 1
    right_bg_w = right_w + 2
    right_bg_h = right_h + 2
    draw_grid_bg(ax, right_bg_x, right_bg_y, right_bg_w, right_bg_h, step=1)

    # B matrix (top of right side)
    b_x = r_x + (right_w - b_n) / 2
    b_y = base_y + right_h / 2 - b_k
    draw_block(ax, b_x, b_y, b_k, b_n, COLOR_B, 'B')
    ax.text(b_x + b_n/2, b_y + b_k + 0.5, f"{b_k}x{b_n}",
            ha='center', va='bottom', fontsize=9, color='#FFD700', fontweight='bold')

    # C/D matrix (bottom of right side)
    c_x = r_x + (right_w - c_n) / 2
    c_y = base_y - right_h / 2
    draw_block(ax, c_x, c_y, c_m, c_n, COLOR_C, 'C/D')
    ax.text(c_x + c_n/2, c_y - 0.7, f"{c_m}x{c_n}",
            ha='center', va='top', fontsize=9, color='#FFD700', fontweight='bold')

    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=6)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect('equal')
    ax.axis('off')


# ============================================================
# One figure, 2x2 layout
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#3d6b00')

# Row 0: Single Cycle
plot_panel(axes[0, 0], "HMMA Single Cycle\n(M=8, N=4, K=8)", (8, 8), (8, 4), (8, 4))
plot_panel(axes[0, 1], "LMMA Single Cycle\n(M=8, N=64, K=8)", (8, 8), (8, 64), (8, 64))

# Row 1: 8 Cycles
plot_panel(axes[1, 0], "HMMA 8 Cycles\n(M=16, N=8, K=16)", (16, 16), (16, 8), (16, 8))
plot_panel(axes[1, 1], "LMMA 8 Cycles\n(M=16, N=128, K=16)", (16, 16), (16, 128), (16, 128))

# Side labels
fig.text(0.015, 0.73, 'Single Cycle\n(Instruction Shape)', va='center', ha='center',
         rotation='vertical', color='white', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='darkgreen', alpha=0.9))
fig.text(0.015, 0.27, 'Fixed 8 Cycles\n(Accumulated Shape)', va='center', ha='center',
         rotation='vertical', color='white', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='darkgreen', alpha=0.9))

fig.suptitle("MMA Instruction Shape Visualization - HMMA vs LMMA", color='white', fontsize=16,
             fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.04, 0.0, 1, 0.94])

fig.savefig("mma_hmma_lmma.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print("Saved: mma_hmma_lmma.png")