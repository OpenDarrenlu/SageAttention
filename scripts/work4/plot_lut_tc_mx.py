"""
LMMA 8 Cycles with Micro Scaling (Final Version)

Micro Scaling Definition:
- A (M x K): scale along K dimension -> [M, K//block_size]
- B (K x N): scale along K dimension -> [K//block_size, N]
- C/D (M x N): no scale grid

Layout:
- Right side (top to bottom): B scale -> B matrix -> C/D matrix
- Left side (bottom aligned with C/D): A scale -> A matrix
- Each scale cell = same size as element cell (1 unit)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.font_manager as fm


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

COLOR_BG = '#5A8F00'
COLOR_GRID = '#FFFFFF'
COLOR_A = '#9B7EBD'
COLOR_B = '#5BC0DE'
COLOR_C = '#8BC34A'
COLOR_EDGE = 'black'
ALPHA_BLOCK = 0.8


def draw_block(ax, ox, oy, m, n, color, label, lw=2.5):
    """Draw matrix block with fine element grid."""
    rect = patches.FancyBboxPatch(
        (ox, oy), n, m, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=COLOR_EDGE, linewidth=lw, alpha=ALPHA_BLOCK, zorder=3
    )
    ax.add_patch(rect)
    for i in range(1, m):
        ax.plot([ox, ox + n], [oy + i, oy + i], color=COLOR_EDGE, lw=0.3, alpha=0.3, zorder=4)
    for j in range(1, n):
        ax.plot([ox + j, ox + j], [oy, oy + m], color=COLOR_EDGE, lw=0.3, alpha=0.3, zorder=4)
    if label:
        ax.text(ox + n/2, oy + m/2, label, ha='center', va='center',
                fontsize=13, fontweight='bold', color='black', zorder=7,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='none', alpha=0.9))


def draw_scale_grid(ax, ox, oy, scale_m, scale_n, color, lw=2.0):
    """Draw scale grid with SAME cell size as element grid (1 unit per cell)."""
    rect = patches.FancyBboxPatch(
        (ox, oy), scale_n, scale_m, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=COLOR_EDGE, linewidth=lw, alpha=ALPHA_BLOCK, zorder=3
    )
    ax.add_patch(rect)
    for i in range(1, scale_m):
        ax.plot([ox, ox + scale_n], [oy + i, oy + i], color=COLOR_EDGE, lw=0.6, alpha=0.5, zorder=4)
    for j in range(1, scale_n):
        ax.plot([ox + j, ox + j], [oy, oy + scale_m], color=COLOR_EDGE, lw=0.6, alpha=0.5, zorder=4)


def draw_grid_bg(ax, x0, y0, w, h, step=1, color=COLOR_GRID, alpha=0.25):
    for x in np.arange(x0, x0 + w + step, step):
        ax.plot([x, x], [y0, y0 + h], color=color, linewidth=0.5, alpha=alpha, zorder=1)
    for y in np.arange(y0, y0 + h + step, step):
        ax.plot([x0, x0 + w], [y, y], color=color, linewidth=0.5, alpha=alpha, zorder=1)


def plot_lmma_micro_scaling(block_size=16, M=16, N=128, K=16,
                            output_file="lmma_micro_scaling.png", show=True):
    """
    Plot LMMA 8 Cycles with micro scaling.

    Parameters:
        block_size: micro scaling block size (must divide K)
        M: M dimension (rows of A/C)
        N: N dimension (cols of B/C)
        K: K dimension (cols of A / rows of B)
    """
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"

    scale_a = [M, K // block_size]
    scale_b = [K // block_size, N]

    fig, ax = plt.subplots(1, 1, figsize=(22, 11), facecolor='#3d6b00')
    ax.set_facecolor(COLOR_BG)

    margin = 4
    gap_h = 6
    gap_v = 3

    # Layout from top to bottom
    total_h = margin + scale_b[0] + gap_v + K + gap_v + M + margin

    # Right side x position
    b_x = margin + scale_a[1] + gap_h + K + gap_h

    # B scale (top)
    scale_b_y = total_h - margin - scale_b[0]
    draw_scale_grid(ax, b_x, scale_b_y, scale_b[0], scale_b[1], COLOR_B)
    ax.text(b_x + scale_b[1]/2, scale_b_y + scale_b[0] + 1.0, 
            f"B scale(UE8M0) [{scale_b[0]}x{scale_b[1]}]",
            ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')

    # B matrix (below scale)
    b_y = scale_b_y - gap_v - K
    draw_grid_bg(ax, b_x - 1, b_y - 1, N + 2, K + 2, step=1)
    draw_block(ax, b_x, b_y, K, N, COLOR_B, 'B')
    ax.text(b_x + N/2, b_y - 1.5, f"B: {K}x{N}",
            ha='center', va='top', fontsize=10, color='#FFD700', fontweight='bold')

    # C/D matrix (below B)
    c_y = b_y - gap_v - M
    draw_grid_bg(ax, b_x - 1, c_y - 1, N + 2, M + 2, step=1)
    draw_block(ax, b_x, c_y, M, N, COLOR_C, 'C/D')
    ax.text(b_x + N/2, c_y - 1.5, f"C/D: {M}x{N}",
            ha='center', va='top', fontsize=10, color='#FFD700', fontweight='bold')

    # A matrix (left side, bottom aligned with C/D)
    a_x = margin + scale_a[1] + gap_h
    a_y = c_y
    draw_grid_bg(ax, a_x - 1, a_y - 1, K + 2, M + 2, step=1)
    draw_block(ax, a_x, a_y, M, K, COLOR_A, 'A')
    ax.text(a_x + K/2, a_y - 1.5, f"A: {M}x{K}",
            ha='center', va='top', fontsize=10, color='#FFD700', fontweight='bold')

    # A scale (left of A, bottom aligned with A)
    scale_a_x = margin
    scale_a_y = a_y
    draw_scale_grid(ax, scale_a_x, scale_a_y, scale_a[0], scale_a[1], COLOR_A)
    ax.text(scale_a_x + scale_a[1]/2, scale_a_y - 1.5, 
            f"A scale(UE8M0) [{scale_a[0]}x{scale_a[1]}]",
            ha='center', va='top', fontsize=10, color='white', fontweight='bold')

    total_w = b_x + N + margin

    ax.set_title(f"LMMA 8 Cycles with Micro Scaling\n"
                 f"(M={M}, N={N}, K={K}, block_size={block_size})", 
                 color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    plot_lmma_micro_scaling(block_size=8, M=16, N=64, K=8,
                            output_file="lmma_micro_scaling.png")