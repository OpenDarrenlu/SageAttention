"""
LMMA 8 Cycles with Micro Scaling Visualization
M=16, N=128, K=16, block_size=16
A scale shape: [16, 1], B scale shape: [16, 8], C/D scale shape: [16, 8]
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


def draw_micro_block(ax, ox, oy, scale_m, scale_n, micro_m, micro_n, color, label, lw=2.5):
    total_m = scale_m * micro_m
    total_n = scale_n * micro_n
    rect = patches.FancyBboxPatch(
        (ox, oy), total_n, total_m, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=COLOR_EDGE, linewidth=lw, alpha=ALPHA_BLOCK, zorder=3
    )
    ax.add_patch(rect)
    for i in range(1, total_m):
        ax.plot([ox, ox + total_n], [oy + i, oy + i], color=COLOR_EDGE, lw=0.3, alpha=0.3, zorder=4)
    for j in range(1, total_n):
        ax.plot([ox + j, ox + j], [oy, oy + total_m], color=COLOR_EDGE, lw=0.3, alpha=0.3, zorder=4)
    for i in range(1, scale_m):
        y = oy + i * micro_m
        ax.plot([ox, ox + total_n], [y, y], color='white', lw=1.8, alpha=0.95, zorder=6)
    for j in range(1, scale_n):
        x = ox + j * micro_n
        ax.plot([x, x], [oy, oy + total_m], color='white', lw=1.8, alpha=0.95, zorder=6)
    if label:
        ax.text(ox + total_n/2, oy + total_m/2, label, ha='center', va='center',
                fontsize=13, fontweight='bold', color='black', zorder=7,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='none', alpha=0.9))


def draw_grid_bg(ax, x0, y0, w, h, step=1, color=COLOR_GRID, alpha=0.25):
    for x in np.arange(x0, x0 + w + step, step):
        ax.plot([x, x], [y0, y0 + h], color=color, linewidth=0.5, alpha=alpha, zorder=1)
    for y in np.arange(y0, y0 + h + step, step):
        ax.plot([x0, x0 + w], [y, y], color=color, linewidth=0.5, alpha=alpha, zorder=1)


fig, ax = plt.subplots(1, 1, figsize=(16, 7), facecolor='#3d6b00')
ax.set_facecolor(COLOR_BG)

scale_m_a, scale_n_a = 16, 1
scale_m_b, scale_n_b = 16, 8
scale_m_c, scale_n_c = 16, 8
micro_m, micro_n = 1, 16

a_m, a_n = scale_m_a * micro_m, scale_n_a * micro_n
b_m, b_n = scale_m_b * micro_m, scale_n_b * micro_n
c_m, c_n = scale_m_c * micro_m, scale_n_c * micro_n

margin = 3
gap_h = 4
gap_v = 2

left_w = a_n + 2
right_w = max(b_n, c_n) + 2
total_w = left_w + right_w + gap_h + margin * 2
right_h = b_m + gap_v + c_m
left_h = a_m
content_h = max(left_h, right_h)
total_h = content_h + margin * 2 + 4
base_y = margin + 2 + content_h / 2

a_x = margin + 1
a_y = base_y - a_m / 2
draw_grid_bg(ax, a_x - 1, a_y - 1, a_n + 2, a_m + 2, step=1)
draw_micro_block(ax, a_x, a_y, scale_m_a, scale_n_a, micro_m, micro_n, COLOR_A, 'A')
ax.text(a_x + a_n/2, a_y - 1.0, f"A: {scale_m_a}x{scale_n_a} (scale) = {a_m}x{a_n} (element)",
        ha='center', va='top', fontsize=10, color='#FFD700', fontweight='bold')

r_x = a_x + a_n + gap_h
right_w_actual = max(b_n, c_n)
r_bg_y = base_y - right_h / 2 - 1
draw_grid_bg(ax, r_x - 1, r_bg_y, right_w_actual + 2, right_h + 2, step=1)

b_x = r_x + (right_w_actual - b_n) / 2
b_y = base_y + right_h / 2 - b_m
draw_micro_block(ax, b_x, b_y, scale_m_b, scale_n_b, micro_m, micro_n, COLOR_B, 'B')
ax.text(b_x + b_n/2, b_y + b_m + 1.0, f"B: {scale_m_b}x{scale_n_b} (scale) = {b_m}x{b_n} (element)",
        ha='center', va='bottom', fontsize=10, color='#FFD700', fontweight='bold')

c_x = r_x + (right_w_actual - c_n) / 2
c_y = base_y - right_h / 2
draw_micro_block(ax, c_x, c_y, scale_m_c, scale_n_c, micro_m, micro_n, COLOR_C, 'C/D')
ax.text(c_x + c_n/2, c_y - 1.0, f"C/D: {scale_m_c}x{scale_n_c} (scale) = {c_m}x{c_n} (element)",
        ha='center', va='top', fontsize=10, color='#FFD700', fontweight='bold')

ax.text(0.98, 0.02, "Micro Scaling: block_size = 16\n"
        "White lines = scale block boundaries\n"
        "Fine grid = 1 element",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=10, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', alpha=0.9, edgecolor='white'))

ax.set_title("LMMA 8 Cycles with Micro Scaling\n(M=16, N=128, K=16, block_size=16)", 
             color='white', fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h)
ax.set_aspect('equal')
ax.axis('off')

fig.tight_layout()
fig.savefig("lmma_micro_scaling.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print("Saved: lmma_micro_scaling.png")