# -*- coding: utf-8 -*-
"""DBond-GT architecture schematic (paper Fig., double-column 183mm, plain-text labels)."""
import sys
sys.path.insert(0, r"C:\Users\gsj23\.zcode\skills\nature-figure\scripts")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 7})
T900, T700, T600, T50 = "#134E4A", "#0F766E", "#0D9488", "#F0FDFA"
A900, A700, A600, A50 = "#78350F", "#B45309", "#D97706", "#FFFBEB"
TXT, MUT, DASH = "#1E293B", "#64748B", "#94A3B8"
fig = plt.figure(figsize=(183/25.4, 88/25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 183); ax.set_ylim(0, 88); ax.axis("off")
def box(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec=ec, lw=lw))
def txt(x, y, s, fs=6.0, bold=False, color=TXT, style="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", style=style)
def arrow(x1, y1, x2, y2, color="#475569", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, shrinkA=1, shrinkB=1))
Y0, H = 38, 26
box(2, Y0, 25, H, T50, T600); txt(14.5, 59.5, "Pre-synthesis", 6.6, True, T900)
txt(14.5, 55.6, "input", 6.6, True, T900)
txt(14.5, 50.0, "sequence + charge +", 5.5)
txt(14.5, 47.2, "pep_mass + NCE", 5.5)
txt(14.5, 42.6, "no intensity /", 5.2, color=MUT, style="italic")
txt(14.5, 40.4, "scan_num / rt", 5.2, color=MUT, style="italic")
box(31, Y0, 37, H, T50, T600); txt(49.5, 59.5, "Residue graph", 6.8, True, T900)
gy, gr = 49.5, 1.9; xs = [36 + i*6.8 for i in range(5)]
ax.add_patch(Circle((49.5, 55.5), 1.8, fc=T900, zorder=5))
ax.text(49.5, 55.5, "G", ha="center", va="center", fontsize=5.5, color="white", fontweight="bold", zorder=6)
for i, (c, x) in enumerate(zip("AKLGF", xs)):
    ax.add_patch(Circle((x, gy), gr, fc=T600, zorder=5))
    ax.text(x, gy, c, ha="center", va="center", fontsize=5.5, color="white", fontweight="bold", zorder=6)
    if i < 4: ax.plot([x+gr, xs[i+1]-gr], [gy, gy], color=T700, lw=1.3, zorder=4)
    ax.plot([x, 49.5], [gy+gr, 55.5-1.8], color=DASH, lw=0.7, ls=(0, (2, 2)), zorder=3)
for i in range(3):
    ax.plot([xs[i], xs[i+2]], [gy+gr+0.5]*2, color=A600, lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.plot([xs[2]+0.6, xs[2]+0.6], [gy+gr+0.3, gy+gr+1.0], color=A700, lw=1.5)
txt(49.5, 45.2, "adjacency + distance edges +", 5.5)
txt(49.5, 42.8, "global node", 5.5)
txt(49.5, 40.2, "d_max = 6 → effective |i−j| ≤ 5", 5.2, color=MUT)
box(72, Y0, 24, H, T50, T600); txt(84, 59.5, "Encoders", 6.8, True, T900)
txt(84, 53.0, "node: AA + pos", 5.6)
txt(84, 50.4, "+ phys + cond", 5.6)
txt(84, 47.8, "→ h (256-d)", 5.6)
txt(84, 43.2, "edge: type + dist", 5.4)
txt(84, 40.8, "+ attrs → e (256-d)", 5.4)
box(100, Y0, 32, H, "white", T700, 1.3); txt(116, 59.5, "Message passing", 6.8, True, T900)
ax.add_patch(FancyBboxPatch((104, 50.2), 24, 6.2, boxstyle="round,pad=0.4", fc="#CCFBF1", ec=T600, lw=0.6))
txt(116, 53.3, "2 × edge-gated GAT\n(4 heads)", 5.6, True)
ax.add_patch(FancyBboxPatch((104, 44.6), 24, 4.0, boxstyle="round,pad=0.4", fc="#CCFBF1", ec=T600, lw=0.6))
txt(116, 46.6, "3 × residual GCN", 5.6, True)
txt(116, 42.4, "residual + LayerNorm,", 5.2, color=MUT)
txt(116, 40.2, "dropout 0.1", 5.2, color=MUT)
box(136, Y0, 26, H, T50, T600); txt(149, 59.5, "Bond head", 6.8, True, T900)
txt(149, 55.8, "(per bond i)", 5.4, color=MUT)
txt(149, 51.6, "[ h_i ‖ h_j ‖ e_ij", 5.6)
txt(149, 49.0, "‖ h_i−h_j ‖ h_i·h_j", 5.6)
txt(149, 46.4, "‖ W_t·t_i ]", 5.6)
txt(149, 42.4, "MLP → logit → σ → p_i", 5.4)
box(166, Y0, 15, H, "white", T600); txt(173.5, 59.5, "Output", 6.8, True, T900)
for i, v in enumerate([0.35, 0.75, 0.5, 0.9, 0.45]):
    ax.add_patch(Rectangle((168.5 + i*2.5, 45.4), 1.6, v*8.5, fc=A600 if v > 0.8 else T600))
ax.plot([168.2, 180.3], [49.65]*2, color=DASH, lw=0.8, ls=(0, (2, 2)))
txt(173.5, 42.6, "p_i, τ = 0.5", 5.2, color=MUT)
txt(173.5, 40.4, "5-seed avg", 5.2, color=MUT)
box(80, 5, 48, 16, A50, A600)
txt(104, 17.6, "FiLM conditioning", 6.6, True, A900)
ax.plot([117.5], [17.6], marker="*", ms=5.5, color=A600, ls="none")
txt(104, 12.4, "[0.1·q, 0.01·NCE] → MLP → (γ, β)", 5.6)
txt(104, 9.8, "h ← (1+γ)·h + β", 5.6)
txt(104, 7.0, "zero-init, ~33K params → all nodes", 5.2, color=MUT)
arrow(104, 21.5, 112, 37.3, A700, 1.8)
box(126, 68, 48, 14, A50, A600)
txt(150, 78.0, "15-d theory features", 6.6, True, A900)
ax.plot([165.0], [78.0], marker="*", ms=5.5, color=A600, ls="none")
txt(150, 74.0, "b/y m/z, masses, position,", 5.4)
txt(150, 71.6, "flanking residues, H2O/NH3-loss,", 5.4)
txt(150, 69.2, "Pro context (sequence-derived)", 5.4)
arrow(150, 67.6, 150, 64.6, A700, 1.8)
for x1, x2 in [(27.6, 30.4), (68.6, 71.4), (96.6, 99.4), (132.6, 135.4), (162.6, 165.4)]:
    arrow(x1, 51, x2, 51, "#475569", 2.0)
from audit_panel_alignment import require_matplotlib_panel_alignment
require_matplotlib_panel_alignment(fig, json_out="figures/model_architecture2.alignment.json",
                                   overlay_svg=None, strict=False)
out = "figures/model_architecture2"
fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".svg", bbox_inches="tight")
fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
print("saved")
