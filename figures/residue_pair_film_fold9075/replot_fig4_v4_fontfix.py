# -*- coding: utf-8 -*-
"""Figure 4 font-legibility fix (20261014): identical content/layout to replot_fig4.py,
but all point sizes are enlarged ~2.3x so that, after LaTeX scales the 30-in canvas to
1.0\\linewidth (442 pt), the rendered text meets the paper's 5-pt glyph floor:
  - axis tick labels  : 11 -> 25 pt  (rendered ~5.3 pt)
  - axis titles       : 12 -> 29 pt  (rendered ~6.2 pt)
  - panel titles      : 13 -> 30 pt  (rendered ~6.4 pt)
  - colorbar label    : 10 -> 25 pt  (rendered ~5.3 pt), colorbar tick labels -> 25 pt
  - in-cell annotations: 9.5 -> 15 pt (rendered ~3.2 pt; bounded by the 34-pt cell width,
    a 3-digit value cannot exceed ~16 pt without overlap)
  - grid / cell-edge line widths bumped proportionally.
Data and panel content unchanged: figures/residue_pair_film_fold9075/residue_pair_stats.csv
(fold 9075 film), N<50 gray-cell mask on all three panels, diff clip +-20 pp.
"""
import io
import os
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = r"D:\UserData\Desktop\Github-project\Dbond2\DBond"
CSV = os.path.join(ROOT, "figures", "residue_pair_film_fold9075", "residue_pair_stats.csv")
OUT = r"G:\paperbib\graphtransformer\residue_pair_matrix_v3.pdf"
BACKUP = r"G:\paperbib\graphtransformer\residue_pair_matrix_v3_old_20261014.pdf"
RARE = (10, 50)
FIG = (30, 10.5)

FS_TICK = 25
FS_AXLAB = 29
FS_TITLE = 30
FS_CBAR = 25
FS_ANNOT = 15

df = pd.read_csv(CSV)
aa_rows = list(dict.fromkeys(df["X"]))
aa_cols = list(dict.fromkeys(df["Y"]))
assert aa_rows == aa_cols, "X/Y order mismatch"
aa = aa_rows
n = len(aa)

emp = df.pivot(index="X", columns="Y", values="empirical").reindex(index=aa, columns=aa).to_numpy()
pred = df.pivot(index="X", columns="Y", values="predicted").reindex(index=aa, columns=aa).to_numpy()
counts = df.pivot(index="X", columns="Y", values="N").reindex(index=aa, columns=aa).to_numpy(dtype=float)

emp = np.nan_to_num(emp, nan=0.0)
pred = np.nan_to_num(pred, nan=0.0)
emp_pct = emp * 100.0
pred_pct = pred * 100.0
diff_pct = pred_pct - emp_pct

rare_aa = {"B", "O", "X", "Z"}
fig, axes = plt.subplots(1, 3, figsize=FIG)
plt.subplots_adjust(wspace=0.40, left=0.05, right=0.97, top=0.90, bottom=0.16)


def draw(ax, matrix, vmin, vmax, cmap, title, annotate):
    im = ax.imshow(matrix, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(aa, fontsize=FS_TICK)
    ax.set_yticklabels(aa, fontsize=FS_TICK)
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        if tl.get_text() in rare_aa:
            tl.set_color("#888888")
    ax.set_xlabel("C-terminal residue Y", fontsize=FS_AXLAB, fontweight="bold", labelpad=10)
    ax.set_ylabel("N-terminal residue X", fontsize=FS_AXLAB, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=16)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(n):
        for j in range(n):
            if int(counts[i, j]) < RARE[1]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                           facecolor="#EEEEEE", edgecolor="#BBBBBB",
                                           linewidth=1.0, zorder=1))
    if annotate:
        for i in range(n):
            for j in range(n):
                c = int(counts[i, j])
                if c < RARE[1]:
                    continue
                v = matrix[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color="black")
    return im


im_a = draw(axes[0], emp_pct, 0.0, 100.0, "YlOrRd", "(a) Empirical", True)
im_b = draw(axes[1], pred_pct, 0.0, 100.0, "YlOrRd", "(b) Model", False)
im_c = draw(axes[2], diff_pct, -20.0, 20.0, "RdBu_r", "(c) Bias", False)

for im, ax, lab in [(im_a, axes[0], "Cleavage rate (%)"),
                    (im_b, axes[1], "Model probability (%)"),
                    (im_c, axes[2], "Bias (pp)")]:
    cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cb.set_label(lab, fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_CBAR)

if os.path.exists(OUT) and os.path.exists(BACKUP) is False:
    shutil.copy2(OUT, BACKUP)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", OUT)
print("backup:", BACKUP if os.path.exists(BACKUP) else "(already existed)")