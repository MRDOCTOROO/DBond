# -*- coding: utf-8 -*-
"""证据包：主图4 的 r / MAE / 274 对与 |diff|<=10pp 占比复算。

数据：figures/residue_pair_film_fold9075/residue_pair_stats.csv（fold 9075 film 臂
测试集逐残基对统计；total_samples=100,539 与 SI 折表一致）。
口径（与绘图代码 plot_residue_pair_matrix 相同）：N>=50 的残基对为展示/统计集；
r = Pearson(empirical, predicted)（百分比尺度不变）；MAE = mean|pred-emp|（pp）。
输出：fig4_recompute_results.json（同目录）
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "residue_pair_film_fold9075", "residue_pair_stats.csv")

df = pd.read_csv(CSV)
N = df["N"].to_numpy()
rel = df[df["N"] >= 50]
emp = rel["empirical"].to_numpy() * 100
pred = rel["predicted"].to_numpy() * 100
r = float(np.corrcoef(emp, pred)[0, 1])
mae = float(np.mean(np.abs(pred - emp)))
frac10 = float(np.mean(np.abs(pred - emp) <= 10))

out = {
    "source_csv": os.path.normpath(CSV),
    "cells_total": int(len(df)),
    "cells_n_zero": int((N == 0).sum()),
    "cells_0_lt_n_lt_50": int(((N > 0) & (N < 50)).sum()),
    "cells_n_ge_50": int((N >= 50).sum()),
    "pearson_r": r,
    "mae_pp": mae,
    "frac_abs_diff_le_10pp": frac10,
}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "fig4_recompute_results.json"), "w", encoding="utf-8"),
          indent=1)
print("saved: fig4_recompute_results.json")