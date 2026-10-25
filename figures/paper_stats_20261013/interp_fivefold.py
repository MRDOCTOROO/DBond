# -*- coding: utf-8 -*-
"""证据包：五组划分的解释性汇总复算（attention 对齐 + occlusion 一致性）。

数据：pod `result/interp_film/fold_{1222,2252,3514,6072,9075}/occlusion/occlusion_summary.json`
      的镜像（已复制到本目录 fold_*/ 下）。
核对对象（正文/SI）：全局 Pearson 0.283±0.054；全局 Spearman 0.355±0.025；
逐样本 r 的划分级均值 0.291±0.056；强一致性（r>=0.5）合计 1/750。
输出：interp_fivefold_results.json（同目录）
"""
import glob
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def mstd(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1))


g_pear, g_spear, m_samp = [], [], []
n_total, n_strong_total = 0, 0
per_fold = {}
for path in sorted(glob.glob(os.path.join(HERE, "fold_*", "occlusion_summary.json"))):
    d = json.load(open(path, encoding="utf-8"))
    fold = os.path.basename(os.path.dirname(path)).replace("fold_", "")
    _psr = d.get("per_sample_pearson_r") or {}
    if isinstance(_psr, dict):
        r_list = [float(v) for v in _psr.values()]
    else:
        r_list = [float(x) for x in _psr]
    n_strong = int(sum(1 for r in r_list if r >= 0.5))
    n_total += len(r_list)
    n_strong_total += n_strong
    g_pear.append(float(d["aggregate"]["global_pearson_r"]))
    g_spear.append(float(d["aggregate"]["global_spearman_rho"]))
    m_samp.append(float(d["mean_per_sample_pearson_r"]))
    per_fold[fold] = {"n": len(r_list), "strong_r_ge_0.5": n_strong,
                      "global_pearson": g_pear[-1], "global_spearman": g_spear[-1],
                      "mean_per_sample_r": m_samp[-1]}

out = {"per_fold": per_fold,
       "five_fold": {
           "global_pearson_mean_std": mstd(g_pear),
           "global_spearman_mean_std": mstd(g_spear),
           "mean_per_sample_r_mean_std": mstd(m_samp)},
       "strong_consistency": {"n_samples_total": n_total, "n_strong_total": n_strong_total}}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(os.path.join(HERE, "interp_fivefold_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved: interp_fivefold_results.json")