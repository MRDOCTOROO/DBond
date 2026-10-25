# -*- coding: utf-8 -*-
"""证据包：遮掩一致性逐样本相关系数的【序列整簇 bootstrap】置信区间（fold 9075）。

背景（审稿 F04）：150 个样本仅对应 80 条唯一序列，样本级重抽样未考虑序列内
重复，区间被低估；本脚本以整条序列为重抽单元给出对照区间。

数据：G:/paperbib/fold_9075/occlusion/occlusion_per_sample.json
      （pod: result/interp_film/fold_9075/occlusion/occlusion_per_sample.json 的镜像）
用法：python cluster_ci.py [per_sample_json]
输出：cluster_ci_results.json（同目录）
"""
import io
import json
import os
import sys
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np

DEFAULT = r"G:/paperbib/fold_9075/occlusion/occlusion_per_sample.json"
ITERS = 5000
SEED = 42


def cluster_ci(rs, seqs, mask, iters=ITERS):
    idx = np.where(mask)[0]
    by_seq = defaultdict(list)
    for i in idx:
        by_seq[seqs[i]].append(rs[i])
    keys = list(by_seq)
    rng = np.random.default_rng(SEED)
    boot = np.empty(iters)
    for k in range(iters):
        draw = rng.choice(len(keys), size=len(keys), replace=True)
        boot[k] = np.concatenate([by_seq[keys[d]] for d in draw]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), float(rs[idx].mean()), len(keys)


def sample_ci(rs, mask, iters=ITERS):
    idx = np.where(mask)[0]
    vals = rs[idx]
    rng = np.random.default_rng(SEED)
    boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(iters)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    occ = json.load(open(path, encoding="utf-8"))
    rs = np.array([x["pearson_r"] for x in occ])
    seqs = [x["sequence"] for x in occ]
    chg = [x["charge"] for x in occ]
    fbr = [x["fbr"] for x in occ]
    ln = [x["seq_len"] for x in occ]

    out = {"source": path, "n_samples": len(rs), "n_unique_sequences": len(set(seqs)),
           "iters": ITERS, "seed": SEED, "groups": {}}

    allm = np.ones(len(rs), bool)
    lo, hi, mean, nclu = cluster_ci(rs, seqs, allm)
    slo, shi = sample_ci(rs, allm)
    out["overall"] = {"mean": mean, "clustered_ci": [lo, hi], "sample_ci": [slo, shi],
                      "n_clusters": nclu}
    print("ALL mean=%.4f clustered=[%.4f,%.4f] sample=[%.4f,%.4f]" % (mean, lo, hi, slo, shi))

    for lab, mask in [("charge", None)]:
        pass
    for z in sorted(set(chg)):
        m = np.array([c == z for c in chg])
        lo, hi, mean, nclu = cluster_ci(rs, seqs, m)
        out["groups"][f"charge_{z:g}"] = {"n": int(m.sum()), "mean": mean,
                                          "clustered_ci": [lo, hi], "sample_ci": list(sample_ci(rs, m))}
    for lab, m in [("fbr_low", np.array([f < 0.3 for f in fbr])),
                   ("fbr_med", np.array([(0.3 <= f <= 0.7) for f in fbr])),
                   ("fbr_high", np.array([f > 0.7 for f in fbr]))]:
        lo, hi, mean, nclu = cluster_ci(rs, seqs, m)
        out["groups"][lab] = {"n": int(m.sum()), "mean": mean,
                              "clustered_ci": [lo, hi], "sample_ci": list(sample_ci(rs, m))}
    order = np.argsort(ln, kind="stable")
    q = np.zeros(len(ln), int)
    q[order[:64]] = 0
    q[order[64:126]] = 1
    q[order[126:]] = 2
    for lab, k in [("len_q1", 0), ("len_q2", 1), ("len_q3", 2)]:
        m = q == k
        lo, hi, mean, nclu = cluster_ci(rs, seqs, m)
        out["groups"][lab] = {"n": int(m.sum()), "mean": mean,
                              "clustered_ci": [lo, hi], "sample_ci": list(sample_ci(rs, m))}

    for k, v in out["groups"].items():
        print("%-9s n=%3d mean=%.4f clustered=[%.4f,%.4f] sample=[%.4f,%.4f]" %
              (k, v["n"], v["mean"], v["clustered_ci"][0], v["clustered_ci"][1],
               v["sample_ci"][0], v["sample_ci"][1]))

    o = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_ci_results.json")
    json.dump(out, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved:", o)


if __name__ == "__main__":
    main()