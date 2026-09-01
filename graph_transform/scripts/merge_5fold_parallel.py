#!/usr/bin/env python3
"""汇聚折级并行（train_5fold_parallel.py）产出的各折 test 指标，计算 mean±std。

扫描 <checkpoint_base>/5fold_par/<fold_id>/5fold/*/fold_<id>/metrics/gt_pre/latest_test_metric.csv，
按 fold_id 去重（取最新），输出 mean±std 表并落盘 CSV。

用法（devpod，项目根目录）：
    .venv/bin/python graph_transform/scripts/merge_5fold_parallel.py \
        --checkpoint_base checkpoints/graph_transform/pre_synthesis \
        --output result/metric/graph_transform/5fold_md6_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_KEYS = [
    "f1", "accuracy", "auc", "pr_auc", "mcc", "brier_score", "ece",
    "spearman_rho", "top10_precision", "top20_precision", "top50_precision",
    "ex_f1", "lab_f1_mi", "subset_acc", "bond_acc",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_base", default="checkpoints/graph_transform/pre_synthesis")
    ap.add_argument("--output", default="result/metric/graph_transform/5fold_parallel_summary.csv")
    ap.add_argument("--tag", default="gt_pre", help="ablation tag 子目录名")
    args = ap.parse_args()

    pattern = os.path.join(
        args.checkpoint_base, "5fold_par", "fold_*", "5fold", "*", "fold_*",
        "metrics", args.tag, "latest_test_metric.csv",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"ERROR: no per-fold metric csv under {pattern}", file=sys.stderr)
        sys.exit(1)

    # fold_id -> 最新 csv（同折多次运行取 mtime 最新）
    by_fold: dict[str, str] = {}
    for p in paths:
        parts = p.split(os.sep)
        outer_fold = parts[-8]   # 5fold_par/fold_<id>
        mtime = os.path.getmtime(p)
        if outer_fold not in by_fold or mtime > os.path.getmtime(by_fold[outer_fold]):
            by_fold[outer_fold] = p

    rows = {}
    for fold_dir, path in sorted(by_fold.items()):
        fold_id = fold_dir.replace("fold_", "")
        df = pd.read_csv(path)
        d = {r["metric"]: float(r["value"]) for _, r in df.iterrows()}
        rows[fold_id] = d
        print("fold {}: f1={:.4f} pr_auc={:.4f} mcc={:.4f} auc={:.4f} bond_acc/lab_acc_mi={:.4f}/{:.4f}".format(
            fold_id, d.get("f1", float("nan")), d.get("pr_auc", float("nan")),
            d.get("mcc", float("nan")), d.get("auc", float("nan")),
            d.get("bond_acc", float("nan")), d.get("lab_acc_mi", float("nan"))))

    print("\n=== {} 折汇总 mean±std ===".format(len(rows)))
    out_rows = []
    for k in DEFAULT_KEYS:
        vals = np.array([rows[f][k] for f in rows if k in rows[f]], dtype=float)
        if vals.size == 0:
            continue
        mean, std = float(vals.mean()), float(vals.std(ddof=0))
        out_rows.append({"metric": k, "mean": mean, "std": std,
                         "min": float(vals.min()), "max": float(vals.max()),
                         "num_folds": int(vals.size)})
        print("{:>12}: {:.4f} ± {:.4f}  [min={:.4f}, max={:.4f}]".format(
            k, mean, std, vals.min(), vals.max()))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pd.DataFrame(out_rows, columns=["metric", "mean", "std", "min", "max", "num_folds"]).to_csv(args.output, index=False)
    print("\nsaved:", args.output)


if __name__ == "__main__":
    main()
