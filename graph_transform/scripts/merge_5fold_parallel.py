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
    ap.add_argument("--run_ts", default=None,
                    help="只汇聚指定 5fold/<时间戳> 目录的那次运行（推荐多次运行后使用）")
    ap.add_argument("--allow_mixed", action="store_true",
                    help="允许混合不同次运行的折（不推荐，会破坏 mean±std 含义）")
    args = ap.parse_args()

    pattern = os.path.join(
        args.checkpoint_base, "5fold_par", "fold_*", "5fold", "*", "fold_*",
        "metrics", args.tag, "latest_test_metric.csv",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"ERROR: no per-fold metric csv under {pattern}", file=sys.stderr)
        sys.exit(1)

    # fold_id -> (run_ts, path)。同折多次运行取 mtime 最新；
    # run_ts = 5fold/<时间戳> 目录名，用于混合运行检测。
    by_fold: dict[str, tuple[str, str]] = {}
    for p in paths:
        parts = p.split(os.sep)
        outer_fold = parts[-8]   # .../5fold_par/fold_<id>/5fold/<ts>/fold_<id>/metrics/<tag>/file
        run_ts = parts[-5]
        mtime = os.path.getmtime(p)
        if outer_fold not in by_fold or mtime > os.path.getmtime(by_fold[outer_fold][1]):
            by_fold[outer_fold] = (run_ts, p)

    # 混合运行检测：各折必须来自同一次 5 折运行（同一 run_ts），否则
    # mean±std 会把不同配置的折静默混在一起。--allow_mixed 可显式放行。
    used_ts = {ts for ts, _ in by_fold.values()}
    if len(used_ts) > 1 and not args.allow_mixed:
        print("ERROR: 检测到来自多次运行的折结果混用:", file=sys.stderr)
        for fold, (ts, p) in sorted(by_fold.items()):
            print(f"  fold_{fold.replace('fold_', '')}: run_ts={ts}  {p}", file=sys.stderr)
        print("请用 --run_ts <时间戳> 指定要汇聚的那次运行，或 --allow_mixed 强制合并。",
              file=sys.stderr)
        sys.exit(2)
    if args.run_ts:
        by_fold = {f: v for f, v in by_fold.items() if v[0] == args.run_ts}
        if not by_fold:
            print(f"ERROR: --run_ts {args.run_ts} 没有匹配的折结果", file=sys.stderr)
            sys.exit(1)

    rows = {}
    for fold_dir, (run_ts, path) in sorted(by_fold.items()):
        fold_id = fold_dir.replace("fold_", "")
        df = pd.read_csv(path)
        d = {r["metric"]: float(r["value"]) for _, r in df.iterrows()}
        rows[fold_id] = d
        print("fold {} (run_ts={}): f1={:.4f} pr_auc={:.4f} mcc={:.4f} auc={:.4f} bond_acc/lab_acc_mi={:.4f}/{:.4f}".format(
            fold_id, run_ts, d.get("f1", float("nan")), d.get("pr_auc", float("nan")),
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
