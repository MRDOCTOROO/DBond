#!/usr/bin/env python3
"""
R-20 五折聚合评估脚本

对每个 fold：评估 best_model.pt（产出含 pr_auc/mcc/brier_score/ece 的 metric CSV
+ 含 pred_prob 的 pred CSV） → 跑 candidate ranking 分析（Spearman + Top-k enrichment）。
最后把 5 折的所有 R-20 指标聚合成 mean±std 表格。

目录约定（来自你的实际结构）：
  {cv_root}/
  ├── fold_{id}/
  │   ├── config.yaml                              # 该 fold 的完整 config（含 ablation.tag）
  │   └── checkpoints/{tag}/{ts}/best_model.pt     # 训练好的权重

输出（默认 {cv_root}/r20_aggregation/）：
  - per_fold/                                       每折原始结果
  │   ├── fold_{id}_metric.csv
  │   ├── fold_{id}.pred.csv
  │   └── fold_{id}_ranking/{ranking_summary.json, per_peptide.csv, enrichment_table.csv}
  - r20_summary.csv                                 R-20 全部指标的 5fold mean±std
  - r20_summary.json                                同上 + 每折明细
  - run.log

用法：
  python graph_transform/scripts/aggregate_r20_5fold.py \
      --cv_root checkpoints/graph_transform/5fold/20260422_232825sqence_graph \
      --folds 1222 2252 3514 6072 9075 \
      --tag sequence_graph \
      --top_k_fractions 0.1 0.2 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("r20_aggregate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s[%(levelname)s]:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def find_best_model(fold_dir: str, tag: str, logger: logging.Logger) -> Optional[str]:
    """在 fold_dir/checkpoints/{tag}/*/best_model.pt 里找权重，若有多个取最新。"""
    pattern = os.path.join(fold_dir, "checkpoints", tag, "*", "best_model.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        # 兜底：不限 tag 子目录
        pattern2 = os.path.join(fold_dir, "checkpoints", "*", "best_model.pt")
        candidates = glob.glob(pattern2)
        if candidates:
            logger.warning(
                f"No best_model.pt under checkpoints/{tag}/*/ ; falling back to {pattern2}"
            )
    if not candidates:
        pattern3 = os.path.join(fold_dir, "**", "best_model.pt", recursive=True)
        candidates = glob.glob(pattern3, recursive=True)
    if not candidates:
        return None
    # 多个取修改时间最新的
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def run_evaluate(
    config: str,
    checkpoint: str,
    out_metric: str,
    out_pred: str,
    logger: logging.Logger,
) -> bool:
    """调用 evaluate_graph_model.py，返回是否成功。"""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "evaluate_graph_model.py"),
        "--config", config,
        "--checkpoint", checkpoint,
        "--out_metric_csv", out_metric,
        "--out_pred_csv", out_pred,
    ]
    logger.info(f"  EVAL CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  EVAL FAILED (rc={result.returncode})")
        logger.error(f"  stderr:\n{result.stderr[-3000:]}")
        logger.error(f"  stdout:\n{result.stdout[-2000:]}")
        return False
    # 评估成功时 evaluate_graph_model.py 会打印这两行
    logger.info(f"  EVAL OK: {out_metric}")
    return True


def run_ranking(
    pred_csv: str,
    output_dir: str,
    top_k_fractions: List[float],
    logger: logging.Logger,
) -> Optional[str]:
    """调用 candidate_ranking_analysis.py，返回 ranking_summary.json 路径或 None。"""
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "candidate_ranking_analysis.py"),
        "--pred_csv", pred_csv,
        "--output_dir", output_dir,
        "--top_k_fractions", *[str(f) for f in top_k_fractions],
    ]
    logger.info(f"  RANK CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  RANK FAILED (rc={result.returncode})")
        logger.error(f"  stderr:\n{result.stderr[-2000:]}")
        return None
    summary_path = os.path.join(output_dir, "ranking_summary.json")
    if not os.path.exists(summary_path):
        logger.error(f"  ranking_summary.json not found at {summary_path}")
        return None
    logger.info(f"  RANK OK: {summary_path}")
    return summary_path


def load_metric_csv(path: str) -> Dict[str, float]:
    """读 metric CSV 为 {metric_name: value} 字典。"""
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        try:
            out[str(row["metric"])] = float(row["value"])
        except (ValueError, TypeError):
            continue
    return out


def load_ranking_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# R-20 论文表需要的指标（key = 显示名, value = (metric_csv 中的 key, 是否该值越高越好)）
# ranking 相关的单独处理
R20_METRIC_KEYS: List[Tuple[str, str]] = [
    ("ROC-AUC", "auc"),
    ("PR-AUC", "pr_auc"),
    ("MCC", "mcc"),
    ("Brier_score", "brier_score"),
    ("ECE", "ece"),
    # 附带几个常规分类指标（论文主表本来就有，聚合方便复用）
    ("accuracy", "accuracy"),
    ("f1", "f1"),
    ("subset_acc", "subset_acc"),
]


def aggregate_metric(per_fold_metrics: List[Tuple[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    """把 5 折的 metric 字典聚合成 mean±std 行。"""
    rows: List[Dict[str, Any]] = []
    for display, key in R20_METRIC_KEYS:
        values = []
        for fold_id, metrics in per_fold_metrics:
            if key in metrics:
                values.append((fold_id, metrics[key]))
        if not values:
            continue
        arr = np.array([v for _, v in values], dtype=float)
        rows.append({
            "metric": display,
            "source_key": key,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_folds": int(arr.size),
            "per_fold": {fid: val for fid, val in values},
        })
    return rows


def aggregate_ranking(per_fold_ranking: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """把 5 折的 ranking_summary 聚合。"""
    # Spearman
    rho_values = []
    for fold_id, summ in per_fold_ranking:
        sp = summ.get("spearman", {})
        if "rho" in sp:
            rho_values.append((fold_id, float(sp["rho"])))
    spearman_agg = {}
    if rho_values:
        arr = np.array([v for _, v in rho_values], dtype=float)
        spearman_agg = {
            "rho_mean": float(arr.mean()),
            "rho_std": float(arr.std(ddof=0)),
            "rho_min": float(arr.min()),
            "rho_max": float(arr.max()),
            "n_folds": int(arr.size),
            "per_fold_rho": {fid: v for fid, v in rho_values},
        }

    # Top-k enrichment：对每个 fraction 聚合 mean_R_true / mean_R_pred / enrichment_ratio
    enrichment_agg: Dict[str, Dict[str, float]] = {}
    # 用第一折的 fraction 列表作为基准（所有折应一致）
    if per_fold_ranking:
        first_summ = per_fold_ranking[0][1]
        for entry in first_summ.get("enrichment", []):
            sel = entry.get("selection")
            if sel == "All":
                continue
            frac = entry.get("fraction")
            true_vals, pred_vals, ratio_vals = [], [], []
            for fold_id, summ in per_fold_ranking:
                for e in summ.get("enrichment", []):
                    if e.get("selection") == sel:
                        true_vals.append((fold_id, float(e["mean_R_true"])))
                        pred_vals.append((fold_id, float(e["mean_R_pred"])))
                        ratio_vals.append((fold_id, float(e["enrichment_ratio"])))
                        break
            if not true_vals:
                continue
            true_arr = np.array([v for _, v in true_vals])
            ratio_arr = np.array([v for _, v in ratio_vals])
            enrichment_agg[str(sel)] = {
                "fraction": frac,
                "mean_R_true_mean": float(true_arr.mean()),
                "mean_R_true_std": float(true_arr.std(ddof=0)),
                "enrichment_ratio_mean": float(ratio_arr.mean()),
                "enrichment_ratio_std": float(ratio_arr.std(ddof=0)),
                "n_folds": int(true_arr.size),
                "per_fold_mean_R_true": {fid: v for fid, v in true_vals},
            }

    # All 基线
    baseline_agg = {}
    all_true_vals = []
    for fold_id, summ in per_fold_ranking:
        for e in summ.get("enrichment", []):
            if e.get("selection") == "All":
                all_true_vals.append((fold_id, float(e["mean_R_true"])))
                break
    if all_true_vals:
        arr = np.array([v for _, v in all_true_vals])
        baseline_agg = {
            "baseline_R_true_mean": float(arr.mean()),
            "baseline_R_true_std": float(arr.std(ddof=0)),
            "per_fold": {fid: v for fid, v in all_true_vals},
        }

    return {"spearman": spearman_agg, "topk": enrichment_agg, "baseline": baseline_agg}


def format_mean_std(mean: float, std: float, fmt: str = ".4f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


def main():
    parser = argparse.ArgumentParser(description="R-20 five-fold aggregation: evaluate + ranking")
    parser.add_argument(
        "--cv_root", type=str, required=True,
        help="5fold 根目录，如 checkpoints/graph_transform/5fold/20260422_232825sqence_graph",
    )
    parser.add_argument(
        "--folds", type=str, nargs="+", default=["1222", "2252", "3514", "6072", "9075"],
        help="fold id 列表（对应目录名 fold_{id}）",
    )
    parser.add_argument(
        "--tag", type=str, default="sequence_graph",
        help="ablation tag（用于定位 checkpoints/{tag}/* 子目录）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录，默认 {cv_root}/r20_aggregation/",
    )
    parser.add_argument(
        "--top_k_fractions", type=float, nargs="+", default=[0.1, 0.2, 0.5],
        help="Top-k 候选比例",
    )
    parser.add_argument(
        "--skip_evaluate", action="store_true",
        help="跳过评估步骤（若 pred/metric CSV 已存在则直接用，仅补 ranking 或聚合）",
    )
    parser.add_argument(
        "--force_rerun", action="store_true",
        help="即使结果已存在也强制重跑（默认：metric/pred 存在则跳过评估）",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.cv_root, "r20_aggregation")
    per_fold_dir = os.path.join(output_dir, "per_fold")
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info(f"R-20 5-fold aggregation | cv_root={args.cv_root} | tag={args.tag}")
    logger.info(f"folds={args.folds} | top_k={args.top_k_fractions}")
    logger.info(f"output={output_dir}")
    logger.info("=" * 70)

    if not os.path.isdir(args.cv_root):
        raise FileNotFoundError(f"cv_root not found: {args.cv_root}")

    per_fold_metrics: List[Tuple[str, Dict[str, float]]] = []
    per_fold_ranking: List[Tuple[str, Dict[str, Any]]] = []
    failed_folds: List[str] = []

    for fold_id in args.folds:
        logger.info("=" * 50)
        logger.info(f"FOLD {fold_id}")
        logger.info("=" * 50)
        fold_dir = os.path.join(args.cv_root, f"fold_{fold_id}")
        config_path = os.path.join(fold_dir, "config.yaml")
        if not os.path.exists(config_path):
            logger.error(f"config.yaml not found: {config_path}")
            failed_folds.append(fold_id)
            continue
        if not args.skip_evaluate:
            checkpoint = find_best_model(fold_dir, args.tag, logger)
            if checkpoint is None:
                logger.error(f"best_model.pt not found under {fold_dir}")
                failed_folds.append(fold_id)
                continue
            logger.info(f"  checkpoint: {checkpoint}")

        fold_out = os.path.join(per_fold_dir, f"fold_{fold_id}")
        os.makedirs(fold_out, exist_ok=True)
        metric_csv = os.path.join(fold_out, "metric.csv")
        pred_csv = os.path.join(fold_out, "pred.csv")
        ranking_dir = os.path.join(fold_out, "ranking")
        ranking_summary = os.path.join(ranking_dir, "ranking_summary.json")

        # 评估（除非已有且未强制重跑）
        need_evaluate = True
        if os.path.exists(metric_csv) and os.path.exists(pred_csv) and not args.force_rerun:
            logger.info(f"  metric/pred CSV exist, skipping evaluate (use --force_rerun to override)")
            need_evaluate = False
        if need_evaluate and not args.skip_evaluate:
            ok = run_evaluate(config_path, checkpoint, metric_csv, pred_csv, logger)
            if not ok:
                failed_folds.append(fold_id)
                continue

        # 读取 metric
        if os.path.exists(metric_csv):
            metrics = load_metric_csv(metric_csv)
            per_fold_metrics.append((fold_id, metrics))
            # 打印 R-20 关键指标
            for disp, key in R20_METRIC_KEYS:
                if key in metrics:
                    logger.info(f"    {disp:<14}: {metrics[key]:.4f}")
        else:
            logger.warning(f"  metric CSV missing: {metric_csv}")

        # ranking
        need_ranking = True
        if os.path.exists(ranking_summary) and not args.force_rerun:
            logger.info(f"  ranking_summary.json exists, skipping ranking")
            need_ranking = False
        if need_ranking and os.path.exists(pred_csv):
            summary_path = run_ranking(pred_csv, ranking_dir, args.top_k_fractions, logger)
            if summary_path is None:
                logger.warning(f"  ranking failed for fold {fold_id}")
        if os.path.exists(ranking_summary):
            summ = load_ranking_summary(ranking_summary)
            per_fold_ranking.append((fold_id, summ))
            sp = summ.get("spearman", {})
            logger.info(f"    Spearman ρ    : {sp.get('rho', float('nan')):.4f} (p={sp.get('p_value', float('nan')):.2e})")
            for e in summ.get("enrichment", []):
                if e.get("selection") != "All":
                    logger.info(
                        f"    {e['selection']:<8} R_true={e['mean_R_true']:.4f} "
                        f"ratio={e['enrichment_ratio']:.3f}"
                    )

    if failed_folds:
        logger.warning(f"\nFailed folds: {failed_folds}")

    if len(per_fold_metrics) < 2 and len(per_fold_ranking) < 2:
        logger.error("Not enough successful folds (<2) to aggregate. Exiting.")
        return

    logger.info("\n" + "=" * 70)
    logger.info("Aggregating 5-fold results")
    logger.info("=" * 70)

    metric_agg = aggregate_metric(per_fold_metrics) if per_fold_metrics else []
    ranking_agg = aggregate_ranking(per_fold_ranking) if per_fold_ranking else {}

    # ---- 写 r20_summary.csv（论文填表直接用）----
    summary_csv_rows: List[Dict[str, str]] = []
    for row in metric_agg:
        summary_csv_rows.append({
            "category": "discrimination_calibration",
            "metric": row["metric"],
            "mean±std": format_mean_std(row["mean"], row["std"]),
            "mean": f"{row['mean']:.6f}",
            "std": f"{row['std']:.6f}",
            "min": f"{row['min']:.6f}",
            "max": f"{row['max']:.6f}",
            "n_folds": str(row["n_folds"]),
        })
    if ranking_agg.get("spearman"):
        sp = ranking_agg["spearman"]
        summary_csv_rows.append({
            "category": "ranking",
            "metric": "Spearman_rho",
            "mean±std": format_mean_std(sp["rho_mean"], sp["rho_std"]),
            "mean": f"{sp['rho_mean']:.6f}",
            "std": f"{sp['rho_std']:.6f}",
            "min": f"{sp['rho_min']:.6f}",
            "max": f"{sp['rho_max']:.6f}",
            "n_folds": str(sp["n_folds"]),
        })
    for sel, agg in ranking_agg.get("topk", {}).items():
        summary_csv_rows.append({
            "category": "ranking_topk",
            "metric": f"{sel}_mean_R_true",
            "mean±std": format_mean_std(agg["mean_R_true_mean"], agg["mean_R_true_std"]),
            "mean": f"{agg['mean_R_true_mean']:.6f}",
            "std": f"{agg['mean_R_true_std']:.6f}",
            "min": "",
            "max": "",
            "n_folds": str(agg["n_folds"]),
        })
        summary_csv_rows.append({
            "category": "ranking_topk",
            "metric": f"{sel}_enrichment_ratio",
            "mean±std": format_mean_std(agg["enrichment_ratio_mean"], agg["enrichment_ratio_std"]),
            "mean": f"{agg['enrichment_ratio_mean']:.6f}",
            "std": f"{agg['enrichment_ratio_std']:.6f}",
            "min": "",
            "max": "",
            "n_folds": str(agg["n_folds"]),
        })
    if ranking_agg.get("baseline"):
        bl = ranking_agg["baseline"]
        summary_csv_rows.append({
            "category": "ranking_baseline",
            "metric": "All_mean_R_true",
            "mean±std": format_mean_std(bl["baseline_R_true_mean"], bl["baseline_R_true_std"]),
            "mean": f"{bl['baseline_R_true_mean']:.6f}",
            "std": f"{bl['baseline_R_true_std']:.6f}",
            "min": "",
            "max": "",
            "n_folds": str(len(bl.get("per_fold", {}))),
        })

    summary_csv = os.path.join(output_dir, "r20_summary.csv")
    pd.DataFrame(summary_csv_rows).to_csv(summary_csv, index=False)
    logger.info(f"\nSaved 5-fold summary CSV: {summary_csv}")

    # ---- 写 r20_summary.json（含每折明细）----
    summary_json = {
        "cv_root": os.path.abspath(args.cv_root),
        "tag": args.tag,
        "folds": args.folds,
        "failed_folds": failed_folds,
        "top_k_fractions": args.top_k_fractions,
        "metric_aggregation": metric_agg,
        "ranking_aggregation": ranking_agg,
    }
    summary_json_path = os.path.join(output_dir, "r20_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved 5-fold summary JSON: {summary_json_path}")

    # ---- 控制台打印最终表格 ----
    logger.info("\n" + "=" * 70)
    logger.info("R-20 Five-fold Summary (mean ± std)")
    logger.info("=" * 70)
    logger.info(f"{'Category':<28} {'Metric':<28} {'mean ± std':<20}")
    logger.info("-" * 80)
    for row in summary_csv_rows:
        logger.info(f"{row['category']:<28} {row['metric']:<28} {row['mean±std']:<20}")
    logger.info("=" * 80)
    logger.info("Done.")


if __name__ == "__main__":
    main()
