#!/usr/bin/env python3
"""
聚合 DBond-GT 5 折 peptide-level ranking 结果为 mean±std。

输入：candidate_ranking_analysis.py（--dedup_by_seq 模式）输出的 5 个 fold 目录，
每个含 ranking_summary.json（带 bond_metrics / spearman / enrichment）。

输出：
  - dbond_gt_ranking_summary.csv   论文填表直接用（mean±std）
  - dbond_gt_ranking_summary.json  含每折明细
  - run.log

用法：
  python graph_transform/scripts/aggregate_dbond_gt_ranking.py \
      --ranking_root result/ranking/dbond_gt_peptide \
      --folds 1222 2252 3514 6072 9075 \
      --output_dir result/ranking/dbond_gt_peptide
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r20_metrics import aggregate_folds, format_summary_table


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("gt_ranking_agg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s[%(levelname)s]:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, "aggregate.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_fold_summary(ranking_root: str, fold_id: str, logger: logging.Logger) -> Dict[str, Any]:
    """读单折 ranking_summary.json，转成 aggregate_folds 需要的格式。"""
    path = os.path.join(ranking_root, f"fold_{fold_id}", "ranking_summary.json")
    if not os.path.exists(path):
        logger.error(f"ranking_summary.json not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        summ = json.load(f)
    return {
        "fold_id": fold_id,
        "bond_metrics": summ.get("bond_metrics", {}),
        "ranking": {
            "spearman": summ.get("spearman", {}),
            "enrichment": summ.get("enrichment", []),
            "baseline": {
                "mean_R_true": summ.get("overall", {}).get("mean_R_true", 0.0),
                "mean_R_pred": summ.get("overall", {}).get("mean_R_pred", 0.0),
                "n_peptides": summ.get("n_peptides", 0),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate DBond-GT 5-fold peptide-level ranking")
    parser.add_argument(
        "--ranking_root", type=str, required=True,
        help="含 fold_{id}/ 子目录的根目录（candidate_ranking_analysis 的 --output_dir 父级）",
    )
    parser.add_argument(
        "--folds", type=str, nargs="+", default=["1222", "2252", "3514", "6072", "9075"],
    )
    parser.add_argument(
        "--model_name", type=str, default="DBond-GT",
        help="输出表里的模型名",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录，默认同 --ranking_root",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.ranking_root
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info(f"Aggregate DBond-GT 5-fold ranking | root={args.ranking_root}")
    logger.info("=" * 60)

    per_fold: List[Dict[str, Any]] = []
    for fold_id in args.folds:
        fold_data = load_fold_summary(args.ranking_root, fold_id, logger)
        if not fold_data:
            continue
        per_fold.append(fold_data)
        bm = fold_data["bond_metrics"]
        sp = fold_data["ranking"]["spearman"]
        logger.info(
            f"  fold {fold_id}: ROC-AUC={bm.get('roc_auc', 0):.4f}  "
            f"PR-AUC={bm.get('pr_auc', 0):.4f}  MCC={bm.get('mcc', 0):.4f}  "
            f"Brier={bm.get('brier_score', 0):.4f}  ECE={bm.get('ece', 0):.4f}  "
            f"ρ={sp.get('rho', 0):.4f} (n_pep={sp.get('n_peptides', 0)})"
        )

    if len(per_fold) < 2:
        logger.error(f"Only {len(per_fold)} folds found, need ≥2 to aggregate")
        return

    summary = aggregate_folds(per_fold)
    rows = format_summary_table(summary, args.model_name)

    # 控制台打印
    logger.info("\n" + "=" * 60)
    logger.info(f"{args.model_name} 5-fold aggregation (mean ± std)")
    logger.info("=" * 60)
    for row in rows:
        logger.info(f"  {row['category']:<28} {row['metric']:<28} {row['mean±std']}")

    # 总 peptide 数（5 折合计唯一序列）
    total_peptides = sum(f["ranking"]["spearman"].get("n_peptides", 0) for f in per_fold)
    logger.info(f"\n  Total unique peptide sequences across {len(per_fold)} folds: {total_peptides}")

    # 写 CSV
    csv_path = os.path.join(output_dir, "dbond_gt_ranking_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"\nSaved: {csv_path}")

    # 写 JSON（含每折明细 + 合计 peptide 数）
    summary["model_name"] = args.model_name
    summary["total_unique_peptides"] = total_peptides
    summary["n_folds"] = len(per_fold)
    json_path = os.path.join(output_dir, "dbond_gt_ranking_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {json_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
