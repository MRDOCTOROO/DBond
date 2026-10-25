#!/usr/bin/env python3
"""
生成 R-20 回应所需的三张论文图：

  1. panel_reliability_diagram.{ext}      概率校准（Brier / ECE）
  2. panel_predicted_vs_observed.{ext}     预测 vs 观测断裂率散点（Spearman）
  3. panel_topk_enrichment.{ext}           Top-k 候选富集曲线

数据来源：
  - 散点图 + Top-k 曲线：candidate_ranking_analysis.py 输出的
    per_peptide.csv / ranking_summary.json / enrichment_table.csv
  - 可靠性图：优先读 calibration JSON（评估时单独导出）；
    缺失则从 pred_csv 的 pred_prob 列重新计算键级校准。

输出目录默认 实验图/ranking/，与 interpretability_revised 风格一致。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_calibration_metrics
from utils.visualization import (
    plot_reliability_diagram,
    plot_predicted_vs_observed_scatter,
    plot_topk_enrichment_curve,
)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s[%(levelname)s]:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("ranking_figures")


def load_or_compute_calibration(
    pred_csv: Optional[str],
    calibration_json: Optional[str],
    n_bins: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """优先读 JSON；否则从 pred_csv 的 pred_prob / true 重新算键级校准。"""
    if calibration_json and os.path.exists(calibration_json):
        with open(calibration_json, "r", encoding="utf-8") as f:
            calib = json.load(f)
        logger.info(f"Loaded calibration from {calibration_json}")
        return calib

    if not pred_csv or not os.path.exists(pred_csv):
        raise FileNotFoundError(
            "Need either --calibration_json or a valid --pred_csv to build the reliability diagram"
        )
    logger.info(f"Recomputing bond-level calibration from {pred_csv}")
    df = pd.read_csv(pred_csv, na_filter=False)
    probs: List[float] = []
    targets: List[int] = []
    for _, row in df.iterrows():
        p = str(row.get("pred_prob", "")).strip()
        t = str(row.get("true", "")).strip()
        if not p or not t:
            continue
        p_parts = [x for x in p.split(";") if x != ""]
        t_parts = [x for x in t.split(";") if x != ""]
        n = min(len(p_parts), len(t_parts))
        if n == 0:
            continue
        probs.extend(float(x) for x in p_parts[:n])
        targets.extend(int(float(x)) for x in t_parts[:n])
    if not probs:
        raise ValueError("No valid pred_prob/true pairs found in pred_csv")
    calib = compute_calibration_metrics(np.array(probs, dtype=np.float32),
                                        np.array(targets, dtype=np.int32),
                                        n_bins=n_bins)
    logger.info(f"Calibration: ECE={calib['ece']:.4f}, Brier={calib['brier_score']:.4f}, "
                f"n_samples={calib['n_samples']}")
    return calib


def main():
    parser = argparse.ArgumentParser(description="Generate R-20 ranking/calibration figures")
    parser.add_argument(
        "--ranking_dir", type=str, default="result/ranking",
        help="candidate_ranking_analysis.py 的输出目录",
    )
    parser.add_argument(
        "--pred_csv", type=str, default="result/pred/graph_transform/latest.pred.csv",
        help="预测 CSV（用于在没有 calibration_json 时重新计算校准）",
    )
    parser.add_argument(
        "--calibration_json", type=str, default=None,
        help="评估时导出的校准 JSON（含 bin_* 字段）。优先使用。",
    )
    parser.add_argument(
        "--output_dir", type=str, default="实验图/ranking",
        help="图表输出目录",
    )
    parser.add_argument(
        "--figure_format", type=str, default="svg", choices=["svg", "png"],
        help="图表格式",
    )
    parser.add_argument(
        "--calibration_bins", type=int, default=10,
        help="可靠性图 bin 数",
    )
    args = parser.parse_args()
    logger = setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)
    ext = "svg" if args.figure_format == "svg" else "png"

    # ---- Panel 1: reliability diagram ----
    calib = load_or_compute_calibration(
        args.pred_csv, args.calibration_json, args.calibration_bins, logger,
    )
    rel_path = os.path.join(args.output_dir, f"panel_reliability_diagram.{ext}")
    plot_reliability_diagram(calib, save_path=rel_path)
    logger.info(f"Saved: {rel_path}")

    # ---- Panel 2: predicted vs observed scatter ----
    per_peptide_path = os.path.join(args.ranking_dir, "per_peptide.csv")
    summary_path = os.path.join(args.ranking_dir, "ranking_summary.json")
    if not os.path.exists(per_peptide_path) or not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Missing ranking outputs in {args.ranking_dir}. "
            "Run candidate_ranking_analysis.py first."
        )
    per_peptide = pd.read_csv(per_peptide_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    scatter_path = os.path.join(args.output_dir, f"panel_predicted_vs_observed.{ext}")
    plot_predicted_vs_observed_scatter(per_peptide, summary["spearman"], save_path=scatter_path)
    logger.info(f"Saved: {scatter_path}")

    # ---- Panel 3: Top-k enrichment curve ----
    enrichment_path = os.path.join(args.ranking_dir, "enrichment_table.csv")
    enrichment_df = pd.read_csv(enrichment_path)
    enrichment_records = enrichment_df.to_dict(orient="records")
    topk_path = os.path.join(args.output_dir, f"panel_topk_enrichment.{ext}")
    plot_topk_enrichment_curve(enrichment_records, save_path=topk_path)
    logger.info(f"Saved: {topk_path}")

    logger.info(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
