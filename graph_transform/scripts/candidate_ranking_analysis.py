#!/usr/bin/env python3
"""
Candidate-level ranking analysis（R-20 核心回应）

目的：验证 DBond-GT 输出的键级概率聚合成的 predicted cleavage ratio，
      能否作为镜像肽候选序列优先级排序的 ranking score。

输入：evaluate_graph_model.py 生成的 latest.pred.csv，至少含列：
      - pred_prob / true：分号拼接的键级概率 / 真实 0-1 标签
      - name / seq / tb  ：肽元信息与键数

输出（默认 result/ranking/）：
  - ranking_summary.json     : Spearman ρ、Top-k enrichment、全集基线
  - per_peptide.csv          : 每条肽的 R_pred / R_true / n_bonds（供散点图与复现）
  - enrichment_table.csv     : Top-k 富集表
  - run.log                  : 运行日志

关键定义（术语与论文统一）：
  R_pred (predicted cleavage ratio) = mean(pred_prob_vec)   # 键级概率均值
  R_true (observed PBCLA ratio)     = mean(true_label_vec)  # 实测断裂比例

注意：本脚本不依赖模型权重，纯后处理；可对任一 fold 的 pred.csv 重跑。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def setup_logging(output_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("candidate_ranking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s[%(levelname)s]:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def parse_semicolon_vector(value: Any, dtype=np.float32) -> np.ndarray:
    """把 '0.12;0.85;...' 风格的字符串解析为一维 ndarray。空串返回空数组。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=dtype)
    text = str(value).strip()
    if not text:
        return np.array([], dtype=dtype)
    parts = [p for p in text.split(";") if p != ""]
    if not parts:
        return np.array([], dtype=dtype)
    return np.array(parts, dtype=dtype)


def compute_per_peptide_ratios(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """逐肽聚合键级概率/标签为 R_pred / R_true。"""
    records: List[Dict[str, Any]] = []
    skipped = 0
    for row_idx, row in df.iterrows():
        prob_vec = parse_semicolon_vector(row.get("pred_prob"), np.float32)
        true_vec = parse_semicolon_vector(row.get("true"), np.int32)
        n = min(prob_vec.size, true_vec.size)
        if n == 0:
            skipped += 1
            continue
        prob_vec = prob_vec[:n]
        true_vec = true_vec[:n]
        records.append({
            "name": row.get("name", ""),
            "seq": str(row.get("seq", "")),
            "charge": row.get("charge", ""),
            "nce": row.get("nce", ""),
            "n_bonds": int(n),
            "R_pred": float(np.mean(prob_vec)),   # predicted cleavage ratio
            "R_true": float(np.mean(true_vec)),   # observed PBCLA cleavage ratio
        })
    if skipped:
        logger.warning(f"Skipped {skipped} rows with empty pred_prob/true vectors")
    if not records:
        raise ValueError("No valid peptide rows found (check pred_prob/true columns)")
    return pd.DataFrame(records)


def compute_spearman(per_peptide: pd.DataFrame) -> Dict[str, Any]:
    """R_pred vs R_true 的 Spearman 秩相关。"""
    if len(per_peptide) < 3:
        logger_warn = "Not enough peptides for Spearman correlation"
        return {"rho": 0.0, "p_value": 1.0, "n_peptides": int(len(per_peptide)), "warning": logger_warn}
    result = stats.spearmanr(per_peptide["R_pred"].values, per_peptide["R_true"].values)
    rho = float(result.correlation) if not np.isnan(result.correlation) else 0.0
    pval = float(result.pvalue) if not np.isnan(result.pvalue) else 1.0
    return {"rho": rho, "p_value": pval, "n_peptides": int(len(per_peptide))}


def compute_topk_enrichment(
    per_peptide: pd.DataFrame,
    fractions: List[float],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """按 R_pred 降序选择 Top-k 候选，比较其 mean(R_true) 与全集基线。"""
    overall_ratio = float(per_peptide["R_true"].mean())
    n_total = len(per_peptide)
    ranked = per_peptide.sort_values("R_pred", ascending=False).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for frac in sorted(set(fractions)):
        frac = float(frac)
        if not (0.0 < frac <= 1.0):
            logger.warning(f"Skipping invalid top_k fraction {frac}")
            continue
        k = max(1, int(round(frac * n_total)))
        subset = ranked.iloc[:k]
        observed = float(subset["R_true"].mean())
        predicted = float(subset["R_pred"].mean())
        rows.append({
            "selection": f"Top {int(frac * 100)}%",
            "fraction": frac,
            "n_peptides": int(k),
            "mean_R_pred": predicted,
            "mean_R_true": observed,
            "delta_vs_all": observed - overall_ratio,   # 富集量（正值=筛选有效）
            "enrichment_ratio": observed / overall_ratio if overall_ratio > 0 else float("inf"),
        })
    # 末行加全集基线作对照
    rows.append({
        "selection": "All",
        "fraction": 1.0,
        "n_peptides": int(n_total),
        "mean_R_pred": float(per_peptide["R_pred"].mean()),
        "mean_R_true": overall_ratio,
        "delta_vs_all": 0.0,
        "enrichment_ratio": 1.0,
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Candidate-level ranking analysis for R-20")
    parser.add_argument(
        "--pred_csv", type=str,
        default="result/pred/graph_transform/latest.pred.csv",
        help="evaluate_graph_model.py 输出的预测 CSV（含 pred_prob/true 列）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="result/ranking",
        help="输出目录",
    )
    parser.add_argument(
        "--top_k_fractions", type=float, nargs="+", default=[0.1, 0.2, 0.5],
        help="Top-k 候选比例（默认 0.1 0.2 0.5）",
    )
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    logger.info("=" * 60)
    logger.info("Candidate-level ranking analysis (R-20)")
    logger.info("=" * 60)

    if not os.path.exists(args.pred_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {args.pred_csv}")

    df = pd.read_csv(args.pred_csv, na_filter=False)
    logger.info(f"Loaded {len(df)} rows from {args.pred_csv}")
    required = {"pred_prob", "true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    per_peptide = compute_per_peptide_ratios(df, logger)
    logger.info(f"Computed ratios for {len(per_peptide)} peptides")

    spearman = compute_spearman(per_peptide)
    logger.info(
        f"Spearman(R_pred, R_true): rho={spearman['rho']:.4f}, "
        f"p={spearman['p_value']:.4e}, n={spearman['n_peptides']}"
    )

    enrichment = compute_topk_enrichment(per_peptide, args.top_k_fractions, logger)
    for row in enrichment:
        logger.info(
            f"  {row['selection']:<10} n={row['n_peptides']:<6} "
            f"R_pred={row['mean_R_pred']:.4f}  R_true={row['mean_R_true']:.4f}  "
            f"delta={row['delta_vs_all']:+.4f}  ratio={row['enrichment_ratio']:.3f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    per_peptide_path = os.path.join(args.output_dir, "per_peptide.csv")
    per_peptide.to_csv(per_peptide_path, index=False)
    logger.info(f"Saved per-peptide table: {per_peptide_path}")

    enrichment_df = pd.DataFrame(enrichment)
    enrichment_path = os.path.join(args.output_dir, "enrichment_table.csv")
    enrichment_df.to_csv(enrichment_path, index=False)
    logger.info(f"Saved enrichment table: {enrichment_path}")

    summary = {
        "pred_csv": os.path.abspath(args.pred_csv),
        "n_peptides": int(len(per_peptide)),
        "spearman": spearman,
        "enrichment": enrichment,
        "overall": {
            "mean_R_pred": float(per_peptide["R_pred"].mean()),
            "mean_R_true": float(per_peptide["R_true"].mean()),
            "std_R_pred": float(per_peptide["R_pred"].std()),
            "std_R_true": float(per_peptide["R_true"].std()),
        },
        "top_k_fractions": args.top_k_fractions,
    }
    summary_path = os.path.join(args.output_dir, "ranking_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved summary: {summary_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
