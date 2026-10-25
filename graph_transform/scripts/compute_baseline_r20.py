#!/usr/bin/env python3
"""
基线模型 R-20 指标计算（DBond-M / DBond-S / DBond-AF / DBond-AF-opt）

复用基线 5fold 已落盘的 test.pred.csv（含 pred_prob 列），无需重跑推理。
对每个 fold：
  1. 读 test CSV（肽级元信息 + true_multi / 单标签 bond_label）
  2. 读 pred CSV（键级概率）
  3. 重建 peptide-level 对应关系
  4. 算键级 R-20 指标（ROC-AUC/PR-AUC/MCC/Brier/ECE）
  5. 按 seq 去重 → peptide-level Spearman + Top-k enrichment
  6. 5 折聚合 mean±std

两种数据格式：
  - 多标签（m/af/af_opt）: pred CSV 每行一键，按 max_len-1=35 切块重建
  - 单标签（s）          : pred CSV 与 test CSV 行对齐，test CSV 含 seq

用法见 main()。
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r20_metrics import (
    compute_bond_level_r20,
    build_peptide_level_table,
    compute_peptide_ranking,
    aggregate_folds,
    format_summary_table,
)


def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("baseline_r20")
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


def parse_true_multi(value: Any) -> np.ndarray:
    """'0;1;1;...' → int ndarray。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=np.int32)
    text = str(value).strip()
    if not text:
        return np.array([], dtype=np.int32)
    return np.array([int(x) for x in text.split(";") if x != ""], dtype=np.int32)


# =============================================================================
# 多标签基线（dbond_m / dbond_af / dbond_af_opt）：扁平 pred → peptide 重建
# =============================================================================

def reconstruct_multilabel(
    test_csv: str,
    pred_csv: str,
    padding_width: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[float], List[float], List[int]]:
    """把扁平 pred CSV 重建回 peptide-level。

    返回:
      flat_probs, flat_targets : 所有有效键的概率/标签（已去 padding，用于键级指标）
      seqs, r_pred_list, r_true_list, n_bonds_list : 每 spectrum 一项（用于 peptide ranking）

    关键：pred CSV 按 padding_width 行/肽 展开，但真实键数 = len(true_multi)，
    用 true_multi 长度 mask 掉 padding 行。
    """
    test_df = pd.read_csv(test_csv, na_filter=False)
    pred_df = pd.read_csv(pred_csv, na_filter=False)

    n_spectra = len(test_df)
    n_pred_rows = len(pred_df)
    expected = n_spectra * padding_width

    # 行数校验：允许 pred 略多/略少（容错），但差异大时警告
    if n_pred_rows != expected:
        ratio = n_pred_rows / n_spectra if n_spectra else 0
        logger.warning(
            f"pred rows ({n_pred_rows}) != spectra×padding ({expected}). "
            f"ratio={ratio:.2f} rows/spectrum — will use true_multi length to mask"
        )

    probs = pred_df["pred_prob"].to_numpy(dtype=np.float32)
    # pred CSV 的 true 列是 0/1（已展平含 padding），test CSV 的 true_multi 是分号串
    pred_true = pred_df["true"].to_numpy(dtype=np.int32) if "true" in pred_df.columns else None

    flat_probs: List[float] = []
    flat_targets: List[int] = []
    seqs: List[str] = []
    r_pred_list: List[float] = []
    r_true_list: List[float] = []
    n_bonds_list: List[int] = []

    cursor = 0
    for row_idx in range(n_spectra):
        test_row = test_df.iloc[row_idx]
        true_multi = parse_true_multi(test_row.get("true_multi"))
        n_bonds = int(true_multi.size)
        if n_bonds == 0:
            # 没有 true_multi，跳过（无法确定有效键）
            cursor += padding_width
            continue
        # 从 pred 取出该 spectrum 的 padding_width 行（或剩余行）
        end = min(cursor + padding_width, n_pred_rows)
        spec_probs = probs[cursor:end]
        spec_probs = spec_probs[:n_bonds]  # 截到真实键数
        cursor = end if end - cursor == padding_width else cursor + padding_width

        if spec_probs.size < n_bonds:
            # pred 行不足，跳过该 spectrum
            continue

        spec_targets = true_multi[:n_bonds]
        flat_probs.extend(spec_probs.tolist())
        flat_targets.extend(spec_targets.tolist())

        seq = str(test_row.get("seq", ""))
        seqs.append(seq)
        r_pred_list.append(float(np.mean(spec_probs)))
        r_true_list.append(float(np.mean(spec_targets)))
        n_bonds_list.append(n_bonds)

    logger.info(
        f"Reconstructed {n_spectra} spectra → {len(seqs)} valid spectra, "
        f"{len(flat_probs)} valid bonds (after masking padding)"
    )
    return (
        np.array(flat_probs, dtype=np.float32),
        np.array(flat_targets, dtype=np.int32),
        seqs,
        r_pred_list,
        r_true_list,
        n_bonds_list,
    )


# =============================================================================
# 单标签基线（dbond_s）：pred 与 test 行对齐，test 含 seq
# =============================================================================

def reconstruct_singlelabel(
    test_csv: str,
    pred_csv: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[float], List[float], List[int]]:
    """dbond_s：test CSV 每行一键（带 seq/bond_pos/bond_label），pred CSV 行对齐。

    按 (seq, charge, pep_mass, nce, scan_num) 聚合成 spectrum（同一 precursor 的多键合并）。
    """
    test_df = pd.read_csv(test_csv, na_filter=False)
    pred_df = pd.read_csv(pred_csv, na_filter=False)

    n_test = len(test_df)
    n_pred = len(pred_df)
    if n_test != n_pred:
        logger.warning(f"Row count mismatch: test={n_test}, pred={n_pred} — using min")

    n = min(n_test, n_pred)
    test_df = test_df.iloc[:n].reset_index(drop=True)
    pred_df = pred_df.iloc[:n].reset_index(drop=True)

    probs = pred_df["pred_prob"].to_numpy(dtype=np.float32)
    # dbond_s pred CSV 的 true 列是单键 0/1；test CSV 的 bond_label 也是单键 0/1
    targets = pred_df["true"].to_numpy(dtype=np.int32) if "true" in pred_df.columns else \
              test_df["bond_label"].to_numpy(dtype=np.int32)

    seqs_per_bond = test_df["seq"].astype(str).to_numpy()

    # 键级扁平（全部键都有效，无 padding）
    flat_probs = probs.tolist()
    flat_targets = targets.tolist()

    # 聚合到 spectrum（同一 precursor 的多键合并）
    # precursor key: seq + charge + pep_mass + nce + scan_num（与 dbond_s _evaluate_on_test 一致）
    precursor_cols = ["seq"]
    for col in ["charge", "pep_mass", "nce", "scan_num"]:
        if col in test_df.columns:
            precursor_cols.append(col)
    test_df = test_df.copy()
    test_df["_prob"] = probs
    test_df["_target"] = targets
    test_df["_precursor"] = test_df[precursor_cols].astype(str).agg("||".join, axis=1)

    grouped = test_df.groupby("_precursor", sort=False)
    seqs: List[str] = []
    r_pred_list: List[float] = []
    r_true_list: List[float] = []
    n_bonds_list: List[int] = []
    for _, group in grouped:
        seq = str(group["seq"].iloc[0])
        spec_probs = group["_prob"].to_numpy(dtype=np.float32)
        spec_targets = group["_target"].to_numpy(dtype=np.int32)
        seqs.append(seq)
        r_pred_list.append(float(np.mean(spec_probs)))
        r_true_list.append(float(np.mean(spec_targets)))
        n_bonds_list.append(int(len(spec_probs)))

    logger.info(
        f"dbond_s: {n} bond rows → {len(seqs)} spectra after precursor aggregation"
    )
    return (
        np.array(flat_probs, dtype=np.float32),
        np.array(flat_targets, dtype=np.int32),
        seqs,
        r_pred_list,
        r_true_list,
        n_bonds_list,
    )


# =============================================================================
# 单 fold 处理 + 5 fold 聚合
# =============================================================================

def process_fold(
    model_type: str,
    test_csv: str,
    pred_csv: str,
    padding_width: int,
    top_k_fractions: List[float],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """处理单折：返回 {bond_metrics, ranking, n_spectra, n_peptides}。"""
    if model_type == "dbond_s":
        flat_probs, flat_targets, seqs, r_pred, r_true, n_bonds = reconstruct_singlelabel(
            test_csv, pred_csv, logger
        )
    else:
        flat_probs, flat_targets, seqs, r_pred, r_true, n_bonds = reconstruct_multilabel(
            test_csv, pred_csv, padding_width, logger
        )

    bond_metrics = compute_bond_level_r20(flat_probs, flat_targets)
    peptide_df = build_peptide_level_table(seqs, r_pred, r_true, n_bonds)
    ranking = compute_peptide_ranking(peptide_df, top_k_fractions)

    return {
        "bond_metrics": bond_metrics,
        "ranking": ranking,
        "n_spectra": len(seqs),
        "n_peptides": len(peptide_df),
    }


def find_pred_csv(cv_root: str, fold_id: str, model_name: str, logger: logging.Logger) -> Optional[str]:
    """在 {cv_root}/{timestamp}/fold_{id}/pred/ 下找 test.pred.csv。

    cv_root 可能有多个 timestamp 子目录，取最新的。
    """
    pattern = os.path.join(cv_root, "*", f"fold_{fold_id}", "pred", "test.pred.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        pattern2 = os.path.join(cv_root, f"fold_{fold_id}", "pred", "test.pred.csv")
        candidates = glob.glob(pattern2)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def find_test_csv(fold_data_dir: str, fold_id: str, model_type: str, logger: logging.Logger) -> Optional[str]:
    """根据模型类型定位 test CSV。"""
    if model_type == "dbond_s":
        # 单标签：.test.csv（每行一键）
        path = os.path.join(fold_data_dir, f"{fold_id}.test.csv")
        if os.path.exists(path):
            return path
        logger.error(f"dbond_s test csv not found: {path}")
        return None
    else:
        # 多标签：.test.fbr.multi.csv（每行一肽）
        path = os.path.join(fold_data_dir, f"{fold_id}.test.fbr.multi.csv")
        if os.path.exists(path):
            return path
        logger.error(f"multilabel test csv not found: {path}")
        return None


def run_model(
    model_name: str,
    model_type: str,
    cv_root: str,
    fold_data_dir: str,
    folds: List[str],
    padding_width: int,
    top_k_fractions: List[float],
    output_dir: str,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """跑一个模型的所有 fold，返回 (per_fold_detail, aggregated_summary)。"""
    per_fold: List[Dict[str, Any]] = []
    for fold_id in folds:
        logger.info(f"  [{model_name}] fold {fold_id}")
        pred_csv = find_pred_csv(cv_root, fold_id, model_name, logger)
        test_csv = find_test_csv(fold_data_dir, fold_id, model_type, logger)
        if not pred_csv:
            logger.warning(f"    pred CSV not found, skipping fold {fold_id}")
            continue
        if not test_csv:
            logger.warning(f"    test CSV not found, skipping fold {fold_id}")
            continue
        logger.info(f"    pred: {pred_csv}")
        logger.info(f"    test: {test_csv}")
        try:
            fold_result = process_fold(
                model_type, test_csv, pred_csv, padding_width, top_k_fractions, logger
            )
            fold_result["fold_id"] = fold_id
            per_fold.append(fold_result)
            bm = fold_result["bond_metrics"]
            sp = fold_result["ranking"]["spearman"]
            logger.info(
                f"    ROC-AUC={bm['roc_auc']:.4f} PR-AUC={bm['pr_auc']:.4f} "
                f"MCC={bm['mcc']:.4f} Brier={bm['brier_score']:.4f} ECE={bm['ece']:.4f}  "
                f"Spearman ρ={sp['rho']:.4f} (n_pep={sp['n_peptides']})"
            )
        except Exception as e:
            logger.error(f"    fold {fold_id} failed: {e}", exc_info=True)

    summary = aggregate_folds(per_fold) if per_fold else {"n_folds": 0, "aggregated": {}}
    return per_fold, summary


def main():
    parser = argparse.ArgumentParser(description="Compute R-20 metrics for 4 baseline models")
    parser.add_argument(
        "--cv_roots", type=str, nargs="+", required=True,
        help="每个基线的 cv_root，按顺序对应 --models。如：result/cv/dbond_m result/cv/dbond_s ...",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="基线名（与 --cv_roots 一一对应）：dbond_m / dbond_s / dbond_af / dbond_af_opt",
    )
    parser.add_argument(
        "--fold_data_dir", type=str, default="dataset/5fold",
        help="5fold 测试数据目录",
    )
    parser.add_argument(
        "--folds", type=str, nargs="+", default=["1222", "2252", "3514", "6072", "9075"],
    )
    parser.add_argument(
        "--padding_width", type=int, default=35,
        help="多标签基线 pred CSV 的每肽行数（= max_len - 1，默认 35）",
    )
    parser.add_argument(
        "--top_k_fractions", type=float, nargs="+", default=[0.1, 0.2, 0.5],
    )
    parser.add_argument(
        "--output_dir", type=str, default="result/r20_baselines",
    )
    args = parser.parse_args()

    if len(args.cv_roots) != len(args.models):
        parser.error("--cv_roots and --models must have the same length")

    logger = setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("Baseline R-20 metrics computation")
    logger.info(f"models: {args.models}")
    logger.info(f"padding_width={args.padding_width}, top_k={args.top_k_fractions}")
    logger.info("=" * 70)

    all_summary_rows: List[Dict[str, str]] = []
    all_models_detail: Dict[str, Any] = {}

    for cv_root, model_name in zip(args.cv_roots, args.models):
        logger.info("\n" + "=" * 60)
        logger.info(f"MODEL: {model_name}  (cv_root={cv_root})")
        logger.info("=" * 60)
        # model_type：dbond_s 是单标签，其余多标签
        model_type = "dbond_s" if model_name == "dbond_s" else "multilabel"
        per_fold, summary = run_model(
            model_name=model_name,
            model_type=model_type,
            cv_root=cv_root,
            fold_data_dir=args.fold_data_dir,
            folds=args.folds,
            padding_width=args.padding_width,
            top_k_fractions=args.top_k_fractions,
            output_dir=args.output_dir,
            logger=logger,
        )
        # 保存该模型的明细
        model_out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)
        with open(os.path.join(model_out_dir, "r20_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        rows = format_summary_table(summary, model_name)
        all_summary_rows.extend(rows)
        all_models_detail[model_name] = summary

        # 控制台打印该模型聚合结果
        agg = summary.get("aggregated", {})
        bm = agg.get("bond_metrics", {})
        rk = agg.get("ranking", {})
        logger.info(f"\n--- {model_name} 5-fold aggregation ---")
        for m in ["roc_auc", "pr_auc", "mcc", "brier_score", "ece"]:
            if m in bm:
                logger.info(f"  {m:<12}: {bm[m]['mean']:.4f} ± {bm[m]['std']:.4f}")
        if rk.get("spearman_rho"):
            logger.info(f"  Spearman ρ  : {rk['spearman_rho']['mean']:.4f} ± {rk['spearman_rho']['std']:.4f}")

    # 汇总 CSV（所有模型所有指标）
    summary_csv = os.path.join(args.output_dir, "r20_all_baselines_summary.csv")
    pd.DataFrame(all_summary_rows).to_csv(summary_csv, index=False)
    logger.info(f"\nSaved all-baselines summary: {summary_csv}")

    detail_path = os.path.join(args.output_dir, "r20_all_baselines_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_models_detail, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved all-baselines detail: {detail_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
