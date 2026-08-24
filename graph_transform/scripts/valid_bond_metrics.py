#!/usr/bin/env python3
"""统一「有效键口径」指标重算（R-01 裁决性脚本）。

背景（lab_acc padding 口径问题）：
  - dbond_m / af / af_opt 的 example/label 指标矩阵固定宽 35（max_len-1），
    padding 格 (gt=0, pred=0) 全部计为 TN，只抬高 lab_acc（precision/recall/F1
    基于 tp/fp/fn，不受 padding-only TN 影响）；
  - DBond-GT 与 dbond_s 的矩阵宽 = 测试集实际最大键数（本数据 31）。
  两侧 padding 占比不同（约 25.4% vs 15.7%），lab_acc 不能直接横向比较。

本脚本从各模型已落盘的 test pred CSV 出发，在【真实存在的 peptide bonds】上
统一重算（同一代码路径、同一 0.5 阈值、padding 无关）：
  bond_acc / bond_precision / bond_recall / bond_f1 / bond_mcc
其中 bond_precision / bond_recall / bond_f1 应与训练汇总里的
lab_precision_mi / lab_recall_mi / lab_f1_mi 一致（自动交叉校验 padding 分析）；
bond_acc 即论文 R-01 表的 "Bond-level Accuracy (valid bonds only)"。

三种 pred 格式：
  - gt         : {cv_root}/r20_aggregation/per_fold/fold_{id}/pred.csv
                 （aggregate_r20_5fold.py 产物；true/pred/pred_prob 为分号串，
                  仅含有效键 —— evaluator.collect_prediction_outputs 已按
                  seq_len-1 截断）
  - multilabel : {cv_root}/[ts/]fold_{id}/pred/test.pred.csv 扁平行（含 padding），
                 用 test CSV 的 true_multi 长度屏蔽 padding（复用 compute_baseline_r20）
  - single     : dbond_s 扁平行（每行一个真实键，无 padding）

用法（两台机器各跑各的，输出 CSV 可直接合并；GT 与基线 cv_root 不同机时分开跑）：

  # graphtrans 机器（GT-pre）
  python graph_transform/scripts/valid_bond_metrics.py \
      --models dbond_gt_pre --cv_roots checkpoints/graph_transform/5fold/<gt_pre_cv_root> \
      --output_dir result/valid_bond_metrics_gt

  # dbond-gt-2 机器（四个基线 pre）
  python graph_transform/scripts/valid_bond_metrics.py \
      --models dbond_s_pre dbond_m_pre dbond_af_pre dbond_af_opt_pre \
      --cv_roots result/cv/dbond_s/<ts> result/cv/dbond_m/<ts> \
                 result/cv/dbond_af/<ts> result/cv/dbond_af_opt/<ts> \
      --fold_data_dir dataset/5fold \
      --output_dir result/valid_bond_metrics_baselines

输出：
  {output_dir}/valid_bond_metrics.csv   所有模型 × 指标 5fold mean±std（论文填表用）
  {output_dir}/per_model/{model}.csv    每折明细
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
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 复用基线 pred 重建逻辑（含 padding 屏蔽与 pred/test CSV 定位）
from compute_baseline_r20 import (
    find_pred_csv,
    find_test_csv,
    reconstruct_multilabel,
    reconstruct_singlelabel,
)

logger = logging.getLogger("valid_bond_metrics")

METRIC_KEYS = ["n_bonds", "positive_rate", "bond_acc", "bond_precision",
               "bond_recall", "bond_f1", "bond_mcc"]


def infer_model_type(model_name: str) -> str:
    """模型名 → pred 格式类型：gt / single / multilabel。"""
    n = model_name.lower()
    if "gt" in n:
        return "gt"
    if "dbond_s" in n or "single" in n:  # dbond_s / dbond_s_pre
        return "single"
    return "multilabel"


# =============================================================================
# GT pred 解析（分号串，仅含有效键）
# =============================================================================

def find_gt_pred_csv(gt_cv_root: str, fold_id: str) -> Optional[str]:
    """定位 GT 某折的 pred CSV（aggregate_r20_5fold.py 产物优先）。"""
    patterns = [
        os.path.join(gt_cv_root, "r20_aggregation", "per_fold", f"fold_{fold_id}", "pred.csv"),
    ]
    for pat in patterns:
        if os.path.exists(pat):
            return pat
    # 兜底：fold 目录下训练/评估直接落盘的 pred csv
    for pat in [
        os.path.join(gt_cv_root, f"fold_{fold_id}", "pred", "*.csv"),
        os.path.join(gt_cv_root, f"fold_{fold_id}", "**", "*pred*.csv"),
    ]:
        candidates = glob.glob(pat, recursive=True)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None


def parse_gt_pred(pred_csv: str, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """GT pred CSV 的 true/pred_prob 分号串 → 扁平 (probs, targets)，仅有效键。"""
    df = pd.read_csv(pred_csv, na_filter=False)
    probs: List[float] = []
    targets: List[int] = []
    for t_str, p_str in zip(df["true"], df["pred_prob"]):
        t_list = [int(x) for x in str(t_str).split(";") if x.strip() != ""]
        p_list = [float(x) for x in str(p_str).split(";") if x.strip() != ""]
        n = min(len(t_list), len(p_list))
        targets.extend(t_list[:n])
        probs.extend(p_list[:n])
    return (np.array(probs, dtype=np.float32),
            np.array(targets, dtype=np.int32))


# =============================================================================
# 有效键指标（threshold 二值化，padding 无关）
# =============================================================================

def compute_valid_bond_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    tgts = np.asarray(targets, dtype=np.int32).reshape(-1)
    n = min(probs.size, tgts.size)
    probs, tgts = probs[:n], tgts[:n]
    preds = (probs > threshold).astype(np.int32)

    tp = int(np.sum((preds == 1) & (tgts == 1)))
    fp = int(np.sum((preds == 1) & (tgts == 0)))
    tn = int(np.sum((preds == 0) & (tgts == 0)))
    fn = int(np.sum((preds == 0) & (tgts == 1)))

    result: Dict[str, float] = {
        "n_bonds": float(n),
        "positive_rate": float(np.mean(tgts)) if n else 0.0,
        "bond_acc": (tp + tn) / n if n else 0.0,
        "bond_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "bond_recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }
    p, r = result["bond_precision"], result["bond_recall"]
    result["bond_f1"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    try:
        result["bond_mcc"] = float(matthews_corrcoef(tgts, preds))
    except ValueError:
        result["bond_mcc"] = 0.0
    return result


# =============================================================================
# 单折处理（按类型分发）
# =============================================================================

def load_fold_valid_bonds(
    model_name: str,
    model_type: str,
    cv_root: str,
    fold_id: str,
    fold_data_dir: str,
    padding_width: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if model_type == "gt":
        pred_csv = find_gt_pred_csv(cv_root, fold_id)
        if not pred_csv:
            raise FileNotFoundError(
                f"GT pred CSV not found for fold {fold_id} under {cv_root} "
                f"(期望 {cv_root}/r20_aggregation/per_fold/fold_{fold_id}/pred.csv；"
                f"若不存在请先跑 aggregate_r20_5fold.py)"
            )
        logger.info(f"    gt pred: {pred_csv}")
        return parse_gt_pred(pred_csv, threshold)

    pred_csv = find_pred_csv(cv_root, fold_id, model_name, logger)
    if not pred_csv:
        raise FileNotFoundError(f"baseline pred CSV not found for fold {fold_id} under {cv_root}")
    if model_type == "single":
        test_csv = find_test_csv(fold_data_dir, fold_id, "dbond_s", logger)
        if not test_csv:
            raise FileNotFoundError(f"dbond_s test csv not found for fold {fold_id}")
        logger.info(f"    pred: {pred_csv}\n    test: {test_csv}")
        probs, targets, *_ = reconstruct_singlelabel(test_csv, pred_csv, logger)
        return probs, targets

    # multilabel（m / af / af_opt）
    test_csv = find_test_csv(fold_data_dir, fold_id, "multilabel", logger)
    if not test_csv:
        raise FileNotFoundError(f"multilabel test csv not found for fold {fold_id}")
    logger.info(f"    pred: {pred_csv}\n    test: {test_csv}")
    probs, targets, *_ = reconstruct_multilabel(test_csv, pred_csv, padding_width, logger)
    return probs, targets


def aggregate_model(per_fold: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """5fold mean±std（ddof=0，与既有 5fold 汇总一致）。"""
    agg: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.array([float(f[key]) for f in per_fold if key in f], dtype=float)
        if values.size == 0:
            continue
        agg[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_folds": int(values.size),
        }
    return agg


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一有效键口径指标重算（bond_acc 等，padding 无关，R-01 裁决）")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="模型名，如 dbond_s_pre dbond_m_pre dbond_af_pre dbond_af_opt_pre dbond_gt_pre")
    parser.add_argument("--cv_roots", type=str, nargs="+", required=True,
                        help="与 --models 一一对应的 cv_root")
    parser.add_argument("--types", type=str, nargs="+", default=None,
                        help="可选，强制指定类型（gt/single/multilabel），默认按名字推断")
    parser.add_argument("--fold_data_dir", type=str, default="dataset/5fold",
                        help="5fold 测试数据目录（single/multilabel 需要）")
    parser.add_argument("--folds", type=str, nargs="+",
                        default=["1222", "2252", "3514", "6072", "9075"])
    parser.add_argument("--padding_width", type=int, default=35,
                        help="多标签 pred 每谱行数（= max_len - 1）")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="result/valid_bond_metrics")
    args = parser.parse_args()

    if len(args.models) != len(args.cv_roots):
        parser.error("--models 与 --cv_roots 数量必须一致")
    if args.types and len(args.types) != len(args.models):
        parser.error("--types 与 --models 数量必须一致")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "per_model"), exist_ok=True)
    # logging
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s[%(levelname)s]:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(args.output_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info(f"Valid-bond metrics (threshold={args.threshold}, padding-free)")
    logger.info(f"models={args.models}")
    logger.info("=" * 70)

    summary_rows: List[Dict[str, Any]] = []
    console_lines: List[str] = []
    all_agg: Dict[str, Any] = {}

    for model_idx, (model_name, cv_root) in enumerate(zip(args.models, args.cv_roots)):
        model_type = (args.types[model_idx] if args.types else infer_model_type(model_name))
        logger.info("\n" + "=" * 60)
        logger.info(f"MODEL: {model_name}  (type={model_type}, cv_root={cv_root})")
        logger.info("=" * 60)

        per_fold: List[Dict[str, Any]] = []
        for fold_id in args.folds:
            logger.info(f"  fold {fold_id}")
            try:
                probs, targets = load_fold_valid_bonds(
                    model_name, model_type, cv_root, fold_id,
                    args.fold_data_dir, args.padding_width, args.threshold,
                )
                m = compute_valid_bond_metrics(probs, targets, args.threshold)
                m["fold_id"] = fold_id
                per_fold.append(m)
                logger.info(
                    f"    n_bonds={int(m['n_bonds'])} bond_acc={m['bond_acc']:.4f} "
                    f"P={m['bond_precision']:.4f} R={m['bond_recall']:.4f} "
                    f"F1={m['bond_f1']:.4f} MCC={m['bond_mcc']:.4f}"
                )
            except Exception as e:
                logger.error(f"    fold {fold_id} failed: {e}", exc_info=True)

        if not per_fold:
            logger.error(f"  {model_name}: no fold succeeded, skipping")
            continue

        per_fold_df = pd.DataFrame(per_fold)[["fold_id"] + [k for k in METRIC_KEYS if k in per_fold[0]]]
        per_fold_csv = os.path.join(args.output_dir, "per_model", f"{model_name}.csv")
        per_fold_df.to_csv(per_fold_csv, index=False)
        logger.info(f"  per-fold detail → {per_fold_csv}")

        agg = aggregate_model(per_fold)
        all_agg[model_name] = {"aggregated": agg,
                               "per_fold": per_fold,
                               "model_type": model_type}
        for key in METRIC_KEYS:
            if key not in agg:
                continue
            a = agg[key]
            summary_rows.append({
                "model": model_name, "metric": key,
                "mean": a["mean"], "std": a["std"],
                "min": a["min"], "max": a["max"], "n_folds": a["n_folds"],
                "mean±std": f"{a['mean']:.4f} ± {a['std']:.4f}",
            })
        if "bond_acc" in agg:
            console_lines.append(
                f"  {model_name:<20} bond_acc = {agg['bond_acc']['mean']*100:.2f} ± {agg['bond_acc']['std']*100:.2f} %"
            )

    if not summary_rows:
        logger.error("所有模型均失败，未产出结果。")
        return

    summary_csv = os.path.join(args.output_dir, "valid_bond_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    logger.info(f"\nSaved summary: {summary_csv}")

    detail_json = os.path.join(args.output_dir, "valid_bond_metrics.json")
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(all_agg, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved detail: {detail_json}")

    logger.info("\n" + "=" * 70)
    logger.info("Bond-level Accuracy (valid bonds only) — 5fold mean ± std")
    logger.info("=" * 70)
    for line in console_lines:
        logger.info(line)
    logger.info("-" * 70)
    logger.info(
        "交叉校验: bond_precision/bond_recall/bond_f1 应与各训练 5fold_summary 的\n"
        "lab_precision_mi/lab_recall_mi/lab_f1_mi 一致（padding 不影响 tp/fp/fn）；\n"
        "若一致，则 lab_acc 差异确系 padding TN 口径所致。"
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
