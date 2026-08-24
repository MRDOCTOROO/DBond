"""
R-20 指标统一计算模块（DBond-GT 与所有基线共用同一口径）

核心原则：
  1. 键级判别力/校准指标（ROC-AUC/PR-AUC/MCC/Brier/ECE）在【有效键】上计算
     - 有效键 = 真实存在的肽键（排除 padding），由 true_multi 长度决定
  2. peptide-level ranking 在【唯一序列】上去重聚合
     - 同一 seq 的多个 spectrum：R_pred/R_true 取均值
     - 这样 n = 唯一序列数（符合审稿人 "candidate sequence ranking" 原意）

本模块不读文件、不做格式适配，只接受已解析好的结构化输入。
格式适配（GT 的分号串 vs 基线的扁平行）由各 compute_xxx 脚本负责。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)


# =============================================================================
# 1. 键级判别力 + 校准指标（扁平 probabilities/targets）
# =============================================================================

def compute_bond_level_r20(
    probabilities: np.ndarray,
    targets: np.ndarray,
    binary_preds: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    """键级 R-20 指标。probabilities/targets 已展平、已去除 padding。

    返回: {roc_auc, pr_auc, mcc, brier_score, ece, n_bonds, positive_rate}
    """
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.int32).reshape(-1)
    n = min(probabilities.size, targets.size)
    probabilities = probabilities[:n]
    targets = targets[:n]

    result: Dict[str, float] = {
        "roc_auc": 0.0,
        "pr_auc": 0.0,
        "mcc": 0.0,
        "brier_score": 0.0,
        "ece": 0.0,
        "n_bonds": int(n),
        "positive_rate": float(np.mean(targets)) if n > 0 else 0.0,
    }
    if n == 0:
        return result

    if binary_preds is None:
        binary_preds = (probabilities > threshold).astype(np.int32)

    # Brier（概率均方误差，不依赖阈值）
    result["brier_score"] = float(np.mean((probabilities - targets.astype(np.float32)) ** 2))

    # ECE（10-bin 加权平均 |置信度-正类频率|）
    result["ece"] = _expected_calibration_error(probabilities, targets.astype(np.float32), n_bins)

    # MCC（依赖阈值化预测）
    try:
        result["mcc"] = float(matthews_corrcoef(targets, binary_preds))
    except ValueError:
        result["mcc"] = 0.0

    # ROC-AUC / PR-AUC（需两类都存在）
    if len(np.unique(targets)) > 1:
        try:
            result["roc_auc"] = float(roc_auc_score(targets, probabilities))
        except ValueError:
            pass
        try:
            result["pr_auc"] = float(average_precision_score(targets, probabilities))
        except ValueError:
            pass

    return result


def _expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, n_bins: int) -> float:
    """ECE = Σ_bin (n_bin/N) × |acc_bin − conf_bin|，键级正类频率作 acc。"""
    if probabilities.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(
        np.digitize(probabilities, bin_edges[1:-1], right=False), 0, n_bins - 1
    )
    total = float(probabilities.size)
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        count = int(np.sum(mask))
        if count == 0:
            continue
        conf = float(np.mean(probabilities[mask]))
        acc = float(np.mean(targets[mask]))
        ece += abs(acc - conf) * (count / total)
    return float(ece)


# =============================================================================
# 2. peptide-level 候选排序（去重 + Spearman + Top-k enrichment）
# =============================================================================

def build_peptide_level_table(
    seqs: List[str],
    r_pred_per_spectrum: List[float],
    r_true_per_spectrum: List[float],
    n_bonds_per_spectrum: List[int],
) -> pd.DataFrame:
    """spectrum 级 R_pred/R_true → peptide-seq 级（去重，同 seq 取均值）。

    符合审稿人 "candidate sequence ranking" 原意：每个唯一序列一个点。
    R-02 聚合定义（回顾性 obs 口径）：
      R_pred(p) = (1/N_p) Σ_{s∈S_p} R_pred(p,s)
      R_true(p) = (1/N_p) Σ_{s∈S_p} R_true(p,s)
      S_p = 该序列全部实验谱记录 —— 与 R_true 用完全相同的谱集合与等权（逐谱平均，
      非保留单条记录），排序结果与输入文件行序无关。
    n_spectra 列记录每候选的谱记录数（R-02 要求披露的"重复记录"数）。
    """
    df = pd.DataFrame({
        "seq": seqs,
        "R_pred": r_pred_per_spectrum,
        "R_true": r_true_per_spectrum,
        "n_bonds": n_bonds_per_spectrum,
        "n_spectra": 1,  # 占位计数列（每行一条谱记录），聚合时 sum 即该候选的谱数
    })
    # 同一 seq 的多个 spectrum 取均值（R_pred/R_true），n_bonds 取 max（同序列键数相同），
    # n_spectra 求和（= 该候选进入聚合的实际条件记录数）。
    # 注：不能写 agg({"n_spectra": "size"}) 依赖"size 不需要列存在"的旧行为——
    # 云端 pandas 2.3.3 实测抛 KeyError（Column(s) do not exist），pandas 3.x 同样。
    peptide_df = df.groupby("seq", as_index=False).agg({
        "R_pred": "mean",
        "R_true": "mean",
        "n_bonds": "max",
        "n_spectra": "sum",
    })
    return peptide_df


def compute_peptide_ranking(peptide_df: pd.DataFrame, top_k_fractions: List[float]) -> Dict[str, Any]:
    """peptide-level Spearman + Top-k enrichment。

    输入 peptide_df 需含 R_pred / R_true 列（已去重，每序列一行）。
    """
    n_peptides = len(peptide_df)
    result: Dict[str, Any] = {
        "spearman": {"rho": 0.0, "p_value": 1.0, "n_peptides": int(n_peptides)},
        "enrichment": [],
        "baseline": {
            "mean_R_pred": float(peptide_df["R_pred"].mean()) if n_peptides else 0.0,
            "mean_R_true": float(peptide_df["R_true"].mean()) if n_peptides else 0.0,
            "n_peptides": int(n_peptides),
        },
    }
    if n_peptides < 3:
        return result

    # Spearman
    sp = stats.spearmanr(peptide_df["R_pred"].values, peptide_df["R_true"].values)
    rho = float(sp.correlation) if not np.isnan(sp.correlation) else 0.0
    pval = float(sp.pvalue) if not np.isnan(sp.pvalue) else 1.0
    result["spearman"] = {"rho": rho, "p_value": pval, "n_peptides": int(n_peptides)}

    # Top-k enrichment
    overall_true = float(peptide_df["R_true"].mean())
    # R-02 tie 稳定化：R_pred 降序、seq 字典序升序（稳定 mergesort），
    # 使排序完全确定、与输入行序无关；并列分数的 Top-k 边界取该字典序前 k 行。
    if "seq" in peptide_df.columns:
        ranked = peptide_df.sort_values(
            ["R_pred", "seq"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
    else:
        ranked = peptide_df.sort_values("R_pred", ascending=False, kind="mergesort").reset_index(drop=True)
    enrichment_rows: List[Dict[str, Any]] = []
    for frac in sorted(set(top_k_fractions)):
        frac = float(frac)
        if not (0.0 < frac <= 1.0):
            continue
        k = max(1, int(round(frac * n_peptides)))
        subset = ranked.iloc[:k]
        observed_true = float(subset["R_true"].mean())
        observed_pred = float(subset["R_pred"].mean())
        enrichment_rows.append({
            "selection": f"Top {int(frac * 100)}%",
            "fraction": frac,
            "n_peptides": int(k),
            "mean_R_pred": observed_pred,
            "mean_R_true": observed_true,
            "delta_vs_all": observed_true - overall_true,
            "enrichment_ratio": observed_true / overall_true if overall_true > 0 else float("inf"),
        })
    # 末行加 All 基线
    enrichment_rows.append({
        "selection": "All",
        "fraction": 1.0,
        "n_peptides": int(n_peptides),
        "mean_R_pred": float(peptide_df["R_pred"].mean()),
        "mean_R_true": overall_true,
        "delta_vs_all": 0.0,
        "enrichment_ratio": 1.0,
    })
    result["enrichment"] = enrichment_rows
    return result


# =============================================================================
# 3. 聚合：5 折 mean ± std
# =============================================================================

R20_BOND_METRICS = ["roc_auc", "pr_auc", "mcc", "brier_score", "ece"]


def aggregate_folds(per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把每折的 {bond_metrics, ranking} 聚合成 mean±std。

    per_fold[i] = {
        "fold_id": str,
        "bond_metrics": {roc_auc, pr_auc, mcc, brier_score, ece, n_bonds, positive_rate},
        "ranking": {spearman, enrichment, baseline},
    }
    """
    n_folds = len(per_fold)
    summary: Dict[str, Any] = {"n_folds": n_folds, "per_fold": per_fold, "aggregated": {}}
    if n_folds == 0:
        return summary

    # 键级指标
    bond_agg: Dict[str, Dict[str, float]] = {}
    for key in R20_BOND_METRICS + ["n_bonds", "positive_rate"]:
        values = [f["bond_metrics"].get(key, np.nan) for f in per_fold]
        arr = np.array([v for v in values if not (isinstance(v, float) and np.isnan(v))], dtype=float)
        if arr.size == 0:
            continue
        bond_agg[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }
    summary["aggregated"]["bond_metrics"] = bond_agg

    # Ranking：Spearman ρ
    rho_values = [f["ranking"]["spearman"]["rho"] for f in per_fold]
    rho_arr = np.array(rho_values, dtype=float)
    baseline_true_values = [f["ranking"]["baseline"]["mean_R_true"] for f in per_fold]
    baseline_arr = np.array(baseline_true_values, dtype=float)
    n_peptides_values = [f["ranking"]["spearman"]["n_peptides"] for f in per_fold]
    n_peptides_total = int(np.sum(n_peptides_values))

    ranking_agg: Dict[str, Any] = {
        "spearman_rho": {
            "mean": float(rho_arr.mean()),
            "std": float(rho_arr.std(ddof=0)),
            "min": float(rho_arr.min()),
            "max": float(rho_arr.max()),
            "n_folds": int(rho_arr.size),
            "total_n_peptides": n_peptides_total,
        },
        "baseline_R_true": {
            "mean": float(baseline_arr.mean()),
            "std": float(baseline_arr.std(ddof=0)),
            "n_folds": int(baseline_arr.size),
        },
        "topk": {},
    }

    # Top-k：用第一折的 fraction 列表作基准
    first_enrichment = per_fold[0]["ranking"]["enrichment"]
    for entry in first_enrichment:
        sel = entry["selection"]
        if sel == "All":
            continue
        true_vals, ratio_vals, pred_vals = [], [], []
        for f in per_fold:
            for e in f["ranking"]["enrichment"]:
                if e["selection"] == sel:
                    true_vals.append(e["mean_R_true"])
                    ratio_vals.append(e["enrichment_ratio"])
                    pred_vals.append(e["mean_R_pred"])
                    break
        if not true_vals:
            continue
        true_arr = np.array(true_vals)
        ratio_arr = np.array(ratio_vals)
        ranking_agg["topk"][sel] = {
            "mean_R_true_mean": float(true_arr.mean()),
            "mean_R_true_std": float(true_arr.std(ddof=0)),
            "enrichment_ratio_mean": float(ratio_arr.mean()),
            "enrichment_ratio_std": float(ratio_arr.std(ddof=0)),
            "n_folds": int(true_arr.size),
        }
    summary["aggregated"]["ranking"] = ranking_agg
    return summary


def format_summary_table(summary: Dict[str, Any], model_name: str) -> List[Dict[str, str]]:
    """把聚合结果展平为 CSV 行（mean±std 字符串），便于论文制表。"""
    rows: List[Dict[str, str]] = []
    agg = summary.get("aggregated", {})
    bond = agg.get("bond_metrics", {})

    def fmt(key: str) -> str:
        if key not in bond:
            return ""
        return f"{bond[key]['mean']:.4f} ± {bond[key]['std']:.4f}"

    for metric in R20_BOND_METRICS:
        rows.append({
            "model": model_name,
            "category": "discrimination_calibration",
            "metric": {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "mcc": "MCC",
                       "brier_score": "Brier_score", "ece": "ECE"}[metric],
            "mean±std": fmt(metric),
        })

    ranking = agg.get("ranking", {})
    sp = ranking.get("spearman_rho", {})
    if sp:
        rows.append({
            "model": model_name,
            "category": "ranking",
            "metric": "Spearman_rho",
            "mean±std": f"{sp['mean']:.4f} ± {sp['std']:.4f}",
        })
    for sel, topk in ranking.get("topk", {}).items():
        rows.append({
            "model": model_name,
            "category": "ranking_topk",
            "metric": f"{sel}_mean_R_true",
            "mean±std": f"{topk['mean_R_true_mean']:.4f} ± {topk['mean_R_true_std']:.4f}",
        })
        rows.append({
            "model": model_name,
            "category": "ranking_topk",
            "metric": f"{sel}_enrichment_ratio",
            "mean±std": f"{topk['enrichment_ratio_mean']:.4f} ± {topk['enrichment_ratio_std']:.4f}",
        })
    bl = ranking.get("baseline_R_true", {})
    if bl:
        rows.append({
            "model": model_name,
            "category": "ranking_baseline",
            "metric": "All_mean_R_true",
            "mean±std": f"{bl['mean']:.4f} ± {bl['std']:.4f}",
        })
    return rows
