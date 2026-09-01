"""
评估指标模块

键级别二分类评估指标，同时补齐与 dbond_m 横向对齐的 example-based / label-based 指标，
以及判别力（PR-AUC/MCC）与校准（Brier/ECE）扩展。

P0 口径修复（R-01 审阅）：
  - 核心指标逻辑统一继承 training.metrics.BinaryBondMetrics（padding-free、
    ex_f1 逐样本平均、from_logits 显式语义），本模块只做扩展，消除两份实现的漂移。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef

from training.metrics import (  # noqa: F401  (re-exported for backward compatibility)
    BinaryBondMetrics as _BaseBinaryBondMetrics,
    DBOND_M_COMPARABLE_METRIC_ORDER,
    TASK_EXTRA_METRIC_ORDER as _BASE_TASK_EXTRA_METRIC_ORDER,
    _sigmoid_if_needed,
    compute_binary_bond_metrics,
    order_binary_bond_metric_dict,
)


# 在基类的指标顺序上追加评估侧扩展指标，保持 CSV 输出列序稳定。
TASK_EXTRA_METRIC_ORDER = _BASE_TASK_EXTRA_METRIC_ORDER + (
    "pr_auc",
    "mcc",
    "brier_score",
    "ece",
)


def metric_display_name(metric_name: str) -> str:
    return "Loss" if metric_name == "loss" else metric_name


def metric_rows(metrics: Dict[str, float]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for key, value in order_binary_bond_metric_dict(metrics).items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            rows.append({
                "metric": metric_display_name(key),
                "value": float(value),
            })
    return rows

logger = logging.getLogger(__name__)


class BinaryBondMetrics(_BaseBinaryBondMetrics):
    """键级别二分类评估指标：基类 padding-free 指标 + PR-AUC/MCC/Brier 扩展。"""

    def compute(self) -> Dict[str, float]:
        metrics = super().compute()
        if not metrics:
            return metrics

        valid_probabilities = self.last_probabilities
        valid_targets = self.last_targets

        if valid_probabilities.size > 0 and len(np.unique(valid_targets)) > 1:
            try:
                pr_auc = average_precision_score(valid_targets, valid_probabilities)
            except ValueError:
                pr_auc = 0.0
        else:
            pr_auc = 0.0

        binary_predictions = (valid_probabilities > self.threshold).astype(np.int32) if valid_probabilities.size else np.array([], dtype=np.int32)
        if valid_targets.size > 1:
            try:
                mcc = float(matthews_corrcoef(valid_targets, binary_predictions))
            except ValueError:
                mcc = 0.0
        else:
            mcc = 0.0

        # 判别力（阈值无关）与校准指标：PR-AUC / MCC / Brier score
        metrics["pr_auc"] = pr_auc
        metrics["mcc"] = mcc
        if valid_probabilities.size > 0:
            metrics["brier_score"] = float(np.mean((valid_probabilities - valid_targets.astype(np.float32)) ** 2))
        else:
            metrics["brier_score"] = 0.0
        return order_binary_bond_metric_dict(metrics)


def compute_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """计算概率校准指标，用于 R-20 要求的概率可信度分析。

    返回：
      - ece: Expected Calibration Error（10-bin 加权平均 |置信度-准确率|）
      - brier_score: Brier 分数（独立口径，与 compute() 内的 brier_score 互校）
      - max_calibration_error: 最差 bin 的绝对偏差
      - n_samples: 有效键数
      - bin_confidences / bin_accuracies / bin_counts / bin_edges: 可靠性图原始数据

    说明：键级二分类的"准确率"= 该 bin 内真实断裂比例（正类频率）。
    空样本时所有数值指标返回 0.0，bin_* 返回空列表，保证可视化层无 NPE。
    """
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=np.int32).reshape(-1)

    result: Dict[str, Any] = {
        "ece": 0.0,
        "brier_score": 0.0,
        "max_calibration_error": 0.0,
        "n_samples": int(probabilities.size),
        "n_bins": int(n_bins),
        "bin_confidences": [],
        "bin_accuracies": [],
        "bin_counts": [],
        "bin_edges": [],
    }

    if probabilities.size == 0 or targets.size == 0:
        return result

    targets_float = targets.astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(probabilities, bin_edges[1:-1], right=False), 0, n_bins - 1)

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    ece = 0.0
    max_calibration_error = 0.0
    total = float(probabilities.size)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        count = int(np.sum(mask))
        if count == 0:
            continue
        conf = float(np.mean(probabilities[mask]))
        acc = float(np.mean(targets_float[mask]))
        bin_confidences.append(conf)
        bin_accuracies.append(acc)
        bin_counts.append(count)
        gap = abs(acc - conf)
        ece += gap * (count / total)
        if gap > max_calibration_error:
            max_calibration_error = gap

    result.update({
        "ece": float(ece),
        "brier_score": float(np.mean((probabilities - targets_float) ** 2)),
        "max_calibration_error": float(max_calibration_error),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
    })
    return result


# Backward-compatible aliases for older imports.
MultiLabelMetrics = BinaryBondMetrics
compute_multilabel_metrics = compute_binary_bond_metrics
