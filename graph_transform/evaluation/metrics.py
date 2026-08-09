"""
评估指标模块

本模块包含键级别二分类评估指标实现，同时补齐与 dbond_m 横向对齐的
example-based / label-based 指标。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPSILON = 1e-8
DBOND_M_COMPARABLE_METRIC_ORDER = (
    "loss",
    "dbond_style_loss",
    "subset_acc",
    "ex_acc",
    "ex_precision",
    "ex_recall",
    "ex_f1",
    "lab_acc_ma",
    "lab_acc_mi",
    "lab_precision_ma",
    "lab_precision_mi",
    "lab_recall_ma",
    "lab_recall_mi",
    "lab_f1_ma",
    "lab_f1_mi",
)
TASK_EXTRA_METRIC_ORDER = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "auc",
    "auc_macro",
    "auc_micro",
    "auc_weighted",
    "pr_auc",
    "mcc",
    "brier_score",
    "ece",
    "hamming_loss",
    "positive_rate",
    "pred_positive_rate",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1",
    "avg_fetch_wait_time",
    "avg_move_time",
    "avg_forward_time",
    "avg_backward_time",
    "avg_batch_time",
    "avg_grad_norm",
    "max_grad_norm",
    "gpu_mem_start_allocated_mb",
    "gpu_mem_end_allocated_mb",
    "gpu_mem_end_reserved_mb",
    "gpu_mem_peak_allocated_mb",
    "gpu_mem_peak_reserved_mb",
    "gpu_mem_end_free_mb",
    "gpu_mem_total_mb",
)

logger = logging.getLogger(__name__)


def order_binary_bond_metric_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    ordered: OrderedDict[str, float] = OrderedDict()
    for key in DBOND_M_COMPARABLE_METRIC_ORDER + TASK_EXTRA_METRIC_ORDER:
        if key in metrics:
            ordered[key] = metrics[key]
    for key, value in metrics.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


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


def _sigmoid_if_needed(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    if values.max() > 1.0 or values.min() < 0.0:
        return torch.sigmoid(torch.from_numpy(values.astype(np.float32))).numpy()
    return values.astype(np.float32)


def _example_subset_accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.all(np.equal(gt, pred), axis=1).astype(np.float32)))


def _example_accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    ex_and = np.sum(np.logical_and(gt, pred), axis=1).astype(np.float32)
    ex_or = np.sum(np.logical_or(gt, pred), axis=1).astype(np.float32)
    return float(np.mean(ex_and / (ex_or + EPSILON)))


def _example_precision(gt: np.ndarray, pred: np.ndarray) -> float:
    ex_and = np.sum(np.logical_and(gt, pred), axis=1).astype(np.float32)
    ex_pred = np.sum(pred, axis=1).astype(np.float32)
    return float(np.mean(ex_and / (ex_pred + EPSILON)))


def _example_recall(gt: np.ndarray, pred: np.ndarray) -> float:
    ex_and = np.sum(np.logical_and(gt, pred), axis=1).astype(np.float32)
    ex_gt = np.sum(gt, axis=1).astype(np.float32)
    return float(np.mean(ex_and / (ex_gt + EPSILON)))


def _example_f1(gt: np.ndarray, pred: np.ndarray, beta: float = 1.0) -> float:
    precision = _example_precision(gt, pred)
    recall = _example_recall(gt, pred)
    return float(((1 + beta ** 2) * precision * recall) / ((beta ** 2) * (precision + recall + EPSILON)))


def _label_quantity(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    tp = np.sum(np.logical_and(gt, pred), axis=0)
    fp = np.sum(np.logical_and(1 - gt, pred), axis=0)
    tn = np.sum(np.logical_and(1 - gt, 1 - pred), axis=0)
    fn = np.sum(np.logical_and(gt, 1 - pred), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype(np.float64)


def _label_accuracy_macro(gt: np.ndarray, pred: np.ndarray) -> float:
    quantity = _label_quantity(gt, pred)
    tp_tn = quantity[0] + quantity[2]
    denom = np.sum(quantity, axis=0)
    return float(np.mean(tp_tn / (denom + EPSILON)))


def _label_accuracy_micro(gt: np.ndarray, pred: np.ndarray) -> float:
    quantity = _label_quantity(gt, pred)
    tp, fp, tn, fn = np.sum(quantity, axis=1)
    return float((tp + tn) / (tp + fp + tn + fn + EPSILON))


def _label_precision_macro(gt: np.ndarray, pred: np.ndarray) -> float:
    quantity = _label_quantity(gt, pred)
    tp = quantity[0]
    tp_fp = quantity[0] + quantity[1]
    return float(np.mean(tp / (tp_fp + EPSILON)))


def _label_precision_micro(gt: np.ndarray, pred: np.ndarray) -> float:
    quantity = _label_quantity(gt, pred)
    tp, fp, _, _ = np.sum(quantity, axis=1)
    return float(tp / (tp + fp + EPSILON))


def _label_recall_macro(gt: np.ndarray, pred: np.ndarray) -> float:
    quantity = _label_quantity(gt, pred)
    tp = quantity[0]
    tp_fn = quantity[0] + quantity[3]
    return float(np.mean(tp / (tp_fn + EPSILON)))


def _label_recall_micro(gt: np.ndarray, pred: np.ndarray) -> float:
    quantity = _label_quantity(gt, pred)
    tp, _, _, fn = np.sum(quantity, axis=1)
    return float(tp / (tp + fn + EPSILON))


def _label_f1_macro(gt: np.ndarray, pred: np.ndarray, beta: float = 1.0) -> float:
    quantity = _label_quantity(gt, pred)
    tp = quantity[0]
    fp = quantity[1]
    fn = quantity[3]
    return float(np.mean((1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + EPSILON)))


def _label_f1_micro(gt: np.ndarray, pred: np.ndarray, beta: float = 1.0) -> float:
    quantity = _label_quantity(gt, pred)
    tp = np.sum(quantity[0])
    fp = np.sum(quantity[1])
    fn = np.sum(quantity[3])
    return float((1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + EPSILON))


class BinaryBondMetrics:
    """键级别二分类评估指标，同时输出 dbond_m 同口径指标。"""

    def __init__(self, config: Dict[str, Any], allow_target_aware_threshold: bool = True):
        self.config = config
        self.threshold = config.get("threshold", 0.5)
        self.threshold_strategy = config.get("threshold_strategy", "fixed")
        # 当为 False 时（如 test 评估），禁止使用依赖被评估集标签/预测分布的策略，
        # 强制回退到固定阈值，避免阈值挑选造成的数据泄露。
        self.allow_target_aware_threshold = allow_target_aware_threshold
        self._target_aware_warned = False
        self.all_valid_predictions: List[np.ndarray] = []
        self.all_valid_targets: List[np.ndarray] = []
        self.sample_predictions: List[np.ndarray] = []
        self.sample_targets: List[np.ndarray] = []
        # 缓存最近一次 compute() 得到的扁平化概率与标签，供 evaluator 计算校准指标(ECE/Brier)
        # 和外部诊断使用，避免重复推理。每次 compute() 都会覆盖。
        self.last_probabilities: np.ndarray = np.array([], dtype=np.float32)
        self.last_targets: np.ndarray = np.array([], dtype=np.int32)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, label_mask: torch.Tensor | None = None):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.detach().cpu().numpy()

        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
            targets = targets.reshape(1, -1)
            if label_mask is None:
                label_mask = np.ones_like(targets, dtype=bool)
            else:
                label_mask = np.asarray(label_mask).reshape(1, -1).astype(bool)
        else:
            if label_mask is None:
                label_mask = np.ones_like(targets, dtype=bool)
            else:
                label_mask = np.asarray(label_mask).astype(bool)

        for pred_row, target_row, mask_row in zip(predictions, targets, label_mask):
            mask_row = mask_row.astype(bool)
            valid_pred = pred_row[mask_row].reshape(-1)
            valid_target = target_row[mask_row].reshape(-1).astype(np.int32)
            self.sample_predictions.append(valid_pred)
            self.sample_targets.append(valid_target)
            if valid_pred.size > 0:
                self.all_valid_predictions.append(valid_pred)
                self.all_valid_targets.append(valid_target)

    def compute(self) -> Dict[str, float]:
        if not self.sample_predictions or not self.sample_targets:
            return {}

        valid_predictions = np.concatenate(self.all_valid_predictions, axis=0).astype(np.float32) if self.all_valid_predictions else np.array([], dtype=np.float32)
        valid_targets = np.concatenate(self.all_valid_targets, axis=0).astype(np.int32) if self.all_valid_targets else np.array([], dtype=np.int32)
        if valid_predictions.size == 0:
            return {}

        valid_probabilities = _sigmoid_if_needed(valid_predictions)
        threshold = self._get_threshold(valid_probabilities, valid_targets)
        binary_valid_predictions = (valid_probabilities > threshold).astype(np.int32)

        # 缓存扁平化概率/标签，供 evaluator 计算校准指标（ECE/Brier）使用
        self.last_probabilities = valid_probabilities
        self.last_targets = valid_targets

        sample_probabilities = [_sigmoid_if_needed(sample.astype(np.float32)) for sample in self.sample_predictions]
        max_len = max((sample.size for sample in sample_probabilities), default=0)
        pred_matrix = np.zeros((len(sample_probabilities), max_len), dtype=np.int32)
        target_matrix = np.zeros((len(self.sample_targets), max_len), dtype=np.int32)

        for idx, (prob_row, target_row) in enumerate(zip(sample_probabilities, self.sample_targets)):
            row_len = prob_row.size
            if row_len == 0:
                continue
            pred_matrix[idx, :row_len] = (prob_row > threshold).astype(np.int32)
            target_matrix[idx, :row_len] = target_row.astype(np.int32)

        metrics = {
            "accuracy": accuracy_score(valid_targets, binary_valid_predictions),
            "precision": precision_score(valid_targets, binary_valid_predictions, zero_division=0),
            "recall": recall_score(valid_targets, binary_valid_predictions, zero_division=0),
            "f1": f1_score(valid_targets, binary_valid_predictions, zero_division=0),
            "precision_micro": precision_score(valid_targets, binary_valid_predictions, average="binary", zero_division=0),
            "recall_micro": recall_score(valid_targets, binary_valid_predictions, average="binary", zero_division=0),
            "f1_micro": f1_score(valid_targets, binary_valid_predictions, average="binary", zero_division=0),
            "hamming_loss": hamming_loss(valid_targets, binary_valid_predictions),
            "positive_rate": float(np.mean(valid_targets)),
            "pred_positive_rate": float(np.mean(binary_valid_predictions)),
            "subset_acc": _example_subset_accuracy(target_matrix, pred_matrix),
            "ex_acc": _example_accuracy(target_matrix, pred_matrix),
            "ex_precision": _example_precision(target_matrix, pred_matrix),
            "ex_recall": _example_recall(target_matrix, pred_matrix),
            "ex_f1": _example_f1(target_matrix, pred_matrix),
            "lab_acc_ma": _label_accuracy_macro(target_matrix, pred_matrix),
            "lab_acc_mi": _label_accuracy_micro(target_matrix, pred_matrix),
            "lab_precision_ma": _label_precision_macro(target_matrix, pred_matrix),
            "lab_precision_mi": _label_precision_micro(target_matrix, pred_matrix),
            "lab_recall_ma": _label_recall_macro(target_matrix, pred_matrix),
            "lab_recall_mi": _label_recall_micro(target_matrix, pred_matrix),
            "lab_f1_ma": _label_f1_macro(target_matrix, pred_matrix),
            "lab_f1_mi": _label_f1_micro(target_matrix, pred_matrix),
        }

        if len(np.unique(valid_targets)) > 1:
            try:
                auc = roc_auc_score(valid_targets, valid_probabilities)
            except ValueError:
                auc = 0.0
            # PR-AUC（average precision）：类别不平衡场景下比 ROC-AUC 更关注正类
            try:
                pr_auc = average_precision_score(valid_targets, valid_probabilities)
            except ValueError:
                pr_auc = 0.0
        else:
            auc = 0.0
            pr_auc = 0.0

        metrics["auc"] = auc
        metrics["auc_macro"] = auc
        metrics["auc_micro"] = auc
        metrics["auc_weighted"] = auc
        # 判别力（阈值无关）与校准指标：PR-AUC / MCC / Brier score
        metrics["pr_auc"] = pr_auc
        metrics["mcc"] = float(matthews_corrcoef(valid_targets, binary_valid_predictions))
        metrics["brier_score"] = float(np.mean((valid_probabilities - valid_targets.astype(np.float32)) ** 2))
        metrics["class_0_precision"] = metrics["precision"]
        metrics["class_0_recall"] = metrics["recall"]
        metrics["class_0_f1"] = metrics["f1"]
        return order_binary_bond_metric_dict(metrics)

    def _get_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        if self.threshold_strategy == "fixed":
            return self.threshold
        # target-aware 策略（adaptive/optimal）会使用被评估集自身的标签或预测分布，
        # 在 test 评估上会构成阈值泄露。当 allow_target_aware_threshold=False 时回退到固定阈值。
        if self.threshold_strategy in ("adaptive", "optimal") and not self.allow_target_aware_threshold:
            if not self._target_aware_warned:
                logger.warning(
                    "threshold_strategy=%r 会使用被评估集的标签/预测分布，"
                    "已强制回退到固定阈值 %.3f 以避免数据泄露。",
                    self.threshold_strategy, self.threshold,
                )
                self._target_aware_warned = True
            return self.threshold
        if self.threshold_strategy == "adaptive":
            return float(np.mean(predictions))
        if self.threshold_strategy == "optimal":
            return self._optimal_threshold(predictions, targets)
        return self.threshold

    def _optimal_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = self.threshold
        best_f1 = -1.0
        for threshold in thresholds:
            binary_predictions = (predictions > threshold).astype(np.int32)
            score = f1_score(targets, binary_predictions, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold

    def reset(self):
        self.all_valid_predictions = []
        self.all_valid_targets = []
        self.sample_predictions = []
        self.sample_targets = []
        self.last_probabilities = np.array([], dtype=np.float32)
        self.last_targets = np.array([], dtype=np.int32)


def compute_binary_bond_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    metrics = BinaryBondMetrics({"threshold": threshold, "threshold_strategy": "fixed"})
    metrics.update(predictions, targets)
    return metrics.compute()


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
