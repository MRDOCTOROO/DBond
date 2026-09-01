"""
训练指标模块

本模块包含键级别二分类任务使用的训练指标与指标历史跟踪器，
并补齐与 dbond_m 横向对齐的多标签指标口径。

P0 口径修复（R-01 审阅）：
  - 所有 example/label 指标只在真实存在的 bond（valid mask 内）上计算，
    padding 位置不再计入 TN，消除短序列对 subset/ex/lab 指标的系统性抬高；
  - ex_f1 改为逐样本 F1 的平均值（旧实现是先平均 precision/recall 再调和，
    数值上不等价于 macro example-F1）；
  - update() 显式区分 logits / probabilities（from_logits，默认 True），
    不再按数值范围猜测输入类型。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score, roc_auc_score


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


def _sigmoid_if_needed(values: np.ndarray) -> np.ndarray:
    """兼容别名：新代码请使用 update(from_logits=...) 显式语义。"""
    if values.size == 0:
        return values.astype(np.float32)
    if values.max() > 1.0 or values.min() < 0.0:
        return torch.sigmoid(torch.from_numpy(values.astype(np.float32))).numpy()
    return values.astype(np.float32)


logger = logging.getLogger(__name__)


def _to_probabilities(values: np.ndarray, from_logits: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if from_logits:
        return torch.sigmoid(torch.from_numpy(values)).numpy()
    return values


# =============================================================================
# padding-free 指标：所有计算都限制在 valid mask 内
# =============================================================================

def _example_confusion(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray):
    """逐样本 (tp, fp, tn, fn)，只统计 valid 位置。"""
    tp = np.sum((gt == 1) & (pred == 1) & valid, axis=1).astype(np.float64)
    fp = np.sum((gt == 0) & (pred == 1) & valid, axis=1).astype(np.float64)
    tn = np.sum((gt == 0) & (pred == 0) & valid, axis=1).astype(np.float64)
    fn = np.sum((gt == 1) & (pred == 0) & valid, axis=1).astype(np.float64)
    return tp, fp, tn, fn


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _example_subset_accuracy(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    """整肽完全匹配率：所有 valid 位置都相等；无 valid bond 的样本不计入。"""
    has_bond = valid.any(axis=1)
    if not has_bond.any():
        return 0.0
    all_eq = np.all((gt == pred) | ~valid, axis=1)
    return float(np.mean(all_eq[has_bond]))


def _example_accuracy(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    tp, fp, tn, fn = _example_confusion(gt, pred, valid)
    has_bond = valid.any(axis=1)
    if not has_bond.any():
        return 0.0
    jaccard_den = tp + fp + fn  # example accuracy = TP / (TP+FP+FN) 的 Jaccard 口径
    return float(np.mean(_safe_div(tp, jaccard_den)[has_bond]))


def _example_precision(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    tp, fp, _, _ = _example_confusion(gt, pred, valid)
    has_bond = valid.any(axis=1)
    if not has_bond.any():
        return 0.0
    return float(np.mean(_safe_div(tp, tp + fp)[has_bond]))


def _example_recall(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    tp, _, _, fn = _example_confusion(gt, pred, valid)
    has_bond = valid.any(axis=1)
    if not has_bond.any():
        return 0.0
    return float(np.mean(_safe_div(tp, tp + fn)[has_bond]))


def _example_f1(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, beta: float = 1.0) -> float:
    """逐样本 F1 后取平均（macro example-F1），空样本不计入。"""
    tp, fp, _, fn = _example_confusion(gt, pred, valid)
    has_bond = valid.any(axis=1)
    if not has_bond.any():
        return 0.0
    f1_den = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    f1 = np.where(f1_den > 0, (1 + beta ** 2) * tp / np.where(f1_den > 0, f1_den, 1.0), 0.0)
    return float(np.mean(f1[has_bond]))


def _label_quantity(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """按键位置（列）聚合 tp/fp/tn/fn，只统计 valid 位置。"""
    tp = np.sum((gt == 1) & (pred == 1) & valid, axis=0)
    fp = np.sum((gt == 0) & (pred == 1) & valid, axis=0)
    tn = np.sum((gt == 0) & (pred == 0) & valid, axis=0)
    fn = np.sum((gt == 1) & (pred == 0) & valid, axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype(np.float64)


def _label_accuracy_macro(quantity: np.ndarray) -> float:
    tp_tn = quantity[0] + quantity[2]
    denom = np.sum(quantity, axis=0)
    valid_cols = denom > 0
    if not valid_cols.any():
        return 0.0
    return float(np.mean(tp_tn[valid_cols] / denom[valid_cols]))


def _label_accuracy_micro(quantity: np.ndarray) -> float:
    tp, fp, tn, fn = np.sum(quantity, axis=1)
    denom = tp + fp + tn + fn
    return float((tp + tn) / denom) if denom > 0 else 0.0


def _label_precision_macro(quantity: np.ndarray) -> float:
    tp, fp = quantity[0], quantity[1]
    valid_cols = (tp + fp) > 0
    if not valid_cols.any():
        return 0.0
    return float(np.mean(tp[valid_cols] / (tp + fp)[valid_cols]))


def _label_precision_micro(quantity: np.ndarray) -> float:
    tp, fp = np.sum(quantity[0]), np.sum(quantity[1])
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def _label_recall_macro(quantity: np.ndarray) -> float:
    tp, fn = quantity[0], quantity[3]
    valid_cols = (tp + fn) > 0
    if not valid_cols.any():
        return 0.0
    return float(np.mean(tp[valid_cols] / (tp + fn)[valid_cols]))


def _label_recall_micro(quantity: np.ndarray) -> float:
    tp, fn = np.sum(quantity[0]), np.sum(quantity[3])
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _label_f1_macro(quantity: np.ndarray, beta: float = 1.0) -> float:
    tp, fp, fn = quantity[0], quantity[1], quantity[3]
    f1_den = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    valid_cols = f1_den > 0
    if not valid_cols.any():
        return 0.0
    return float(np.mean((1 + beta ** 2) * tp[valid_cols] / f1_den[valid_cols]))


def _label_f1_micro(quantity: np.ndarray, beta: float = 1.0) -> float:
    tp = np.sum(quantity[0])
    fp = np.sum(quantity[1])
    fn = np.sum(quantity[3])
    f1_den = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    return float((1 + beta ** 2) * tp / f1_den) if f1_den > 0 else 0.0


class BinaryBondMetrics:
    """键级别二分类指标（padding-free 口径），同时输出 dbond_m 同口径指标。"""

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
        # 缓存最近一次 compute() 得到的扁平化概率与标签，供 evaluator 计算校准指标
        # (ECE/Brier) 和外部诊断使用。每次 compute() 都会覆盖。
        self.last_probabilities: np.ndarray = np.array([], dtype=np.float32)
        self.last_targets: np.ndarray = np.array([], dtype=np.int32)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor | None = None,
        from_logits: bool = True,
    ):
        """累积一个 batch 的预测。

        Args:
            predictions: 模型输出。``from_logits=True``（默认）时视为未过 sigmoid 的
                logits；显式传 ``from_logits=False`` 表示已是概率。
            targets: 与 predictions 同形状的标签。
            label_mask: 有效 bond 掩码（True=参与统计）。
            from_logits: 输入是否为 logits。
        """
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

        valid_probabilities = _to_probabilities(valid_predictions, from_logits=True)
        threshold = self._get_threshold(valid_probabilities, valid_targets)
        binary_valid_predictions = (valid_probabilities > threshold).astype(np.int32)

        # 缓存扁平化概率/标签，供 evaluator 计算校准指标（ECE/Brier）使用
        self.last_probabilities = valid_probabilities
        self.last_targets = valid_targets

        # padding-free 矩阵：每个样本只在其真实键数内参与统计
        max_len = max((sample.size for sample in self.sample_predictions), default=0)
        n_samples = len(self.sample_predictions)
        pred_matrix = np.zeros((n_samples, max_len), dtype=np.int32)
        target_matrix = np.zeros((n_samples, max_len), dtype=np.int32)
        valid_matrix = np.zeros((n_samples, max_len), dtype=bool)
        for idx, (prob_row, target_row) in enumerate(zip(self.sample_predictions, self.sample_targets)):
            row_len = prob_row.size
            if row_len == 0:
                continue
            pred_matrix[idx, :row_len] = (prob_row > threshold).astype(np.int32)
            target_matrix[idx, :row_len] = target_row.astype(np.int32)
            valid_matrix[idx, :row_len] = True

        quantity = _label_quantity(target_matrix, pred_matrix, valid_matrix)
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
            "subset_acc": _example_subset_accuracy(target_matrix, pred_matrix, valid_matrix),
            "ex_acc": _example_accuracy(target_matrix, pred_matrix, valid_matrix),
            "ex_precision": _example_precision(target_matrix, pred_matrix, valid_matrix),
            "ex_recall": _example_recall(target_matrix, pred_matrix, valid_matrix),
            "ex_f1": _example_f1(target_matrix, pred_matrix, valid_matrix),
            "lab_acc_ma": _label_accuracy_macro(quantity),
            "lab_acc_mi": _label_accuracy_micro(quantity),
            "lab_precision_ma": _label_precision_macro(quantity),
            "lab_precision_mi": _label_precision_micro(quantity),
            "lab_recall_ma": _label_recall_macro(quantity),
            "lab_recall_mi": _label_recall_micro(quantity),
            "lab_f1_ma": _label_f1_macro(quantity),
            "lab_f1_mi": _label_f1_micro(quantity),
        }

        if len(np.unique(valid_targets)) > 1:
            try:
                auc = roc_auc_score(valid_targets, valid_probabilities)
            except ValueError:
                auc = 0.0
        else:
            auc = 0.0

        metrics["auc"] = auc
        metrics["auc_macro"] = auc
        metrics["auc_micro"] = auc
        metrics["auc_weighted"] = auc
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


class MetricTracker:
    """指标跟踪器"""

    def __init__(self):
        self.metrics_history = {}
        self.best_metrics = {}
        self.current_epoch = 0

    def update(self, epoch: int, metrics: Dict[str, float], mode: str = 'train'):
        self.current_epoch = epoch

        for metric_name, value in metrics.items():
            key = f"{mode}_{metric_name}"
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

            if metric_name in ['f1', 'auc', 'auc_macro', 'auc_micro', 'lab_f1_mi', 'ex_f1'] and mode == 'val':
                best_key = f"best_{metric_name}"
                if best_key not in self.best_metrics or value > self.best_metrics[best_key]:
                    self.best_metrics[best_key] = value
                    self.best_metrics[f"{best_key}_epoch"] = epoch

    def get_best_metrics(self) -> Dict[str, float]:
        return self.best_metrics

    def get_metric_history(self, metric_name: str, mode: str = 'train') -> List[float]:
        key = f"{mode}_{metric_name}"
        return self.metrics_history.get(key, [])

    def print_summary(self):
        print("\n=== Training Summary ===")
        print(f"Total epochs: {self.current_epoch}")

        for key, value in self.best_metrics.items():
            if not key.endswith('_epoch'):
                epoch_key = f"{key}_epoch"
                epoch = self.best_metrics.get(epoch_key, 'N/A')
                print(f"{key}: {value:.4f} (epoch {epoch})")


def compute_binary_bond_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    metrics = BinaryBondMetrics({"threshold": threshold, "threshold_strategy": "fixed"})
    metrics.update(predictions, targets)
    return metrics.compute()


# Backward-compatible aliases for older imports.
MultiLabelMetrics = BinaryBondMetrics
compute_multilabel_metrics = compute_binary_bond_metrics
