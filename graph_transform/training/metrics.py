"""
训练指标模块

本模块包含键级别二分类任务使用的训练指标与指标历史跟踪器。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score, roc_auc_score


class BinaryBondMetrics:
    """键级别二分类指标。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get("threshold", 0.5)
        self.threshold_strategy = config.get("threshold_strategy", "fixed")
        self.all_predictions = []
        self.all_targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.all_predictions.append(predictions.reshape(-1))
        self.all_targets.append(targets.reshape(-1))

    def compute(self) -> Dict[str, float]:
        if not self.all_predictions or not self.all_targets:
            return {}

        predictions = np.concatenate(self.all_predictions, axis=0).astype(np.float32)
        targets = np.concatenate(self.all_targets, axis=0).astype(np.int32)

        if predictions.size == 0:
            return {}

        if predictions.max() > 1.0 or predictions.min() < 0.0:
            probabilities = torch.sigmoid(torch.from_numpy(predictions)).numpy()
        else:
            probabilities = predictions

        threshold = self._get_threshold(probabilities, targets)
        binary_predictions = (probabilities >= threshold).astype(np.int32)

        metrics = {
            "accuracy": accuracy_score(targets, binary_predictions),
            "precision": precision_score(targets, binary_predictions, zero_division=0),
            "recall": recall_score(targets, binary_predictions, zero_division=0),
            "f1": f1_score(targets, binary_predictions, zero_division=0),
            "precision_micro": precision_score(targets, binary_predictions, average="binary", zero_division=0),
            "recall_micro": recall_score(targets, binary_predictions, average="binary", zero_division=0),
            "f1_micro": f1_score(targets, binary_predictions, average="binary", zero_division=0),
            "hamming_loss": hamming_loss(targets, binary_predictions),
            "positive_rate": float(np.mean(targets)),
            "pred_positive_rate": float(np.mean(binary_predictions)),
        }

        if len(np.unique(targets)) > 1:
            try:
                auc = roc_auc_score(targets, probabilities)
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
        return metrics

    def _get_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        if self.threshold_strategy == "fixed":
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
            binary_predictions = (predictions >= threshold).astype(np.int32)
            score = f1_score(targets, binary_predictions, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        return best_threshold

    def reset(self):
        self.all_predictions = []
        self.all_targets = []


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

            if metric_name in ['f1', 'auc', 'auc_macro', 'auc_micro'] and mode == 'val':
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
