"""
评估指标模块

本模块包含多标签分类的评估指标实现。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, hamming_loss, jaccard_score, coverage_error,
    label_ranking_average_precision_score, label_ranking_loss
)


class MultiLabelMetrics:
    """多标签分类评估指标"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get('threshold', 0.5)
        self.threshold_strategy = config.get('threshold_strategy', 'fixed')
        
        # 存储所有批次的预测和标签
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新预测和标签
        
        Args:
            predictions: 模型预测 [batch_size, num_classes]
            targets: 真实标签 [batch_size, num_classes]
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        self.all_predictions.append(predictions)
        self.all_targets.append(targets)
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            Dict[str, float]: 所有指标的值
        """
        if not self.all_predictions or not self.all_targets:
            return {}
        
        # 合并所有批次
        predictions = np.vstack(self.all_predictions)
        targets = np.vstack(self.all_targets)
        
        # 应用sigmoid到预测
        if predictions.max() > 1 or predictions.min() < 0:
            predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        # 确定阈值
        threshold = self._get_threshold(predictions, targets)
        
        # 二值化预测
        binary_predictions = (predictions >= threshold).astype(int)
        
        # 计算指标
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(targets, binary_predictions)
        metrics['precision'] = precision_score(targets, binary_predictions, average='macro', zero_division=0)
        metrics['recall'] = recall_score(targets, binary_predictions, average='macro', zero_division=0)
        metrics['f1'] = f1_score(targets, binary_predictions, average='macro', zero_division=0)
        
        # 微平均指标
        metrics['precision_micro'] = precision_score(targets, binary_predictions, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(targets, binary_predictions, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(targets, binary_predictions, average='micro', zero_division=0)
        
        # 多标签特定指标
        try:
            metrics['hamming_loss'] = hamming_loss(targets, binary_predictions)
            metrics['jaccard'] = jaccard_score(targets, binary_predictions, average='macro', zero_division=0)
            
            # 排序指标
            if len(np.unique(targets)) > 1:  # 确保有正负样本
                metrics['coverage_error'] = coverage_error(targets, predictions)
                metrics['label_ranking_average_precision'] = label_ranking_average_precision_score(targets, predictions)
                metrics['label_ranking_loss'] = label_ranking_loss(targets, predictions)
        except ValueError as e:
            print(f"Warning: Could not compute some metrics: {e}")
        
        # AUC指标
        try:
            metrics['auc_macro'] = roc_auc_score(targets, predictions, average='macro')
            metrics['auc_micro'] = roc_auc_score(targets, predictions, average='micro')
            metrics['auc_weighted'] = roc_auc_score(targets, predictions, average='weighted')
        except ValueError as e:
            print(f"Warning: Could not compute AUC: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_micro'] = 0.0
            metrics['auc_weighted'] = 0.0
        
        # 按类别计算指标
        class_metrics = self._compute_class_metrics(targets, binary_predictions)
        metrics.update(class_metrics)
        
        return metrics
    
    def _get_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        获取分类阈值
        
        Args:
            predictions: 预测概率
            targets: 真实标签
            
        Returns:
            float: 分类阈值
        """
        if self.threshold_strategy == 'fixed':
            return self.threshold
        elif self.threshold_strategy == 'adaptive':
            return self._adaptive_threshold(predictions, targets)
        elif self.threshold_strategy == 'optimal':
            return self._optimal_threshold(predictions, targets)
        else:
            return self.threshold
    
    def _adaptive_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """自适应阈值"""
        # 使用预测概率的均值作为阈值
        return np.mean(predictions)
    
    def _optimal_threshold(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """最优阈值（基于F1分数）"""
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = self.threshold
        best_f1 = 0.0
        
        for threshold in thresholds:
            binary_preds = (predictions >= threshold).astype(int)
            f1 = f1_score(targets, binary_preds, average='macro', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def _compute_class_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """计算每个类别的指标"""
        num_classes = targets.shape[1]
        class_metrics = {}
        
        for i in range(num_classes):
            class_targets = targets[:, i]
            class_preds = predictions[:, i]
            
            # 跳过没有正样本的类别
            if np.sum(class_targets) == 0:
                class_metrics[f'class_{i}_precision'] = 0.0
                class_metrics[f'class_{i}_recall'] = 0.0
                class_metrics[f'class_{i}_f1'] = 0.0
            else:
                precision = precision_score(class_targets, class_preds, zero_division=0)
                recall = recall_score(class_targets, class_preds, zero_division=0)
                f1 = f1_score(class_targets, class_preds, zero_division=0)
                
                class_metrics[f'class_{i}_precision'] = precision
                class_metrics[f'class_{i}_recall'] = recall
                class_metrics[f'class_{i}_f1'] = f1
        
        return class_metrics
    
    def reset(self):
        """重置存储的预测和标签"""
        self.all_predictions = []
        self.all_targets = []


def compute_multilabel_metrics(predictions: torch.Tensor, 
                              targets: torch.Tensor,
                              threshold: float = 0.5) -> Dict[str, float]:
    """
    计算多标签指标（便捷函数）
    
    Args:
        predictions: 模型预测 [batch_size, num_classes]
        targets: 真实标签 [batch_size, num_classes]
        threshold: 分类阈值
        
    Returns:
        Dict[str, float]: 指标字典
    """
    # 转换为numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # 应用sigmoid
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # 二值化
    binary_predictions = (predictions >= threshold).astype(int)
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(targets, binary_predictions),
        'precision': precision_score(targets, binary_predictions, average='macro', zero_division=0),
        'recall': recall_score(targets, binary_predictions, average='macro', zero_division=0),
        'f1': f1_score(targets, binary_predictions, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(targets, binary_predictions),
    }
    
    try:
        metrics['auc_macro'] = roc_auc_score(targets, predictions, average='macro')
        metrics['auc_micro'] = roc_auc_score(targets, predictions, average='micro')
    except ValueError:
        metrics['auc_macro'] = 0.0
        metrics['auc_micro'] = 0.0
    
    return metrics
