"""
评估器模块

本模块包含图神经网络的评估器实现。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np

from .metrics import MultiLabelMetrics
from training import MultiLabelLoss


class Evaluator:
    """图神经网络评估器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device = torch.device('cpu'),
                 config: Dict[str, Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化评估器
        
        Args:
            model: 模型
            device: 计算设备
            config: 配置
            logger: 日志记录器
        """
        self.model = model
        self.device = device
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 评估配置
        self.eval_config = self.config.get('evaluation', {})
        self.use_amp = self.config.get('device', {}).get('use_amp', False)
        if self.device.type != 'cuda':
            self.use_amp = False
        self.amp_device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        # 指标计算器
        self.metrics_calculator = MultiLabelMetrics(self.eval_config)
        
        # 将模型移动到设备
        self.model.to(self.device)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            Dict[str, float]: 评估指标
        """
        self.model.eval()
        
        # 重置指标
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        # 损失函数（与训练保持一致）
        loss_config = self.config.get('loss', {})
        criterion = MultiLabelLoss(loss_config).to(self.device)
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating")
            
            for batch_idx, batch_data in enumerate(pbar):
                # 将数据移动到设备
                batch_data = self._move_to_device(batch_data)
                
                # 前向传播
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(batch_data)
                        targets = batch_data['labels']
                        targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
                        loss = criterion(predictions, targets)
                else:
                    predictions = self.model(batch_data)
                    targets = batch_data['labels']
                    targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
                    loss = criterion(predictions, targets)
                
                # 更新指标
                self.metrics_calculator.update(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / num_batches:.4f}'
                })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 计算指标
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss
        
        # 记录结果
        self.logger.info(f"Evaluation - Loss: {avg_loss:.4f}")
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'loss'])
            self.logger.info(f"Evaluation Metrics - {metric_str}")
        
        return metrics
    
    def evaluate_with_thresholds(self, data_loader: DataLoader, 
                               thresholds: list = None) -> Dict[str, Any]:
        """
        使用不同阈值评估模型
        
        Args:
            data_loader: 数据加载器
            thresholds: 阈值列表
            
        Returns:
            Dict[str, Any]: 不同阈值下的评估结果
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        self.model.eval()
        
        # 收集所有预测和标签
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Collecting predictions"):
                batch_data = self._move_to_device(batch_data)
                predictions = self.model(batch_data)
                targets = batch_data['labels']
                targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # 合并所有批次
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 应用sigmoid
        all_predictions = torch.sigmoid(all_predictions)
        
        # 计算不同阈值下的指标
        results = {}
        
        for threshold in thresholds:
            binary_predictions = (all_predictions >= threshold).float()
            
            # 计算指标
            metrics = self._compute_metrics_threshold(
                all_predictions, all_targets, binary_predictions, threshold
            )
            results[f'threshold_{threshold}'] = metrics
        
        # 找到最佳阈值
        best_threshold = self._find_best_threshold(results)
        results['best_threshold'] = best_threshold
        
        self.logger.info(f"Best threshold: {best_threshold}")
        
        return results
    
    def evaluate_per_class(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        按类别评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            Dict[str, Any]: 按类别的评估结果
        """
        self.model.eval()
        
        # 收集所有预测和标签
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Collecting predictions for per-class evaluation"):
                batch_data = self._move_to_device(batch_data)
                predictions = self.model(batch_data)
                targets = batch_data['labels']
                targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # 合并所有批次
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 应用sigmoid和二值化
        all_predictions = torch.sigmoid(all_predictions)
        threshold = self.eval_config.get('threshold', 0.5)
        binary_predictions = (all_predictions >= threshold).float()
        
        # 按类别计算指标
        num_classes = all_targets.shape[1]
        per_class_metrics = {}
        
        for i in range(num_classes):
            class_targets = all_targets[:, i]
            class_preds = binary_predictions[:, i]
            class_probs = all_predictions[:, i]
            
            # 计算指标
            class_metrics = self._compute_single_class_metrics(
                class_targets, class_preds, class_probs, i
            )
            per_class_metrics[f'class_{i}'] = class_metrics
        
        # 计算类别统计
        class_stats = self._compute_class_statistics(all_targets)
        per_class_metrics['class_statistics'] = class_stats
        
        return per_class_metrics
    
    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        对数据集进行预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            torch.Tensor: 预测结果
        """
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Predicting"):
                batch_data = self._move_to_device(batch_data)
                
                if self.use_amp:
                    with torch.amp.autocast(self.amp_device_type, enabled=True):
                        predictions = self.model(batch_data)
                else:
                    predictions = self.model(batch_data)
                predictions = self._apply_label_mask_predict(batch_data, predictions)
                all_predictions.append(predictions.cpu())
        
        # 合并所有批次
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # 应用sigmoid
        all_predictions = torch.sigmoid(all_predictions)
        
        return all_predictions
    
    def predict_with_labels(self, data_loader: DataLoader) -> tuple:
        """
        对数据集进行预测并返回标签
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            tuple: (预测结果, 真实标签)
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Predicting with labels"):
                batch_data = self._move_to_device(batch_data)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(batch_data)
                else:
                    predictions = self.model(batch_data)
                targets = batch_data['labels']
                targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # 合并所有批次
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 应用sigmoid
        all_predictions = torch.sigmoid(all_predictions)
        
        return all_predictions, all_targets
    
    def _move_to_device(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """将批次数据移动到设备"""
        device_batch = {}
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        
        return device_batch

    def _apply_label_mask(self, batch_data: Dict[str, Any],
                          targets: torch.Tensor,
                          predictions: torch.Tensor) -> tuple:
        """应用标签掩码，过滤padding位置"""
        mask = batch_data.get('label_mask')
        if mask is None:
            return targets, predictions

        mask = mask.bool()
        masked_preds = predictions[mask]
        masked_targets = targets[mask]

        if masked_preds.dim() == 1:
            masked_preds = masked_preds.unsqueeze(1)
        if masked_targets.dim() == 1:
            masked_targets = masked_targets.unsqueeze(1)

        return masked_targets, masked_preds

    def _apply_label_mask_predict(self, batch_data: Dict[str, Any],
                                  predictions: torch.Tensor) -> torch.Tensor:
        """预测时应用标签掩码，过滤padding位置"""
        mask = batch_data.get('label_mask')
        if mask is None:
            return predictions

        mask = mask.bool()
        masked_preds = predictions[mask]
        if masked_preds.dim() == 1:
            masked_preds = masked_preds.unsqueeze(1)
        return masked_preds
    
    def _compute_metrics_threshold(self, predictions: torch.Tensor, 
                                 targets: torch.Tensor,
                                 binary_predictions: torch.Tensor,
                                 threshold: float) -> Dict[str, float]:
        """计算特定阈值下的指标"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 转换为numpy
        preds_np = binary_predictions.numpy()
        targets_np = targets.numpy()
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(targets_np, preds_np),
            'precision': precision_score(targets_np, preds_np, average='macro', zero_division=0),
            'recall': recall_score(targets_np, preds_np, average='macro', zero_division=0),
            'f1': f1_score(targets_np, preds_np, average='macro', zero_division=0),
        }
        
        return metrics
    
    def _compute_single_class_metrics(self, targets: torch.Tensor,
                                    predictions: torch.Tensor,
                                    probabilities: torch.Tensor,
                                    class_idx: int) -> Dict[str, float]:
        """计算单个类别的指标"""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # 转换为numpy
        targets_np = targets.numpy()
        preds_np = predictions.numpy()
        probs_np = probabilities.numpy()
        
        # 计算指标
        metrics = {}
        
        if np.sum(targets_np) > 0:  # 如果有正样本
            metrics['precision'] = precision_score(targets_np, preds_np, zero_division=0)
            metrics['recall'] = recall_score(targets_np, preds_np, zero_division=0)
            metrics['f1'] = f1_score(targets_np, preds_np, zero_division=0)
            
            # AUC
            try:
                metrics['auc'] = roc_auc_score(targets_np, probs_np)
            except ValueError:
                metrics['auc'] = 0.0
        else:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
            metrics['auc'] = 0.0
        
        # 基础统计
        metrics['total_samples'] = len(targets_np)
        metrics['positive_samples'] = int(np.sum(targets_np))
        metrics['negative_samples'] = int(len(targets_np) - np.sum(targets_np))
        metrics['positive_rate'] = float(np.mean(targets_np))
        
        return metrics
    
    def _compute_class_statistics(self, targets: torch.Tensor) -> Dict[str, Any]:
        """计算类别统计信息"""
        targets_np = targets.numpy()
        
        stats = {
            'num_classes': targets_np.shape[1],
            'total_samples': targets_np.shape[0],
            'class_positive_counts': [],
            'class_positive_rates': [],
            'class_negative_counts': [],
            'class_negative_rates': []
        }
        
        for i in range(targets_np.shape[1]):
            class_targets = targets_np[:, i]
            positive_count = int(np.sum(class_targets))
            total_count = len(class_targets)
            positive_rate = float(np.mean(class_targets))
            
            stats['class_positive_counts'].append(positive_count)
            stats['class_positive_rates'].append(positive_rate)
            stats['class_negative_counts'].append(total_count - positive_count)
            stats['class_negative_rates'].append(1.0 - positive_rate)
        
        return stats
    
    def _find_best_threshold(self, results: Dict[str, Any]) -> float:
        """找到最佳阈值"""
        best_threshold = 0.5
        best_f1 = 0.0
        
        for key, metrics in results.items():
            if key.startswith('threshold_') and 'f1' in metrics:
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_threshold = metrics['threshold']
        
        return best_threshold
