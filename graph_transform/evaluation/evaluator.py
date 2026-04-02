"""
评估器模块

本模块包含图神经网络的评估器实现。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import gc
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np

from .metrics import BinaryBondMetrics, order_binary_bond_metric_dict, _sigmoid_if_needed

try:
    from training import BinaryBondLoss
except ImportError:
    from graph_transform.training import BinaryBondLoss


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
        self.debug_config = self.config.get('debug', {})
        self.profile_memory = self.debug_config.get('profile_memory', False)
        self.force_gc_on_eval_end = self.debug_config.get('force_gc_on_eval_end', False)
        
        # 指标计算器
        self.metrics_calculator = BinaryBondMetrics(self.eval_config)
        
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
        
        total_weighted_loss = 0.0
        total_valid_bonds = 0
        total_dbond_style_loss = 0.0
        total_samples = 0
        num_batches = 0
        memory_eval_start = self._get_memory_stats(reset_peak=True)
        
        # 损失函数（与训练保持一致）
        loss_config = self.config.get('loss', {})
        criterion = BinaryBondLoss(loss_config).to(self.device)
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating")
            
            for batch_idx, batch_data in enumerate(pbar):
                # 将数据移动到设备
                batch_data = self._move_to_device(batch_data)
                
                # 前向传播
                if self.use_amp:
                    with torch.amp.autocast(self.amp_device_type, enabled=True):
                        predictions_full = self.model(batch_data)
                        targets_full = batch_data['labels']
                        targets, predictions = self._apply_label_mask(batch_data, targets_full, predictions_full)
                        self._ensure_finite_tensor(predictions, "predictions")
                        loss = criterion(predictions, targets)
                else:
                    predictions_full = self.model(batch_data)
                    targets_full = batch_data['labels']
                    targets, predictions = self._apply_label_mask(batch_data, targets_full, predictions_full)
                    self._ensure_finite_tensor(predictions, "predictions")
                    loss = criterion(predictions, targets)
                self._ensure_finite_tensor(loss, "loss")
                dbond_style_loss = self._compute_dbond_style_loss(criterion, batch_data, predictions_full, targets_full)
                self._ensure_finite_tensor(dbond_style_loss, "dbond_style_loss")
                
                # 更新指标
                self.metrics_calculator.update(
                    predictions_full,
                    targets_full,
                    label_mask=batch_data.get('label_mask'),
                )
                valid_bond_count = int(targets.numel())
                sample_count = int(targets_full.shape[0])
                total_weighted_loss += loss.item() * valid_bond_count
                total_valid_bonds += valid_bond_count
                total_dbond_style_loss += dbond_style_loss.item() * sample_count
                total_samples += sample_count
                num_batches += 1
                
                # 更新进度条
                avg_loss = total_weighted_loss / total_valid_bonds if total_valid_bonds > 0 else 0.0
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{avg_loss:.4f}'
                })
                if self.profile_memory and batch_idx % self.config.get('logging', {}).get('log_interval', 10) == 0:
                    memory_stats = self._get_memory_stats()
                    if memory_stats is not None:
                        self.logger.info(
                            "Eval Batch %s GPU Memory - allocated: %.2fMB, reserved: %.2fMB, peak_allocated: %.2fMB, peak_reserved: %.2fMB, free: %.2fMB, total: %.2fMB",
                            batch_idx,
                            memory_stats['allocated_mb'],
                            memory_stats['reserved_mb'],
                            memory_stats['peak_allocated_mb'],
                            memory_stats['peak_reserved_mb'],
                            memory_stats['free_mb'],
                            memory_stats['total_mb'],
                        )
                del loss
                del predictions
                del targets
                del predictions_full
                del targets_full
                del batch_data
        
        # 计算平均损失
        avg_loss = total_weighted_loss / total_valid_bonds if total_valid_bonds > 0 else 0.0
        
        # 计算指标
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss
        metrics['dbond_style_loss'] = total_dbond_style_loss / total_samples if total_samples > 0 else 0.0
        memory_eval_end = self._get_memory_stats()
        if memory_eval_start is not None and memory_eval_end is not None:
            metrics['gpu_mem_start_allocated_mb'] = memory_eval_start['allocated_mb']
            metrics['gpu_mem_end_allocated_mb'] = memory_eval_end['allocated_mb']
            metrics['gpu_mem_end_reserved_mb'] = memory_eval_end['reserved_mb']
            metrics['gpu_mem_peak_allocated_mb'] = memory_eval_end['peak_allocated_mb']
            metrics['gpu_mem_peak_reserved_mb'] = memory_eval_end['peak_reserved_mb']
        metrics = order_binary_bond_metric_dict(metrics)
        
        # 记录结果
        self.logger.info(f"Evaluation - Loss: {avg_loss:.4f}")
        if memory_eval_start is not None and memory_eval_end is not None:
            self.logger.info(
                "Evaluation GPU Memory - start_allocated: %.2fMB, end_allocated: %.2fMB, end_reserved: %.2fMB, peak_allocated: %.2fMB, peak_reserved: %.2fMB, free: %.2fMB",
                memory_eval_start['allocated_mb'],
                memory_eval_end['allocated_mb'],
                memory_eval_end['reserved_mb'],
                memory_eval_end['peak_allocated_mb'],
                memory_eval_end['peak_reserved_mb'],
                memory_eval_end['free_mb'],
            )
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'loss'])
            self.logger.info(f"Evaluation Metrics - {metric_str}")
        if self.force_gc_on_eval_end:
            gc.collect()
        
        return metrics

    def collect_prediction_outputs(self, data_loader: DataLoader, threshold: Optional[float] = None) -> Dict[str, Any]:
        """收集逐样本评估输出，便于保存预测结果文件。"""
        self.model.eval()

        sample_logits = []
        sample_targets = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Collecting evaluation outputs"):
                batch_data = self._move_to_device(batch_data)

                if self.use_amp:
                    with torch.amp.autocast(self.amp_device_type, enabled=True):
                        logits = self.model(batch_data)
                else:
                    logits = self.model(batch_data)

                labels = batch_data['labels']
                seq_lens = batch_data['seq_lens']

                for row_idx, seq_len in enumerate(seq_lens.tolist()):
                    bond_len = max(int(seq_len) - 1, 0)
                    if bond_len == 0:
                        sample_logits.append(np.array([], dtype=np.float32))
                        sample_targets.append(np.array([], dtype=np.int32))
                        continue

                    sample_logits.append(
                        logits[row_idx, :bond_len].detach().float().cpu().numpy().astype(np.float32)
                    )
                    sample_targets.append(
                        labels[row_idx, :bond_len].detach().cpu().numpy().astype(np.int32)
                    )

        if threshold is None:
            valid_logits = [row for row in sample_logits if row.size > 0]
            valid_targets = [row for row in sample_targets if row.size > 0]
            if valid_logits and valid_targets:
                flat_logits = np.concatenate(valid_logits, axis=0).astype(np.float32)
                flat_targets = np.concatenate(valid_targets, axis=0).astype(np.int32)
                flat_probabilities = _sigmoid_if_needed(flat_logits)
                threshold = BinaryBondMetrics(self.eval_config)._get_threshold(flat_probabilities, flat_targets)
            else:
                threshold = float(self.eval_config.get('threshold', 0.5))

        pred_strings = []
        true_strings = []
        prob_strings = []

        for logit_row, target_row in zip(sample_logits, sample_targets):
            if logit_row.size == 0:
                pred_strings.append("")
                true_strings.append("")
                prob_strings.append("")
                continue

            prob_row = _sigmoid_if_needed(logit_row.astype(np.float32))
            pred_row = (prob_row > threshold).astype(np.int32)
            pred_strings.append(";".join(map(str, pred_row.tolist())))
            true_strings.append(";".join(map(str, target_row.astype(np.int32).tolist())))
            prob_strings.append(";".join(f"{value:.6f}" for value in prob_row.tolist()))

        return {
            'threshold': float(threshold),
            'pred_strings': pred_strings,
            'true_strings': true_strings,
            'prob_strings': prob_strings,
        }
    
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
            binary_predictions = (all_predictions > threshold).float()
            
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
        binary_predictions = (all_predictions > threshold).float()
        
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
                    with torch.amp.autocast(self.amp_device_type, enabled=True):
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

    def _compute_dbond_style_loss(self,
                                  criterion: nn.Module,
                                  batch_data: Dict[str, Any],
                                  predictions_full: torch.Tensor,
                                  targets_full: torch.Tensor) -> torch.Tensor:
        """按 dbond_m 风格在固定宽度标签上计算 loss，仅用于报表对比。"""
        label_mask = batch_data.get('label_mask')
        if label_mask is None:
            return criterion(predictions_full, targets_full)

        invalid_mask = ~label_mask.bool()
        padded_predictions = predictions_full.masked_fill(invalid_mask, -1e9)
        return criterion(padded_predictions, targets_full)
    
    def _compute_metrics_threshold(self, predictions: torch.Tensor, 
                                 targets: torch.Tensor,
                                 binary_predictions: torch.Tensor,
                                 threshold: float) -> Dict[str, float]:
        """计算特定阈值下的指标"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 转换为numpy
        preds_np = binary_predictions.numpy().reshape(-1)
        targets_np = targets.numpy().reshape(-1)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(targets_np, preds_np),
            'precision': precision_score(targets_np, preds_np, zero_division=0),
            'recall': recall_score(targets_np, preds_np, zero_division=0),
            'f1': f1_score(targets_np, preds_np, zero_division=0),
        }
        
        return metrics

    def _ensure_finite_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """评估阶段遇到 NaN/Inf 直接失败，避免输出无意义指标。"""
        if torch.isfinite(tensor).all():
            return
        raise FloatingPointError(f"Non-finite {name} detected during evaluation.")

    def _get_memory_stats(self, reset_peak: bool = False) -> Optional[Dict[str, float]]:
        """返回当前 CUDA 显存统计。"""
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return None
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / (1024 ** 2),
            'reserved_mb': torch.cuda.memory_reserved(self.device) / (1024 ** 2),
            'peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            'peak_reserved_mb': torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
            'free_mb': free_bytes / (1024 ** 2),
            'total_mb': total_bytes / (1024 ** 2),
        }
    
    def _compute_single_class_metrics(self, targets: torch.Tensor,
                                    predictions: torch.Tensor,
                                    probabilities: torch.Tensor,
                                    class_idx: int) -> Dict[str, float]:
        """计算单个类别的指标"""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # 转换为numpy
        targets_np = targets.numpy().reshape(-1)
        preds_np = predictions.numpy().reshape(-1)
        probs_np = probabilities.numpy().reshape(-1)
        
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
            'num_classes': 1,
            'total_samples': targets_np.shape[0],
            'class_positive_counts': [],
            'class_positive_rates': [],
            'class_negative_counts': [],
            'class_negative_rates': []
        }
        
        positive_count = int(np.sum(targets_np))
        total_count = len(targets_np)
        positive_rate = float(np.mean(targets_np)) if total_count > 0 else 0.0

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
