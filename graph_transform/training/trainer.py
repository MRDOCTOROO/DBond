"""
训练器模块

本模块包含图神经网络的训练器实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

from .metrics import MultiLabelMetrics, MetricTracker


class Trainer:
    """图神经网络训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = torch.device('cpu'),
                 config: Dict[str, Any] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化训练器
        
        Args:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器
            device: 计算设备
            config: 配置
            logger: 日志记录器
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 训练配置
        self.training_config = self.config.get('training', {})
        self.gradient_clip_norm = self.training_config.get('gradient_clip_norm', 1.0)
        self.use_amp = self.config.get('device', {}).get('use_amp', False)
        
        # 混合精度训练
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 指标跟踪
        self.metrics_calculator = MultiLabelMetrics(self.config.get('evaluation', {}))
        self.metric_tracker = MetricTracker()
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # 将模型移动到设备
        self.model.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        self.current_epoch = epoch
        
        # 重置指标
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch_data in enumerate(pbar):
            # 将数据移动到设备
            batch_data = self._move_to_device(batch_data)
            
            # 前向传播
            loss = self._forward_pass(batch_data)
            
            # 反向传播
            self._backward_pass(loss)
            
            # 更新指标
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / num_batches:.4f}'
            })
            
            # 记录日志
            if batch_idx % self.training_config.get('log_interval', 10) == 0:
                self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 计算指标
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss
        
        # 更新指标跟踪器
        self.metric_tracker.update(epoch, metrics, mode='train')
        
        # 更新学习率
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)
                else:
                    self.scheduler.step()
        
        # 记录epoch总结
        self.logger.info(f"Epoch {epoch} Training - Loss: {avg_loss:.4f}")
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'loss'])
            self.logger.info(f"Epoch {epoch} Training Metrics - {metric_str}")
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            Dict[str, float]: 验证指标
        """
        self.model.eval()
        
        # 重置指标
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
            
            for batch_idx, batch_data in enumerate(pbar):
                # 将数据移动到设备
                batch_data = self._move_to_device(batch_data)
                
                # 前向传播
                predictions = self.model(batch_data)
                targets = batch_data['labels']
                
                # 计算损失
                loss = self.criterion(predictions, targets)
                
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
        
        # 更新指标跟踪器
        self.metric_tracker.update(epoch, metrics, mode='val')
        
        # 记录验证结果
        self.logger.info(f"Epoch {epoch} Validation - Loss: {avg_loss:.4f}")
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'loss'])
            self.logger.info(f"Epoch {epoch} Validation Metrics - {metric_str}")
        
        return metrics
    
    def _forward_pass(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(batch_data)
                targets = batch_data['labels']
                targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
                loss = self.criterion(predictions, targets)
        else:
            predictions = self.model(batch_data)
            targets = batch_data['labels']
            targets, predictions = self._apply_label_mask(batch_data, targets, predictions)
            loss = self.criterion(predictions, targets)
        
        # 更新训练指标
        self.metrics_calculator.update(predictions, targets)
        
        return loss
    
    def _backward_pass(self, loss: torch.Tensor):
        """反向传播"""
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
    
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
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        保存检查点
        
        Args:
            filepath: 文件路径
            epoch: 当前epoch
            metrics: 指标
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
        
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        Args:
            filepath: 检查点文件路径
            
        Returns:
            Dict: 检查点信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        
        self.logger.info(f"Loaded checkpoint from {filepath}, epoch {self.current_epoch}")
        
        return checkpoint
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'model_type': type(self.model).__name__,
            'optimizer_type': type(self.optimizer).__name__
        }
        
        if self.scheduler is not None:
            summary['scheduler_type'] = type(self.scheduler).__name__
        
        return summary
    
    def print_training_summary(self):
        """打印训练摘要"""
        self.metric_tracker.print_summary()
        
        summary = self.get_model_summary()
        print("\n=== Model Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
