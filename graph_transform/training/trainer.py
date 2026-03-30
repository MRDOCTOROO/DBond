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
import os
import json
import gc
from typing import Dict, Any, Optional
from tqdm import tqdm

from .metrics import BinaryBondMetrics, MetricTracker, order_binary_bond_metric_dict


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
        if self.device.type != 'cuda':
            self.use_amp = False
        self.amp_device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        # 混合精度训练
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=True)
        
        # 指标跟踪
        self.metrics_calculator = BinaryBondMetrics(self.config.get('evaluation', {}))
        self.metric_tracker = MetricTracker()
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # 将模型移动到设备
        self.model.to(self.device)
        self.debug_config = self.config.get('debug', {})
        self.profile_time = self.debug_config.get('profile_time', False)
        self.profile_memory = self.debug_config.get('profile_memory', False)
        self.log_grad_norm = self.debug_config.get('log_grad_norm', True)
        self.save_nonfinite_batch = self.debug_config.get('save_nonfinite_batch', True)
        self.force_gc_on_epoch_end = self.debug_config.get('force_gc_on_epoch_end', False)
        self.empty_cache_on_epoch_end = self.debug_config.get('empty_cache_on_epoch_end', False)
        self.diagnostic_dir = self.debug_config.get(
            'diagnostic_dir',
            os.path.join(self.config.get('logging', {}).get('log_dir', 'logs/graph_transform'), 'diagnostics'),
        )
        os.makedirs(self.diagnostic_dir, exist_ok=True)
        self.last_grad_norm = 0.0
        if hasattr(self.model, 'enable_timing'):
            self.model.enable_timing = self.profile_time
        if self.profile_time:
            self.logger.info("Detailed timing profiling is enabled; CUDA synchronization will add overhead.")
    
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
        
        total_weighted_loss = 0.0
        total_valid_bonds = 0
        total_dbond_style_loss = 0.0
        total_samples = 0
        num_batches = 0
        total_fetch_wait_time = 0.0
        total_move_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_batch_time = 0.0
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        last_batch_end = time.perf_counter()
        memory_epoch_start = self._get_memory_stats(reset_peak=True)
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch_data in enumerate(pbar):
            batch_start = time.perf_counter()
            fetch_wait_time = batch_start - last_batch_end
            total_fetch_wait_time += fetch_wait_time

            # 将数据移动到设备
            self._maybe_sync_device()
            move_start = time.perf_counter()
            batch_data = self._move_to_device(batch_data)
            self._maybe_sync_device()
            move_time = time.perf_counter() - move_start
            
            # 前向传播
            forward_start = time.perf_counter()
            loss, loss_stats = self._forward_pass(batch_data)
            self._maybe_sync_device()
            forward_time = time.perf_counter() - forward_start
            
            # 反向传播
            backward_start = time.perf_counter()
            self._backward_pass(loss)
            self._maybe_sync_device()
            backward_time = time.perf_counter() - backward_start
            batch_time = time.perf_counter() - batch_start
            last_batch_end = time.perf_counter()
            
            # 更新指标
            valid_bond_count = loss_stats['valid_bond_count']
            sample_count = loss_stats['sample_count']
            total_weighted_loss += loss.item() * valid_bond_count
            total_valid_bonds += valid_bond_count
            total_dbond_style_loss += loss_stats['dbond_style_loss'] * sample_count
            total_samples += sample_count
            num_batches += 1
            total_move_time += move_time
            total_forward_time += forward_time
            total_backward_time += backward_time
            total_batch_time += batch_time
            total_grad_norm += self.last_grad_norm
            max_grad_norm = max(max_grad_norm, self.last_grad_norm)
            
            # 更新进度条
            avg_loss = total_weighted_loss / total_valid_bonds if total_valid_bonds > 0 else 0.0
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'Batch s': f'{batch_time:.3f}'
            })
            
            # 记录日志
            if batch_idx % self.training_config.get('log_interval', 10) == 0:
                if self.profile_time:
                    batch_stats = self._extract_batch_stats(batch_data)
                    self.logger.info(
                        "Epoch %s Batch %s Timing - fetch_wait: %.4fs, move: %.4fs, forward: %.4fs, backward+opt: %.4fs, total: %.4fs, loss: %.4f, batch_stats: %s",
                        epoch,
                        batch_idx,
                        fetch_wait_time,
                        move_time,
                        forward_time,
                        backward_time,
                        batch_time,
                        loss.item(),
                        batch_stats,
                    )
                    if self.log_grad_norm:
                        self.logger.info(
                            "Epoch %s Batch %s GradNorm - total_grad_norm: %.4f",
                            epoch,
                            batch_idx,
                            self.last_grad_norm,
                        )
                    model_timing = getattr(self.model, 'last_forward_timing', {})
                    if model_timing:
                        self.logger.info(
                            "Epoch %s Batch %s ModelTiming - %s",
                            epoch,
                            batch_idx,
                            self._format_timing_dict(model_timing),
                        )
                else:
                    self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

                if self.profile_memory:
                    memory_stats = self._get_memory_stats()
                    if memory_stats is not None:
                        self.logger.info(
                            "Epoch %s Batch %s GPU Memory - allocated: %.2fMB, reserved: %.2fMB, peak_allocated: %.2fMB, peak_reserved: %.2fMB, free: %.2fMB, total: %.2fMB",
                            epoch,
                            batch_idx,
                            memory_stats['allocated_mb'],
                            memory_stats['reserved_mb'],
                            memory_stats['peak_allocated_mb'],
                            memory_stats['peak_reserved_mb'],
                            memory_stats['free_mb'],
                            memory_stats['total_mb'],
                        )

            del loss
            del batch_data
        
        # 计算平均损失
        avg_loss = total_weighted_loss / total_valid_bonds if total_valid_bonds > 0 else 0.0
        
        # 计算指标
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss
        metrics['dbond_style_loss'] = total_dbond_style_loss / total_samples if total_samples > 0 else 0.0
        if num_batches > 0:
            metrics['avg_fetch_wait_time'] = total_fetch_wait_time / num_batches
            metrics['avg_move_time'] = total_move_time / num_batches
            metrics['avg_forward_time'] = total_forward_time / num_batches
            metrics['avg_backward_time'] = total_backward_time / num_batches
            metrics['avg_batch_time'] = total_batch_time / num_batches
            metrics['avg_grad_norm'] = total_grad_norm / num_batches
            metrics['max_grad_norm'] = max_grad_norm
        memory_epoch_end = self._get_memory_stats()
        if memory_epoch_start is not None and memory_epoch_end is not None:
            metrics['gpu_mem_start_allocated_mb'] = memory_epoch_start['allocated_mb']
            metrics['gpu_mem_end_allocated_mb'] = memory_epoch_end['allocated_mb']
            metrics['gpu_mem_end_reserved_mb'] = memory_epoch_end['reserved_mb']
            metrics['gpu_mem_peak_allocated_mb'] = memory_epoch_end['peak_allocated_mb']
            metrics['gpu_mem_peak_reserved_mb'] = memory_epoch_end['peak_reserved_mb']
            metrics['gpu_mem_end_free_mb'] = memory_epoch_end['free_mb']
            metrics['gpu_mem_total_mb'] = memory_epoch_end['total_mb']
        metrics = order_binary_bond_metric_dict(metrics)
        
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
        self.logger.info(
            "Epoch %s Training - Loss: %.4f, AvgBatch: %.4fs, FetchWait: %.4fs, Move: %.4fs, Forward: %.4fs, Backward: %.4fs",
            epoch,
            avg_loss,
            metrics.get('avg_batch_time', 0.0),
            metrics.get('avg_fetch_wait_time', 0.0),
            metrics.get('avg_move_time', 0.0),
            metrics.get('avg_forward_time', 0.0),
            metrics.get('avg_backward_time', 0.0),
        )
        if self.log_grad_norm:
            self.logger.info(
                "Epoch %s Gradient - AvgGradNorm: %.4f, MaxGradNorm: %.4f",
                epoch,
                metrics.get('avg_grad_norm', 0.0),
                metrics.get('max_grad_norm', 0.0),
            )
        if memory_epoch_start is not None and memory_epoch_end is not None:
            self.logger.info(
                "Epoch %s GPU Memory - start_allocated: %.2fMB, end_allocated: %.2fMB, end_reserved: %.2fMB, peak_allocated: %.2fMB, peak_reserved: %.2fMB, free: %.2fMB",
                epoch,
                memory_epoch_start['allocated_mb'],
                memory_epoch_end['allocated_mb'],
                memory_epoch_end['reserved_mb'],
                memory_epoch_end['peak_allocated_mb'],
                memory_epoch_end['peak_reserved_mb'],
                memory_epoch_end['free_mb'],
            )
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'loss'])
            self.logger.info(f"Epoch {epoch} Training Metrics - {metric_str}")

        if self.force_gc_on_epoch_end:
            gc.collect()
        if self.empty_cache_on_epoch_end and self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        memory_eval_start = self._get_memory_stats(reset_peak=True)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
            
            for batch_idx, batch_data in enumerate(pbar):
                # 将数据移动到设备
                batch_data = self._move_to_device(batch_data)
                
                # 前向传播
                predictions_full = self.model(batch_data)
                targets_full = batch_data['labels']
                targets, predictions = self._apply_label_mask(batch_data, targets_full, predictions_full)

                # 计算损失
                loss = self.criterion(predictions, targets)
                
                # 更新指标
                self.metrics_calculator.update(
                    predictions_full,
                    targets_full,
                    label_mask=batch_data.get('label_mask'),
                )
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / num_batches:.4f}'
                })
                del loss
                del predictions
                del targets
                del predictions_full
                del targets_full
                del batch_data
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 计算指标
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss
        memory_eval_end = self._get_memory_stats()
        if memory_eval_start is not None and memory_eval_end is not None:
            metrics['gpu_mem_start_allocated_mb'] = memory_eval_start['allocated_mb']
            metrics['gpu_mem_end_allocated_mb'] = memory_eval_end['allocated_mb']
            metrics['gpu_mem_end_reserved_mb'] = memory_eval_end['reserved_mb']
            metrics['gpu_mem_peak_allocated_mb'] = memory_eval_end['peak_allocated_mb']
            metrics['gpu_mem_peak_reserved_mb'] = memory_eval_end['peak_reserved_mb']
        metrics = order_binary_bond_metric_dict(metrics)
        
        # 更新指标跟踪器
        self.metric_tracker.update(epoch, metrics, mode='val')
        
        # 记录验证结果
        self.logger.info(f"Epoch {epoch} Validation - Loss: {avg_loss:.4f}")
        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'loss'])
            self.logger.info(f"Epoch {epoch} Validation Metrics - {metric_str}")
        
        return metrics
    
    def _forward_pass(self, batch_data: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        """前向传播"""
        if self.use_amp:
            with torch.amp.autocast(self.amp_device_type, enabled=True):
                predictions_full = self.model(batch_data)
                targets_full = batch_data['labels']
                targets, predictions = self._apply_label_mask(batch_data, targets_full, predictions_full)
                self._ensure_finite_tensor(predictions, "predictions", batch_data)
                loss = self.criterion(predictions, targets)
        else:
            predictions_full = self.model(batch_data)
            targets_full = batch_data['labels']
            targets, predictions = self._apply_label_mask(batch_data, targets_full, predictions_full)
            self._ensure_finite_tensor(predictions, "predictions", batch_data)
            loss = self.criterion(predictions, targets)

        self._ensure_finite_tensor(loss, "loss", batch_data)
        with torch.no_grad():
            dbond_style_loss = self._compute_dbond_style_loss(batch_data, predictions_full, targets_full)
        self._ensure_finite_tensor(dbond_style_loss, "dbond_style_loss", batch_data)
        
        # 更新训练指标
        self.metrics_calculator.update(
            predictions_full,
            targets_full,
            label_mask=batch_data.get('label_mask'),
        )
        
        return loss, {
            'valid_bond_count': int(targets.numel()),
            'sample_count': int(targets_full.shape[0]),
            'dbond_style_loss': float(dbond_style_loss.item()),
        }
    
    def _backward_pass(self, loss: torch.Tensor):
        """反向传播"""
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            self.last_grad_norm = self._compute_grad_norm()
            
            # 梯度裁剪
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.last_grad_norm = self._compute_grad_norm()
            
            # 梯度裁剪
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()

    def _maybe_sync_device(self):
        """仅在 profiling 时同步 CUDA，保证计时可信。"""
        if self.profile_time and self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
    
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

    def _compute_dbond_style_loss(self,
                                  batch_data: Dict[str, Any],
                                  predictions_full: torch.Tensor,
                                  targets_full: torch.Tensor) -> torch.Tensor:
        """按 dbond_m 风格在固定宽度标签上计算 loss，仅用于报表对比。"""
        label_mask = batch_data.get('label_mask')
        if label_mask is None:
            return self.criterion(predictions_full, targets_full)

        invalid_mask = ~label_mask.bool()
        padded_predictions = predictions_full.masked_fill(invalid_mask, -1e9)
        return self.criterion(padded_predictions, targets_full)

    def _extract_batch_stats(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取当前 batch 的基础统计，方便排查数据和吞吐问题。"""
        stats = {}
        seq_lens = batch_data.get('seq_lens')
        if isinstance(seq_lens, torch.Tensor) and seq_lens.numel() > 0:
            stats['batch_size'] = int(seq_lens.numel())
            stats['seq_min'] = int(seq_lens.min().item())
            stats['seq_max'] = int(seq_lens.max().item())
            stats['seq_mean'] = round(float(seq_lens.float().mean().item()), 2)
        edge_index = batch_data.get('edge_index')
        if isinstance(edge_index, torch.Tensor):
            stats['edges'] = int(edge_index.size(1))
        label_mask = batch_data.get('label_mask')
        if isinstance(label_mask, torch.Tensor):
            stats['valid_bonds'] = int(label_mask.sum().item())
        return stats

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

    def _format_timing_dict(self, timing: Dict[str, Any]) -> str:
        """将模型内部 timing dict 格式化为紧凑字符串。"""
        parts = []
        count_keys = {'batch_size', 'total_nodes', 'total_edges', 'num_nodes', 'num_edges'}
        for key, value in timing.items():
            if isinstance(value, float):
                if key in count_keys:
                    parts.append(f"{key}={value:.0f}")
                else:
                    parts.append(f"{key}={value:.4f}s")
            else:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _compute_grad_norm(self) -> float:
        """计算当前参数梯度的 L2 范数。"""
        total = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                return float('inf')
            grad_norm = grad.norm(2)
            total += float(grad_norm.item()) ** 2
        return total ** 0.5

    def _ensure_finite_tensor(self, tensor: torch.Tensor, name: str, batch_data: Dict[str, Any]) -> None:
        """检测非有限值并尽早失败，避免后续整轮训练都被 NaN 污染。"""
        if torch.isfinite(tensor).all():
            return

        batch_stats = self._extract_batch_stats(batch_data)
        finite_mask = torch.isfinite(tensor)
        finite_values = tensor[finite_mask]
        detail = {
            'shape': tuple(tensor.shape),
            'batch_stats': batch_stats,
            'model_timing': getattr(self.model, 'last_forward_timing', {}),
            'gpu_memory': self._get_memory_stats(),
        }
        if finite_values.numel() > 0:
            detail['finite_min'] = float(finite_values.min().detach().cpu().item())
            detail['finite_max'] = float(finite_values.max().detach().cpu().item())
        detail['gradient_norm'] = self.last_grad_norm
        detail['tensor_stats'] = self._collect_tensor_stats(batch_data)
        diagnostic_path = None
        if self.save_nonfinite_batch:
            diagnostic_path = self._write_nonfinite_diagnostic(name, detail)
            detail['diagnostic_path'] = diagnostic_path
        self.logger.error("Non-finite %s detected: %s", name, detail)
        raise FloatingPointError(f"Non-finite {name} detected; see training log for batch statistics.")

    def _collect_tensor_stats(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集 batch 主要张量的统计信息，便于远程排查。"""
        stats = {}
        keys_of_interest = {
            'labels',
            'label_mask',
            'seq_lens',
            'edge_index',
            'edge_attr',
            'edge_types',
            'edge_distances',
            'state_vars',
            'env_vars',
            'charges',
            'pep_masses',
            'intensities',
            'nces',
            'rts',
        }
        for key, value in batch_data.items():
            if key not in keys_of_interest or not isinstance(value, torch.Tensor):
                continue
            detached = value.detach()
            entry = {
                'shape': list(detached.shape),
                'dtype': str(detached.dtype),
                'device': str(detached.device),
                'numel': int(detached.numel()),
                'finite': bool(torch.isfinite(detached).all().item()) if detached.is_floating_point() else True,
            }
            if detached.numel() > 0 and detached.is_floating_point():
                finite = detached[torch.isfinite(detached)]
                if finite.numel() > 0:
                    entry.update({
                        'min': float(finite.min().cpu().item()),
                        'max': float(finite.max().cpu().item()),
                        'mean': float(finite.mean().cpu().item()),
                    })
            elif detached.numel() > 0:
                entry.update({
                    'min': float(detached.min().cpu().item()),
                    'max': float(detached.max().cpu().item()),
                })
            stats[key] = entry
        return stats

    def _write_nonfinite_diagnostic(self, name: str, detail: Dict[str, Any]) -> str:
        """将非有限 batch 的统计信息写到诊断文件。"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"nonfinite_{name}_epoch{self.current_epoch}_{timestamp}.json"
        filepath = os.path.join(self.diagnostic_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=True, indent=2)
        return filepath
    
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
