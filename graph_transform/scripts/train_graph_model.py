#!/usr/bin/env python3
"""
图神经网络训练脚本

本脚本用于训练图神经网络键级别二分类模型。
支持配置文件、命令行参数和完整的训练流程。
"""

import os
import sys
import argparse
import re
import yaml
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from torch.nn.parameter import UninitializedParameter
from models.utils import ModelConfig, CheckpointManager, LearningRateScheduler
from data import GraphDataset, GraphDataLoader, CachedGraphDataset
from training import Trainer, BinaryBondLoss
from evaluation import Evaluator
from evaluation.metrics import metric_rows, metric_display_name


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """设置日志"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志目录
    log_dir = log_config.get('log_dir', 'logs/graph_transform')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置logger
    logger = logging.getLogger('graph_transform')
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # 文件处理器
    log_file = os.path.join(log_dir, log_config.get('log_file', 'training.log'))
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    return logger


def resolve_plot_dir(checkpoint_dir: str, log_config: Dict[str, Any]) -> str:
    """解析训练曲线输出目录。"""
    plot_dir = log_config.get('plot_dir')
    if not plot_dir:
        plot_dir = os.path.join(checkpoint_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def resolve_tensorboard_dir(checkpoint_dir: str, log_config: Dict[str, Any]) -> str:
    """解析TensorBoard日志目录，按运行目录隔离。"""
    base_dir = log_config.get('tensorboard_log_dir') or os.path.join(checkpoint_dir, 'tensorboard')
    tb_dir = os.path.join(base_dir, os.path.basename(checkpoint_dir))
    os.makedirs(tb_dir, exist_ok=True)
    return tb_dir


def create_tensorboard_writers(
    checkpoint_dir: str,
    log_config: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, 'SummaryWriter']:
    """仿照 dbond_m，为 train/val/test 分别创建 TensorBoard writer。"""
    if SummaryWriter is None:
        logger.warning("TensorBoard is enabled in config but tensorboard is not installed; event logging is disabled.")
        return {}

    tb_root = resolve_tensorboard_dir(checkpoint_dir, log_config)
    config_yaml = yaml.safe_dump(config, sort_keys=False)
    writers: Dict[str, SummaryWriter] = {}
    for mode in ('train', 'val', 'test'):
        mode_dir = os.path.join(tb_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        writers[mode] = SummaryWriter(log_dir=mode_dir)
        writers[mode].add_text("config/yaml", config_yaml, 0)
    logger.info(f"TensorBoard root dir: {tb_root}")
    return writers


def flush_tensorboard_writers(writers: Dict[str, 'SummaryWriter']) -> None:
    for writer in writers.values():
        writer.flush()


def close_tensorboard_writers(writers: Dict[str, 'SummaryWriter']) -> None:
    for writer in writers.values():
        writer.close()


def sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return sanitized or "unknown"


def build_evaluation_id(checkpoint_reference: str, phase: str) -> str:
    checkpoint_stem = sanitize_filename_component(os.path.splitext(os.path.basename(checkpoint_reference))[0])
    checkpoint_parent = sanitize_filename_component(os.path.basename(os.path.dirname(checkpoint_reference)))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{checkpoint_parent}_{checkpoint_stem}_{phase}"


def append_eval_id_to_path(filepath: str, evaluation_id: str) -> str:
    directory, filename = os.path.split(filepath)
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}__{evaluation_id}{ext}")


def save_evaluation_outputs(
    *,
    metrics: Dict[str, Any],
    output_df: pd.DataFrame,
    metric_csv_path: str,
    pred_csv_path: str,
    evaluation_id: str,
    logger: logging.Logger,
) -> None:
    metric_dir = os.path.dirname(metric_csv_path)
    pred_dir = os.path.dirname(pred_csv_path)
    if metric_dir:
        os.makedirs(metric_dir, exist_ok=True)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)

    metric_df = pd.DataFrame(metric_rows(metrics))
    metric_df.to_csv(metric_csv_path, index=False)
    output_df.to_csv(pred_csv_path, index=False)

    archive_metric_path = append_eval_id_to_path(metric_csv_path, evaluation_id)
    archive_pred_path = append_eval_id_to_path(pred_csv_path, evaluation_id)
    metric_df.to_csv(archive_metric_path, index=False)
    output_df.to_csv(archive_pred_path, index=False)

    logger.info(f"Saved latest metrics to {metric_csv_path}")
    logger.info(f"Saved latest predictions to {pred_csv_path}")
    logger.info(f"Archived metrics to {archive_metric_path}")
    logger.info(f"Archived predictions to {archive_pred_path}")


def _save_line_plot(
    epochs: list,
    series: Dict[str, list],
    title: str,
    ylabel: str,
    save_path: str,
) -> None:
    """保存折线图，自动跳过None值。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    has_data = False
    for label, values in series.items():
        valid_pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if not valid_pairs:
            continue
        has_data = True
        plt.plot(
            [e for e, _ in valid_pairs],
            [v for _, v in valid_pairs],
            label=label,
            linewidth=2,
        )

    if not has_data:
        plt.close()
        return

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_training_curves(
    history: Dict[str, list],
    config: Dict[str, Any],
    logger: logging.Logger,
    checkpoint_dir: str,
) -> None:
    """保存训练曲线与历史表。"""
    log_config = config.get('logging', {})
    if not log_config.get('save_training_curves', log_config.get('save_loss_curves', True)):
        return

    plot_dir = resolve_plot_dir(checkpoint_dir, log_config)

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(plot_dir, 'training_history.csv')
    history_df.to_csv(history_csv, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Could not save training curves because matplotlib is unavailable: {exc}")
        return

    epochs = history.get('epoch', [])
    if not epochs:
        return

    _save_line_plot(
        epochs,
        {
            'Train Loss': history.get('train_loss', []),
            'Val Loss': history.get('val_loss', []),
        },
        'GraphTransformer Loss Curve',
        'Loss',
        os.path.join(plot_dir, 'loss_curve.png'),
    )
    _save_line_plot(
        epochs,
        {
            'Train F1': history.get('train_f1', []),
            'Val F1': history.get('val_f1', []),
        },
        'GraphTransformer F1 Curve',
        'F1',
        os.path.join(plot_dir, 'f1_curve.png'),
    )
    _save_line_plot(
        epochs,
        {
            'Learning Rate': history.get('learning_rate', []),
        },
        'Learning Rate Schedule',
        'Learning Rate',
        os.path.join(plot_dir, 'learning_rate_curve.png'),
    )
    _save_line_plot(
        epochs,
        {
            'Avg Grad Norm': history.get('train_avg_grad_norm', []),
            'Max Grad Norm': history.get('train_max_grad_norm', []),
        },
        'Gradient Norm Curve',
        'Grad Norm',
        os.path.join(plot_dir, 'grad_norm_curve.png'),
    )
    _save_line_plot(
        epochs,
        {
            'Train Precision': history.get('train_precision', []),
            'Train Recall': history.get('train_recall', []),
            'Val Precision': history.get('val_precision', []),
            'Val Recall': history.get('val_recall', []),
        },
        'Precision Recall Curve',
        'Score',
        os.path.join(plot_dir, 'precision_recall_curve.png'),
    )
    _save_line_plot(
        epochs,
        {
            'Train AUC': history.get('train_auc', []),
            'Val AUC': history.get('val_auc', []),
        },
        'AUC Curve',
        'AUC',
        os.path.join(plot_dir, 'auc_curve.png'),
    )
    _save_line_plot(
        epochs,
        {
            'Avg Batch Time': history.get('train_avg_batch_time', []),
            'Avg Forward Time': history.get('train_avg_forward_time', []),
            'Avg Backward Time': history.get('train_avg_backward_time', []),
            'Avg Fetch Wait Time': history.get('train_avg_fetch_wait_time', []),
        },
        'Timing Curve',
        'Seconds',
        os.path.join(plot_dir, 'timing_curve.png'),
    )

    logger.info(f"Saved training curves to {plot_dir}")


def log_metrics_to_tensorboard(
    writers: Dict[str, 'SummaryWriter'],
    mode: str,
    metrics: Dict[str, Any],
    step: int,
) -> None:
    """写入 TensorBoard 标量；按 split 分 writer，贴近 dbond_m 视图。"""
    writer = writers.get(mode)
    if writer is None:
        return

    for key, value in metrics.items():
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(metric_display_name(key), float(value), step)


def setup_device(config: Dict[str, Any]) -> torch.device:
    """设置计算设备"""
    device_config = config.get('device', {})
    
    if device_config.get('auto_detect', True):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_id = device_config.get('gpu_id', 0)
            torch.cuda.set_device(gpu_id)
            logger.info(f"Using CUDA device: {gpu_id}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
    else:
        device_type = device_config.get('device_type', 'cpu')
        device = torch.device(device_type)
        if device_type == 'cuda':
            gpu_id = device_config.get('gpu_id', 0)
            torch.cuda.set_device(gpu_id)
    
    return device


def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['device']['device_type'] = args.device
    
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """创建模型"""
    model_config = ModelConfig(config['model'])
    
    model = GraphTransformer(model_config)
    model = model.to(device)
    
    total_params = sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, UninitializedParameter)
    )
    logger.info(f"Created model with {total_params} parameters")
    
    return model


def create_datasets(config: Dict[str, Any]) -> tuple:
    """创建数据集"""
    data_config = config['data']
    model_config = ModelConfig(config['model'])

    dataset_cls = CachedGraphDataset if data_config.get('cache_graphs', False) else GraphDataset
    
    # 训练数据集
    train_kwargs = {
        'csv_path': data_config['train_csv_path'],
        'config': model_config,
        'max_seq_len': data_config['max_seq_len'],
        'graph_strategy': data_config['graph_strategy'],
        'augmentation': data_config.get('augmentation', False),
        'split': 'train'
    }
    if dataset_cls is CachedGraphDataset:
        train_kwargs.update({
            'cache_dir': data_config.get('cache_dir', 'cache/graph_data'),
            'rebuild_cache': data_config.get('rebuild_cache', False),
            'cache_full_graphs': data_config.get('cache_full_graphs', False),
        })
    train_dataset = dataset_cls(**train_kwargs)
    
    # 验证数据集
    if data_config.get('val_csv_path'):
        val_kwargs = {
            'csv_path': data_config['val_csv_path'],
            'config': model_config,
            'max_seq_len': data_config['max_seq_len'],
            'graph_strategy': data_config['graph_strategy'],
            'augmentation': False,
            'split': 'val'
        }
        if dataset_cls is CachedGraphDataset:
            val_kwargs.update({
                'cache_dir': data_config.get('cache_dir', 'cache/graph_data'),
                'rebuild_cache': data_config.get('rebuild_cache', False),
                'cache_full_graphs': data_config.get('cache_full_graphs', False),
            })
        val_dataset = dataset_cls(**val_kwargs)
    else:
        # 从训练集分割验证集
        val_split = data_config.get('validation_split', 0.2)
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # 测试数据集
    test_dataset = None
    if data_config.get('test_csv_path'):
        test_kwargs = {
            'csv_path': data_config['test_csv_path'],
            'config': model_config,
            'max_seq_len': data_config['max_seq_len'],
            'graph_strategy': data_config['graph_strategy'],
            'augmentation': False,
            'split': 'test'
        }
        if dataset_cls is CachedGraphDataset:
            test_kwargs.update({
                'cache_dir': data_config.get('cache_dir', 'cache/graph_data'),
                'rebuild_cache': data_config.get('rebuild_cache', False),
                'cache_full_graphs': data_config.get('cache_full_graphs', False),
            })
        test_dataset = dataset_cls(**test_kwargs)
    
    # 打印数据集统计信息
    logger.info("Dataset Statistics:")
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")
    
    # 获取训练数据集的详细统计信息
    if hasattr(train_dataset, 'get_statistics'):
        stats = train_dataset.get_statistics()
        logger.info(f"Average sequence length: {stats.get('avg_seq_length', 'N/A'):.2f}")
        logger.info(f"Max sequence length: {stats.get('max_seq_length', 'N/A')}")
        if 'length_counts' in stats:
            logger.info(f"Length counts: {stats['length_counts']}")
        if 'positive_ratio' in stats:
            logger.info(f"Positive label ratio: {stats['positive_ratio']:.4f}")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, 
                        config: Dict[str, Any]) -> tuple:
    """创建数据加载器"""
    data_config = config['data']
    training_config = config['training']
    performance_config = config.get('performance', {})
    persistent_workers = performance_config.get('persistent_workers', False)
    prefetch_factor = performance_config.get('prefetch_factor')
    
    # 训练数据加载器
    train_loader = GraphDataLoader(
        dataset=train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        drop_last=data_config.get('drop_last', False),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    # 验证数据加载器
    val_loader = None
    if val_dataset is not None:
        val_loader = GraphDataLoader(
            dataset=val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    
    # 测试数据加载器
    test_loader = None
    if test_dataset is not None:
        test_loader = GraphDataLoader(
            dataset=test_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    
    logger.info(
        "DataLoader config - batch_size: %s, num_workers: %s, pin_memory: %s, drop_last: %s, persistent_workers: %s, prefetch_factor: %s",
        training_config['batch_size'],
        data_config.get('num_workers', 4),
        data_config.get('pin_memory', True),
        data_config.get('drop_last', False),
        persistent_workers if data_config.get('num_workers', 4) > 0 else False,
        prefetch_factor if data_config.get('num_workers', 4) > 0 else None,
    )

    return train_loader, val_loader, test_loader


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """创建优化器"""
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    learning_rate = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.0)
    eps = optimizer_config.get('eps', 1e-8)
    if isinstance(eps, str):
        eps = float(eps)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=eps
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=eps
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.9),
            nesterov=optimizer_config.get('nesterov', True)
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """创建学习率调度器"""
    training_config = config['training']
    optimizer_config = config['optimizer']
    
    scheduler_type = training_config.get('scheduler', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        scheduler = LearningRateScheduler.cosine_annealing_with_warmup(
            optimizer=optimizer,
            warmup_epochs=training_config.get('warmup_epochs', 10),
            max_epochs=training_config['epochs'],
            min_lr=training_config.get('min_lr', 0.0)
        )
    elif scheduler_type == 'step':
        scheduler = LearningRateScheduler.step_with_warmup(
            optimizer=optimizer,
            warmup_epochs=training_config.get('warmup_epochs', 10),
            step_size=optimizer_config.get('step_size', 30),
            gamma=optimizer_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=optimizer_config.get('gamma', 0.95)
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=optimizer_config.get('gamma', 0.5),
            patience=training_config.get('patience', 10),
            min_lr=training_config.get('min_lr', 1e-8)
        )
    else:
        scheduler = None
    
    return scheduler


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练图神经网络键级别二分类模型')
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    
    # 可选参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='计算设备')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config, args)
    
    # 设置随机种子
    seed = args.seed or config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    global logger
    logger = setup_logging(config)
    logger.info("Starting training...")
    logger.info(f"Config: {config}")
    
    # 设置设备
    device = setup_device(config)
    logger.info(f"Using device: {device}")
    
    # 创建模型
    model_start = time.perf_counter()
    model = create_model(config, device)
    logger.info(f"Model creation took {time.perf_counter() - model_start:.2f}s")
    
    # 创建数据集和数据加载器
    dataset_start = time.perf_counter()
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    logger.info(f"Dataset creation took {time.perf_counter() - dataset_start:.2f}s")

    dataloader_start = time.perf_counter()
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config
    )
    logger.info(f"DataLoader creation took {time.perf_counter() - dataloader_start:.2f}s")
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # 创建损失函数
    loss_config = config.get('loss', {})
    criterion = BinaryBondLoss(loss_config)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger
    )
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        device=device,
        config=config,
        logger=logger
    )
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        checkpoint = CheckpointManager.load_checkpoint(
            filepath=args.resume,
            model=model,
            optimizer=optimizer
        )
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['metrics'].get('f1', 0.0)
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # 训练循环
    logger.info("Starting training loop...")
    training_config = config['training']
    early_stopping_enabled = training_config.get('early_stopping', False)
    patience = training_config.get('patience', 10)
    min_delta = training_config.get('min_delta', 0.0)
    best_metric = best_metric if args.resume else float('-inf')
    best_epoch = start_epoch
    no_improve_epochs = 0
    if args.resume:
        training_config['checkpoint_dir'] = os.path.dirname(args.resume)
    else:
        base_checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints/graph_transform')
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_config['checkpoint_dir'] = os.path.join(base_checkpoint_dir, run_id)
    os.makedirs(training_config['checkpoint_dir'], exist_ok=True)
    logger.info(f"Checkpoint dir: {training_config['checkpoint_dir']}")
    log_config = config.get('logging', {})
    tb_writers: Dict[str, SummaryWriter] = {}
    if log_config.get('use_tensorboard', False):
        tb_writers = create_tensorboard_writers(training_config['checkpoint_dir'], log_config, config, logger)
    epochs = training_config['epochs']
    training_start = time.perf_counter()
    history = {
        'epoch': [],
        'train_loss': [],
        'train_f1': [],
        'train_precision': [],
        'train_recall': [],
        'train_auc': [],
        'train_avg_grad_norm': [],
        'train_max_grad_norm': [],
        'train_avg_batch_time': [],
        'train_avg_forward_time': [],
        'train_avg_backward_time': [],
        'train_avg_fetch_wait_time': [],
        'learning_rate': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': [],
    }
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        epoch_start = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]['lr']
        
        # 训练阶段
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        log_metrics_to_tensorboard(tb_writers, 'train', train_metrics, epoch + 1)
        train_writer = tb_writers.get('train')
        if train_writer is not None:
            train_writer.add_scalar('learning_rate', epoch_lr, epoch + 1)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics.get('loss'))
        history['train_f1'].append(train_metrics.get('f1'))
        history['train_precision'].append(train_metrics.get('precision_micro', train_metrics.get('precision')))
        history['train_recall'].append(train_metrics.get('recall_micro', train_metrics.get('recall')))
        history['train_auc'].append(train_metrics.get('auc_micro', train_metrics.get('auc')))
        history['train_avg_grad_norm'].append(train_metrics.get('avg_grad_norm'))
        history['train_max_grad_norm'].append(train_metrics.get('max_grad_norm'))
        history['train_avg_batch_time'].append(train_metrics.get('avg_batch_time'))
        history['train_avg_forward_time'].append(train_metrics.get('avg_forward_time'))
        history['train_avg_backward_time'].append(train_metrics.get('avg_backward_time'))
        history['train_avg_fetch_wait_time'].append(train_metrics.get('avg_fetch_wait_time'))
        history['learning_rate'].append(epoch_lr)
        current_val_loss: Optional[float] = None
        current_val_f1: Optional[float] = None
        current_val_precision: Optional[float] = None
        current_val_recall: Optional[float] = None
        current_val_auc: Optional[float] = None
        
        # 验证阶段
        if val_loader is not None and epoch % training_config.get('validation_interval', 1) == 0:
            val_metrics = evaluator.evaluate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            current_val_loss = val_metrics.get('loss')
            current_val_f1 = val_metrics.get('f1')
            current_val_precision = val_metrics.get('precision_micro', val_metrics.get('precision'))
            current_val_recall = val_metrics.get('recall_micro', val_metrics.get('recall'))
            current_val_auc = val_metrics.get('auc_micro', val_metrics.get('auc'))
            log_metrics_to_tensorboard(tb_writers, 'val', val_metrics, epoch + 1)
            
            # 保存最佳模型
            current_f1 = val_metrics['f1']
            if current_f1 > best_metric + min_delta:
                best_metric = current_f1
                best_epoch = epoch + 1
                no_improve_epochs = 0
                checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints/graph_transform')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                CheckpointManager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    metrics=val_metrics,
                    filepath=checkpoint_path,
                    is_best=True
                )
                logger.info(f"Saved best model with F1: {best_metric:.4f}")
            else:
                no_improve_epochs += 1
                logger.info(f"No improvement for {no_improve_epochs} epoch(s)")

            if early_stopping_enabled and no_improve_epochs >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best F1 {best_metric:.4f} at epoch {best_epoch}."
                )
                history['val_loss'].append(current_val_loss)
                history['val_f1'].append(current_val_f1)
                history['val_precision'].append(current_val_precision)
                history['val_recall'].append(current_val_recall)
                history['val_auc'].append(current_val_auc)
                save_training_curves(history, config, logger, training_config['checkpoint_dir'])
                logger.info(f"Epoch {epoch + 1} wall time: {time.perf_counter() - epoch_start:.2f}s")
                break
        
        history['val_loss'].append(current_val_loss)
        history['val_f1'].append(current_val_f1)
        history['val_precision'].append(current_val_precision)
        history['val_recall'].append(current_val_recall)
        history['val_auc'].append(current_val_auc)
        save_training_curves(history, config, logger, training_config['checkpoint_dir'])
        if tb_writers:
            flush_tensorboard_writers(tb_writers)
        
        # 定期保存检查点
        if (epoch + 1) % training_config.get('save_interval', 10) == 0:
            checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints/graph_transform')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            CheckpointManager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_metrics['loss'],
                metrics=train_metrics,
                filepath=checkpoint_path,
                is_best=False
            )
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")

        logger.info(f"Epoch {epoch + 1} wall time: {time.perf_counter() - epoch_start:.2f}s")
    
    # 测试阶段
    if test_loader is not None:
        checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints/graph_transform')
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if val_loader is not None:
            if os.path.exists(best_checkpoint_path):
                logger.info(f"Reloading best checkpoint before final test evaluation: {best_checkpoint_path}")
                CheckpointManager.load_checkpoint(
                    filepath=best_checkpoint_path,
                    model=model,
                    device=device,
                )
            else:
                logger.warning(
                    "Validation was enabled but best checkpoint was not found at %s; "
                    "final test evaluation will use the current in-memory model.",
                    best_checkpoint_path,
                )
        else:
            logger.info("No validation loader configured; final test evaluation will use the current in-memory model.")

        logger.info("Starting final evaluation on test set...")
        test_start = time.perf_counter()
        test_metrics = evaluator.evaluate(test_loader)
        logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}")
        test_step = history['epoch'][-1] if history['epoch'] else 0
        log_metrics_to_tensorboard(tb_writers, 'test', test_metrics, test_step)
        if config.get('evaluation', {}).get('save_outputs', True):
            prediction_outputs = evaluator.collect_prediction_outputs(test_loader)
            checkpoint_reference = best_checkpoint_path if (val_loader is not None and os.path.exists(best_checkpoint_path)) else os.path.join(checkpoint_dir, 'current_in_memory_model')
            evaluation_id = build_evaluation_id(checkpoint_reference, 'test')
            evaluation_config = config.get('evaluation', {})
            metric_csv_path = os.path.join(
                evaluation_config.get('output_metric_dir', 'result/metric/graph_transform'),
                'latest_test_metric.csv',
            )
            pred_csv_path = os.path.join(
                evaluation_config.get('output_pred_dir', 'result/pred/graph_transform'),
                'latest_test.pred.csv',
            )
            output_df = test_dataset.data.copy()
            output_df['evaluation_id'] = evaluation_id
            output_df['checkpoint_path'] = os.path.abspath(checkpoint_reference)
            output_df['threshold'] = prediction_outputs['threshold']
            output_df['true'] = prediction_outputs['true_strings']
            output_df['pred'] = prediction_outputs['pred_strings']
            output_df['pred_prob'] = prediction_outputs['prob_strings']
            save_evaluation_outputs(
                metrics=test_metrics,
                output_df=output_df,
                metric_csv_path=metric_csv_path,
                pred_csv_path=pred_csv_path,
                evaluation_id=evaluation_id,
                logger=logger,
            )
        logger.info(f"Final test evaluation took {time.perf_counter() - test_start:.2f}s")
    
    logger.info(f"Training completed in {time.perf_counter() - training_start:.2f}s")
    if tb_writers:
        close_tensorboard_writers(tb_writers)


if __name__ == '__main__':
    main()
