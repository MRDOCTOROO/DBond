#!/usr/bin/env python3
"""
图神经网络训练脚本

本脚本用于训练图神经网络多标签分类模型。
支持配置文件、命令行参数和完整的训练流程。
"""

import os
import sys
import argparse
import yaml
import logging
import time
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GraphTransformer
from models.utils import ModelConfig, CheckpointManager, LearningRateScheduler
from data import GraphDataset, GraphDataLoader
from training import Trainer, MultiLabelLoss
from evaluation import Evaluator, MultiLabelMetrics


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
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def create_datasets(config: Dict[str, Any]) -> tuple:
    """创建数据集"""
    data_config = config['data']
    model_config = ModelConfig(config['model'])
    
    # 训练数据集
    train_dataset = GraphDataset(
        csv_path=data_config['train_csv_path'],
        config=model_config,
        max_seq_len=data_config['max_seq_len'],
        graph_strategy=data_config['graph_strategy'],
        augmentation=data_config.get('augmentation', False),
        split='train'
    )
    
    # 验证数据集
    if data_config.get('val_csv_path'):
        val_dataset = GraphDataset(
            csv_path=data_config['val_csv_path'],
            config=model_config,
            max_seq_len=data_config['max_seq_len'],
            graph_strategy=data_config['graph_strategy'],
            augmentation=False,
            split='val'
        )
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
        test_dataset = GraphDataset(
            csv_path=data_config['test_csv_path'],
            config=model_config,
            max_seq_len=data_config['max_seq_len'],
            graph_strategy=data_config['graph_strategy'],
            augmentation=False,
            split='test'
        )
    
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
        if 'positive_ratio' in stats:
            logger.info(f"Positive label ratio: {stats['positive_ratio']:.4f}")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, 
                        config: Dict[str, Any]) -> tuple:
    """创建数据加载器"""
    data_config = config['data']
    training_config = config['training']
    
    # 训练数据加载器
    train_loader = GraphDataLoader(
        dataset=train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    # 验证数据加载器
    val_loader = None
    if val_dataset is not None:
        val_loader = GraphDataLoader(
            dataset=val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )
    
    # 测试数据加载器
    test_loader = None
    if test_dataset is not None:
        test_loader = GraphDataLoader(
            dataset=test_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )
    
    return train_loader, val_loader, test_loader


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """创建优化器"""
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    learning_rate = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8)
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
    parser = argparse.ArgumentParser(description='训练图神经网络多标签分类模型')
    
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
    model = create_model(config, device)
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset, test_dataset = create_datasets(config)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
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
    criterion = MultiLabelLoss(loss_config)
    
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
    epochs = training_config['epochs']
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # 训练阶段
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        
        # 验证阶段
        if val_loader is not None and epoch % training_config.get('validation_interval', 1) == 0:
            val_metrics = evaluator.evaluate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            current_f1 = val_metrics['f1']
            if current_f1 > best_metric:
                best_metric = current_f1
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
    
    # 测试阶段
    if test_loader is not None:
        logger.info("Starting final evaluation on test set...")
        test_metrics = evaluator.evaluate(test_loader)
        logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
