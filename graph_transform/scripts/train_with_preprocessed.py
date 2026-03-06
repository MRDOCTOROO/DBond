"""
使用预处理数据的训练示例脚本

展示如何在训练脚本中使用预处理数据，提升训练效率。

Author: DBond Project Team
Date: 2026-03-06
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from tqdm import tqdm
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graph_transform.data.optimized_graph_dataset import create_optimized_dataloader
from graph_transform.models.graph_transformer import GraphTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerWithPreprocessedData:
    """使用预处理数据的训练器"""

    def __init__(self, config_path: str, use_preprocessed: bool = True):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
            use_preprocessed: 是否使用预处理数据
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.use_preprocessed = use_preprocessed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"训练器初始化完成")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 使用预处理数据: {use_preprocessed}")

    def setup_model(self):
        """设置模型"""
        logger.info("设置模型...")

        self.model = GraphTransformer(self.config).to(self.device)

        # 打印模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"  - 总参数: {total_params:,}")
        logger.info(f"  - 可训练参数: {trainable_params:,}")

    def setup_data_loaders(self):
        """设置数据加载器"""
        logger.info("设置数据加载器...")

        # 创建训练数据加载器
        self.train_loader = create_optimized_dataloader(
            config=self.config,
            split='train',
            use_preprocessed=self.use_preprocessed
        )

        # 创建测试数据加载器
        self.test_loader = create_optimized_dataloader(
            config=self.config,
            split='test',
            use_preprocessed=self.use_preprocessed
        )

        logger.info(f"  - 训练批次数: {len(self.train_loader)}")
        logger.info(f"  - 测试批次数: {len(self.test_loader)}")

    def setup_optimizer(self):
        """设置优化器"""
        logger.info("设置优化器...")

        optimizer_config = self.config['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        if optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_config['type']}")

        # 设置学习率调度器
        scheduler_config = self.config['training']
        if scheduler_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['epochs'],
                eta_min=scheduler_config.get('min_lr', 0.00001)
            )
        else:
            self.scheduler = None

        logger.info(f"  - 优化器: {optimizer_config['type']}")
        logger.info(f"  - 学习率: {lr}")

    def setup_loss_function(self):
        """设置损失函数"""
        logger.info("设置损失函数...")

        loss_config = self.config['loss']

        if loss_config['main_loss'] == 'binary_cross_entropy':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_config['main_loss']}")

        logger.info(f"  - 损失函数: {loss_config['main_loss']}")

    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # 数据移到设备
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device)
            edge_types = batch['edge_types'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_types=edge_types,
                batch_size=batch['batch_size']
            )

            # 计算损失
            loss = self.criterion(predictions, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config['training'].get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )

            self.optimizer.step()

            # 统计
            total_loss += loss.item() * batch['batch_size']
            total_samples += batch['batch_size']

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_samples:.4f}'
            })

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples

        metrics = {
            'loss': avg_loss,
            'time': epoch_time
        }

        return metrics

    def evaluate(self) -> dict:
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # 数据移到设备
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device)
                edge_types = batch['edge_types'].to(self.device)
                labels = batch['labels'].to(self.device)
                label_mask = batch['label_mask'].to(self.device)

                # 前向传播
                predictions = self.model(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_types=edge_types,
                    batch_size=batch['batch_size']
                )

                # 计算损失
                loss = self.criterion(predictions, labels)

                # 统计
                total_loss += loss.item() * batch['batch_size']
                total_samples += batch['batch_size']

                # 保存预测结果
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # 计算指标
        avg_loss = total_loss / total_samples

        # 这里可以添加更多评估指标

        metrics = {
            'loss': avg_loss
        }

        return metrics

    def train(self, num_epochs: int = None):
        """训练模型"""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']

        logger.info(f"开始训练，共 {num_epochs} 个epochs")
        logger.info(f"数据模式: {'预处理数据' if self.use_preprocessed else '原始CSV数据'}")

        best_loss = float('inf')
        training_start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 评估
            eval_metrics = self.evaluate()

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 打印统计
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Eval Loss: {eval_metrics['loss']:.4f} "
                f"Time: {train_metrics['time']:.2f}s"
            )

            # 保存最佳模型
            if eval_metrics['loss'] < best_loss:
                best_loss = eval_metrics['loss']
                self.save_model(epoch, eval_metrics['loss'])

        total_time = time.time() - training_start_time
        logger.info(f"训练完成！总时间: {total_time/60:.2f}分钟")

    def save_model(self, epoch: int, loss: float):
        """保存模型"""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }, checkpoint_path)

        logger.info(f"模型已保存: {checkpoint_path}")

    def run(self):
        """运行完整训练流程"""
        logger.info("="*80)
        logger.info("训练流程开始")
        logger.info("="*80)

        # 设置组件
        self.setup_model()
        self.setup_data_loaders()
        self.setup_optimizer()
        self.setup_loss_function()

        # 开始训练
        self.train()

        logger.info("="*80)
        logger.info("训练流程完成")
        logger.info("="*80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用预处理数据的训练脚本')
    parser.add_argument('--config', type=str,
                       default='graph_transform/config/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--use_preprocessed', action='store_true', default=True,
                       help='使用预处理数据（默认开启）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')

    args = parser.parse_args()

    # 创建训练器
    trainer = TrainerWithPreprocessedData(
        config_path=args.config,
        use_preprocessed=args.use_preprocessed
    )

    # 运行训练
    trainer.run()

    if args.epochs:
        trainer.train(num_epochs=args.epochs)
    else:
        trainer.run()


if __name__ == "__main__":
    main()