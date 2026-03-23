"""
预处理功能测试脚本

用于验证图数据预处理功能是否正常工作。

Author: DBond Project Team
Date: 2026-03-06
"""

import os
import sys
import torch
import yaml
import time
from pathlib import Path
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graph_transform.data.optimized_graph_dataset import (
    HybridGraphDataset,
    OptimizedGraphDataLoader
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_preprocessed_data(config_path: str,
                          preprocessed_dir: str,
                          split: str = 'train'):
    """
    测试预处理数据加载

    Args:
        config_path: 配置文件路径
        preprocessed_dir: 预处理数据目录
        split: 数据集划分
    """
    logger.info(f"测试预处理数据加载: {split}")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 检查预处理文件是否存在
    preprocessed_file = Path(preprocessed_dir) / f"{split}_graph_data.pt"

    if not preprocessed_file.exists():
        logger.error(f"预处理文件不存在: {preprocessed_file}")
        logger.info("请先运行预处理脚本:")
        logger.info(f"python graph_transform/scripts/preprocess_graph_data.py \\")
        logger.info(f"    --config {config_path} \\")
        logger.info(f"    --train_csv dataset/dbond_m.train.shuffle.csv \\")
        logger.info(f"    --test_csv dataset/dbond_m.test.csv \\")
        logger.info(f"    --output_dir {preprocessed_dir}")
        return False

    # 创建数据集
    try:
        dataset = HybridGraphDataset(
            data_path=str(preprocessed_file),
            config=config,
            use_preprocessed=True
        )

        logger.info(f"✅ 数据集创建成功")
        logger.info(f"   - 样本数: {len(dataset)}")
        logger.info(f"   - 数据类型: {dataset.dataset_type}")

        # 测试单个样本加载
        sample = dataset[0]
        logger.info(f"✅ 单个样本加载成功")
        logger.info(f"   - 样本键: {list(sample.keys())}")
        logger.info(f"   - 边索引形状: {sample['edge_index'].shape}")
        logger.info(f"   - 边属性形状: {sample['edge_attr'].shape}")
        logger.info(f"   - 标签形状: {sample['labels'].shape}")

        return True

    except Exception as e:
        logger.error(f"❌ 数据集创建失败: {e}")
        return False


def test_dataloader(config_path: str,
                   preprocessed_dir: str,
                   split: str = 'train',
                   batch_size: int = 32):
    """
    测试数据加载器

    Args:
        config_path: 配置文件路径
        preprocessed_dir: 预处理数据目录
        split: 数据集划分
        batch_size: 批大小
    """
    logger.info(f"测试数据加载器: {split}")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建数据加载器
    try:
        dataset = HybridGraphDataset(
            data_path=str(Path(preprocessed_dir) / f"{split}_graph_data.pt"),
            config=config,
            use_preprocessed=True
        )

        dataloader = OptimizedGraphDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2,  # 测试时使用较少工作进程
            pin_memory=False  # 测试时不使用pin_memory
        )

        logger.info(f"✅ 数据加载器创建成功")
        logger.info(f"   - 批大小: {batch_size}")
        logger.info(f"   - 批次数: {len(dataloader)}")

        # 测试批处理
        start_time = time.time()
        batch = next(iter(dataloader))
        load_time = time.time() - start_time

        logger.info(f"✅ 批处理加载成功")
        logger.info(f"   - 加载时间: {load_time:.3f}秒")
        logger.info(f"   - 批次键: {list(batch.keys())}")
        logger.info(f"   - 边索引形状: {batch['edge_index'].shape}")
        logger.info(f"   - 标签形状: {batch['labels'].shape}")

        return True

    except Exception as e:
        logger.error(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_loading_speed(config_path: str,
                           preprocessed_dir: str,
                           num_batches: int = 10):
    """
    基准测试加载速度

    Args:
        config_path: 配置文件路径
        preprocessed_dir: 预处理数据目录
        num_batches: 测试批次数
    """
    logger.info(f"基准测试加载速度 (测试{num_batches}批次)")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        # 创建数据加载器
        dataset = HybridGraphDataset(
            data_path=str(Path(preprocessed_dir) / "train_graph_data.pt"),
            config=config,
            use_preprocessed=True
        )

        dataloader = OptimizedGraphDataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # 测试加载速度
        start_time = time.time()
        batch_count = 0

        for i, batch in enumerate(dataloader):
            batch_count += 1
            if batch_count >= num_batches:
                break

        total_time = time.time() - start_time
        avg_time_per_batch = total_time / num_batches

        logger.info(f"✅ 基准测试完成:")
        logger.info(f"   - 总时间: {total_time:.3f}秒")
        logger.info(f"   - 平均每批次: {avg_time_per_batch:.3f}秒")
        logger.info(f"   - 吞吐量: {num_batches/total_time:.2f} 批次/秒")

        return True

    except Exception as e:
        logger.error(f"❌ 基准测试失败: {e}")
        return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='预处理功能测试脚本')
    parser.add_argument('--config', type=str,
                       default='graph_transform/config/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--preprocessed_dir', type=str,
                       default='cache/preprocessed_graph_data',
                       help='预处理数据目录')
    parser.add_argument('--test_split', type=str, default='train',
                       choices=['train', 'test', 'val'],
                       help='测试的数据集划分')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='测试批大小')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='基准测试批次数')
    parser.add_argument('--skip_dataloader', action='store_true',
                       help='跳过数据加载器测试')
    parser.add_argument('--skip_benchmark', action='store_true',
                       help='跳过基准测试')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("预处理功能测试")
    logger.info("="*80)

    all_passed = True

    # 测试1: 数据集加载
    logger.info("\n测试1: 数据集加载")
    logger.info("-"*80)
    if not test_preprocessed_data(args.config, args.preprocessed_dir, args.test_split):
        all_passed = False

    # 测试2: 数据加载器
    if not args.skip_dataloader:
        logger.info("\n测试2: 数据加载器")
        logger.info("-"*80)
        if not test_dataloader(args.config, args.preprocessed_dir,
                             args.test_split, args.batch_size):
            all_passed = False

    # 测试3: 基准测试
    if not args.skip_benchmark:
        logger.info("\n测试3: 加载速度基准测试")
        logger.info("-"*80)
        if not benchmark_loading_speed(args.config, args.preprocessed_dir,
                                      args.num_batches):
            all_passed = False

    # 总结
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("✅ 所有测试通过！预处理功能正常工作。")
        logger.info("\n下一步:")
        logger.info("1. 开始训练模型")
        logger.info("2. 在配置文件中设置 data.use_preprocessed: true")
        logger.info("3. 运行训练脚本")
    else:
        logger.info("❌ 部分测试失败，请检查错误信息。")
        logger.info("\n故障排除:")
        logger.info("1. 确认已运行预处理脚本")
        logger.info("2. 检查文件路径是否正确")
        logger.info("3. 查看详细错误信息")
    logger.info("="*80)


if __name__ == "__main__":
    main()