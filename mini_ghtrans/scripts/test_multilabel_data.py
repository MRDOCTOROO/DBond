#!/usr/bin/env python3
"""
多标签数据格式测试脚本

本脚本用于测试多标签数据格式的处理是否正确。
"""

import sys
import os
import pandas as pd
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelConfig
from data import GraphDataset
from training import MultiLabelLoss


def test_multilabel_format():
    """测试多标签数据格式"""
    print("Testing multi-label data format...")
    
    # 创建示例数据
    sample_data = {
        'name': ['YP-092', 'YP-092'],
        'seq': ['STKABDFYPQGTATSDAAEFGYED', 'STKABDFYPQGTATSDAAEFGYED'],
        'charge': [2, 4],
        'pep_mass': [1279.057610968719, 640.031853431055],
        'intensity': [40333286.38721, 10812201.61865],
        'nce': [30, 30],
        'scan_num': [2, 3],
        'rt': [0.82725786, 0.90697998],
        'fbr': [0.9130434782608695, 0.8695652173913043],
        'tb': [23, 23],
        'fb': [21, 20],
        'mb': ['1;23;', '1;18;23;'],
        'true_multi': ['0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0', 
                     '0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;1;1;1;1;0']
    }
    
    # 创建DataFrame
    df = pd.DataFrame(sample_data)
    print("Sample data:")
    print(df[['name', 'seq', 'charge', 'true_multi']].head())
    
    # 创建配置
    config_dict = {
        'hidden_dim': 256,
        'num_classes': 24,
        'max_seq_len': 100,
        'alphabet': "ACDEFGHIKLMNPQRSTVWY",
        'pad_char': "U",
        'aa_embedding_dim': 64,
        'position_embedding_dim': 32,
        'physicochemical_dim': 32,
        'num_physicochemical_features': 4,
        'num_env_features': 5,
        'edge_types': ['sequence', 'distance', 'functional'],
        'edge_embedding_dim': 16,
        'distance_embedding_dim': 16,
        'max_distance': 10,
        'dropout': 0.1
    }
    
    config = ModelConfig(config_dict)
    
    # 保存测试数据
    test_csv_path = 'test_multilabel_data.csv'
    df.to_csv(test_csv_path, index=False)
    print(f"Saved test data to {test_csv_path}")
    
    try:
        # 创建数据集
        dataset = GraphDataset(
            csv_path=test_csv_path,
            config=config,
            max_seq_len=100,
            graph_strategy='distance',
            augmentation=False,
            split='test'
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # 测试数据加载
        sample = dataset[0]
        print("\nSample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape} {value.dtype}")
            else:
                print(f"{key}: {value}")
        
        # 测试损失函数
        loss_config = {
            'main_loss': 'binary_cross_entropy',
            'use_auxiliary_losses': False,
            'handle_imbalance': True,
            'num_classes': 24
        }
        
        criterion = MultiLabelLoss(loss_config)
        
        # 创建虚拟预测
        dummy_prediction = torch.randn(1, 24)  # [batch_size, num_classes]
        dummy_target = sample['labels'][:1, :24]  # 确保形状匹配
        
        # 计算损失
        loss = criterion(dummy_prediction, dummy_target)
        print(f"\nLoss calculation successful: {loss.item():.4f}")
        
        # 测试标签解析
        print("\nLabel parsing test:")
        for i in range(min(3, len(df))):
            label_str = df.iloc[i]['true_multi']
            labels = dataset._parse_labels(label_str)
            print(f"Sample {i}: {label_str[:50]}... -> {labels[:10]}")
        
        print("\n✅ Multi-label data format test passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
            print(f"Cleaned up {test_csv_path}")


def test_data_format_requirements():
    """测试数据格式要求"""
    print("\n" + "="*50)
    print("MULTI-LABEL DATA FORMAT REQUIREMENTS")
    print("="*50)
    
    required_columns = [
        'name',        # 样本名称
        'seq',         # 氨基酸序列
        'charge',      # 电荷
        'pep_mass',    # 肽段质量
        'intensity',   # 强度
        'nce',         # 碰撞能量
        'scan_num',    # 扫描号
        'rt',          # 保留时间
        'fbr',         # 碎裂比例
        'tb',          # 总断裂数
        'fb',          # 前向断裂
        'mb',          # 中间断裂
        'true_multi'   # 多标签目标（分号分隔）
    ]
    
    print("Required columns in CSV file:")
    for i, col in enumerate(required_columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\nMulti-label format for 'true_multi' column:")
    print("- Use semicolon (;) as separator")
    print("- Example: '0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0'")
    print("- Each number represents a class label")
    print("- Position i corresponds to label for position i in sequence")
    
    print("\nSequence processing:")
    print("- Standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y")
    print("- Maximum sequence length: configurable (default: 100)")
    print("- Sequences longer than max_length will be truncated")


if __name__ == '__main__':
    test_data_format_requirements()
    test_multilabel_format()
