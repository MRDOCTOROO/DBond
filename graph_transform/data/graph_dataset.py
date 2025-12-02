"""
图数据集类

本文件包含了用于图神经网络训练的数据集类和数据加载器。
支持蛋白质序列到图结构的转换，以及批处理功能。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import os

from .graph_builder import SequenceGraphBuilder
from .preprocessing import SequencePreprocessor


class GraphDataset(Dataset):
    """图数据集类"""
    
    def __init__(self, 
                 csv_path: str,
                 config: Any,
                 max_seq_len: int = 100,
                 graph_strategy: str = 'distance',
                 augmentation: bool = False,
                 split: str = 'train'):
        """
        初始化图数据集
        
        Args:
            csv_path: CSV数据文件路径
            config: 配置对象
            max_seq_len: 最大序列长度
            graph_strategy: 图构建策略 ('sequence', 'distance', 'hybrid')
            augmentation: 是否使用数据增强
            split: 数据分割 ('train', 'val', 'test')
        """
        self.csv_path = csv_path
        self.config = config
        self.max_seq_len = max_seq_len
        self.graph_strategy = graph_strategy
        self.augmentation = augmentation
        self.split = split
        
        # 加载数据
        self.data = self._load_data()
        
        # 初始化组件
        self.graph_builder = SequenceGraphBuilder(config)
        self.preprocessor = SequencePreprocessor(config)
        
        # 数据增强器
        if self.augmentation and split == 'train':
            from .augmentation import SequenceAugmentation
            self.augmentor = SequenceAugmentation(config)
        else:
            self.augmentor = None
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Data file not found: {self.csv_path}")
        
        data = pd.read_csv(self.csv_path)
        
        # 数据验证
        required_columns = ['seq', 'charge', 'pep_mass', 'nce', 'rt', 'fbr', 'true_multi']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 过滤过长的序列
        data = data[data['seq'].str.len() <= self.max_seq_len]
        
        return data.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        row = self.data.iloc[idx]
        
        # 提取序列和标签
        sequence = str(row['seq'])
        labels = self._parse_labels(str(row['true_multi']))
        
        # 提取环境变量
        env_vars = {
            'charge': float(row['charge']),
            'pep_mass': float(row['pep_mass']),
            'nce': float(row['nce']),
            'rt': float(row['rt']),
            'fbr': float(row['fbr'])
        }
        
        # 数据增强
        if self.augmentor is not None:
            sequence, labels, env_vars = self.augmentor.augment(sequence, labels, env_vars)
        
        # 构建图
        graph_data = self.graph_builder.build_graph(sequence, env_vars, self.graph_strategy)
        
        # 准备标签
        label_tensor = self._prepare_labels(labels, len(sequence))
        
        # 组合数据
        sample = {
            'sequence': sequence,
            'node_features': graph_data['node_features'],
            'edge_index': graph_data['edge_index'],
            'edge_attr': graph_data['edge_attr'],
            'edge_types': graph_data['edge_types'],
            'edge_distances': graph_data['edge_distances'],
            'labels': label_tensor,
            'charge': env_vars['charge'],
            'pep_mass': env_vars['pep_mass'],
            'nce': env_vars['nce'],
            'rt': env_vars['rt'],
            'fbr': env_vars['fbr'],
            'seq_len': len(sequence)
        }
        
        return sample
    
    def _parse_labels(self, label_str: str) -> List[int]:
        """解析多标签字符串"""
        if pd.isna(label_str) or label_str == '':
            return []
        
        # 解析分号分隔的多标签
        labels = [int(x.strip()) for x in str(label_str).split(';') if x.strip().isdigit()]
        return labels
    
    def _prepare_labels(self, labels: List[int], seq_len: int) -> torch.Tensor:
        """准备多标签张量"""
        # 创建多标签二进制矩阵 [seq_len, num_classes]
        multi_hot = torch.zeros(seq_len, self.config.num_classes)
        
        # 处理每个位置的标签
        for i, label in enumerate(labels):
            if i >= seq_len:
                break
            if 0 <= label < self.config.num_classes:
                multi_hot[i, label] = 1
        
        # 转换为float类型
        label_tensor = multi_hot.float()
        
        return label_tensor
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        seq_lengths = [len(seq) for seq in self.data['seq']]
        
        stats = {
            'num_samples': len(self.data),
            'avg_seq_length': np.mean(seq_lengths),
            'max_seq_length': np.max(seq_lengths),
            'min_seq_length': np.min(seq_lengths),
            'sequence_length_distribution': np.histogram(seq_lengths, bins=20)
        }
        
        # 标签统计
        all_labels = []
        for label_str in self.data['true_multi']:
            if not pd.isna(label_str):
                labels = self._parse_labels(str(label_str))
                all_labels.extend(labels)
        
        if all_labels:
            stats['label_distribution'] = np.bincount(all_labels)
            stats['positive_ratio'] = np.mean(all_labels)
        
        return stats


class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, 
                 dataset: GraphDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 collate_fn: Optional[callable] = None,
                 pin_memory: bool = True):
        """
        初始化图数据加载器
        
        Args:
            dataset: 图数据集
            batch_size: 批大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            collate_fn: 批处理函数
            pin_memory: 是否固定内存
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn or self._default_collate_fn
        self.pin_memory = pin_memory
        
        # 创建PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            drop_last=True if shuffle else False
        )
    
    def _default_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """默认批处理函数"""
        # 收集所有数据
        sequences = [item['sequence'] for item in batch]
        node_features_list = [item['node_features'] for item in batch]
        edge_index_list = [item['edge_index'] for item in batch]
        edge_attr_list = [item['edge_attr'] for item in batch]
        edge_types_list = [item['edge_types'] for item in batch]
        edge_distances_list = [item['edge_distances'] for item in batch]
        labels_list = [item['labels'] for item in batch]
        
        charges = [item['charge'] for item in batch]
        pep_masses = [item['pep_mass'] for item in batch]
        nces = [item['nce'] for item in batch]
        rts = [item['rt'] for item in batch]
        fbrs = [item['fbr'] for item in batch]
        seq_lens = [item['seq_len'] for item in batch]
        
        # 批处理节点特征
        batch_size = len(batch)
        max_seq_len = max(seq_lens)
        
        # 填充节点特征
        padded_node_features = []
        for node_features in node_features_list:
            if node_features.size(0) < max_seq_len:
                padding = torch.zeros(max_seq_len - node_features.size(0), node_features.size(1))
                padded_node_features.append(torch.cat([node_features, padding], dim=0))
            else:
                padded_node_features.append(node_features[:max_seq_len])
        
        batch_node_features = torch.stack(padded_node_features, dim=0)
        
        # 批处理边索引（需要偏移）
        batch_edge_indices = []
        batch_edge_attrs = []
        batch_edge_types = []
        batch_edge_distances = []
        
        node_offset = 0
        for i, (edge_index, edge_attr, edge_types, edge_distances) in enumerate(
            zip(edge_index_list, edge_attr_list, edge_types_list, edge_distances_list)
        ):
            # 添加节点偏移
            offset_edge_index = edge_index + node_offset
            batch_edge_indices.append(offset_edge_index)
            batch_edge_attrs.append(edge_attr)
            batch_edge_types.append(edge_types)
            batch_edge_distances.append(edge_distances)
            
            node_offset += seq_lens[i]
        
        # 拼接所有边
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        batch_edge_attr = torch.cat(batch_edge_attrs, dim=0)
        batch_edge_types = torch.cat(batch_edge_types, dim=0)
        batch_edge_distances = torch.cat(batch_edge_distances, dim=0)
        
        # 填充标签
        if labels_list[0].dim() == 1:  # 单标签
            padded_labels = []
            for labels in labels_list:
                if labels.size(0) < max_seq_len:
                    padding = torch.zeros(max_seq_len - labels.size(0))
                    padded_labels.append(torch.cat([labels, padding], dim=0))
                else:
                    padded_labels.append(labels[:max_seq_len])
            batch_labels = torch.stack(padded_labels, dim=0)
        else:  # 多标签
            padded_labels = []
            for labels in labels_list:
                if labels.size(0) < max_seq_len:
                    padding = torch.zeros(max_seq_len - labels.size(0), labels.size(1))
                    padded_labels.append(torch.cat([labels, padding], dim=0))
                else:
                    padded_labels.append(labels[:max_seq_len])
            batch_labels = torch.stack(padded_labels, dim=0)
        
        # 创建批次数据
        batch_data = {
            'sequences': sequences,
            'node_features': batch_node_features,
            'edge_index': batch_edge_index,
            'edge_attr': batch_edge_attr,
            'edge_types': batch_edge_types,
            'edge_distances': batch_edge_distances,
            'labels': batch_labels,
            'charges': torch.tensor(charges, dtype=torch.float32),
            'pep_masses': torch.tensor(pep_masses, dtype=torch.float32),
            'nces': torch.tensor(nces, dtype=torch.float32),
            'rts': torch.tensor(rts, dtype=torch.float32),
            'fbrs': torch.tensor(fbrs, dtype=torch.float32),
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long),
            'batch_size': batch_size
        }
        
        return batch_data
    
    def __iter__(self):
        """迭代器"""
        return iter(self.dataloader)
    
    def __len__(self):
        """数据加载器长度"""
        return len(self.dataloader)


class MultiTaskGraphDataset(GraphDataset):
    """多任务图数据集"""
    
    def __init__(self, 
                 csv_path: str,
                 config: Any,
                 tasks: List[str],
                 max_seq_len: int = 100,
                 graph_strategy: str = 'distance',
                 augmentation: bool = False,
                 split: str = 'train'):
        """
        初始化多任务图数据集
        
        Args:
            csv_path: CSV数据文件路径
            config: 配置对象
            tasks: 任务列表
            max_seq_len: 最大序列长度
            graph_strategy: 图构建策略
            augmentation: 是否使用数据增强
            split: 数据分割
        """
        super().__init__(csv_path, config, max_seq_len, graph_strategy, augmentation, split)
        
        self.tasks = tasks
        
        # 验证任务列是否存在
        missing_tasks = [task for task in tasks if task not in self.data.columns]
        if missing_tasks:
            raise ValueError(f"Missing task columns: {missing_tasks}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取多任务样本"""
        sample = super().__getitem__(idx)
        
        # 添加多任务标签
        row = self.data.iloc[idx]
        task_labels = {}
        
        for task in self.tasks:
            if task in row and not pd.isna(row[task]):
                task_labels[task] = torch.tensor(float(row[task]), dtype=torch.float32)
            else:
                task_labels[task] = torch.tensor(0.0, dtype=torch.float32)
        
        sample['task_labels'] = task_labels
        
        return sample


class CachedGraphDataset(GraphDataset):
    """缓存的图数据集"""
    
    def __init__(self, 
                 csv_path: str,
                 config: Any,
                 cache_dir: str,
                 max_seq_len: int = 100,
                 graph_strategy: str = 'distance',
                 augmentation: bool = False,
                 split: str = 'train',
                 rebuild_cache: bool = False):
        """
        初始化缓存图数据集
        
        Args:
            csv_path: CSV数据文件路径
            config: 配置对象
            cache_dir: 缓存目录
            max_seq_len: 最大序列长度
            graph_strategy: 图构建策略
            augmentation: 是否使用数据增强
            split: 数据分割
            rebuild_cache: 是否重建缓存
        """
        super().__init__(csv_path, config, max_seq_len, graph_strategy, augmentation, split)
        
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载或构建缓存
        self.cache_file = os.path.join(cache_dir, f"{split}_{graph_strategy}_cache.pt")
        self._load_or_build_cache()
    
    def _load_or_build_cache(self):
        """加载或构建缓存"""
        if os.path.exists(self.cache_file) and not self.rebuild_cache:
            try:
                self.cached_data = torch.load(self.cache_file)
                print(f"Loaded cache from {self.cache_file}")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        # 构建缓存
        print("Building cache...")
        self.cached_data = {}
        
        for idx in range(len(self.data)):
            sample = super().__getitem__(idx)
            self.cached_data[idx] = {
                'node_features': sample['node_features'],
                'edge_index': sample['edge_index'],
                'edge_attr': sample['edge_attr'],
                'edge_types': sample['edge_types'],
                'edge_distances': sample['edge_distances']
            }
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.data)} samples")
        
        # 保存缓存
        torch.save(self.cached_data, self.cache_file)
        print(f"Cache saved to {self.cache_file}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取缓存样本"""
        row = self.data.iloc[idx]
        
        # 提取序列和标签
        sequence = str(row['seq'])
        labels = self._parse_labels(str(row['true_multi']))
        
        # 提取环境变量
        env_vars = {
            'charge': float(row['charge']),
            'pep_mass': float(row['pep_mass']),
            'nce': float(row['nce']),
            'rt': float(row['rt']),
            'fbr': float(row['fbr'])
        }
        
        # 从缓存获取图数据
        cached_graph = self.cached_data[idx]
        
        # 准备标签
        label_tensor = self._prepare_labels(labels, len(sequence))
        
        # 组合数据
        sample = {
            'sequence': sequence,
            'node_features': cached_graph['node_features'],
            'edge_index': cached_graph['edge_index'],
            'edge_attr': cached_graph['edge_attr'],
            'edge_types': cached_graph['edge_types'],
            'edge_distances': cached_graph['edge_distances'],
            'labels': label_tensor,
            'charge': env_vars['charge'],
            'pep_mass': env_vars['pep_mass'],
            'nce': env_vars['nce'],
            'rt': env_vars['rt'],
            'fbr': env_vars['fbr'],
            'seq_len': len(sequence)
        }
        
        return sample
