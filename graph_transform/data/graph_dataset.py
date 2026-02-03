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
            'seq_len': len(sequence),
            'node_len': len(sequence) + (1 if getattr(self.config, 'use_global_node', False) else 0)
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
        """准备断裂位置标签张量（序列相邻键）"""
        bond_len = max(seq_len - 1, 0)
        label_values = labels

        # 有些数据包含seq_len长度，取前seq_len-1作为键标签
        if len(label_values) >= seq_len:
            label_values = label_values[:bond_len]

        # 过短则补0
        if len(label_values) < bond_len:
            label_values = label_values + [0] * (bond_len - len(label_values))

        label_tensor = torch.tensor(label_values[:bond_len], dtype=torch.float32)
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
                 pin_memory: bool = True,
                 drop_last: bool = False):
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
        self.drop_last = drop_last
        
        # 创建PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    def _default_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """默认批处理函数"""
        # 收集所有数据
        sequences = [item['sequence'] for item in batch]
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
        node_lens = [item.get('node_len', item['seq_len']) for item in batch]
        
        batch_size = len(batch)
        max_seq_len = max(seq_lens)
        
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
            
            node_offset += node_lens[i]
        
        # 拼接所有边
        batch_edge_index = torch.cat(batch_edge_indices, dim=1)
        batch_edge_attr = torch.cat(batch_edge_attrs, dim=0)
        batch_edge_types = torch.cat(batch_edge_types, dim=0)
        batch_edge_distances = torch.cat(batch_edge_distances, dim=0)
        
        # 填充标签（键级别，长度=seq_len-1）
        max_bonds = max(max_seq_len - 1, 0)
        padded_labels = []
        label_masks = []
        for labels, seq_len in zip(labels_list, seq_lens):
            bond_len = max(seq_len - 1, 0)
            if labels.size(0) < max_bonds:
                padding = torch.zeros(max_bonds - labels.size(0))
                padded = torch.cat([labels, padding], dim=0)
            else:
                padded = labels[:max_bonds]
            padded_labels.append(padded)

            mask = torch.zeros(max_bonds)
            if bond_len > 0:
                mask[:bond_len] = 1.0
            label_masks.append(mask)

        batch_labels = torch.stack(padded_labels, dim=0)
        batch_label_masks = torch.stack(label_masks, dim=0)
        
        # 创建批次数据
        batch_data = {
            'sequences': sequences,
            'edge_index': batch_edge_index,
            'edge_attr': batch_edge_attr,
            'edge_types': batch_edge_types,
            'edge_distances': batch_edge_distances,
            'labels': batch_labels,
            'label_mask': batch_label_masks,
            'charges': torch.tensor(charges, dtype=torch.float32),
            'pep_masses': torch.tensor(pep_masses, dtype=torch.float32),
            'nces': torch.tensor(nces, dtype=torch.float32),
            'rts': torch.tensor(rts, dtype=torch.float32),
            'fbrs': torch.tensor(fbrs, dtype=torch.float32),
            'seq_lens': torch.tensor(seq_lens, dtype=torch.long),
            'node_lens': torch.tensor(node_lens, dtype=torch.long),
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
                 rebuild_cache: bool = False,
                 cache_full_graphs: bool = False):
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
            cache_full_graphs: 是否缓存完整图（大数据集可能较慢）
        """
        super().__init__(csv_path, config, max_seq_len, graph_strategy, augmentation, split)
        
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.rebuild_cache = rebuild_cache
        self.cache_full_graphs = cache_full_graphs
        self.use_long_range_edges = getattr(self.config, 'use_long_range_edges', False)
        self.long_range_stride = getattr(self.config, 'long_range_stride', 10)
        self.long_range_hops = getattr(self.config, 'long_range_hops', 1)
        self.use_global_node = getattr(self.config, 'use_global_node', False)
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载或构建边缓存（按长度）
        edge_suffix = self._cache_suffix()
        self.edge_cache_file = os.path.join(
            self.cache_dir,
            f"edges_{self.graph_strategy}_{edge_suffix}.pt"
        )
        self._load_or_build_edge_cache()
        
        # 加载或构建缓存
        self.cache_file = os.path.join(
            self.cache_dir,
            f"{split}_{graph_strategy}_{edge_suffix}_cache.pt"
        )
        if self.cache_full_graphs:
            self._load_or_build_cache()
        else:
            self.cached_data = None

    def _resolve_cache_dir(self, cache_dir: str) -> str:
        """将相对路径缓存目录解析为项目根目录下的绝对路径"""
        if os.path.isabs(cache_dir):
            return cache_dir
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(project_root, cache_dir)

    def _edge_cache_meta(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "graph_strategy": self.graph_strategy,
            "max_seq_len": self.max_seq_len,
            "max_distance": getattr(self.config, "max_distance", 0),
            "edge_types": list(getattr(self.config, "edge_types", [])),
            "use_long_range_edges": self.use_long_range_edges,
            "long_range_stride": self.long_range_stride,
            "long_range_hops": self.long_range_hops,
            "use_global_node": self.use_global_node,
        }

    def _cache_suffix(self) -> str:
        parts = [
            f"maxlen{self.max_seq_len}",
            f"maxdist{getattr(self.config, 'max_distance', 0)}",
        ]
        if self.use_long_range_edges:
            parts.append(f"lr{self.long_range_stride}x{self.long_range_hops}")
        if self.use_global_node:
            parts.append("g1")
        return "_".join(parts)

    def _populate_edge_cache(self, edge_cache: Dict[Any, Any]):
        max_distance = getattr(self.config, "max_distance", 0)
        for seq_len, edge_data in edge_cache.items():
            if isinstance(seq_len, str):
                seq_len = int(seq_len)
            if isinstance(edge_data, dict):
                edge_index = edge_data["edge_index"]
                edge_types = edge_data["edge_types"]
                edge_distances = edge_data["edge_distances"]
            else:
                edge_index, edge_types, edge_distances = edge_data
            key = (
                self.graph_strategy,
                seq_len,
                max_distance,
                self.use_long_range_edges,
                self.long_range_stride,
                self.long_range_hops,
                self.use_global_node,
            )
            self.graph_builder._edge_cache[key] = (edge_index, edge_types, edge_distances)

    def _load_or_build_edge_cache(self):
        """加载或构建按序列长度缓存的边结构"""
        meta = self._edge_cache_meta()
        if os.path.exists(self.edge_cache_file) and not self.rebuild_cache:
            try:
                payload = torch.load(self.edge_cache_file, map_location="cpu")
                cached_meta = payload.get("meta", {})
                cached_edges = payload.get("edges", None)
                if cached_edges is not None and cached_meta == meta:
                    self._populate_edge_cache(cached_edges)
                    print(f"Loaded edge cache from {self.edge_cache_file}")
                    return
            except Exception as e:
                print(f"Failed to load edge cache: {e}")

        # 构建边缓存（按长度）
        print("Building edge cache (by sequence length)...")
        edge_cache: Dict[int, Any] = {}
        for seq_len in range(1, self.max_seq_len + 1):
            edge_index, edge_types, edge_distances = self.graph_builder._get_or_build_edges(
                seq_len, self.graph_strategy
            )
            edge_cache[seq_len] = {
                "edge_index": edge_index,
                "edge_types": edge_types,
                "edge_distances": edge_distances,
            }

        self._populate_edge_cache(edge_cache)
        torch.save({"meta": meta, "edges": edge_cache}, self.edge_cache_file)
        print(f"Edge cache saved to {self.edge_cache_file}")
    
    def _load_or_build_cache(self):
        """加载或构建缓存"""
        meta = {
            "version": 1,
            "csv_path": os.path.abspath(self.csv_path),
            "graph_strategy": self.graph_strategy,
            "max_seq_len": self.max_seq_len,
            "max_distance": getattr(self.config, "max_distance", 0),
            "edge_types": list(getattr(self.config, "edge_types", [])),
            "use_long_range_edges": self.use_long_range_edges,
            "long_range_stride": self.long_range_stride,
            "long_range_hops": self.long_range_hops,
            "use_global_node": self.use_global_node,
        }
        if os.path.exists(self.cache_file) and not self.rebuild_cache:
            try:
                payload = torch.load(self.cache_file, map_location="cpu")
                cached_meta = payload.get("meta", {})
                cached_data = payload.get("data", None)
                if cached_data is not None and cached_meta == meta:
                    self.cached_data = cached_data
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
                'edge_index': sample['edge_index'],
                'edge_attr': sample['edge_attr'],
                'edge_types': sample['edge_types'],
                'edge_distances': sample['edge_distances']
            }
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.data)} samples")
        
        # 保存缓存
        torch.save({"meta": meta, "data": self.cached_data}, self.cache_file)
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
        
        if self.cached_data is not None:
            cached_graph = self.cached_data[idx]
            edge_index = cached_graph['edge_index']
            edge_attr = cached_graph['edge_attr']
            edge_types = cached_graph['edge_types']
            edge_distances = cached_graph['edge_distances']
        else:
            graph_data = self.graph_builder.build_graph(sequence, env_vars, self.graph_strategy)
            edge_index = graph_data['edge_index']
            edge_attr = graph_data['edge_attr']
            edge_types = graph_data['edge_types']
            edge_distances = graph_data['edge_distances']
        
        # 准备标签
        label_tensor = self._prepare_labels(labels, len(sequence))
        
        # 组合数据
        sample = {
            'sequence': sequence,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_types': edge_types,
            'edge_distances': edge_distances,
            'labels': label_tensor,
            'charge': env_vars['charge'],
            'pep_mass': env_vars['pep_mass'],
            'nce': env_vars['nce'],
            'rt': env_vars['rt'],
            'fbr': env_vars['fbr'],
            'seq_len': len(sequence),
            'node_len': len(sequence) + (1 if getattr(self.config, 'use_global_node', False) else 0)
        }
        
        return sample
