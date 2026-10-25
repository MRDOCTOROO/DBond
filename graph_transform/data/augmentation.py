"""
数据增强模块

本模块包含序列和图数据的增强功能。
"""

import torch
import numpy as np
import random
from typing import Dict, List, Optional, Any, Tuple
import re


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        if key in config:
            return config.get(key, default)
        data_config = config.get('data', {})
        if isinstance(data_config, dict) and key in data_config:
            return data_config[key]
        model_config = config.get('model', {})
        if isinstance(model_config, dict) and key in model_config:
            return model_config[key]
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    if hasattr(config, key):
        return getattr(config, key)
    return default


class SequenceAugmentation:
    """序列数据增强器"""
    
    def __init__(self, config):
        self.config = config
        self.augmentation_prob = _get_config_value(config, 'augmentation_prob', 0.3)
        self.max_trials = _get_config_value(config, 'max_augmentation_trials', 3)
        self.env_feature_name = _get_config_value(config, 'env_feature_name', 'rt')
        
        # 氨基酸替代矩阵
        self.amino_acid_groups = {
            'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'],
            'polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
            'positive': ['K', 'R', 'H'],
            'negative': ['D', 'E'],
            'special': ['G', 'P']
        }
        
        # 相似氨基酸替换映射
        self.similar_substitutions = {
            'A': ['G', 'S'],
            'R': ['K', 'H'],
            'N': ['D', 'Q', 'S'],
            'D': ['N', 'E', 'Q'],
            'C': ['S', 'T'],
            'Q': ['E', 'N', 'H'],
            'E': ['D', 'Q', 'K'],
            'G': ['A', 'S', 'P'],
            'H': ['R', 'K', 'Q', 'Y'],
            'I': ['L', 'V', 'M'],
            'L': ['I', 'V', 'M'],
            'K': ['R', 'H', 'E'],
            'M': ['I', 'L', 'V'],
            'F': ['Y', 'W', 'H'],
            'P': ['G', 'A'],
            'S': ['T', 'N', 'C'],
            'T': ['S', 'N', 'C'],
            'W': ['F', 'Y', 'H'],
            'Y': ['F', 'W', 'H'],
            'V': ['I', 'L', 'M']
        }
    
    def augment(self, sequence: str, labels: List[int], sample_features: Dict[str, float]) -> Tuple[str, List[int], Dict[str, float]]:
        """
        增强单个样本
        
        Args:
            sequence: 氨基酸序列
            labels: 标签列表
            env_vars: 环境变量
            
        Returns:
            Tuple[str, List[int], Dict[str, float]]: 增强后的序列、标签和环境变量
        """
        if random.random() > self.augmentation_prob:
            return sequence, labels, sample_features
        
        # 选择增强策略
        augmentation_methods = [
            self._amino_acid_substitution,
            self._sequence_truncation,
            self._noise_injection,
            self._reverse_complement
        ]
        
        method = random.choice(augmentation_methods)
        return method(sequence, labels, sample_features)
    
    def _amino_acid_substitution(self, sequence: str, labels: List[int], sample_features: Dict[str, float]) -> Tuple[str, List[int], Dict[str, float]]:
        """氨基酸替换"""
        if len(sequence) < 3:
            return sequence, labels, sample_features
        
        # 随机选择替换位置
        num_substitutions = random.randint(1, min(3, len(sequence) // 3))
        positions = random.sample(range(len(sequence)), num_substitutions)
        
        sequence_list = list(sequence)
        
        for pos in positions:
            original_aa = sequence_list[pos]
            if original_aa in self.similar_substitutions:
                substitutes = self.similar_substitutions[original_aa]
                sequence_list[pos] = random.choice(substitutes)
        
        augmented_sequence = ''.join(sequence_list)
        
        # 保持标签不变
        return augmented_sequence, labels, sample_features
    
    def _sequence_truncation(self, sequence: str, labels: List[int], sample_features: Dict[str, float]) -> Tuple[str, List[int], Dict[str, float]]:
        """序列截断"""
        if len(sequence) <= 10:
            return sequence, labels, sample_features
        
        # 随机截断比例
        trunc_ratio = random.uniform(0.8, 0.95)
        new_length = int(len(sequence) * trunc_ratio)
        
        # 截断序列
        augmented_sequence = sequence[:new_length]
        
        # 截断标签
        augmented_labels = labels[: max(new_length - 1, 0)]
        
        return augmented_sequence, augmented_labels, sample_features
    
    def _noise_injection(self, sequence: str, labels: List[int], sample_features: Dict[str, float]) -> Tuple[str, List[int], Dict[str, float]]:
        """注入噪声到环境变量"""
        # 复制环境变量
        augmented_features = sample_features.copy()
        
        # 对数值特征添加噪声
        noise_features = ['charge', 'pep_mass', 'intensity', 'nce', self.env_feature_name]
        
        for feature in noise_features:
            if feature in augmented_features:
                value = augmented_features[feature]
                noise_std = value * 0.05  # 5%的标准差
                noise = random.gauss(0, noise_std)
                augmented_features[feature] = max(0, value + noise)
        
        return sequence, labels, augmented_features
    
    def _reverse_complement(self, sequence: str, labels: List[int], sample_features: Dict[str, float]) -> Tuple[str, List[int], Dict[str, float]]:
        """序列反转"""
        augmented_sequence = sequence[::-1]
        augmented_labels = labels[::-1]
        
        return augmented_sequence, augmented_labels, sample_features
    
    def batch_augment(self, sequences: List[str], labels_list: List[List[int]], sample_features_list: List[Dict[str, float]]) -> Tuple[List[str], List[List[int]], List[Dict[str, float]]]:
        """
        批量增强数据
        
        Args:
            sequences: 序列列表
            labels_list: 标签列表
            env_vars_list: 环境变量列表
            
        Returns:
            Tuple: 增强后的数据
        """
        augmented_sequences = []
        augmented_labels = []
        augmented_feature_dicts = []
        
        for seq, labels, sample_features in zip(sequences, labels_list, sample_features_list):
            aug_seq, aug_labels, aug_features = self.augment(seq, labels, sample_features)
            augmented_sequences.append(aug_seq)
            augmented_labels.append(aug_labels)
            augmented_feature_dicts.append(aug_features)
        
        return augmented_sequences, augmented_labels, augmented_feature_dicts


class GraphAugmentation:
    """图数据增强器"""
    
    def __init__(self, config):
        self.config = config
        self.augmentation_prob = _get_config_value(config, 'augmentation_prob', 0.3)
    
    def augment_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强边数据
        
        Args:
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 增强后的边索引和属性
        """
        if random.random() > self.augmentation_prob:
            return edge_index, edge_attr
        
        # 随机丢弃一些边（边dropout）
        num_edges = edge_index.size(1)
        dropout_prob = 0.1
        
        mask = torch.rand(num_edges) > dropout_prob
        augmented_edge_index = edge_index[:, mask]
        augmented_edge_attr = edge_attr[mask]
        
        return augmented_edge_index, augmented_edge_attr
    
    def augment_nodes(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        增强节点特征
        
        Args:
            node_features: 节点特征 [num_nodes, feature_dim]
            
        Returns:
            torch.Tensor: 增强后的节点特征
        """
        if random.random() > self.augmentation_prob:
            return node_features
        
        # 添加高斯噪声
        noise_std = 0.01
        noise = torch.randn_like(node_features) * noise_std
        augmented_features = node_features + noise
        
        return augmented_features
    
    def augment_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        增强整个图结构
        
        Args:
            node_features: 节点特征
            edge_index: 边索引
            edge_attr: 边属性
            
        Returns:
            Tuple: 增强后的图数据
        """
        # 增强节点特征
        augmented_node_features = self.augment_nodes(node_features)
        
        # 增强边数据
        augmented_edge_index, augmented_edge_attr = self.augment_edges(edge_index, edge_attr)
        
        return augmented_node_features, augmented_edge_index, augmented_edge_attr
