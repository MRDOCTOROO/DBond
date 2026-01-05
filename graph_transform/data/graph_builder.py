"""
图构建器

本文件包含了将蛋白质序列转换为图结构的各种策略和实现。
支持多种图构建方法，包括序列连接、距离基线和混合策略。
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import math


class GraphBuilder:
    """基础图构建器"""
    
    def __init__(self, config: Any):
        """
        初始化图构建器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.alphabet = config.alphabet
        self.max_distance = getattr(config, 'max_distance', 10)
        self.edge_types = getattr(config, 'edge_types', ['sequence', 'distance', 'functional'])
    
    def build_graph(self, 
                   sequence: str,
                   env_vars: Dict[str, float],
                   strategy: str = 'sequence') -> Dict[str, torch.Tensor]:
        """
        构建图结构
        
        Args:
            sequence: 蛋白质序列
            env_vars: 环境变量
            strategy: 构建策略
            
        Returns:
            Dict: 包含图数据的字典
        """
        if strategy == 'sequence':
            return self._build_sequence_graph(sequence, env_vars)
        elif strategy == 'distance':
            return self._build_distance_graph(sequence, env_vars)
        elif strategy == 'hybrid':
            return self._build_hybrid_graph(sequence, env_vars)
        else:
            raise ValueError(f"Unknown graph building strategy: {strategy}")
    
    def _build_sequence_graph(self, sequence: str, env_vars: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """构建基于序列连接的图"""
        seq_len = len(sequence)
        
        # 创建序列边（相邻氨基酸）
        edge_indices = []
        edge_types = []
        edge_distances = []
        
        for i in range(seq_len - 1):
            # 前向连接
            edge_indices.append([i, i + 1])
            edge_indices.append([i + 1, i])  # 反向连接
            
            edge_types.extend([0, 0])  # 序列边类型
            edge_distances.extend([1, 1])  # 相邻距离
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        edge_distance_tensor = torch.tensor(edge_distances, dtype=torch.long)
        
        # 边特征
        edge_attr = self._create_edge_features(edge_type_tensor, edge_distance_tensor, env_vars)
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_types': edge_type_tensor,
            'edge_distances': edge_distance_tensor
        }
    
    def _build_distance_graph(self, sequence: str, env_vars: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """构建基于距离的图"""
        seq_len = len(sequence)
        
        # 预测距离矩阵（简化版本）
        distance_matrix = self._predict_distance_matrix(sequence)
        
        edge_indices = []
        edge_types = []
        edge_distances = []
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                distance = distance_matrix[i, j]
                if distance <= self.max_distance:
                    # 添加边
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])
                    
                    # 确定边类型
                    if distance <= 2:
                        edge_type = 1  # 近距离
                    else:
                        edge_type = 2  # 中距离及以上
                    
                    edge_types.extend([edge_type, edge_type])
                    clipped_distance = min(int(distance), self.max_distance)
                    edge_distances.extend([clipped_distance, clipped_distance])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        edge_distance_tensor = torch.tensor(edge_distances, dtype=torch.long)
        
        # 边特征
        edge_attr = self._create_edge_features(edge_type_tensor, edge_distance_tensor, env_vars)
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_types': edge_type_tensor,
            'edge_distances': edge_distance_tensor
        }
    
    def _build_hybrid_graph(self, sequence: str, env_vars: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """构建混合图"""
        # 组合序列图和距离图
        seq_graph = self._build_sequence_graph(sequence, env_vars)
        dist_graph = self._build_distance_graph(sequence, env_vars)
        
        # 合并边
        all_edges = torch.cat([seq_graph['edge_index'], dist_graph['edge_index']], dim=1)
        all_types = torch.cat([seq_graph['edge_types'], dist_graph['edge_types']])
        all_distances = torch.cat([seq_graph['edge_distances'], dist_graph['edge_distances']])
        
        # 去重边
        unique_edges, unique_indices = torch.unique(all_edges, dim=1, return_inverse=True)
        
        # 合并边特征
        edge_attr = self._create_edge_features(all_types, all_distances, env_vars)
        
        return {
            'edge_index': unique_edges,
            'edge_attr': edge_attr[unique_indices],
            'edge_types': all_types[unique_indices],
            'edge_distances': all_distances[unique_indices]
        }
    
    def _predict_distance_matrix(self, sequence: str) -> np.ndarray:
        """预测氨基酸距离矩阵"""
        seq_len = len(sequence)
        distance_matrix = np.zeros((seq_len, seq_len))
        
        # 简化的距离预测（实际应用中应使用更复杂的方法）
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # 基于序列距离的简化预测
                    seq_dist = abs(i - j)
                    # 添加一些随机性来模拟蛋白质折叠
                    folding_factor = 1.0 + 0.2 * np.sin(seq_dist * 0.5)
                    distance_matrix[i, j] = seq_dist * folding_factor
        
        return distance_matrix
    
    def _create_edge_features(self, 
                           edge_types: torch.Tensor,
                           edge_distances: torch.Tensor,
                           env_vars: Dict[str, float]) -> torch.Tensor:
        """创建边特征"""
        num_edges = len(edge_types)
        edge_features = []
        
        for i in range(num_edges):
            edge_type = edge_types[i].item()
            distance = edge_distances[i].item()
            
            # 基础特征
            features = [
                float(edge_type),  # 边类型
                float(distance),  # 距离
                1.0 / (1.0 + distance),  # 距离倒数
            ]
            
            # 环境变量影响
            features.extend([
                env_vars.get('charge', 0.0) * 0.1,
                env_vars.get('nce', 0.0) * 0.01,
                env_vars.get('fbr', 0.0),
            ])
            
            edge_features.append(features)
        
        return torch.tensor(edge_features, dtype=torch.float32)


class SequenceGraphBuilder(GraphBuilder):
    """专门的序列图构建器"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
        # 氨基酸相似性矩阵
        self.aa_similarity = self._create_aa_similarity_matrix()
        
        # 物理化学性质
        self.aa_properties = self._get_aa_properties()
    
    def build_advanced_graph(self, 
                          sequence: str,
                          env_vars: Dict[str, float],
                          include_structure: bool = True) -> Dict[str, torch.Tensor]:
        """
        构建高级图结构
        
        Args:
            sequence: 蛋白质序列
            env_vars: 环境变量
            include_structure: 是否包含结构信息
            
        Returns:
            Dict: 包含图数据的字典
        """
        seq_len = len(sequence)
        
        # 基础序列边
        edge_indices = []
        edge_types = []
        edge_distances = []
        edge_weights = []
        
        # 1. 序列连接边
        for i in range(seq_len - 1):
            edge_indices.extend([[i, i + 1], [i + 1, i]])
            edge_types.extend([0, 0])
            edge_distances.extend([1, 1])
            edge_weights.extend([1.0, 1.0])
        
        # 2. 相似性边
        similarity_threshold = 0.7
        for i in range(seq_len):
            for j in range(i + 2, min(i + 6, seq_len)):  # 跳过相邻的
                aa_i, aa_j = sequence[i], sequence[j]
                similarity = self.aa_similarity[aa_i][aa_j]
                
                if similarity > similarity_threshold:
                    edge_indices.extend([[i, j], [j, i]])
                    edge_types.extend([1, 1])  # 相似性边
                    edge_distances.extend([j - i, j - i])
                    edge_weights.extend([similarity, similarity])
        
        # 3. 功能边（基于物理化学性质）
        if include_structure:
            functional_edges = self._build_functional_edges(sequence)
            for src, dst, weight in functional_edges:
                edge_indices.extend([[src, dst], [dst, src]])
                edge_types.extend([2, 2])  # 功能边
                edge_distances.extend([abs(dst - src), abs(dst - src)])
                edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        edge_distance_tensor = torch.tensor(edge_distances, dtype=torch.long)
        edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
        
        # 增强的边特征
        edge_attr = self._create_enhanced_edge_features(
            edge_type_tensor, edge_distance_tensor, edge_weight_tensor, sequence, env_vars
        )
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'edge_types': edge_type_tensor,
            'edge_distances': edge_distance_tensor,
            'edge_weights': edge_weight_tensor
        }
    
    def _build_functional_edges(self, sequence: str) -> List[Tuple[int, int, float]]:
        """构建功能边"""
        functional_edges = []
        seq_len = len(sequence)
        
        # 寻找功能相似的氨基酸区域
        window_size = 3
        for i in range(seq_len - window_size):
            window1 = sequence[i:i + window_size]
            
            for j in range(i + window_size + 1, seq_len - window_size):
                window2 = sequence[j:j + window_size]
                
                # 计算窗口间的功能相似性
                similarity = self._compute_functional_similarity(window1, window2)
                
                if similarity > 0.6:
                    functional_edges.append((i + 1, j + 1, similarity))
        
        return functional_edges
    
    def _compute_functional_similarity(self, window1: str, window2: str) -> float:
        """计算功能相似性"""
        similarity = 0.0
        
        for aa1, aa2 in zip(window1, window2):
            # 基于物理化学性质的相似性
            prop1 = self.aa_properties.get(aa1, [0, 0, 0, 0])
            prop2 = self.aa_properties.get(aa2, [0, 0, 0, 0])
            
            # 计算欧几里得距离的倒数
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(prop1, prop2)))
            similarity += 1.0 / (1.0 + dist)
        
        return similarity / len(window1)
    
    def _create_enhanced_edge_features(self,
                                   edge_types: torch.Tensor,
                                   edge_distances: torch.Tensor,
                                   edge_weights: torch.Tensor,
                                   sequence: str,
                                   env_vars: Dict[str, float]) -> torch.Tensor:
        """创建增强的边特征"""
        num_edges = len(edge_types)
        edge_features = []
        
        for i in range(num_edges):
            edge_type = edge_types[i].item()
            distance = edge_distances[i].item()
            weight = edge_weights[i].item()
            
            # 基础特征
            features = [
                float(edge_type),  # 边类型
                float(distance),  # 距离
                float(weight),  # 权重
                1.0 / (1.0 + distance),  # 距离倒数
                math.log(1.0 + distance),  # 对数距离
            ]
            
            # 环境调制特征
            charge_effect = env_vars.get('charge', 0.0) * 0.1
            nce_effect = env_vars.get('nce', 0.0) * 0.01
            fbr_effect = env_vars.get('fbr', 0.0)
            
            features.extend([
                charge_effect * (1.0 + weight),
                nce_effect * (1.0 / (1.0 + distance)),
                fbr_effect * weight,
                charge_effect * nce_effect,
            ])
            
            # 边类型独热编码
            type_one_hot = [0.0] * len(self.edge_types)
            if edge_type < len(self.edge_types):
                type_one_hot[edge_type] = 1.0
            features.extend(type_one_hot)
            
            edge_features.append(features)
        
        return torch.tensor(edge_features, dtype=torch.float32)
    
    def _create_aa_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """创建氨基酸相似性矩阵"""
        # 简化的BLOSUM62相似性矩阵
        similarity = {
            'A': {'A': 4.0, 'R': -1.0, 'N': -2.0, 'D': -2.0, 'C': 0.0, 'Q': -1.0, 'E': -1.0, 'G': 0.0, 'H': -2.0, 'I': -1.0, 'L': -1.0, 'K': -1.0, 'M': -1.0, 'F': -2.0, 'P': -1.0, 'S': 1.0, 'T': 0.0, 'W': -3.0, 'Y': -2.0, 'V': 0.0},
            'R': {'A': -1.0, 'R': 5.0, 'N': 0.0, 'D': -2.0, 'C': -3.0, 'Q': 1.0, 'E': 0.0, 'G': -2.0, 'H': 0.0, 'I': -3.0, 'L': -2.0, 'K': 2.0, 'M': -1.0, 'F': -3.0, 'P': -2.0, 'S': -1.0, 'T': -1.0, 'W': -3.0, 'Y': -2.0, 'V': -3.0},
            'N': {'A': -2.0, 'R': 0.0, 'N': 6.0, 'D': 1.0, 'C': -3.0, 'Q': 0.0, 'E': 0.0, 'G': 0.0, 'H': 1.0, 'I': -3.0, 'L': -3.0, 'K': 0.0, 'M': -2.0, 'F': -3.0, 'P': -2.0, 'S': 1.0, 'T': 0.0, 'W': -4.0, 'Y': -2.0, 'V': -3.0},
            'D': {'A': -2.0, 'R': -2.0, 'N': 1.0, 'D': 6.0, 'C': -3.0, 'Q': 0.0, 'E': 2.0, 'G': -1.0, 'H': -1.0, 'I': -3.0, 'L': -4.0, 'K': -1.0, 'M': -3.0, 'F': -3.0, 'P': -1.0, 'S': 0.0, 'T': -1.0, 'W': -4.0, 'Y': -3.0, 'V': -3.0},
            'C': {'A': 0.0, 'R': -3.0, 'N': -3.0, 'D': -3.0, 'C': 9.0, 'Q': -3.0, 'E': -4.0, 'G': -3.0, 'H': -3.0, 'I': -1.0, 'L': -1.0, 'K': -3.0, 'M': -1.0, 'F': -2.0, 'P': -3.0, 'S': -1.0, 'T': -1.0, 'W': -2.0, 'Y': -2.0, 'V': -1.0},
            'Q': {'A': -1.0, 'R': 1.0, 'N': 0.0, 'D': 0.0, 'C': -3.0, 'Q': 5.0, 'E': 2.0, 'G': -2.0, 'H': 0.0, 'I': -3.0, 'L': -2.0, 'K': 1.0, 'M': 0.0, 'F': -3.0, 'P': -1.0, 'S': 0.0, 'T': -1.0, 'W': -2.0, 'Y': -1.0, 'V': -2.0},
            'E': {'A': -1.0, 'R': 0.0, 'N': 0.0, 'D': 2.0, 'C': -4.0, 'Q': 2.0, 'E': 5.0, 'G': -2.0, 'H': 0.0, 'I': -3.0, 'L': -3.0, 'K': 1.0, 'M': -2.0, 'F': -3.0, 'P': -1.0, 'S': 0.0, 'T': -1.0, 'W': -3.0, 'Y': -2.0, 'V': -2.0},
            'G': {'A': 0.0, 'R': -2.0, 'N': 0.0, 'D': -1.0, 'C': -3.0, 'Q': -2.0, 'E': -2.0, 'G': 6.0, 'H': -2.0, 'I': -4.0, 'L': -4.0, 'K': -2.0, 'M': -3.0, 'F': -3.0, 'P': -2.0, 'S': 0.0, 'T': -2.0, 'W': -2.0, 'Y': -3.0, 'V': -3.0},
            'H': {'A': -2.0, 'R': 0.0, 'N': 1.0, 'D': -1.0, 'C': -3.0, 'Q': 0.0, 'E': 0.0, 'G': -2.0, 'H': 8.0, 'I': -3.0, 'L': -3.0, 'K': -1.0, 'M': -2.0, 'F': -1.0, 'P': -2.0, 'S': -1.0, 'T': -2.0, 'W': -2.0, 'Y': 2.0, 'V': -3.0},
            'I': {'A': -1.0, 'R': -3.0, 'N': -3.0, 'D': -3.0, 'C': -1.0, 'Q': -3.0, 'E': -3.0, 'G': -4.0, 'H': -3.0, 'I': 4.0, 'L': 2.0, 'K': -3.0, 'M': 1.0, 'F': 0.0, 'P': -2.0, 'S': -2.0, 'T': -1.0, 'W': -3.0, 'Y': -1.0, 'V': 3.0},
            'L': {'A': -1.0, 'R': -2.0, 'N': -3.0, 'D': -4.0, 'C': -1.0, 'Q': -2.0, 'E': -3.0, 'G': -4.0, 'H': -3.0, 'I': 2.0, 'L': 4.0, 'K': -2.0, 'M': 2.0, 'F': 0.0, 'P': -3.0, 'S': -2.0, 'T': -1.0, 'W': -2.0, 'Y': -1.0, 'V': 1.0},
            'K': {'A': -1.0, 'R': 2.0, 'N': 0.0, 'D': -1.0, 'C': -3.0, 'Q': 1.0, 'E': 1.0, 'G': -2.0, 'H': -1.0, 'I': -3.0, 'L': -2.0, 'K': 5.0, 'M': -1.0, 'F': -3.0, 'P': -1.0, 'S': 0.0, 'T': -1.0, 'W': -3.0, 'Y': -2.0, 'V': -2.0},
            'M': {'A': -1.0, 'R': -1.0, 'N': -2.0, 'D': -3.0, 'C': -1.0, 'Q': 0.0, 'E': -2.0, 'G': -3.0, 'H': -2.0, 'I': 1.0, 'L': 2.0, 'K': -1.0, 'M': 5.0, 'F': 0.0, 'P': -2.0, 'S': -1.0, 'T': -1.0, 'W': -1.0, 'Y': -1.0, 'V': 1.0},
            'F': {'A': -2.0, 'R': -3.0, 'N': -3.0, 'D': -3.0, 'C': -2.0, 'Q': -3.0, 'E': -3.0, 'G': -3.0, 'H': -1.0, 'I': 0.0, 'L': 0.0, 'K': -3.0, 'M': 0.0, 'F': 6.0, 'P': -4.0, 'S': -2.0, 'T': -2.0, 'W': 1.0, 'Y': 3.0, 'V': -1.0},
            'P': {'A': -1.0, 'R': -2.0, 'N': -2.0, 'D': -1.0, 'C': -3.0, 'Q': -1.0, 'E': -1.0, 'G': -2.0, 'H': -2.0, 'I': -2.0, 'L': -3.0, 'K': -1.0, 'M': -2.0, 'F': -4.0, 'P': 7.0, 'S': -1.0, 'T': -1.0, 'W': -4.0, 'Y': -3.0, 'V': -2.0},
            'S': {'A': 1.0, 'R': -1.0, 'N': 1.0, 'D': 0.0, 'C': -1.0, 'Q': 0.0, 'E': 0.0, 'G': 0.0, 'H': -1.0, 'I': -2.0, 'L': -2.0, 'K': 0.0, 'M': -1.0, 'F': -2.0, 'P': -1.0, 'S': 4.0, 'T': 1.0, 'W': -3.0, 'Y': -2.0, 'V': -2.0},
            'T': {'A': 0.0, 'R': -1.0, 'N': 0.0, 'D': -1.0, 'C': -1.0, 'Q': -1.0, 'E': -1.0, 'G': -2.0, 'H': -2.0, 'I': -1.0, 'L': -1.0, 'K': -1.0, 'M': -1.0, 'F': -2.0, 'P': -1.0, 'S': 1.0, 'T': 5.0, 'W': -2.0, 'Y': -2.0, 'V': 0.0},
            'W': {'A': -3.0, 'R': -3.0, 'N': -4.0, 'D': -4.0, 'C': -2.0, 'Q': -2.0, 'E': -3.0, 'G': -2.0, 'H': -2.0, 'I': -3.0, 'L': -2.0, 'K': -3.0, 'M': -1.0, 'F': 1.0, 'P': -4.0, 'S': -3.0, 'T': -2.0, 'W': 11.0, 'Y': 2.0, 'V': -3.0},
            'Y': {'A': -2.0, 'R': -2.0, 'N': -2.0, 'D': -3.0, 'C': -2.0, 'Q': -1.0, 'E': -2.0, 'G': -3.0, 'H': 2.0, 'I': -1.0, 'L': -1.0, 'K': -2.0, 'M': -1.0, 'F': 3.0, 'P': -3.0, 'S': -2.0, 'T': -2.0, 'W': 2.0, 'Y': 7.0, 'V': -2.0},
            'V': {'A': 0.0, 'R': -3.0, 'N': -3.0, 'D': -3.0, 'C': -1.0, 'Q': -2.0, 'E': -2.0, 'G': -3.0, 'H': -3.0, 'I': 3.0, 'L': 1.0, 'K': -2.0, 'M': 1.0, 'F': -1.0, 'P': -2.0, 'S': -2.0, 'T': 0.0, 'W': -3.0, 'Y': -2.0, 'V': 4.0}
        }
        
        # 归一化相似性分数
        for aa1 in similarity:
            max_score = max(similarity[aa1].values())
            min_score = min(similarity[aa1].values())
            for aa2 in similarity[aa1]:
                if max_score != min_score:
                    similarity[aa1][aa2] = (similarity[aa1][aa2] - min_score) / (max_score - min_score)
        
        return similarity
    
    def _get_aa_properties(self) -> Dict[str, List[float]]:
        """获取氨基酸物理化学性质"""
        return {
            'A': [1.8, 0.0, 0.0, 89.1],  # 疏水性, 电荷, 极性, 分子量
            'R': [-4.5, 1.0, 1.0, 174.2],
            'N': [-3.5, 0.0, 1.0, 132.1],
            'D': [-3.5, -1.0, 1.0, 133.1],
            'C': [2.5, 0.0, 0.0, 121.2],
            'Q': [-3.5, 0.0, 1.0, 146.1],
            'E': [-3.5, -1.0, 1.0, 147.1],
            'G': [-0.4, 0.0, 0.0, 75.1],
            'H': [-3.2, 0.1, 1.0, 155.2],
            'I': [4.5, 0.0, 0.0, 131.2],
            'L': [3.8, 0.0, 0.0, 131.2],
            'K': [-3.9, 1.0, 1.0, 146.2],
            'M': [1.9, 0.0, 0.0, 149.2],
            'F': [2.8, 0.0, 0.0, 165.2],
            'P': [-1.6, 0.0, 0.0, 115.1],
            'S': [-0.8, 0.0, 1.0, 105.1],
            'T': [-0.7, 0.0, 1.0, 119.1],
            'W': [-0.9, 0.0, 0.0, 204.2],
            'Y': [-1.3, 0.0, 1.0, 181.2],
            'V': [4.2, 0.0, 0.0, 117.1]
        }


class KnowledgeGraphBuilder(GraphBuilder):
    """基于知识的图构建器"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
        # 加载蛋白质结构知识库
        self.structure_knowledge = self._load_structure_knowledge()
        self.domain_knowledge = self._load_domain_knowledge()
    
    def _load_structure_knowledge(self) -> Dict[str, Any]:
        """加载结构知识"""
        # 简化的结构知识库
        return {
            'motifs': {
                'CXXC': 'disulfide_bond',
                'HXH': 'metal_binding',
                'GGXGG': 'glycine_rich',
                'PXK': 'kinase_motif'
            },
            'secondary_structures': {
                'helix_favoring': ['A', 'L', 'M', 'Q', 'E', 'K'],
                'strand_favoring': ['V', 'I', 'Y', 'F', 'T', 'W'],
                'turn_favoring': ['G', 'P', 'S', 'D', 'N']
            }
        }
    
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """加载领域知识"""
        return {
            'functional_sites': {
                'active_site': ['H', 'D', 'S', 'C'],
                'binding_site': ['R', 'K', 'D', 'E'],
                'catalytic_triad': ['H', 'D', 'S']
            },
            'conservation_scores': {
                'highly_conserved': ['W', 'C', 'H', 'Y'],
                'moderately_conserved': ['F', 'I', 'L', 'V', 'M'],
                'variable': ['A', 'S', 'T', 'G', 'P']
            }
        }
    
    def build_knowledge_graph(self, 
                           sequence: str,
                           env_vars: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """构建基于知识的图"""
        # 基础图
        base_graph = self.build_advanced_graph(sequence, env_vars)
        
        # 添加知识边
        knowledge_edges = self._extract_knowledge_edges(sequence)
        
        # 合并边
        all_edge_indices = [base_graph['edge_index']]
        all_edge_attrs = [base_graph['edge_attr']]
        all_edge_types = [base_graph['edge_types']]
        all_edge_distances = [base_graph['edge_distances']]
        
        for edge_info in knowledge_edges:
            src, dst, edge_type, weight, features = edge_info
            
            edge_index = torch.tensor([[src, dst]], dtype=torch.long)
            edge_attr = torch.tensor([features], dtype=torch.float32)
            edge_type_tensor = torch.tensor([edge_type], dtype=torch.long)
            edge_distance_tensor = torch.tensor([abs(dst - src)], dtype=torch.long)
            
            all_edge_indices.append(edge_index)
            all_edge_attrs.append(edge_attr)
            all_edge_types.append(edge_type_tensor)
            all_edge_distances.append(edge_distance_tensor)
        
        # 合并所有边
        final_edge_index = torch.cat(all_edge_indices, dim=1)
        final_edge_attr = torch.cat(all_edge_attrs, dim=0)
        final_edge_types = torch.cat(all_edge_types, dim=0)
        final_edge_distances = torch.cat(all_edge_distances, dim=0)
        
        return {
            'edge_index': final_edge_index,
            'edge_attr': final_edge_attr,
            'edge_types': final_edge_types,
            'edge_distances': final_edge_distances
        }
    
    def _extract_knowledge_edges(self, sequence: str) -> List[Tuple]:
        """提取知识边"""
        knowledge_edges = []
        seq_len = len(sequence)
        
        # 1. 基于motif的边
        for motif, motif_type in self.structure_knowledge['motifs'].items():
            positions = self._find_motif_positions(sequence, motif)
            for start_pos in positions:
                end_pos = start_pos + len(motif) - 1
                if end_pos < seq_len:
                    edge_type = 3 + list(self.structure_knowledge['motifs'].keys()).index(motif_type)
                    weight = 0.8
                    features = self._create_knowledge_edge_features(motif_type, weight, sequence[start_pos:end_pos+1])
                    knowledge_edges.append((start_pos, end_pos, edge_type, weight, features))
        
        # 2. 基于二级结构的边
        ss_edges = self._build_secondary_structure_edges(sequence)
        knowledge_edges.extend(ss_edges)
        
        # 3. 基于功能位点的边
        functional_edges = self._build_functional_site_edges(sequence)
        knowledge_edges.extend(functional_edges)
        
        return knowledge_edges
    
    def _find_motif_positions(self, sequence: str, motif: str) -> List[int]:
        """查找motif位置"""
        positions = []
        motif_len = len(motif)
        
        for i in range(len(sequence) - motif_len + 1):
            match = True
            for j, aa in enumerate(motif):
                if aa == 'X':  # 通配符
                    continue
                if sequence[i + j] != aa:
                    match = False
                    break
            if match:
                positions.append(i)
        
        return positions
    
    def _build_secondary_structure_edges(self, sequence: str) -> List[Tuple]:
        """构建二级结构边"""
        edges = []
        
        # 简化的二级结构预测
        helix_regions = self._predict_helices(sequence)
        strand_regions = self._predict_strands(sequence)
        
        for region in helix_regions:
            start, end = region
            for i in range(start, end):
                for j in range(i + 3, min(end + 1, i + 8)):  # 螺旋内i,i+3,i+4连接
                    edge_type = 7  # 螺旋边
                    weight = 0.6
                    features = self._create_knowledge_edge_features('helix', weight, sequence[i:j+1])
                    edges.append((i, j, edge_type, weight, features))
        
        for region in strand_regions:
            start, end = region
            for i in range(start, end):
                for j in range(start, end):
                    if abs(i - j) >= 2:  # 折叠连接
                        edge_type = 8  # 折叠边
                        weight = 0.5
                        features = self._create_knowledge_edge_features('strand', weight, sequence[i:j+1])
                        edges.append((i, j, edge_type, weight, features))
        
        return edges
    
    def _predict_helices(self, sequence: str) -> List[Tuple[int, int]]:
        """预测α螺旋区域"""
        helices = []
        helix_favoring = set(self.structure_knowledge['secondary_structures']['helix_favoring'])
        
        i = 0
        while i < len(sequence) - 5:
            # 检查连续的螺旋倾向氨基酸
            helix_score = 0
            for j in range(i, min(i + 6, len(sequence))):
                if sequence[j] in helix_favoring:
                    helix_score += 1
            
            if helix_score >= 4:  # 至少4个螺旋倾向氨基酸
                start = i
                # 扩展螺旋区域
                while i < len(sequence) and sequence[i] in helix_favoring:
                    i += 1
                end = i - 1
                if end - start >= 5:  # 至少5个残基
                    helices.append((start, end))
            else:
                i += 1
        
        return helices
    
    def _predict_strands(self, sequence: str) -> List[Tuple[int, int]]:
        """预测β折叠区域"""
        strands = []
        strand_favoring = set(self.structure_knowledge['secondary_structures']['strand_favoring'])
        
        i = 0
        while i < len(sequence) - 3:
            strand_score = 0
            for j in range(i, min(i + 4, len(sequence))):
                if sequence[j] in strand_favoring:
                    strand_score += 1
            
            if strand_score >= 3:
                start = i
                while i < len(sequence) and sequence[i] in strand_favoring:
                    i += 1
                end = i - 1
                if end - start >= 3:
                    strands.append((start, end))
            else:
                i += 1
        
        return strands
    
    def _build_functional_site_edges(self, sequence: str) -> List[Tuple]:
        """构建功能位点边"""
        edges = []
        
        functional_sites = self.domain_knowledge['functional_sites']
        
        for site_type, residues in functional_sites.items():
            positions = [i for i, aa in enumerate(sequence) if aa in residues]
            
            # 连接相同类型的功能残基
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i+1:]:
                    if pos2 - pos1 <= 10:  # 距离限制
                        edge_type = 9 + list(functional_sites.keys()).index(site_type)
                        weight = 0.7
                        features = self._create_knowledge_edge_features(site_type, weight, sequence[pos1:pos2+1])
                        edges.append((pos1, pos2, edge_type, weight, features))
        
        return edges
    
    def _create_knowledge_edge_features(self, 
                                   edge_type: str,
                                   weight: float,
                                   sequence_segment: str) -> List[float]:
        """创建知识边特征"""
        features = [
            float(edge_type),  # 边类型编码
            weight,  # 权重
            len(sequence_segment),  # 序列段长度
        ]
        
        # 序列段组成特征
        aa_counts = {}
        for aa in sequence_segment:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # 常见氨基酸比例
        common_aas = ['A', 'G', 'L', 'V', 'S', 'E', 'K']
        for aa in common_aas:
            features.append(aa_counts.get(aa, 0) / len(sequence_segment))
        
        return features
