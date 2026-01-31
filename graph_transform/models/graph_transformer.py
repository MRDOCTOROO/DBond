"""
主要的图神经网络模型实现

本文件包含了GraphTransformer模型的完整实现，用于蛋白质序列的多标签分类。
模型将蛋白质序列转换为图结构，然后使用图神经网络进行特征学习和预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .gcn_layers import GraphConvLayer, ResidualGCNLayer
from .attention_layers import GraphAttentionLayer, GlobalAttentionPool


class NodeEncoder(nn.Module):
    """节点编码器：将氨基酸残基编码为特征向量"""
    
    def __init__(self, config):
        super(NodeEncoder, self).__init__()
        
        # 氨基酸字母表
        self.alphabet = config.alphabet
        self.vocab_size = len(self.alphabet) + 1  # +1 for padding
        self.pad_char = config.pad_char
        
        # 基础特征维度
        self.aa_embedding_dim = config.aa_embedding_dim
        self.position_embedding_dim = config.position_embedding_dim
        self.physicochemical_dim = config.physicochemical_dim
        
        # 嵌入层
        self.aa_embedding = nn.Embedding(
            self.vocab_size, 
            self.aa_embedding_dim,
            padding_idx=0
        )
        
        # 位置编码
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            self.position_embedding_dim
        )
        
        # 物理化学性质编码
        self.physicochemical_encoder = nn.Linear(
            config.num_physicochemical_features,
            self.physicochemical_dim
        )
        
        # 环境变量编码
        self.env_encoder = nn.Sequential(
            nn.Linear(config.num_env_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 输出维度
        self.output_dim = (
            self.aa_embedding_dim + 
            self.position_embedding_dim + 
            self.physicochemical_dim + 32
        )
        
        # 最终编码层
        self.node_encoder = nn.Sequential(
            nn.Linear(self.output_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.aa_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
    
    def forward(self, batch_data: Dict) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch_data: 包含序列和特征的批次数据
            
        Returns:
            torch.Tensor: 节点特征张量 [batch_size, seq_len, hidden_dim]
        """
        device = self._get_batch_device(batch_data)
        # 氨基酸序列编码
        seq_tokens = self._encode_sequences(batch_data['sequences'], device)
        aa_features = self.aa_embedding(seq_tokens)
        
        # 位置编码
        positions = torch.arange(
            seq_tokens.size(1), 
            device=seq_tokens.device
        ).unsqueeze(0).expand(seq_tokens.size(0), -1)
        pos_features = self.position_embedding(positions)
        
        # 物理化学性质编码
        physico_features = self._encode_physicochemical(batch_data, device)
        physico_features = self.physicochemical_encoder(physico_features)
        
        # 环境变量编码
        env_features = self._encode_environmental(batch_data, device)
        env_features = self.env_encoder(env_features)
        
        # 扩展环境特征到每个位置
        env_features = env_features.unsqueeze(1).expand(-1, seq_tokens.size(1), -1)
        
        # 拼接所有特征
        combined_features = torch.cat([
            aa_features, pos_features, physico_features, env_features
        ], dim=-1)
        
        # 最终编码
        node_features = self.node_encoder(combined_features)
        
        return node_features
    
    def _encode_sequences(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        """将氨基酸序列编码为token序列"""
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        # 创建字符到索引的映射
        char_to_idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        char_to_idx[self.pad_char] = 0
        
        # 编码序列（与dbond一致：未知字符直接报错）
        encoded_seqs = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq):
                if char not in char_to_idx:
                    raise ValueError(f"Unknown amino acid: {char}")
                encoded_seqs[i, j] = char_to_idx[char]
        
        return encoded_seqs
    
    def _encode_physicochemical(self, batch_data: Dict, device: torch.device) -> torch.Tensor:
        """编码物理化学性质"""
        batch_size = len(batch_data['sequences'])
        max_len = max(len(seq) for seq in batch_data['sequences'])
        
        # 氨基酸物理化学性质表
        aa_properties = self._get_aa_properties()
        
        # 创建物理化学特征张量
        features = torch.zeros(batch_size, max_len, len(aa_properties['A']), device=device)
        
        for i, seq in enumerate(batch_data['sequences']):
            for j, aa in enumerate(seq):
                if aa in aa_properties:
                    features[i, j] = torch.tensor(aa_properties[aa], device=device)
                else:
                    features[i, j] = torch.zeros(len(aa_properties['A']), device=device)
        
        return features
    
    def _encode_environmental(self, batch_data: Dict, device: torch.device) -> torch.Tensor:
        """编码环境变量"""
        return torch.stack(
            [
                batch_data['charges'],
                batch_data['pep_masses'],
                batch_data['nces'],
                batch_data['rts'],
                batch_data['fbrs']
            ],
            dim=1
        ).to(device=device, dtype=torch.float32)

    def _get_batch_device(self, batch_data: Dict) -> torch.device:
        """从批次数据中推断设备"""
        for value in batch_data.values():
            if torch.is_tensor(value):
                return value.device
        return self.aa_embedding.weight.device
    
    def _get_aa_properties(self) -> Dict[str, List[float]]:
        """获取氨基酸物理化学性质"""
        return {
            'A': [1.8, 0.0, 0.0, 89.1],  # 疏水性, 电荷, 极性, 分子量
            'C': [2.5, 0.0, 0.0, 121.2],
            'D': [-3.5, -1.0, 1.0, 133.1],
            'E': [-3.5, -1.0, 1.0, 147.1],
            'F': [2.8, 0.0, 0.0, 165.2],
            'G': [-0.4, 0.0, 0.0, 75.1],
            'H': [-3.2, 0.1, 1.0, 155.2],
            'I': [4.5, 0.0, 0.0, 131.2],
            'K': [-3.9, 1.0, 1.0, 146.2],
            'L': [3.8, 0.0, 0.0, 131.2],
            'M': [1.9, 0.0, 0.0, 149.2],
            'N': [-3.5, 0.0, 1.0, 132.1],
            'P': [-1.6, 0.0, 0.0, 115.1],
            'Q': [-3.5, 0.0, 1.0, 146.1],
            'R': [-4.5, 1.0, 1.0, 174.2],
            'S': [-0.8, 0.0, 1.0, 105.1],
            'T': [-0.7, 0.0, 1.0, 119.1],
            'V': [4.2, 0.0, 0.0, 117.1],
            'W': [-0.9, 0.0, 0.0, 204.2],
            'Y': [-1.3, 0.0, 1.0, 181.2]
        }


class EdgeEncoder(nn.Module):
    """边编码器：编码图中边的特征"""
    
    def __init__(self, config):
        super(EdgeEncoder, self).__init__()
        
        self.edge_types = config.edge_types
        self.edge_embedding_dim = config.edge_embedding_dim
        self.distance_embedding_dim = config.distance_embedding_dim
        
        # 边类型嵌入
        self.edge_type_embedding = nn.Embedding(
            len(self.edge_types),
            self.edge_embedding_dim
        )
        
        # 距离嵌入
        self.distance_embedding = nn.Embedding(
            config.max_distance + 1,
            self.distance_embedding_dim
        )
        
        # 输出维度
        self.output_dim = self.edge_embedding_dim + self.distance_embedding_dim
        
        # 最终编码层
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # 物理环境等原始边特征编码（延迟推断输入维度）
        self.edge_attr_encoder = nn.LazyLinear(config.hidden_dim)
        self.edge_attr_norm = nn.LayerNorm(config.hidden_dim)
        self.edge_attr_fuse = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, edge_indices: torch.Tensor, 
                edge_types: torch.Tensor,
                distances: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            edge_indices: 边索引 [2, num_edges]
            edge_types: 边类型 [num_edges]
            distances: 距离信息 [num_edges]
            edge_attr: 原始边特征 [num_edges, edge_attr_dim] (可选)
            
        Returns:
            torch.Tensor: 边特征 [num_edges, hidden_dim]
        """
        type_features = self.edge_type_embedding(edge_types)
        distance_features = self.distance_embedding(distances)
        
        combined_features = torch.cat([type_features, distance_features], dim=-1)
        edge_features = self.edge_encoder(combined_features)

        if edge_attr is not None:
            attr_features = self.edge_attr_encoder(edge_attr.to(dtype=edge_features.dtype))
            attr_features = self.edge_attr_norm(attr_features)
            edge_features = self.edge_attr_fuse(torch.cat([edge_features, attr_features], dim=-1))
        
        return edge_features


class MultiLabelHead(nn.Module):
    """多标签预测头"""
    
    def __init__(self, config):
        super(MultiLabelHead, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        
        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, node_features: torch.Tensor, 
                batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, hidden_dim] 或 [batch_size, seq_len, hidden_dim]
            batch_indices: 批次索引 [num_nodes]
            
        Returns:
            torch.Tensor: 预测结果 [batch_size, seq_len, num_classes] 或 [batch_size, num_classes]
        """
        if batch_indices is not None:
            # 处理图级别的节点特征 [num_nodes, hidden_dim]
            # 需要重构为序列级别 [batch_size, seq_len, hidden_dim]
            if node_features.dim() == 2:
                # 获取批次大小和序列长度信息
                batch_size = batch_indices.max().item() + 1
                seq_len = self._estimate_seq_len(batch_indices, batch_size)
                
                # 重构为 [batch_size, seq_len, hidden_dim]
                reshaped_features = torch.zeros(batch_size, seq_len, node_features.size(1), 
                                              device=node_features.device)
                
                # 填充特征
                for i in range(batch_size):
                    mask = (batch_indices == i)
                    seq_features = node_features[mask]
                    actual_len = min(seq_features.size(0), seq_len)
                    reshaped_features[i, :actual_len] = seq_features[:actual_len]
                
                node_features = reshaped_features
            
            # 全局池化到序列级别
            pooled_features = self._global_pool_sequence(node_features)
        else:
            # 假设已经是序列级别的特征 [batch_size, seq_len, hidden_dim]
            pooled_features = node_features
        
        # 多标签预测
        predictions = self.mlp(pooled_features)
        
        return predictions
    
    def _estimate_seq_len(self, batch_indices: torch.Tensor, batch_size: int) -> int:
        """估计序列长度"""
        seq_lengths = []
        for i in range(batch_size):
            seq_lengths.append((batch_indices == i).sum().item())
        return max(seq_lengths) if seq_lengths else 1
    
    def _global_pool_sequence(self, node_features: torch.Tensor) -> torch.Tensor:
        """序列级别的全局池化"""
        # 如果已经是 [batch_size, seq_len, hidden_dim] 格式，直接返回
        if node_features.dim() == 3:
            return node_features
        
        # 否则进行全局池化
        return node_features.mean(dim=0, keepdim=True)
    
    def _global_pool(self, node_features: torch.Tensor, 
                    batch_indices: torch.Tensor) -> torch.Tensor:
        """全局池化操作"""
        batch_size = batch_indices.max().item() + 1
        pooled = torch.zeros(batch_size, node_features.size(1), 
                           device=node_features.device)
        
        for i in range(batch_size):
            mask = (batch_indices == i)
            if mask.sum() > 0:
                pooled[i] = node_features[mask].mean(dim=0)
        
        return pooled


class GraphTransformer(nn.Module):
    """主要的图神经网络模型"""
    
    def __init__(self, config):
        super(GraphTransformer, self).__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # 编码器
        self.node_encoder = NodeEncoder(config)
        self.edge_encoder = EdgeEncoder(config)

        self.use_global_node = getattr(config, 'use_global_node', False)
        if self.use_global_node:
            env_out_dim = self.node_encoder.env_encoder[-1].out_features
            self.global_node_embedding = nn.Parameter(torch.zeros(1, config.hidden_dim))
            self.global_node_proj = nn.Sequential(
                nn.Linear(env_out_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList([
            ResidualGCNLayer(config) for _ in range(config.num_gcn_layers)
        ])
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(config) for _ in range(config.num_gat_layers)
        ])
        
        # 全局池化（保留，可能用于图级别任务）
        self.global_pool = GlobalAttentionPool(config)
        
        # 键级别断裂预测头（相邻残基对）
        self.bond_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_gcn_layers + config.num_gat_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, batch_data: Dict) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch_data: 包含序列、图结构和特征的批次数据
            
        Returns:
            torch.Tensor: 键级别断裂预测结果 [batch_size, max_bonds]
        """
        # 节点特征编码
        node_features = self.node_encoder(batch_data)
        global_nodes = None
        if self.use_global_node:
            env_raw = self.node_encoder._encode_environmental(batch_data, node_features.device)
            env_embed = self.node_encoder.env_encoder(env_raw)
            global_nodes = self.global_node_embedding + self.global_node_proj(env_embed)
        
        # 边特征编码
        edge_features = self.edge_encoder(
            batch_data['edge_index'],
            batch_data['edge_types'],
            batch_data['edge_distances'],
            batch_data.get('edge_attr')
        )
        
        # 根据真实序列长度裁剪节点特征
        seq_lens = batch_data['seq_lens'].tolist()
        node_lens = batch_data.get('node_lens')
        if node_lens is not None:
            node_lens = node_lens.tolist()
        else:
            node_lens = [seq_len + (1 if self.use_global_node else 0) for seq_len in seq_lens]
        batch_size = len(seq_lens)
        hidden_dim = node_features.size(-1)

        trimmed_nodes = []
        batch_indices = []
        for i, seq_len in enumerate(seq_lens):
            if seq_len > 0:
                nodes = node_features[i, :seq_len]
            else:
                nodes = node_features.new_empty((0, hidden_dim))
            if self.use_global_node:
                nodes = torch.cat([nodes, global_nodes[i:i + 1]], dim=0)
            if nodes.numel() > 0:
                trimmed_nodes.append(nodes)
                batch_indices.extend([i] * nodes.size(0))

        node_features = torch.cat(trimmed_nodes, dim=0) if trimmed_nodes else node_features.new_empty((0, hidden_dim))
        batch_indices = torch.tensor(batch_indices, device=node_features.device, dtype=torch.long)
        
        # 图卷积层
        for i, gcn_layer in enumerate(self.gcn_layers):
            residual = node_features
            node_features = gcn_layer(node_features, batch_data['edge_index'], edge_features)
            node_features = self.layer_norms[i](node_features + residual)
            node_features = F.dropout(node_features, p=self.config.dropout, training=self.training)
        
        # 图注意力层
        for i, gat_layer in enumerate(self.gat_layers):
            residual = node_features
            node_features = gat_layer(node_features, batch_data['edge_index'])
            node_features = self.layer_norms[len(self.gcn_layers) + i](node_features + residual)
            node_features = F.dropout(node_features, p=self.config.dropout, training=self.training)
        
        # 构建相邻键的特征并预测断裂
        seq_lens = batch_data['seq_lens'].tolist()
        bond_src = []
        bond_dst = []
        offset = 0
        for seq_len, node_len in zip(seq_lens, node_lens):
            for i in range(max(seq_len - 1, 0)):
                bond_src.append(offset + i)
                bond_dst.append(offset + i + 1)
            offset += node_len

        if bond_src:
            bond_src = torch.tensor(bond_src, device=node_features.device)
            bond_dst = torch.tensor(bond_dst, device=node_features.device)
            bond_features = torch.cat(
                [node_features[bond_src], node_features[bond_dst]], dim=-1
            )
            bond_logits_flat = self.bond_head(bond_features).squeeze(-1)
        else:
            bond_logits_flat = torch.empty(0, device=node_features.device)

        # 还原为 [batch_size, max_bonds]
        max_bonds = max(max(seq_lens) - 1, 0) if seq_lens else 0
        predictions = torch.zeros(batch_size, max_bonds, device=node_features.device)
        cursor = 0
        for i, seq_len in enumerate(seq_lens):
            bond_len = max(seq_len - 1, 0)
            if bond_len > 0:
                predictions[i, :bond_len] = bond_logits_flat[cursor:cursor + bond_len]
                cursor += bond_len

        return predictions
    
    def get_attention_weights(self, batch_data: Dict) -> List[torch.Tensor]:
        """获取注意力权重用于可视化"""
        attention_weights = []
        
        # 节点特征编码
        node_features = self.node_encoder(batch_data)
        
        # 边特征编码
        edge_features = self.edge_encoder(
            batch_data['edge_index'],
            batch_data['edge_types'],
            batch_data['edge_distances']
        )
        
        # 重塑节点特征
        batch_size, seq_len, hidden_dim = node_features.shape
        node_features = node_features.view(-1, hidden_dim)
        
        # 通过注意力层收集权重
        for gat_layer in self.gat_layers:
            weights = gat_layer.get_attention_weights(
                node_features, batch_data['edge_index']
            )
            attention_weights.append(weights)
            node_features = gat_layer(node_features, batch_data['edge_index'])
        
        return attention_weights
