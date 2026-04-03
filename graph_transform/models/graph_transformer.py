"""
主要的图神经网络模型实现

本文件包含了GraphTransformer模型的完整实现，用于蛋白质序列的键级别二分类。
模型将蛋白质序列转换为图结构，然后使用图神经网络进行特征学习和预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Tuple
import numpy as np
import time

from .gcn_layers import GraphConvLayer, ResidualGCNLayer
from .attention_layers import GraphAttentionLayer, GlobalAttentionPool


def _maybe_sync(device: torch.device, enabled: bool) -> None:
    """仅在 profiling 时同步 CUDA，避免常规训练开销。"""
    if enabled and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class NodeEncoder(nn.Module):
    """节点编码器：将氨基酸残基编码为特征向量"""
    
    def __init__(self, config):
        super(NodeEncoder, self).__init__()
        
        # 氨基酸字母表
        self.alphabet = config.alphabet
        self.vocab_size = len(self.alphabet) + 1  # +1 for padding
        self.pad_char = config.pad_char
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.alphabet)}
        self.char_to_idx[self.pad_char] = 0
        
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
        self.env_feature_names = list(getattr(config, 'env_feature_names', [getattr(config, 'env_feature_name', 'rt')]))
        self.env_feature_scales = dict(getattr(config, 'env_feature_scales', {'nce': 0.01, self.env_feature_names[0]: getattr(config, 'env_feature_scale', 0.01)}))
        self.env_feature_name = getattr(config, 'env_feature_name', 'rt')
        self.env_feature_scale = float(getattr(config, 'env_feature_scale', 0.01))
        
        # 状态变量编码
        self.state_encoder = nn.Sequential(
            nn.Linear(config.num_state_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
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
            + 32
        )
        
        # 最终编码层
        self.node_encoder = nn.Sequential(
            nn.Linear(self.output_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.register_buffer(
            'ascii_lookup',
            self._build_ascii_lookup_table(),
            persistent=False
        )
        self.register_buffer(
            'physicochemical_lookup',
            self._build_physicochemical_lookup_table(config.num_physicochemical_features),
            persistent=False
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
        physico_features = self._encode_physicochemical(seq_tokens)
        physico_features = self.physicochemical_encoder(physico_features)
        
        # 状态变量与环境变量编码
        state_features = self._encode_state(batch_data, device)
        state_features = self.state_encoder(state_features)

        env_features = self._encode_environmental(batch_data, device)
        env_features = self.env_encoder(env_features)
        
        # 扩展样本级特征到每个位置
        state_features = state_features.unsqueeze(1).expand(-1, seq_tokens.size(1), -1)
        env_features = env_features.unsqueeze(1).expand(-1, seq_tokens.size(1), -1)
        
        # 拼接所有特征
        combined_features = torch.cat([
            aa_features, pos_features, physico_features, state_features, env_features
        ], dim=-1)
        
        # 最终编码
        node_features = self.node_encoder(combined_features)
        
        return node_features
    
    def _encode_sequences(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        """将氨基酸序列编码为token序列"""
        encoded_sequences = []
        lookup = self.ascii_lookup.to(device=device)

        for seq in sequences:
            try:
                seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8).copy()
            except UnicodeEncodeError as exc:
                raise ValueError(f"Sequence contains non-ASCII amino acid symbols: {seq}") from exc

            if seq_bytes.size == 0:
                encoded_sequences.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            byte_tensor = torch.as_tensor(seq_bytes, device=device, dtype=torch.long)
            encoded = lookup[byte_tensor]
            invalid_mask = encoded < 0
            if invalid_mask.any():
                invalid_pos = int(invalid_mask.nonzero(as_tuple=False)[0].item())
                raise ValueError(f"Unknown amino acid: {seq[invalid_pos]}")
            encoded_sequences.append(encoded)

        return pad_sequence(encoded_sequences, batch_first=True, padding_value=0)
    
    def _encode_physicochemical(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """编码物理化学性质"""
        return F.embedding(seq_tokens, self.physicochemical_lookup)
    
    def _encode_state(self, batch_data: Dict, device: torch.device) -> torch.Tensor:
        """编码状态变量"""
        if 'state_vars' in batch_data:
            state_vars = batch_data['state_vars'].to(device=device, dtype=torch.float32)
        else:
            state_vars = torch.stack(
                [
                    batch_data['charges'],
                    batch_data['pep_masses'],
                    batch_data['intensities'],
                ],
                dim=1,
            ).to(device=device, dtype=torch.float32)

        # 与 edge_attr 使用同尺度归一化，避免 intensity 等大数值在 AMP 下放大为 NaN。
        charge = state_vars[:, 0] * 0.1
        pep_mass = state_vars[:, 1] / 2000.0
        intensity = torch.log1p(torch.clamp_min(state_vars[:, 2], 0.0)) / 20.0
        normalized = torch.stack([charge, pep_mass, intensity], dim=1)
        return torch.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)

    def _encode_environmental(self, batch_data: Dict, device: torch.device) -> torch.Tensor:
        """编码环境变量"""
        if 'env_vars' in batch_data:
            env_vars = batch_data['env_vars'].to(device=device, dtype=torch.float32)
        else:
            env_columns = [batch_data['nces']]
            for feature_name in self.env_feature_names:
                feature_key = f'{feature_name}s'
                if feature_key in batch_data:
                    env_columns.append(batch_data[feature_key])
                elif feature_name == 'rt' and 'rts' in batch_data:
                    env_columns.append(batch_data['rts'])
                elif feature_name == 'scan_num' and 'scan_nums' in batch_data:
                    env_columns.append(batch_data['scan_nums'])
                else:
                    raise KeyError(f"Missing batch data for environment feature: {feature_name}")
            env_vars = torch.stack(env_columns, dim=1).to(device=device, dtype=torch.float32)

        normalized_features = [env_vars[:, 0] * float(self.env_feature_scales.get('nce', 0.01))]
        for feature_idx, feature_name in enumerate(self.env_feature_names, start=1):
            scale = float(self.env_feature_scales.get(feature_name, 1.0))
            normalized_features.append(env_vars[:, feature_idx] * scale)
        normalized = torch.stack(normalized_features, dim=1)
        return torch.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)

    def _get_batch_device(self, batch_data: Dict) -> torch.device:
        """从批次数据中推断设备"""
        for value in batch_data.values():
            if torch.is_tensor(value):
                return value.device
        return self.aa_embedding.weight.device

    def _build_ascii_lookup_table(self) -> torch.Tensor:
        """构建 ASCII 到词表索引的查找表。"""
        lookup = torch.full((256,), -1, dtype=torch.long)
        for char, idx in self.char_to_idx.items():
            encoded = char.encode('ascii')
            if len(encoded) != 1:
                raise ValueError(f"Only single-byte ASCII amino acid symbols are supported: {char}")
            lookup[encoded[0]] = idx
        return lookup

    def _build_physicochemical_lookup_table(self, num_features: int) -> torch.Tensor:
        """构建 token 到物化属性的查找表。"""
        table = torch.zeros(self.vocab_size, num_features, dtype=torch.float32)
        for aa, props in self._get_aa_properties().items():
            idx = self.char_to_idx.get(aa)
            if idx is not None:
                table[idx] = torch.tensor(props, dtype=torch.float32)
        return table
    
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
    """保留的通用预测头，当前主路径未使用。"""
    
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
        
        # 通用逐位置预测
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
            state_out_dim = self.node_encoder.state_encoder[-1].out_features
            env_out_dim = self.node_encoder.env_encoder[-1].out_features
            self.global_node_embedding = nn.Parameter(torch.zeros(1, config.hidden_dim))
            self.global_node_proj = nn.Sequential(
                nn.Linear(state_out_dim + env_out_dim, config.hidden_dim),
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

        self.enable_timing = False
        self.last_forward_timing = {}
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(module.weight, UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None and not isinstance(module.bias, UninitializedParameter):
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, batch_data: Dict) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch_data: 包含序列、图结构和特征的批次数据
            
        Returns:
            torch.Tensor: 键级别断裂预测结果 [batch_size, max_bonds]
        """
        timing_enabled = bool(getattr(self, 'enable_timing', False))
        device = self._infer_batch_device(batch_data)
        if timing_enabled:
            _maybe_sync(device, True)
            total_start = time.perf_counter()

        # 节点特征编码
        node_features = self.node_encoder(batch_data)
        if timing_enabled:
            _maybe_sync(node_features.device, True)
            node_encode_end = time.perf_counter()
        global_nodes = None
        if self.use_global_node:
            state_raw = self.node_encoder._encode_state(batch_data, node_features.device)
            state_embed = self.node_encoder.state_encoder(state_raw)
            env_raw = self.node_encoder._encode_environmental(batch_data, node_features.device)
            env_embed = self.node_encoder.env_encoder(env_raw)
            global_context = torch.cat([state_embed, env_embed], dim=-1)
            global_nodes = self.global_node_embedding + self.global_node_proj(global_context)
        if timing_enabled:
            _maybe_sync(node_features.device, True)
            global_node_end = time.perf_counter()
        
        # 边特征编码
        edge_features = self.edge_encoder(
            batch_data['edge_index'],
            batch_data['edge_types'],
            batch_data['edge_distances'],
            batch_data.get('edge_attr')
        )
        if timing_enabled:
            _maybe_sync(node_features.device, True)
            edge_encode_end = time.perf_counter()
        
        # 根据真实序列长度裁剪节点特征
        seq_lens_tensor = batch_data['seq_lens'].to(device=node_features.device, dtype=torch.long)
        node_lens_tensor = batch_data.get('node_lens')
        if node_lens_tensor is not None:
            node_lens_tensor = node_lens_tensor.to(device=node_features.device, dtype=torch.long)
        else:
            node_lens_tensor = seq_lens_tensor + (1 if self.use_global_node else 0)
        seq_lens = seq_lens_tensor.tolist()
        node_lens = node_lens_tensor.tolist()
        batch_size = len(seq_lens)
        hidden_dim = node_features.size(-1)
        max_seq_len = node_features.size(1)
        seq_positions = torch.arange(max_seq_len, device=node_features.device)
        valid_seq_mask = seq_positions.unsqueeze(0) < seq_lens_tensor.unsqueeze(1)

        if self.use_global_node:
            max_node_len = int(node_lens_tensor.max().item()) if batch_size > 0 else 0
            packed_nodes = node_features.new_zeros((batch_size, max_node_len, hidden_dim))
            if max_seq_len > 0:
                packed_nodes[:, :max_seq_len] = node_features[:, :max_seq_len]
            global_positions = seq_lens_tensor.view(-1, 1, 1).expand(-1, 1, hidden_dim)
            packed_nodes.scatter_(1, global_positions, global_nodes.unsqueeze(1))
            node_positions = torch.arange(max_node_len, device=node_features.device)
            valid_node_mask = node_positions.unsqueeze(0) < node_lens_tensor.unsqueeze(1)
            node_features = packed_nodes[valid_node_mask]
            batch_indices = torch.repeat_interleave(
                torch.arange(batch_size, device=node_features.device),
                node_lens_tensor,
                output_size=int(node_lens_tensor.sum().item()),
            )
        else:
            node_features = node_features[valid_seq_mask]
            batch_indices = torch.repeat_interleave(
                torch.arange(batch_size, device=node_features.device),
                seq_lens_tensor,
                output_size=int(seq_lens_tensor.sum().item()),
            )
        if timing_enabled:
            _maybe_sync(node_features.device, True)
            trim_end = time.perf_counter()
        
        # 图卷积层
        gcn_timings = {}
        for i, gcn_layer in enumerate(self.gcn_layers):
            layer_start = time.perf_counter() if timing_enabled else None
            residual = node_features
            node_features = gcn_layer(node_features, batch_data['edge_index'], edge_features)
            node_features = self.layer_norms[i](node_features + residual)
            node_features = F.dropout(node_features, p=self.config.dropout, training=self.training)
            if timing_enabled:
                _maybe_sync(node_features.device, True)
                gcn_timings[f'gcn_layer_{i}'] = time.perf_counter() - layer_start
        if timing_enabled:
            gcn_end = time.perf_counter()
        
        # 图注意力层
        gat_timings = {}
        for i, gat_layer in enumerate(self.gat_layers):
            if hasattr(gat_layer, 'enable_timing'):
                gat_layer.enable_timing = timing_enabled
            layer_start = time.perf_counter() if timing_enabled else None
            residual = node_features
            node_features = gat_layer(node_features, batch_data['edge_index'])
            node_features = self.layer_norms[len(self.gcn_layers) + i](node_features + residual)
            node_features = F.dropout(node_features, p=self.config.dropout, training=self.training)
            if timing_enabled:
                _maybe_sync(node_features.device, True)
                gat_timings[f'gat_layer_{i}_total'] = time.perf_counter() - layer_start
                layer_timing = getattr(gat_layer, 'last_forward_timing', {})
                for key, value in layer_timing.items():
                    if key in {'num_nodes', 'num_edges'}:
                        continue
                    gat_timings[f'gat_layer_{i}_{key}'] = float(value)
        if timing_enabled:
            gat_end = time.perf_counter()
        
        # 构建相邻键的特征并预测断裂
        bond_counts = torch.clamp(seq_lens_tensor - 1, min=0)
        max_bonds = int(bond_counts.max().item()) if batch_size > 0 else 0
        predictions = torch.zeros(batch_size, max_bonds, device=node_features.device)

        if max_bonds > 0:
            bond_positions = torch.arange(max_bonds, device=node_features.device)
            valid_bond_mask = bond_positions.unsqueeze(0) < bond_counts.unsqueeze(1)
            node_offsets = torch.cumsum(
                torch.cat([node_lens_tensor.new_zeros(1), node_lens_tensor[:-1]], dim=0),
                dim=0,
            )
            bond_src = (node_offsets.unsqueeze(1) + bond_positions.unsqueeze(0))[valid_bond_mask]
            bond_dst = bond_src + 1
            bond_features = torch.cat(
                [node_features[bond_src], node_features[bond_dst]], dim=-1
            )
            bond_logits_flat = self.bond_head(bond_features).squeeze(-1)
            predictions[valid_bond_mask] = bond_logits_flat
        else:
            bond_logits_flat = torch.empty(0, device=node_features.device)

        if timing_enabled:
            _maybe_sync(node_features.device, True)
            output_end = time.perf_counter()
            self.last_forward_timing = {
                'node_encode': node_encode_end - total_start,
                'global_node': global_node_end - node_encode_end,
                'edge_encode': edge_encode_end - global_node_end,
                'trim_nodes': trim_end - edge_encode_end,
                'gcn_total': gcn_end - trim_end,
                'gat_total': gat_end - gcn_end,
                'bond_predict_restore': output_end - gat_end,
                'forward_total': output_end - total_start,
                'batch_size': float(batch_size),
                'total_nodes': float(node_features.size(0)),
                'total_edges': float(batch_data['edge_index'].size(1)),
            }
            self.last_forward_timing.update(gcn_timings)
            self.last_forward_timing.update(gat_timings)
        else:
            self.last_forward_timing = {}

        return predictions

    def _infer_batch_device(self, batch_data: Dict) -> torch.device:
        """从 batch 中推断当前设备。"""
        for value in batch_data.values():
            if torch.is_tensor(value):
                return value.device
        return self.global_node_embedding.device if self.use_global_node else next(self.parameters()).device
    
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
