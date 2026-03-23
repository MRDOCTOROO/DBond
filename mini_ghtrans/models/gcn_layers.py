"""
图卷积层实现

本文件包含了图神经网络中的各种图卷积层实现，包括基础GCN层和带残差连接的GCN层。
这些层用于处理蛋白质序列转换得到的图结构数据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphConvLayer(nn.Module):
    """基础图卷积层 (GCN)"""
    
    def __init__(self, config):
        super(GraphConvLayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.dropout = config.dropout
        self.use_edge_features = config.use_edge_features
        
        if self.use_edge_features:
            # 使用边特征的线性变换
            self.linear = nn.Linear(self.input_dim, self.output_dim)
            self.edge_transform = nn.Linear(config.hidden_dim, self.output_dim)
        else:
            # 标准GCN线性变换
            self.linear = nn.Linear(self.input_dim, self.output_dim)
        
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.xavier_uniform_(self.linear.weight)
        if hasattr(self, 'edge_transform'):
            nn.init.xavier_uniform_(self.edge_transform.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim] (可选)
            
        Returns:
            torch.Tensor: 更新后的节点特征 [num_nodes, output_dim]
        """
        # 线性变换
        x = self.linear(x)
        
        # 消息传递
        if self.use_edge_features and edge_attr is not None:
            # 使用边特征的消息传递
            edge_attr = self.edge_transform(edge_attr)
            x = self.propagate_with_edges(x, edge_index, edge_attr)
        else:
            # 标准消息传递
            x = self.propagate(x, edge_index)
        
        # 添加偏置并激活
        x = x + self.bias
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def propagate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """标准消息传递"""
        row, col = edge_index
        
        # 归一化（度归一化）
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, col, torch.ones_like(col, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        messages = x[row] * norm.unsqueeze(1)
        
        # 聚合邻域信息
        out = torch.zeros_like(x)
        out.index_add_(0, col, messages)
        
        return out
    
    def propagate_with_edges(self, x: torch.Tensor, 
                           edge_index: torch.Tensor,
                           edge_attr: torch.Tensor) -> torch.Tensor:
        """使用边特征的消息传递"""
        row, col = edge_index
        
        # 度归一化
        edge_attr = edge_attr.to(dtype=x.dtype)
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        edge_weights = torch.sum(edge_attr, dim=1, keepdim=True).to(dtype=x.dtype)
        deg.index_add_(0, col, torch.abs(edge_weights.squeeze()).to(dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).to(dtype=x.dtype)
        node_messages = (x[row] * edge_weights * norm.unsqueeze(1)).to(dtype=x.dtype)
        
        # 聚合消息
        out = torch.zeros_like(x)
        out.index_add_(0, col, node_messages)
        
        return out


class ResidualGCNLayer(nn.Module):
    """带残差连接的图卷积层"""
    
    def __init__(self, config):
        super(ResidualGCNLayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.dropout = config.dropout
        self.use_edge_features = config.use_edge_features
        
        # 主图卷积层
        self.gcn = GraphConvLayer(config)
        
        # 残差连接的投影层（如果输入输出维度不同）
        if self.input_dim != self.output_dim:
            self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.residual_projection = nn.Identity()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim + self.output_dim, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim] (可选)
            
        Returns:
            torch.Tensor: 更新后的节点特征 [num_nodes, output_dim]
        """
        # 保存残差
        residual = self.residual_projection(x)
        
        # 图卷积
        gcn_out = self.gcn(x, edge_index, edge_attr)
        
        # 门控机制
        gate_input = torch.cat([x, gcn_out], dim=-1)
        gate_values = self.gate(gate_input)
        
        # 残差连接
        output = residual + gate_values * gcn_out
        
        # 层归一化
        output = self.layer_norm(output)
        
        return output


class GraphSAGELayer(nn.Module):
    """GraphSAGE层实现"""
    
    def __init__(self, config):
        super(GraphSAGELayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.dropout = config.dropout
        self.aggregator_type = config.aggregator_type  # 'mean', 'max', 'lstm'
        
        # 聚合器
        if self.aggregator_type == 'mean':
            self.aggregator = MeanAggregator(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'max':
            self.aggregator = MaxAggregator(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'lstm':
            self.aggregator = LSTMAggregator(self.input_dim, self.output_dim)
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")
        
        # 线性变换
        self.linear = nn.Linear(self.input_dim + self.output_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 聚合邻域信息
        agg_features = self.aggregator(x, edge_index)
        
        # 拼接自身特征和聚合特征
        combined = torch.cat([x, agg_features], dim=-1)
        
        # 线性变换
        output = self.linear(combined)
        output = F.relu(output)
        output = self.layer_norm(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output


class MeanAggregator(nn.Module):
    """均值聚合器"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(MeanAggregator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        # 聚合邻域特征
        neighborhood_features = torch.zeros_like(x)
        neighborhood_features.index_add_(0, col, x[row])
        
        # 计算度
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float))
        
        # 均值聚合（避免除零）
        deg = deg.clamp(min=1)
        aggregated = neighborhood_features / deg.unsqueeze(1)
        
        # 线性变换
        output = self.linear(aggregated)
        
        return output


class MaxAggregator(nn.Module):
    """最大值聚合器"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(MaxAggregator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        # 最大值聚合
        aggregated = torch.zeros_like(x)
        
        for i in range(x.size(0)):
            neighbors = row == i
            if neighbors.sum() > 0:
                aggregated[i] = torch.max(x[neighbors], dim=0)[0]
            else:
                aggregated[i] = x[i]  # 如果没有邻居，使用自身特征
        
        output = self.linear(aggregated)
        return output


class LSTMAggregator(nn.Module):
    """LSTM聚合器"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(LSTMAggregator, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        self.linear = nn.Linear(output_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        
        aggregated = torch.zeros(num_nodes, x.size(1), device=x.device)
        
        for i in range(num_nodes):
            neighbors = row == i
            if neighbors.sum() > 0:
                neighbor_features = x[neighbors].unsqueeze(0)
                lstm_out, _ = self.lstm(neighbor_features)
                aggregated[i] = lstm_out[:, -1, :]  # 使用最后一个时间步的输出
            else:
                aggregated[i] = x[i]
        
        output = self.linear(aggregated)
        return output


class GraphDiffusionConvLayer(nn.Module):
    """图扩散卷积层"""
    
    def __init__(self, config):
        super(GraphDiffusionConvLayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.num_steps = config.diffusion_steps
        self.dropout = config.dropout
        
        # 扩散变换
        self.transform = nn.Linear(self.input_dim, self.output_dim)
        
        # 扩散权重
        self.diffusion_weights = nn.Parameter(
            torch.ones(self.num_steps + 1) / (self.num_steps + 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.transform(x)
        
        # 构建归一化的邻接矩阵
        adj = self._build_normalized_adjacency(edge_index, x.size(0))
        
        # 计算扩散序列
        diffused_features = [x]
        current_x = x
        
        for step in range(self.num_steps):
            current_x = torch.matmul(adj, current_x)
            diffused_features.append(current_x)
        
        # 加权聚合
        output = torch.zeros_like(x)
        for i, features in enumerate(diffused_features):
            output += self.diffusion_weights[i] * features
        
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output
    
    def _build_normalized_adjacency(self, edge_index: torch.Tensor, 
                                  num_nodes: int) -> torch.Tensor:
        """构建归一化的邻接矩阵"""
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        
        row, col = edge_index
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # 对称化
        
        # 添加自环
        adj = adj + torch.eye(num_nodes, device=edge_index.device)
        
        # 行归一化
        row_sum = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sum.clamp(min=1)
        
        return adj


class EdgeConditionedConvLayer(nn.Module):
    """边条件卷积层"""
    
    def __init__(self, config):
        super(EdgeConditionedConvLayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.edge_dim = config.edge_dim
        self.dropout = config.dropout
        
        # 核心网络
        self.core_net = nn.Sequential(
            nn.Linear(self.input_dim + self.edge_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(self.input_dim + self.edge_dim, self.output_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        row, col = edge_index
        
        # 消息传递
        messages = []
        for i in range(edge_index.size(1)):
            src, dst = row[i], col[i]
            edge_input = torch.cat([x[src], edge_attr[i]], dim=-1)
            
            # 核心变换
            core_out = self.core_net(edge_input)
            
            # 门控
            gate_out = self.gate_net(edge_input)
            
            # 消息
            message = gate_out * core_out
            messages.append((dst, message))
        
        # 聚合消息
        output = torch.zeros_like(x)
        for dst, message in messages:
            output[dst] += message
        
        # 残差连接
        output = output + x
        output = self.layer_norm(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        return output


class AdaptiveGraphConvLayer(nn.Module):
    """自适应图卷积层"""
    
    def __init__(self, config):
        super(AdaptiveGraphConvLayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        
        # 多头注意力
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.input_dim,
                num_heads=1,
                dropout=self.dropout,
                batch_first=True
            ) for _ in range(self.num_heads)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(
            self.input_dim * self.num_heads,
            self.output_dim
        )
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 构建注意力掩码
        attention_mask = self._build_attention_mask(edge_index, x.size(0))
        
        # 多头注意力
        head_outputs = []
        for attention_head in self.attention_heads:
            # 重塑为序列格式 [batch_size, seq_len, features]
            x_seq = x.unsqueeze(0)
            attn_output, _ = attention_head(
                x_seq, x_seq, x_seq,
                attn_mask=attention_mask
            )
            head_outputs.append(attn_output.squeeze(0))
        
        # 拼接多头输出
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 输出投影
        output = self.output_projection(concatenated)
        output = self.layer_norm(output)
        
        return output
    
    def _build_attention_mask(self, edge_index: torch.Tensor, 
                            num_nodes: int) -> torch.Tensor:
        """构建注意力掩码"""
        mask = torch.full((num_nodes, num_nodes), float('-inf'), 
                         device=edge_index.device)
        
        row, col = edge_index
        mask[row, col] = 0.0  # 允许已连接的节点互相注意
        mask[col, row] = 0.0  # 对称化
        
        # 允许自身注意
        mask = mask.fill_diagonal_(0.0)
        
        return mask
