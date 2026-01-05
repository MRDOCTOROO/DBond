"""
图注意力层实现

本文件包含了图神经网络中的各种注意力层实现，包括图注意力网络(GAT)层、
多头注意力机制和全局注意力池化等组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphAttentionLayer(nn.Module):
    """图注意力层 (Graph Attention Network)"""
    
    def __init__(self, config):
        super(GraphAttentionLayer, self).__init__()
        
        self.input_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.alpha = config.leaky_relu_slope
        self.concat = config.concat_heads  # 是否拼接多头输出
        
        # 多头注意力参数
        self.head_dim = self.output_dim // self.num_heads
        
        # 每个头的线性变换
        self.W = nn.Parameter(torch.Tensor(self.num_heads, self.input_dim, self.head_dim))
        self.a = nn.Parameter(torch.Tensor(self.num_heads, 2 * self.head_dim, 1))
        
        # 残差连接的投影层
        if self.input_dim != self.output_dim:
            self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.residual_projection = nn.Identity()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim] (可选)
            return_attention: 是否返回注意力权重
            
        Returns:
            torch.Tensor: 更新后的节点特征 [num_nodes, output_dim]
            Optional[torch.Tensor]: 注意力权重 [num_heads, num_edges]
        """
        num_nodes, _ = x.shape
        
        # 多头线性变换
        x_heads = torch.einsum('ni,hij->nhj', x, self.W)  # [num_nodes, num_heads, head_dim]
        
        # 计算注意力权重
        attention_weights = self._compute_attention(x_heads, edge_index, edge_attr)
        
        # 应用注意力权重更新节点特征
        output_heads = self._propagate_with_attention(x_heads, edge_index, attention_weights)
        
        # 拼接或平均多头输出
        if self.concat:
            output = output_heads.reshape(num_nodes, self.output_dim)
        else:
            output = output_heads.mean(dim=1)
        
        # 残差连接
        residual = self.residual_projection(x)
        output = output + residual
        
        # 层归一化和dropout
        output = self.layer_norm(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        
        if return_attention:
            return output, attention_weights
        else:
            return output
    
    def _compute_attention(self, x_heads: torch.Tensor, 
                          edge_index: torch.Tensor,
                          edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算注意力权重"""
        row, col = edge_index
        
        # 获取源节点和目标节点的特征
        x_i = x_heads[row]  # [num_edges, num_heads, head_dim]
        x_j = x_heads[col]  # [num_edges, num_heads, head_dim]
        
        # 拼接源节点和目标节点特征
        x_ij = torch.cat([x_i, x_j], dim=-1)  # [num_edges, num_heads, 2*head_dim]
        
        # 计算注意力分数
        e = torch.einsum('ehd,hdk->ehk', x_ij, self.a).squeeze(-1)  # [num_edges, num_heads]
        
        # 如果有边特征，加上边特征的贡献
        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(1).expand(-1, self.num_heads, -1)
            edge_scores = torch.sum(edge_attr, dim=-1)
            e = e + edge_scores
        
        # 应用LeakyReLU
        e = F.leaky_relu(e, negative_slope=self.alpha)
        
        # 计算softmax注意力权重
        attention_weights = self._softmax_attention(e, row, x_heads.size(0))
        
        return attention_weights
    
    def _softmax_attention(self, e: torch.Tensor, 
                           edge_index_row: torch.Tensor,
                           num_nodes: int) -> torch.Tensor:
        """计算softmax注意力权重"""
        # 初始化注意力权重
        attention = torch.zeros_like(e)
        
        # 对每个目标节点计算softmax
        for node in range(num_nodes):
            mask = (edge_index_row == node)
            if mask.sum() > 0:
                attention[mask] = F.softmax(e[mask], dim=0)
        
        return attention
    
    def _propagate_with_attention(self, x_heads: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  attention_weights: torch.Tensor) -> torch.Tensor:
        """使用注意力权重传播消息"""
        row, col = edge_index
        
        # 加权的源节点特征
        weighted_features = x_heads[row] * attention_weights.unsqueeze(-1)
        
        # 聚合到目标节点
        num_nodes, num_heads, head_dim = x_heads.shape
        output = torch.zeros(num_nodes, num_heads, head_dim, device=x_heads.device)
        
        # 使用scatter_add聚合
        output.index_add_(0, col, weighted_features)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, 
                              edge_index: torch.Tensor,
                              edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取注意力权重用于可视化"""
        with torch.no_grad():
            # 多头线性变换
            x_heads = torch.einsum('ni,hij->nhj', x, self.W)
            
            # 计算注意力权重
            attention_weights = self._compute_attention(x_heads, edge_index, edge_attr)
            
            return attention_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = config.dropout
        
        # 查询、键、值的线性变换
        self.q_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 输出投影
        self.out_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, hidden_dim]
            key: 键张量 [batch_size, seq_len, hidden_dim]
            value: 值张量 [batch_size, seq_len, hidden_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, hidden_dim]
            torch.Tensor: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        # 线性变换
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 重塑并投影输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        output = self.out_linear(attention_output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class GlobalAttentionPool(nn.Module):
    """全局注意力池化层"""
    
    def __init__(self, config):
        super(GlobalAttentionPool, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        
        # 查询向量（可学习的全局查询）
        self.global_query = nn.Parameter(torch.Tensor(1, self.num_heads, 1, self.hidden_dim // self.num_heads))
        
        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        nn.init.xavier_uniform_(self.global_query)
    
    def forward(self, node_features: torch.Tensor, 
                batch_indices: torch.Tensor) -> torch.Tensor:
        """
        全局注意力池化
        
        Args:
            node_features: 节点特征 [num_nodes, hidden_dim]
            batch_indices: 批次索引 [num_nodes]
            
        Returns:
            torch.Tensor: 池化后的图特征 [batch_size, hidden_dim]
        """
        batch_size = batch_indices.max().item() + 1
        
        # 按批次分组节点特征
        pooled_features = torch.zeros(batch_size, self.hidden_dim, device=node_features.device)
        
        for i in range(batch_size):
            # 获取当前批次的节点
            mask = (batch_indices == i)
            if mask.sum() > 0:
                batch_nodes = node_features[mask].unsqueeze(0)  # [1, num_batch_nodes, hidden_dim]
                
                # 全局查询扩展到与节点数量相同
                query = self.global_query.expand(1, self.num_heads, batch_nodes.size(1), -1)
                query = query.reshape(1, batch_nodes.size(1), self.hidden_dim)
                
                # 多头注意力
                attended_features, _ = self.multihead_attention(
                    query, batch_nodes, batch_nodes
                )
                
                # 平均池化
                pooled_features[i] = attended_features.mean(dim=1)
            else:
                pooled_features[i] = torch.zeros(self.hidden_dim, device=node_features.device)
        
        # 输出投影
        output = self.output_projection(pooled_features)
        
        return output


class EdgeAttention(nn.Module):
    """边注意力机制"""
    
    def __init__(self, config):
        super(EdgeAttention, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.edge_dim = getattr(config, 'edge_dim', 32)
        
        # 边特征变换
        self.edge_transform = nn.Linear(self.edge_dim, self.hidden_dim)
        
        # 注意力计算
        self.attention_mlp = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        计算边注意力权重
        
        Args:
            node_features: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim]
            
        Returns:
            torch.Tensor: 边注意力权重 [num_edges, 1]
        """
        row, col = edge_index
        
        # 获取源节点和目标节点特征
        src_features = node_features[row]
        dst_features = node_features[col]
        
        # 边特征变换
        edge_features = self.edge_transform(edge_attr)
        
        # 拼接源节点、边、目标节点特征
        combined = torch.cat([src_features, edge_features, dst_features], dim=-1)
        
        # 计算注意力权重
        attention_weights = self.attention_mlp(combined)
        
        return attention_weights


class HierarchicalAttention(nn.Module):
    """分层注意力机制"""
    
    def __init__(self, config):
        super(HierarchicalAttention, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_hierarchical_layers
        
        # 每层的注意力
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # 层间融合
        self.layer_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_layers, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        分层注意力前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: 输出特征 [batch_size, seq_len, hidden_dim]
        """
        layer_outputs = []
        
        # 逐层处理
        for i, attention_layer in enumerate(self.attention_layers):
            if i == 0:
                # 第一层使用原始输入
                layer_output, _ = attention_layer(x, x, x)
            else:
                # 后续层使用前一层的输出
                layer_output, _ = attention_layer(layer_outputs[-1], layer_outputs[-1], layer_outputs[-1])
            
            layer_outputs.append(layer_output)
        
        # 融合所有层的输出
        concatenated = torch.cat(layer_outputs, dim=-1)
        output = self.layer_fusion(concatenated)
        
        return output


class CrossAttention(nn.Module):
    """交叉注意力机制"""
    
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        交叉注意力前向传播
        
        Args:
            query: 查询张量 [batch_size, query_len, hidden_dim]
            key: 键张量 [batch_size, key_len, hidden_dim]
            value: 值张量 [batch_size, value_len, hidden_dim]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, query_len, hidden_dim]
        """
        # 交叉注意力
        cross_output, _ = self.cross_attention(query, key, value)
        
        # 自注意力
        self_output, _ = self.self_attention(query, query, query)
        
        # 融合输出
        combined = torch.cat([cross_output, self_output], dim=-1)
        output = self.fusion(combined)
        
        return output


class AdaptiveAttention(nn.Module):
    """自适应注意力机制"""
    
    def __init__(self, config):
        super(AdaptiveAttention, self).__init__()
        
        self.hidden_dim = config.hidden_dim
        self.num_attention_types = getattr(config, 'num_attention_types', 3)
        
        # 不同类型的注意力机制
        self.attention_types = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(self.num_attention_types)
        ])
        
        # 自适应权重网络
        self.adaptive_weights = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_attention_types),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自适应注意力前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: 输出特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算自适应权重
        global_context = x.mean(dim=1)  # [batch_size, hidden_dim]
        weights = self.adaptive_weights(global_context)  # [batch_size, num_attention_types]
        
        # 计算不同类型的注意力输出
        attention_outputs = []
        for attention_type in self.attention_types:
            output, _ = attention_type(x, x, x)
            attention_outputs.append(output)
        
        # 加权融合
        weighted_output = torch.zeros_like(x)
        for i, output in enumerate(attention_outputs):
            weight = weights[:, i].unsqueeze(1).unsqueeze(2)
            weighted_output += weight * output
        
        # 输出投影
        output = self.output_projection(weighted_output)
        
        return output
