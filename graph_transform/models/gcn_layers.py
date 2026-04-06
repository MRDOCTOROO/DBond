"""
图卷积层实现

本文件包含了图神经网络中的各种图卷积层实现，包括基础GCN层和带残差连接的GCN层。
这些层用于处理蛋白质序列转换得到的图结构数据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_compute = x.to(compute_dtype)
        
        # 归一化（度归一化）
        deg = torch.zeros(x.size(0), device=x.device, dtype=compute_dtype)
        deg.index_add_(0, col, torch.ones_like(col, dtype=compute_dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        messages = x_compute[row] * norm.unsqueeze(1)
        
        # 聚合邻域信息
        out = torch.zeros_like(x_compute)
        out.index_add_(0, col, messages)
        
        return out.to(dtype=x.dtype)
    
    def propagate_with_edges(self, x: torch.Tensor, 
                           edge_index: torch.Tensor,
                           edge_attr: torch.Tensor) -> torch.Tensor:
        """使用边特征的消息传递"""
        row, col = edge_index
        compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x_compute = x.to(compute_dtype)
        edge_attr = edge_attr.to(dtype=compute_dtype)
        
        # 度归一化
        deg = torch.zeros(x.size(0), device=x.device, dtype=compute_dtype)

        # 原实现直接 sum(hidden_dim) 作为边权，训练后期容易数值爆炸。
        # 这里改成有界的 sigmoid(mean) 权重，让边特征只做门控，不做无界放大。
        edge_logits = edge_attr.mean(dim=1, keepdim=True)
        edge_weights = torch.sigmoid(edge_logits).clamp(1e-4, 1.0)
        deg.index_add_(0, col, edge_weights.squeeze())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        node_messages = x_compute[row] * edge_weights * norm.unsqueeze(1)
        
        # 聚合消息
        out = torch.zeros_like(x_compute)
        out.index_add_(0, col, node_messages)
        
        return out.to(dtype=x.dtype)


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

