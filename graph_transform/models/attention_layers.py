"""
图注意力层实现

本文件包含了图神经网络中的各种注意力层实现，包括图注意力网络(GAT)层、
多头注意力机制和全局注意力池化等组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import time


def _maybe_sync(device: torch.device, enabled: bool) -> None:
    """仅在 profiling 时同步 CUDA，避免常规训练引入额外开销。"""
    if enabled and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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
        self.use_edge_bias = getattr(config, 'gat_use_edge_bias', True)
        self.use_edge_gate = getattr(config, 'gat_use_edge_gate', True)
        
        # 多头注意力参数
        self.head_dim = self.output_dim // self.num_heads
        
        # 每个头的线性变换
        self.W = nn.Parameter(torch.Tensor(self.num_heads, self.input_dim, self.head_dim))
        self.a = nn.Parameter(torch.Tensor(self.num_heads, 2 * self.head_dim, 1))
        self.edge_attention_bias = nn.Linear(self.input_dim, self.num_heads)
        self.edge_message_gate = nn.Linear(self.input_dim, self.num_heads * self.head_dim)
        
        # 残差连接的投影层
        if self.input_dim != self.output_dim:
            self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.residual_projection = nn.Identity()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)

        # 可选的前向耗时分析
        self.enable_timing = False
        self.last_forward_timing = {}
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.edge_attention_bias.weight)
        nn.init.zeros_(self.edge_attention_bias.bias)
        nn.init.xavier_uniform_(self.edge_message_gate.weight)
        nn.init.zeros_(self.edge_message_gate.bias)
    
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
        timing_enabled = bool(getattr(self, 'enable_timing', False))
        total_start = time.perf_counter() if timing_enabled else None
        if timing_enabled:
            _maybe_sync(x.device, True)
        
        # 多头线性变换
        x_heads = torch.einsum('ni,hij->nhj', x, self.W)  # [num_nodes, num_heads, head_dim]
        row, col = edge_index
        src_heads = x_heads[row]
        dst_heads = x_heads[col]
        if timing_enabled:
            _maybe_sync(x.device, True)
            project_end = time.perf_counter()
        
        edge_bias, edge_gate = self._compute_edge_modulation(
            edge_attr,
            src_heads.dtype,
            src_heads.device,
        )

        # 计算注意力权重
        attention_weights = self._compute_attention(
            src_heads,
            dst_heads,
            col,
            edge_bias,
            num_nodes,
        )
        if timing_enabled:
            _maybe_sync(x.device, True)
            attention_end = time.perf_counter()
        
        # 应用注意力权重更新节点特征
        output_heads = self._propagate_with_attention(
            src_heads,
            col,
            attention_weights,
            num_nodes,
            edge_gate=edge_gate,
        )
        if timing_enabled:
            _maybe_sync(x.device, True)
            propagate_end = time.perf_counter()
        
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

        if timing_enabled:
            _maybe_sync(x.device, True)
            output_end = time.perf_counter()
            self.last_forward_timing = {
                'project': project_end - total_start,
                'attention': attention_end - project_end,
                'propagate': propagate_end - attention_end,
                'output': output_end - propagate_end,
                'total': output_end - total_start,
                'num_nodes': float(num_nodes),
                'num_edges': float(edge_index.size(1)),
            }
        else:
            self.last_forward_timing = {}
        
        if return_attention:
            return output, attention_weights
        else:
            return output
    
    def _compute_attention(self,
                           src_heads: torch.Tensor,
                           dst_heads: torch.Tensor,
                           target_index: torch.Tensor,
                           edge_bias: Optional[torch.Tensor],
                           num_nodes: int) -> torch.Tensor:
        """计算注意力权重"""
        if src_heads.numel() == 0:
            return src_heads.new_empty((0, self.num_heads))

        compute_dtype = torch.float32 if src_heads.dtype in (torch.float16, torch.bfloat16) else src_heads.dtype
        src_heads_compute = src_heads.to(compute_dtype)
        dst_heads_compute = dst_heads.to(compute_dtype)

        # 将 attention 向量拆成 source/target 两半，避免 cat + einsum 的额外开销。
        attention_vector = self.a.squeeze(-1).to(compute_dtype)
        a_src, a_dst = attention_vector.split(self.head_dim, dim=1)
        e = (src_heads_compute * a_src.unsqueeze(0)).sum(dim=-1)
        e = e + (dst_heads_compute * a_dst.unsqueeze(0)).sum(dim=-1)
        
        if edge_bias is not None:
            e = e + edge_bias.to(compute_dtype)
        
        # 应用LeakyReLU
        e = F.leaky_relu(e, negative_slope=self.alpha)
        
        # 计算softmax注意力权重
        attention_weights = self._softmax_attention(e, target_index, num_nodes).to(dtype=src_heads.dtype)
        
        return attention_weights

    def _compute_edge_modulation(self,
                                 edge_attr: Optional[torch.Tensor],
                                 dtype: torch.dtype,
                                 device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """将边特征映射到每个注意力头的 bias 和 message gate。"""
        if edge_attr is None:
            return None, None

        edge_attr = edge_attr.to(device=device)
        compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        edge_attr_compute = edge_attr.to(compute_dtype)

        edge_bias = None
        edge_gate = None
        if self.use_edge_bias:
            edge_bias = self.edge_attention_bias(edge_attr_compute)
        if self.use_edge_gate:
            edge_gate = torch.sigmoid(
                self.edge_message_gate(edge_attr_compute).view(-1, self.num_heads, self.head_dim)
            )
        return edge_bias, edge_gate
    
    def _softmax_attention(self, e: torch.Tensor, 
                           target_index: torch.Tensor,
                           num_nodes: int) -> torch.Tensor:
        """按目标节点分组计算softmax注意力权重。"""
        if e.numel() == 0:
            return e

        num_heads = e.size(1)
        expanded_index = target_index.unsqueeze(-1).expand(-1, num_heads)

        # 先按目标节点求每个head的最大值，避免softmax上溢。
        max_per_node = torch.full(
            (num_nodes, num_heads),
            float("-inf"),
            device=e.device,
            dtype=e.dtype,
        )
        max_per_node.scatter_reduce_(0, expanded_index, e, reduce="amax", include_self=True)
        stabilized = e - max_per_node[target_index]
        exp_scores = stabilized.exp()

        sum_per_node = torch.zeros((num_nodes, num_heads), device=e.device, dtype=e.dtype)
        sum_per_node.scatter_add_(0, expanded_index, exp_scores)

        return exp_scores / sum_per_node[target_index].clamp_min(1e-12)
    
    def _propagate_with_attention(self,
                                  src_heads: torch.Tensor,
                                  target_index: torch.Tensor,
                                  attention_weights: torch.Tensor,
                                  num_nodes: int,
                                  edge_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """使用注意力权重传播消息"""
        if edge_gate is not None:
            src_heads = src_heads * edge_gate.to(dtype=src_heads.dtype)

        # 加权的源节点特征
        weighted_features = src_heads * attention_weights.unsqueeze(-1)
        
        # 聚合到目标节点
        num_heads, head_dim = weighted_features.size(1), weighted_features.size(2)
        output = torch.zeros(num_nodes, num_heads * head_dim, device=weighted_features.device, dtype=weighted_features.dtype)
        output.index_add_(0, target_index, weighted_features.reshape(weighted_features.size(0), -1))
        
        return output.view(num_nodes, num_heads, head_dim)
    
    def get_attention_weights(self, x: torch.Tensor, 
                              edge_index: torch.Tensor,
                              edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取注意力权重用于可视化"""
        with torch.no_grad():
            # 多头线性变换
            x_heads = torch.einsum('ni,hij->nhj', x, self.W)
            row, col = edge_index
            
            # 计算注意力权重
            attention_weights = self._compute_attention(
                x_heads[row],
                x_heads[col],
                col,
                self._compute_edge_modulation(edge_attr, x_heads.dtype, x_heads.device)[0],
                x_heads.size(0),
            )
            
            return attention_weights

