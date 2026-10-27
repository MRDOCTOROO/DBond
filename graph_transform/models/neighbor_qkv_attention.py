"""Destination-query, incoming-neighbor-key/value graph attention.

Every head projects the complete node vector. The original graph (including
its one global node), edge bias/value gate, residual, LayerNorm and dropout
are retained. No padding nodes or additional edges are constructed here.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def validate_neighbor_mode(mode):
    if mode not in ('legacy', 'qkv8'):
        raise ValueError(f'Unknown neighbor_attention_mode: {mode!r}')
    return mode


MARKERS = ('neighbor_attention_mode_code', 'neighbor_attention_schema_code',
           'neighbor_attention_heads_code')


def validate_neighbor_checkpoint(model, state):
    mode = validate_neighbor_mode(getattr(model, 'neighbor_attention_mode', 'legacy'))
    found = {k for k in state if k.startswith('neighbor_attention_')}
    expected = set(MARKERS) if mode == 'qkv8' else set()
    if found != expected:
        raise ValueError('Neighbor QKV attention checkpoint mode/schema mismatch')
    if expected:
        if any(not torch.is_tensor(state[k]) or state[k].dtype != torch.int64
               or state[k].shape != torch.Size([]) for k in MARKERS):
            raise ValueError('Neighbor QKV checkpoint markers must be scalar int64 tensors')
        if [int(state[k]) for k in MARKERS] != [1, 1, 8]:
            raise ValueError('Neighbor QKV checkpoint requires mode1/schema1/eight heads')


class NeighborQKVAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = self.output_dim = int(config.hidden_dim)
        self.num_heads = int(getattr(config, 'neighbor_qkv_heads', 8))
        if self.num_heads != 8 or self.input_dim % self.num_heads:
            raise ValueError('This frozen experiment requires 8 heads and divisible hidden size')
        if not config.concat_heads:
            raise ValueError('This experiment concatenates the eight attention heads')
        self.head_dim = self.input_dim // self.num_heads
        self.dropout = config.dropout
        self.use_edge_bias = getattr(config, 'gat_use_edge_bias', True)
        self.use_edge_gate = getattr(config, 'gat_use_edge_gate', True)
        self.q_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.k_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.v_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.out_proj = nn.Linear(self.output_dim, self.output_dim)
        self.edge_attention_bias = nn.Linear(self.input_dim, self.num_heads)
        self.edge_message_gate = nn.Linear(self.input_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.enable_timing = False
        self.last_forward_timing = {}
        self.enable_diagnostics = False
        self.last_diagnostics = {}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _parts(self, x, edge_index, edge_attr):
        source, target = edge_index
        n = x.shape[0]
        q = self.q_proj(x).reshape(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(n, self.num_heads, self.head_dim)
        # Accumulate dot products and neighbor softmax in FP32 under AMP.
        dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        logits = (q[target].to(dtype) * k[source].to(dtype)).sum(-1) / math.sqrt(self.head_dim)
        values = v[source].to(dtype)
        if edge_attr is not None:
            if self.use_edge_bias:
                logits = logits + self.edge_attention_bias(edge_attr).to(dtype)
            if self.use_edge_gate:
                values = values * torch.sigmoid(self.edge_message_gate(edge_attr)).reshape(
                    -1, self.num_heads, self.head_dim).to(dtype)
        if logits.numel():
            index = target[:, None].expand(-1, self.num_heads)
            maxima = logits.new_full((n, self.num_heads), -torch.inf)
            maxima.scatter_reduce_(0, index, logits, reduce='amax', include_self=True)
            exp = (logits - maxima[target]).exp()
            denom = logits.new_zeros((n, self.num_heads))
            denom.scatter_add_(0, index, exp)
            attention = exp / denom[target].clamp_min(1e-12)
        else:
            attention = logits
        aggregate = values.new_zeros((n, self.num_heads, self.head_dim))
        aggregate.index_add_(0, target, attention[..., None] * values)
        return aggregate.reshape(n, self.output_dim).to(v.dtype), attention

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        aggregate, attention = self._parts(x, edge_index, edge_attr)
        # Isolated nodes get the same residual+normalization convention as GAT.
        update = self.out_proj(aggregate)
        degree = torch.bincount(edge_index[1], minlength=x.shape[0])
        update = update.masked_fill((degree == 0)[:, None], 0.)
        output = F.dropout(self.layer_norm(x + update), self.dropout, self.training)
        if self.enable_diagnostics:
            self.last_diagnostics = {
                'mode': 'qkv8', 'heads': self.num_heads, 'head_dim': self.head_dim,
                'num_nodes': x.shape[0], 'num_edges': edge_index.shape[1],
                'attention_entropy_per_edge_mean': float(
                    -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean())
                    if attention.numel() else 0.,
                'aggregation_l2_mean': float(aggregate.float().norm(dim=-1).mean()) if x.shape[0] else 0.,
            }
        if return_attention:
            return output, attention
        return output

    def get_attention_weights(self, x, edge_index, edge_attr=None):
        with torch.no_grad():
            return self._parts(x, edge_index, edge_attr)[1]
