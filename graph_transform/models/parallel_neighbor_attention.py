"""A zero-output-initialized QKV8 context branch parallel to intact GAT4.

Q(center), K/V(incoming neighbors) read the same post-GCN packed features.
The branch returns only a residual context; no self residual, LN, dropout or
FFN is added here. Its zero projection initially preserves the original model.
"""
import math
import torch
from torch import nn

MARKERS = ('parallel_neighbor_attention_mode_code',
           'parallel_neighbor_attention_schema_code',
           'parallel_neighbor_attention_heads_code',
           'parallel_neighbor_attention_fusion_code')


def validate_parallel_neighbor_mode(mode):
    if mode not in ('none', 'qkv8'):
        raise ValueError(f'Unknown parallel_neighbor_attention_mode: {mode!r}')
    return mode


def validate_parallel_neighbor_checkpoint(model, state):
    enabled = getattr(model, 'parallel_neighbor_attention_mode', 'none') == 'qkv8'
    found = {k for k in state if k.startswith('parallel_neighbor_attention_')}
    if found != (set(MARKERS) if enabled else set()):
        raise ValueError('Parallel neighbor attention checkpoint mode/schema mismatch')
    actual_branch = {k for k in state if k.startswith('parallel_neighbor_context.')}
    expected_branch = ({k for k in model.state_dict() if k.startswith('parallel_neighbor_context.')}
                       if enabled else set())
    if actual_branch != expected_branch:
        raise ValueError('Incomplete or unexpected parallel neighbor context parameters')
    if enabled and any(state[k].shape != model.state_dict()[k].shape for k in expected_branch):
        raise ValueError('Parallel neighbor context parameter shape mismatch')
    if enabled:
        if any(not torch.is_tensor(state[k]) or state[k].dtype != torch.int64
               or state[k].shape != torch.Size([]) for k in MARKERS):
            raise ValueError('Parallel neighbor attention requires scalar int64 markers')
        if [int(state[k]) for k in MARKERS] != [1, 1, 8, 1]:
            raise ValueError('Parallel neighbor attention requires mode1/schema1/heads8/residual-fusion1')


class ParallelNeighborQKV8(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_dim != 256 or not config.concat_heads:
            raise ValueError('Frozen parallel candidate requires H256 and concatenated heads')
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = 32
        self.q_proj = nn.Linear(256, 256, bias=False)
        self.k_proj = nn.Linear(256, 256, bias=False)
        self.v_proj = nn.Linear(256, 256, bias=False)
        self.edge_attention_bias = nn.Linear(256, 8)
        self.edge_message_gate = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 256)
        if not getattr(config, 'gat_use_edge_bias', True) or not getattr(config, 'gat_use_edge_gate', True):
            raise ValueError('Frozen parallel candidate retains the existing edge bias/value gate')
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.enable_diagnostics = False
        self.last_diagnostics = {}

    def aggregate(self, x, edge_index, edge_attr):
        source, target = edge_index
        n = x.shape[0]
        q = self.q_proj(x).reshape(n, 8, 32)
        k = self.k_proj(x).reshape(n, 8, 32)
        v = self.v_proj(x).reshape(n, 8, 32)
        dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        scores = (q[target].to(dtype) * k[source].to(dtype)).sum(-1) / math.sqrt(32)
        scores = scores + self.edge_attention_bias(edge_attr).to(dtype)
        values = v[source].to(dtype) * self.edge_message_gate(edge_attr).sigmoid().reshape(-1, 8, 32).to(dtype)
        if scores.numel():
            index = target[:, None].expand(-1, 8)
            maxima = scores.new_full((n, 8), -torch.inf)
            maxima.scatter_reduce_(0, index, scores, reduce='amax', include_self=True)
            exp = (scores - maxima[target]).exp()
            denominator = scores.new_zeros((n, 8))
            denominator.scatter_add_(0, index, exp)
            attention = exp / denominator[target].clamp_min(1e-12)
        else:
            attention = scores
        output = values.new_zeros((n, 8, 32))
        output.index_add_(0, target, values * attention[..., None])
        return output.reshape(n, 256).to(v.dtype), attention

    def forward(self, x, edge_index, edge_attr):
        context, attention = self.aggregate(x, edge_index, edge_attr)
        residual = self.out_proj(context)
        degree = torch.bincount(edge_index[1], minlength=x.shape[0])
        residual = residual.masked_fill((degree == 0)[:, None], 0.)
        self.last_diagnostics = {}
        if self.enable_diagnostics:
            self.last_diagnostics = {
                'mode': 'parallel_qkv8', 'num_heads': 8, 'head_dim': 32,
                'num_packed_nodes': x.shape[0], 'num_edges': edge_index.shape[1],
                'context_l2_mean': float(context.detach().float().norm(dim=-1).mean()) if x.shape[0] else 0.,
                'projected_residual_l2_mean': float(residual.detach().float().norm(dim=-1).mean()) if x.shape[0] else 0.,
            }
        return residual
