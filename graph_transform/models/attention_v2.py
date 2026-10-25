"""Shared-projection GATv2 scoring with DBond's existing edge messages.

Score = a_v2^T LeakyReLU(W h_source + W h_target) + scalar_edge_bias.
The projection W is shared by source, target and messages. The scalar edge bias
is added AFTER the nonlinear content score, unlike the legacy GAT placement.
This is a restricted GATv2 variant, not a claim of universal expressivity.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .attention_layers import GraphAttentionLayer


ATTENTION_MODE_KEY = 'gat_attention_mode_code'


def validate_attention_type(value):
    if value not in ('gat', 'gatv2'):
        raise ValueError(f'Unknown gat_attention_type: {value!r}')
    return value


def validate_attention_checkpoint(model, state_dict, checkpoint_config=None):
    """Reject cross-mode/partial attention loads before mutating any weights.

Legacy GAT has no marker; keeping this convention preserves its state schema.
GATv2 requires both its scalar int64 marker and every a_v2 parameter. Config
metadata, when supplied, must agree with the actual state (absent flag = GAT).
"""
    requested = validate_attention_type(model.gat_attention_type)
    marker = state_dict.get(ATTENTION_MODE_KEY)
    if ATTENTION_MODE_KEY in state_dict:
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or int(marker.item()) != 1):
            raise ValueError('Invalid checkpoint gat_attention_mode_code')
        saved = 'gatv2'
    else:
        saved = 'gat'
    if saved != requested:
        raise ValueError(f'Checkpoint gat_attention_type mismatch: {saved!r} != {requested!r}')
    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        saved_model_config = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(saved_model_config, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        declared = validate_attention_type(saved_model_config.get('gat_attention_type', 'gat'))
        if declared != saved:
            raise ValueError('Checkpoint gat_attention_type config/state mismatch')
    suffix = 'a_v2' if requested == 'gatv2' else 'a'
    expected = {f'gat_layers.{i}.{suffix}': getattr(layer, suffix)
                for i, layer in enumerate(model.gat_layers)}
    actual = {key for key in state_dict if key.startswith('gat_layers.')
              and key.rsplit('.', 1)[-1] in ('a', 'a_v2')}
    if actual != set(expected):
        raise ValueError('Checkpoint attention parameter keys do not match gat_attention_type')
    for key, parameter in expected.items():
        value = state_dict[key]
        if not torch.is_tensor(value) or value.shape != parameter.shape:
            raise ValueError(f'Checkpoint attention shape mismatch: {key}')


class GraphAttentionV2Layer(GraphAttentionLayer):
    """Change the content score only; reuse the legacy propagation path."""

    def __init__(self, config):
        # Consume exactly the legacy initialization stream for all shared
        # tensors, then replace (not retain) its scoring vector in an RNG fork.
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            a_v2 = nn.Parameter(self.W.new_empty(self.num_heads, self.head_dim, 1))
            nn.init.xavier_uniform_(a_v2)
        del self.a
        self.a_v2 = a_v2

    def reset_parameters(self):
        # The parent constructor invokes this before a_v2 exists. Later public
        # resets must initialize the live vector rather than the removed a.
        if not hasattr(self, 'a_v2'):
            return super().reset_parameters()
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_v2)
        nn.init.xavier_uniform_(self.edge_attention_bias.weight)
        nn.init.zeros_(self.edge_attention_bias.bias)
        nn.init.xavier_uniform_(self.edge_message_gate.weight)
        nn.init.zeros_(self.edge_message_gate.bias)

    def _compute_attention(self, src_heads, dst_heads, target_index,
                           edge_bias, num_nodes):
        if src_heads.numel() == 0:
            return src_heads.new_empty((0, self.num_heads))
        dtype = torch.float32 if src_heads.dtype in (torch.float16, torch.bfloat16) else src_heads.dtype
        content = F.leaky_relu(src_heads.to(dtype) + dst_heads.to(dtype),
                               negative_slope=self.alpha)
        scores = (content * self.a_v2.squeeze(-1).to(dtype).unsqueeze(0)).sum(dim=-1)
        if edge_bias is not None:
            scores = scores + edge_bias.to(dtype)
        return self._softmax_attention(scores, target_index, num_nodes).to(src_heads.dtype)
