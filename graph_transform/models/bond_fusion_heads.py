"""Four-state bond-local fusion ablations with legacy concat as the default.

All source slots are ordered: left, right, original edge, difference, product,
and projected theory. Source weights describe a fitted predictor; they are not
causal feature importances or evidence of physical contacts.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .bond_cross_attention import BondCrossAttentionHead
from .four_state import replace_binary_output


BOND_FUSION_SOURCES = ('left', 'right', 'edge', 'difference', 'product', 'theory')
_MODE_CODES = {'cross_attention': 1, 'gated': 2, 'residual_mlp': 3}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_CODES.items()}
_MARKERS = ('bond_fusion_mode_code', 'bond_fusion_schema_code',
            'bond_fusion_source_roles', 'bond_fusion_heads_code')


def validate_bond_fusion_mode(value):
    if value not in ('concat', 'cross_attention', 'gated', 'residual_mlp'):
        raise ValueError(f'Unknown bond_fusion_mode: {value!r}')
    return value


def bond_fusion_mode_code(mode):
    return _MODE_CODES.get(validate_bond_fusion_mode(mode))


def validate_bond_fusion_checkpoint(model, state_dict, checkpoint_config=None):
    """Validate routing, schema, source order and branches before copying.

    Missing fusion markers mean the legacy concat route, including the older
    binary cross-attention head which retains its independent head-type marker.
    strict=False may not omit an experimental branch or change its semantics.
    """
    requested = validate_bond_fusion_mode(model.bond_fusion_mode)
    mode_key = _MARKERS[0]
    if mode_key in state_dict:
        marker = state_dict[mode_key]
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() not in _CODE_TO_MODE):
            raise ValueError('Invalid checkpoint bond_fusion_mode_code')
        saved = _CODE_TO_MODE[int(marker.item())]
    else:
        saved = 'concat'
    if saved != requested:
        raise ValueError(f'Checkpoint bond_fusion_mode mismatch: {saved!r} != {requested!r}')

    expected_state = model.state_dict()
    actual_keys = {key for key in state_dict if key.startswith('bond_fusion_')}
    expected_keys = {key for key in expected_state if key.startswith('bond_fusion_')}
    if actual_keys != expected_keys:
        raise ValueError('Checkpoint bond-fusion marker/branch keys mismatch')
    for key in expected_keys:
        value = state_dict[key]
        if not torch.is_tensor(value) or value.shape != expected_state[key].shape:
            raise ValueError(f'Checkpoint bond-fusion tensor shape mismatch: {key}')
        if key in _MARKERS and (value.dtype != torch.int64 or not torch.equal(
                value.detach().cpu(), expected_state[key].detach().cpu())):
            raise ValueError(f'Checkpoint bond-fusion semantics mismatch: {key}')

    if checkpoint_config is not None:
        if not isinstance(checkpoint_config, dict):
            raise ValueError('Checkpoint config must be a dictionary')
        declared = checkpoint_config.get('model', checkpoint_config)
        if not isinstance(declared, dict):
            raise ValueError('Checkpoint model config must be a dictionary')
        declared_mode = validate_bond_fusion_mode(declared.get('bond_fusion_mode', 'concat'))
        if declared_mode != saved:
            raise ValueError('Checkpoint bond_fusion_mode config/state mismatch')
        if saved != 'concat':
            if (declared.get('bond_head_type', 'concat_mlp') != 'four_state_mlp'
                    or declared.get('hidden_dim', 256) != model.hidden_dim
                    or not all(declared.get(key, default) for key, default in (
                        ('bond_use_edge_repr', True), ('bond_use_diff_feature', True),
                        ('bond_use_product_feature', True), ('use_theory_features', False)))):
                raise ValueError('Checkpoint bond-fusion source/output config mismatch')
            if saved == 'cross_attention' and declared.get('bond_cross_attention_heads', 4) != model.bond_head.cross_attention.num_heads:
                raise ValueError('Checkpoint bond-fusion attention-head config mismatch')


class BondFusionResidual(nn.Module):
    """One zero-output LN -> H/2 -> ReLU -> H residual block."""

    def __init__(self, hidden_dim):
        super().__init__()
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 2:
            raise ValueError('Residual fusion requires hidden_dim >= 2')
        self.normalization = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation = nn.ReLU()
        self.up = nn.Linear(hidden_dim // 2, hidden_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        delta = self.up(self.activation(self.down(self.normalization(hidden))))
        return hidden + delta.to(dtype=hidden.dtype)


class FourStateBondCrossAttentionHead(BondCrossAttentionHead):
    """The historical local 2Q/4KV architecture, now outputting four states."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__(hidden_dim, num_heads, dropout, [0, 1, 2, 3])
        self.mlp[3] = replace_binary_output(self.mlp[3])
        self.enable_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, feature_parts):
        if len(feature_parts) != 6:
            raise ValueError('Four-state cross-attention fusion requires six ordered sources')
        self.last_diagnostics = {}
        query = torch.stack(feature_parts[:2], dim=1)
        memory = torch.stack(feature_parts[2:], dim=1)
        memory = memory + self.memory_type_embedding.to(memory.dtype).unsqueeze(0)
        if query.size(0) == 0:
            if self.enable_diagnostics:
                self.last_diagnostics = cross_attention_diagnostics(self.cross_attention, query, memory)
            return feature_parts[0].new_empty((0, 4))
        # Keep the production attention call identical when diagnostics are on.
        # Detached Q/K probabilities are summarized separately without dropout
        # or random draws; eval diagnostics therefore leave output/RNG unchanged.
        fused, _ = self.cross_attention(query, memory, memory, need_weights=False)
        result = self.mlp(self.layer_norm(query + fused).flatten(start_dim=1))
        if self.enable_diagnostics:
            self.last_diagnostics = cross_attention_diagnostics(self.cross_attention, query, memory)
        return result


@torch.no_grad()
def cross_attention_diagnostics(attention, query, memory):
    """Pre-dropout probabilities; opt-in, detached, no retained training graph."""
    heads = attention.num_heads
    count, query_count, hidden_dim = query.shape
    memory_count = memory.size(1)
    if count:
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.detach().float()
            k = memory.detach().float()
            weight = attention.in_proj_weight.detach().float()
            bias = attention.in_proj_bias.detach().float()
            q = F.linear(q, weight[:hidden_dim], bias[:hidden_dim])
            k = F.linear(k, weight[hidden_dim:2 * hidden_dim], bias[hidden_dim:2 * hidden_dim])
            head_dim = hidden_dim // heads
            q = q.reshape(count, query_count, heads, head_dim).transpose(1, 2)
            k = k.reshape(count, memory_count, heads, head_dim).transpose(1, 2)
            weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * head_dim ** -0.5, dim=-1)
        per_head = weights.mean(dim=0)
        per_query = per_head.mean(dim=0)
        per_source = per_query.mean(dim=0)
    else:
        per_head = query.new_zeros((heads, query_count, memory_count), dtype=torch.float32)
        per_query = per_head.mean(dim=0)
        per_source = per_query.mean(dim=0)
    return {
        'query_sources': list(BOND_FUSION_SOURCES[:2]),
        'memory_sources': list(BOND_FUSION_SOURCES[2:]),
        'attention_heads': heads,
        'attention_scope': 'per-bond, pre-dropout Q/K softmax; descriptive only',
        'attention_mean_per_head_query_source': per_head.cpu().tolist(),
        'attention_mean_per_query_source': per_query.cpu().tolist(),
        'source_attention_mean': per_source.cpu().tolist(),
    }


@torch.no_grad()
def source_norm_diagnostics(feature_parts):
    return [float(part.detach().float().norm(dim=-1).mean().item()) if part.size(0) else 0.
            for part in feature_parts]


@torch.no_grad()
def gate_diagnostics(weights):
    count, sources = weights.shape
    values = weights.detach().float()
    if count:
        means = values.mean(dim=0)
        minima = values.amin(dim=0)
        maxima = values.amax(dim=0)
    else:
        means = minima = maxima = values.new_zeros((sources,))
    return {'source_weight_mean': means.cpu().tolist(),
            'source_weight_min': minima.cpu().tolist(),
            'source_weight_max': maxima.cpu().tolist(),
            'scaled_gate_mean': (sources * means).cpu().tolist()}


@torch.no_grad()
def residual_diagnostics(before, after):
    before = before.detach().float()
    after = after.detach().float()
    if before.size(0):
        delta = (after - before).norm(dim=-1)
        initial = before.norm(dim=-1)
        return {'residual_delta_l2_mean': float(delta.mean().item()),
                'residual_input_l2_mean': float(initial.mean().item()),
                'residual_output_l2_mean': float(after.norm(dim=-1).mean().item()),
                'residual_relative_l2_mean': float((delta / initial.clamp_min(torch.finfo(initial.dtype).eps)).mean().item())}
    return {key: 0. for key in ('residual_delta_l2_mean', 'residual_input_l2_mean',
                               'residual_output_l2_mean', 'residual_relative_l2_mean')}
