"""Bond-local cross-attention: endpoint queries attend to bond feature sources."""
import torch
from torch import nn


def validate_bond_head_type(value):
    if value not in ('concat_mlp', 'cross_attention_mlp'):
        raise ValueError(f'Unknown bond_head_type: {value!r}')
    return value


class BondCrossAttentionHead(nn.Module):
    """Each valid bond is a separate batch item, never a sequence of bonds.

    Queries: [left residue, right residue], shape [N, 2, H].
    Keys/values: enabled [edge, left-right, left*right, projected theory],
    shape [N, sources, H], with learned source type embeddings.
    Endpoint residual + LayerNorm -> flatten the ordered pair -> MLP -> logit.
    """
    def __init__(self, hidden_dim, num_heads, dropout, memory_role_ids):
        super().__init__()
        if (not isinstance(num_heads, int) or isinstance(num_heads, bool)
                or num_heads < 1 or hidden_dim % num_heads):
            raise ValueError('bond_cross_attention_heads must be positive and divide hidden_dim')
        if not memory_role_ids:
            raise ValueError('cross_attention_mlp requires at least one bond memory source')
        self.register_buffer('memory_role_ids', torch.tensor(memory_role_ids, dtype=torch.int64))
        self.register_buffer('num_heads_code', torch.tensor(num_heads, dtype=torch.int64))
        self.memory_type_embedding = nn.Parameter(torch.empty(len(memory_role_ids), hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.memory_type_embedding, std=0.02)
        nn.init.xavier_uniform_(self.cross_attention.in_proj_weight)
        nn.init.zeros_(self.cross_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feature_parts):
        if len(feature_parts) != 2 + self.memory_type_embedding.size(0):
            raise ValueError('Bond cross-attention source count does not match its configuration')
        query = torch.stack(feature_parts[:2], dim=1)
        memory = torch.stack(feature_parts[2:], dim=1)
        memory = memory + self.memory_type_embedding.to(memory.dtype).unsqueeze(0)
        fused, _ = self.cross_attention(query, memory, memory, need_weights=False)
        fused = self.layer_norm(query + fused)
        return self.mlp(fused.flatten(start_dim=1))


def validate_bond_head_checkpoint(model, state_dict):
    """Prevent strict=False from silently leaving a fusion head uninitialized."""
    key = 'bond_head_mode_code'
    if key in state_dict:
        marker = state_dict[key]
        if (not torch.is_tensor(marker) or marker.dtype != torch.int64
                or marker.shape != torch.Size([]) or marker.item() != 1):
            raise ValueError('Invalid checkpoint bond_head_mode_code')
    if (key in state_dict) != (model.bond_head_type == 'cross_attention_mlp'):
        raise ValueError('Checkpoint bond_head_type mismatch')
    expected = {k: v for k, v in model.state_dict().items() if k.startswith('bond_head.')}
    actual = {k: v for k, v in state_dict.items() if k.startswith('bond_head.')}
    if set(expected) != set(actual):
        raise ValueError('Checkpoint bond head parameter keys mismatch')
    for name, tensor in actual.items():
        if not torch.is_tensor(tensor) or tensor.shape != expected[name].shape:
            raise ValueError(f'Checkpoint bond head tensor shape mismatch: {name}')
        if name.endswith(('memory_role_ids', 'num_heads_code')):
            if tensor.dtype != torch.int64 or not torch.equal(tensor.detach().cpu(), expected[name].detach().cpu()):
                raise ValueError(f'Checkpoint bond head semantics mismatch: {name}')
