"""Opt-in late NCE adaptation; no new input information or shared RNG draws."""
import torch
from torch import nn


class NCEStateResidual(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(4, 4))
        self.register_buffer('nce_values', torch.tensor([20., 30., 40., 50.]))

    def forward(self, hidden, nces, bond_samples):
        values = nces.reshape(-1).float()
        match = (values[:, None] - self.nce_values[None, :]).abs() <= 1e-4
        if not bool(torch.isfinite(values).all()) or not bool((match.sum(-1) == 1).all()):
            raise ValueError('stratified4 requires NCE exactly 20/30/40/50; no silent extrapolation')
        index = match.long().argmax(-1)[bond_samples]
        # Each adapter reads the full shared H-dimensional bond representation.
        return torch.einsum('boh,bh->bo', self.weight[index], hidden) + self.bias[index]


def validate_nce_head_checkpoint(model, state):
    keys = {k for k in state if k.startswith('nce_state_residual.') or k == 'nce_head_mode_code'}
    expected = {k for k in model.state_dict() if k.startswith('nce_state_residual.') or k == 'nce_head_mode_code'}
    if keys != expected:
        raise ValueError('Checkpoint NCE head mode/schema mismatch')
    for name in ('nce_head_mode_code', 'nce_state_residual.nce_values'):
        if name in expected and not torch.equal(state[name].detach().cpu(), model.state_dict()[name].detach().cpu()):
            raise ValueError('Checkpoint NCE head semantic marker mismatch')
