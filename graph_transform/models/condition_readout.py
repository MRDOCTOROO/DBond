"""Matched-budget late condition interactions for the existing bond head.

The residual consumes the head's hidden activation before its legacy dropout.
It introduces no new dropout and starts at an exactly zero logit correction.
Only charge and NCE enter this branch; masked values are replaced before any
arithmetic so a masked NaN cannot contaminate the condition representation.
"""

import torch
from torch import nn


class BondConditionReadout(nn.Module):
    MODES = {'concat': 1, 'gated': 2}

    def __init__(self, hidden_dim, mode, width=32, use_charge=True, use_nce=True):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'Unknown bond_condition_mode: {mode}')
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError('bond_condition_dim must be a positive integer')
        self.mode = mode
        # Both modes have identical parameter shapes. Persist the operation as
        # well as the weights, preventing a silent concat/gated checkpoint swap.
        self.register_buffer('mode_code', torch.tensor(self.MODES[mode], dtype=torch.long))
        self.register_buffer('condition_mask', torch.tensor([use_charge, use_nce], dtype=torch.bool),
                             persistent=False)
        self.bond_project = nn.Sequential(nn.Linear(hidden_dim, width), nn.SiLU())
        self.condition_encoder = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + 'mode_code')
        expected = torch.tensor(self.MODES[self.mode], dtype=torch.long)
        if not torch.is_tensor(saved) or not torch.equal(saved.detach().cpu(), expected):
            error_msgs.append(f'{prefix}bond condition mode mismatch: expected {self.mode!r}')
            # A failed load must not overwrite the marker and let a later load
            # accidentally accept weights for the other operation.
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, bond_hidden, charges, nces, bond_sample_indices):
        """Return [N_valid_bonds] corrections with explicit sample ownership."""
        values = torch.stack([
            charges.to(device=bond_hidden.device, dtype=torch.float32),
            nces.to(device=bond_hidden.device, dtype=torch.float32),
        ], dim=-1)
        values = values.masked_fill(~self.condition_mask, 0.0)
        scaled = torch.stack([values[:, 0] * 0.1, values[:, 1] * 0.01], dim=-1)
        condition = self.condition_encoder(scaled)[bond_sample_indices]
        bond = self.bond_project(bond_hidden)
        if self.mode == 'gated':
            bond = bond * (1.0 + torch.tanh(condition))
        return self.head(torch.cat([bond, condition], dim=-1)).squeeze(-1)
