"""Sequence/charge descriptors inspired by the mobile-proton/PIC framework.

These are composition counts and signed charge/count differences, NOT gas-phase
basicities, predicted proton locations, or hard fragmentation rules. B/O primary
amines and X pyridyl groups are structural extrapolations for this project.
Paizs & Suhai (2005), DOI 10.1002/mas.20024, motivates the variables; it does not
validate this feature set or residual architecture for synthetic D-peptides.
"""
import torch
from torch import nn


GROUPS = ('R', 'KBO', 'H', 'X', 'D', 'E')
FEATURE_NAMES = tuple(f'{side}_{group}_per_length' for side in ('b', 'y') for group in GROUPS) + (
    'b_length_over_32', 'y_length_over_32', 'charge_over_10',
    'charge_minus_arg_count_per_length', 'charge_minus_basic_group_count_per_length',
)
FEATURE_DIM = len(FEATURE_NAMES)


class ChargePartitionReadout(nn.Module):
    """A 17->32->3 residual on b-only/y-only/both logits; none is the reference.

    Zero final projection gives exactly the original four-state initial function.
    No dropout is added, so the original model's RNG stream stays unchanged.
    """
    def __init__(self, char_to_idx, vocab_size):
        super().__init__()
        lookup = torch.zeros(vocab_size, len(GROUPS))
        for residue, idx in char_to_idx.items():
            if idx:
                lookup[idx] = torch.tensor([float(residue in group) for group in GROUPS])
        self.register_buffer('lookup', lookup, persistent=False)
        self.register_buffer('feature_version', torch.tensor(1, dtype=torch.int64))
        self.mlp = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.SiLU(), nn.Linear(32, 3))
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def features(self, tokens, charges, valid_bond_mask):
        """Return [N_valid_bonds,17], in the main head's row-major mask order."""
        batch, width = tokens.shape
        if charges.shape != (batch,) or valid_bond_mask.shape != (batch, max(width-1, 0)):
            raise ValueError('Charge-partition descriptor shape mismatch')
        lengths = (tokens != 0).sum(-1).float()
        denominator = lengths.clamp_min(1).view(-1, 1, 1)
        composition = self.lookup[tokens]
        total = composition.sum(1)
        prefix = composition.cumsum(1)[:, :-1]
        suffix = total[:, None] - prefix
        left_len = torch.arange(1, width, device=tokens.device).float()[None, :].expand(batch, -1)
        right_len = lengths[:, None] - left_len
        charge = charges.float()
        global_features = torch.stack((charge / 10., (charge-total[:, 0])/lengths.clamp_min(1),
                                      (charge-total[:, :4].sum(-1))/lengths.clamp_min(1)), -1)
        features = torch.cat((prefix/denominator, suffix/denominator,
                              left_len[..., None]/32., right_len[..., None]/32.,
                              global_features[:, None, :].expand(-1, max(width-1, 0), -1)), -1)
        return features[valid_bond_mask]

    def forward(self, tokens, charges, valid_bond_mask):
        delta = self.mlp(self.features(tokens, charges, valid_bond_mask))
        return torch.cat((torch.zeros_like(delta[:, :1]), delta), -1).float()


def validate_charge_partition_checkpoint(model, state_dict):
    expected = {k: v for k, v in model.state_dict().items() if k.startswith('charge_partition.')}
    actual = {k: v for k, v in state_dict.items() if k.startswith('charge_partition.')}
    if set(actual) != set(expected):
        raise ValueError('Checkpoint charge-partition configuration mismatch')
    for key, value in actual.items():
        if not torch.is_tensor(value) or value.shape != expected[key].shape:
            raise ValueError(f'Invalid charge-partition checkpoint tensor: {key}')
        if key.endswith('feature_version') and (value.dtype != torch.int64 or value.item() != 1):
            raise ValueError('Invalid charge-partition descriptor version')
