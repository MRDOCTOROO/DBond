"""Ordered N/C fragment contexts for the existing bond classifier.

These are pools of contextual GNN states, not independently simulated fragments.
Global nodes and padding are never directly pooled. No spectrum is required.
"""

import torch
from torch import nn


def fragment_means(packed_nodes, seq_lengths, node_offsets, max_seq_len):
    """Return [B, L-1, D] prefix/suffix means for cuts after residue i.

Packing may contain a global node after each peptide. Explicit sequence lengths
exclude it. FP32, separate directional sums avoid AMP accumulation/cancellation.
Invalid cuts are zero; both sides include at least one residue at valid cuts.
"""
    batch_size = seq_lengths.numel()
    width = max(int(max_seq_len) - 1, 0)
    if max_seq_len < 2 or packed_nodes.size(0) == 0:
        empty = packed_nodes.new_zeros((batch_size, width, packed_nodes.size(-1)), dtype=torch.float32)
        return empty, empty.clone()
    positions = torch.arange(max_seq_len, device=packed_nodes.device)
    real_mask = positions.unsqueeze(0) < seq_lengths.unsqueeze(1)
    # Index zero is only a safe placeholder; masked_fill also removes NaN/Inf
    # from a placeholder/global node (multiplication by zero would not).
    indices = node_offsets.unsqueeze(1) + positions.unsqueeze(0)
    indices = indices.masked_fill(~real_mask, 0)
    residues = packed_nodes[indices].float().masked_fill(~real_mask.unsqueeze(-1), 0.0)
    prefix = residues.cumsum(dim=1)[:, :-1]
    suffix = residues.flip([1]).cumsum(dim=1).flip([1])[:, 1:]
    n_count = torch.arange(1, max_seq_len, device=packed_nodes.device).unsqueeze(0)
    c_count = seq_lengths.unsqueeze(1) - n_count
    valid = c_count > 0
    prefix = prefix / n_count.unsqueeze(-1)
    suffix = suffix / c_count.clamp_min(1).unsqueeze(-1)
    return (prefix.masked_fill(~valid.unsqueeze(-1), 0.0),
            suffix.masked_fill(~valid.unsqueeze(-1), 0.0))


class DualFragmentReadout(nn.Module):
    """Add a small learned logit correction using ordered fragment contexts."""

    def __init__(self, hidden_dim, fragment_dim=64, max_seq_len=32):
        super().__init__()
        if fragment_dim <= 0 or max_seq_len <= 0:
            raise ValueError("fragment_dim and max_seq_len must be positive")
        self.max_seq_len = max_seq_len
        self.project = nn.Sequential(nn.Linear(hidden_dim, fragment_dim), nn.SiLU())
        # Local bond states let the correction depend on both cleavage-site
        # identity and whole-fragment composition. N and C order is preserved.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4 * fragment_dim + 2, fragment_dim),
            nn.SiLU(),
            nn.Linear(fragment_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, nodes, seq_lengths, node_offsets, max_seq_len,
                valid_bond_mask, bond_src, bond_dst):
        prefix, suffix = fragment_means(
            self.project(nodes), seq_lengths, node_offsets, max_seq_len)
        # Normal collate uses max_bonds=L-1; accept a shorter prediction grid.
        width = valid_bond_mask.size(1)
        prefix, suffix = prefix[:, :width][valid_bond_mask], suffix[:, :width][valid_bond_mask]
        n_count = torch.arange(1, width + 1, device=nodes.device).unsqueeze(0).expand_as(valid_bond_mask)
        c_count = seq_lengths.unsqueeze(1) - n_count
        lengths = torch.stack([n_count, c_count], dim=-1)[valid_bond_mask].float() / self.max_seq_len
        features = torch.cat([
            nodes[bond_src], nodes[bond_dst], prefix, suffix,
            prefix - suffix, prefix * suffix, lengths,
        ], dim=-1)
        return self.head(features).squeeze(-1)
