"""Condition-uniform auxiliary BCE for operational fragment-match frequencies."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ion12_condition_bce(logits: torch.Tensor, targets: torch.Tensor,
                        label_mask: torch.Tensor) -> torch.Tensor:
    """Mean over conditions of mean BCE over valid bonds and 12 channels.

    ``logits`` has packed valid-bond order; ``targets`` is [B, max_bonds, 12].
    group_n and main-task sample weights intentionally do not enter this loss.
    Padding is never read, so padded targets need not be meaningful.
    """
    if label_mask is None or label_mask.ndim != 2:
        raise ValueError('ion12 requires a two-dimensional valid-bond mask')
    mask = label_mask.bool()
    if targets.ndim != 3 or targets.shape != (*mask.shape, 12):
        raise ValueError('ion12 targets must have shape [B, max_bonds, 12]')
    lengths = mask.sum(dim=1)
    if lengths.numel() == 0 or torch.any(lengths == 0):
        raise ValueError('ion12 requires at least one valid bond per condition')
    selected = targets[mask].float()
    if logits.shape != selected.shape:
        raise ValueError('ion12 logits and valid targets are misaligned')
    if not torch.isfinite(selected).all() or torch.any((selected < 0) | (selected > 1)):
        raise ValueError('ion12 frequencies must be finite and lie in [0, 1]')
    if not torch.isfinite(logits).all():
        raise ValueError('ion12 logits must be finite')
    per_bond = F.binary_cross_entropy_with_logits(logits.float(), selected, reduction='none').mean(dim=-1)
    # Each condition receives equal total weight, independently of peptide length.
    inverse_lengths = lengths.to(per_bond.dtype).reciprocal()
    packed_weights = inverse_lengths[:, None].expand_as(mask)[mask]
    return (per_bond * packed_weights).sum() / lengths.numel()
