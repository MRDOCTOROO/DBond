"""Train-only sequence ranking objective for the four-state bond model.

Unlike the historical batch-local hinge loss, this objective uses a temperature
scaled logistic pair loss and excludes pairs whose training q difference is too
small.  It is a batch-local proxy for q_spearman_pep_seq, not the metric itself.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from .metrics import batch_group_keys


def pairwise_sequence_logistic_loss(
    union_logits: torch.Tensor,
    batch_data: dict[str, Any],
    *,
    temperature: float,
    min_target_delta: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Rank distinct training sequences by soft-q cleavage ratio.

    Each condition (seq, charge, NCE) contributes equally within a sequence.
    Training is on folded rows, normally one row per condition.  Repeated
    condition rows in a batch are first averaged so that their multiplicity
    cannot silently change the sequence-level target.  Valid bonds alone enter
    each row mean.  Only unordered *different-sequence* pairs with |Δq| >
    min_target_delta contribute to the logistic loss.
    """
    if union_logits.ndim != 2:
        raise ValueError('Sequence ranking expects [batch, bonds] union logits')
    soft = batch_data.get('soft_labels')
    mask = batch_data.get('label_mask')
    seqs = batch_data.get('sequences')
    keys = batch_group_keys(batch_data)
    if soft is None or mask is None or seqs is None or keys is None:
        raise ValueError('Pairwise ranking requires train-only soft q, valid-bond mask, sequence and condition keys')
    if soft.shape != union_logits.shape or mask.shape != union_logits.shape:
        raise ValueError('Pairwise ranking q/mask/logit shapes differ')
    if len(seqs) != union_logits.size(0) or len(keys) != len(seqs):
        raise ValueError('Pairwise ranking row keys do not align')
    valid = mask.bool()
    if (not torch.isfinite(soft[valid]).all()
            or not torch.all((soft[valid] >= 0) & (soft[valid] <= 1))
            or not torch.isfinite(union_logits[valid]).all()):
        raise ValueError('Pairwise ranking valid soft q/logits must be finite and q in [0,1]')
    safe_logits = union_logits.float().masked_fill(~valid, 0)
    safe_soft = soft.float().detach().masked_fill(~valid, 0)
    row_valid = valid.sum(dim=1) > 0
    row_prob = torch.sigmoid(safe_logits).masked_fill(~valid, 0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    row_target = safe_soft.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    condition_rows: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    valid_rows = row_valid.detach().cpu().tolist()
    for row, (seq, key) in enumerate(zip(seqs, keys)):
        if valid_rows[row]:
            condition_rows[str(seq)][key].append(row)

    scores, targets = [], []
    for groups in condition_rows.values():
        condition_scores = [row_prob[rows].mean() for rows in groups.values()]
        condition_targets = [row_target[rows].mean() for rows in groups.values()]
        scores.append(torch.stack(condition_scores).mean())
        targets.append(torch.stack(condition_targets).mean())

    n_sequences = len(scores)
    stats = {'valid_sequences': n_sequences, 'candidate_pairs': n_sequences * (n_sequences - 1) // 2,
             'eligible_pairs': 0, 'zero_pair_batch': 1}
    if n_sequences < 2:
        return safe_logits.sum() * 0.0, stats

    score = torch.stack(scores)
    target = torch.stack(targets).detach()
    ii, jj = torch.triu_indices(n_sequences, n_sequences, offset=1, device=score.device)
    target_diff = target[ii] - target[jj]
    keep = target_diff.abs() > min_target_delta
    n_pairs = int(keep.sum().item())
    stats['eligible_pairs'] = n_pairs
    stats['zero_pair_batch'] = int(n_pairs == 0)
    if n_pairs == 0:
        return safe_logits.sum() * 0.0, stats
    ordered_margin = torch.sign(target_diff[keep]) * (score[ii[keep]] - score[jj[keep]])
    return F.softplus(-ordered_margin / temperature).mean(), stats
