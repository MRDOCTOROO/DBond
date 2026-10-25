"""Valid-bond joint CE with optional conditional weighting and union Brier."""
import math
import torch
import torch.nn.functional as F


def four_state_cross_entropy(logits, targets, mask, weights=None, conditional_weight=1.0,
                             union_brier_weight=0.0):
    """BCE(union q) + alpha * q * CE(state | positive) + lambda * Brier(union).

    Alpha=1 uses the original categorical-CE expression exactly, preserving the
    existing control's floating-point computation. Alpha changes supervision,
    never the inference probabilities or the definition of the four targets.
    Lambda=0 skips the Brier computation entirely and preserves the historical
    loss/gradient path. Both terms use the same valid-bond targets and weights.
    """
    if isinstance(conditional_weight, bool) or not isinstance(conditional_weight, (int, float)):
        raise ValueError('four_state_conditional_weight must be a finite nonnegative number')
    if not math.isfinite(conditional_weight) or conditional_weight < 0:
        raise ValueError('four_state_conditional_weight must be a finite nonnegative number')
    if (isinstance(union_brier_weight, bool) or not isinstance(union_brier_weight, (int, float))
            or not math.isfinite(union_brier_weight) or union_brier_weight < 0):
        raise ValueError('four_state_union_brier_weight must be a finite nonnegative number')
    if mask is None or targets.shape != (*mask.shape, 4) or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('four-state target/mask shape mismatch')
    mask = mask.bool()
    target = targets[mask].float()
    if logits.shape != target.shape or not target.numel():
        raise ValueError('four-state valid logit/target count mismatch')
    if (not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any()
            or not torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=2e-6, rtol=0)):
        raise ValueError('Invalid four-state probabilities')
    logits = logits.float()
    if conditional_weight == 1.0:
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1)
    else:
        union = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        # Retain the serialized q, including q=0 and q=1, without division by q.
        union_loss = F.binary_cross_entropy_with_logits(union, 1 - target[:, 0], reduction='none')
        conditional_loss = -(target[:, 1:] * F.log_softmax(logits[:, 1:], dim=-1)).sum(-1)
        loss = union_loss + conditional_weight * conditional_loss
    if union_brier_weight > 0:
        union_logits = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
        union_probability = torch.sigmoid(union_logits)
        union_target = 1 - target[:, 0]
        loss = loss + union_brier_weight * (union_probability - union_target).square()
    if weights is None:
        return loss.mean()
    weights = weights.float().reshape(loss.shape)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid four-state sample weights')
    return (loss * weights).sum() / weights.sum()
