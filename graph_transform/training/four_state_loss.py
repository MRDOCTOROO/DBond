"""Valid-bond mean soft categorical CE, with the legacy group weighting."""
import torch
import torch.nn.functional as F


def four_state_cross_entropy(logits, targets, mask, weights=None):
    if mask is None or targets.shape != (*mask.shape, 4) or not ((mask == 0) | (mask == 1)).all():
        raise ValueError('four-state target/mask shape mismatch')
    mask = mask.bool()
    target = targets[mask].float()
    if logits.shape != target.shape or not target.numel():
        raise ValueError('four-state valid logit/target count mismatch')
    if (not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any()
            or not torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=2e-6, rtol=0)):
        raise ValueError('Invalid four-state probabilities')
    loss = -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    if weights is None:
        return loss.mean()
    weights = weights.float().reshape(loss.shape)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid four-state sample weights')
    return (loss * weights).sum() / weights.sum()
