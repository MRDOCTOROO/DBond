"""Training-only empirical count-distribution CE; no inference decoder."""
import torch
import torch.nn.functional as F


def cardinality_cross_entropy(logits, mask, histogram):
    """Uniform-condition mean of count CE divided by each row's valid length.

    A log-domain Poisson-binomial DP uses the existing four-state union
    probabilities. Finite unreachable states avoid undefined -inf gradients;
    no clipping, threshold, count prior or learned parameter is introduced.
    """
    mask = mask.bool()
    lengths = mask.sum(-1)
    if (mask.ndim != 2 or (lengths < 1).any()
            or not torch.equal(mask, torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None])
            or logits.shape != (int(lengths.sum()), 4)
            or histogram.shape != (mask.shape[0], mask.shape[1] + 1)):
        raise ValueError('Invalid count-loss valid-bond layout')
    histogram = histogram.float()
    unsupported = torch.arange(histogram.shape[1], device=mask.device)[None] > lengths[:, None]
    if (not torch.isfinite(histogram).all() or (histogram < 0).any()
            or histogram[unsupported].count_nonzero()
            or not torch.allclose(histogram.sum(-1), torch.ones_like(lengths, dtype=torch.float32), atol=2e-6, rtol=0)):
        raise ValueError('Invalid empirical cardinality target')
    logstate = F.log_softmax(logits.float(), -1)
    logzero = logits.new_zeros(mask.shape, dtype=torch.float32)
    logone = logits.new_zeros(mask.shape, dtype=torch.float32)
    logzero[mask] = logstate[:, 0]
    logone[mask] = torch.logsumexp(logstate[:, 1:], -1)
    dp = logstate.new_zeros((mask.shape[0], 1))
    for j in range(mask.shape[1]):
        a = dp + logzero[:, j:j+1]
        b = dp + logone[:, j:j+1]
        update = torch.cat((a[:, :1], torch.logaddexp(a[:, 1:], b[:, :-1]), b[:, -1:]), -1)
        unchanged = torch.cat((dp, dp.new_full((len(dp), 1), -1e30)), -1)
        dp = torch.where(mask[:, j:j+1], update, unchanged)
    return (-(histogram * dp).sum(-1) / lengths).mean()
