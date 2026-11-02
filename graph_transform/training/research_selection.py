"""Frozen validation-only selectors; no prediction/test/threshold optimization."""
import math

WEIGHTS = (0.4, 0.4, 0.2)


def components(metrics):
    p, r = float(metrics['ex_precision']), float(metrics['ex_recall'])
    values = (float(metrics['f1']), 2*p*r/(p+r) if p+r else 0., float(metrics['subset_acc']))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid validation selector component')
    return values


def composite_score(metrics):
    return sum(w*v for w, v in zip(WEIGHTS, components(metrics)))


class ResearchSelection:
    """Parallel shadow selection does not affect the original F1 early stopping."""
    def __init__(self, config):
        if config.get('schema') != 1 or config.get('weights') != list(WEIGHTS):
            raise ValueError('Research selector requires frozen schema=1, weights=.4/.4/.2')
        self.best_score = float('-inf')
        self.best_epoch = 0

    def update(self, metrics, epoch):
        score = composite_score(metrics)
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
        return improved, dict(schema=1, weights=list(WEIGHTS), score=score,
                              best_score=self.best_score, best_epoch=self.best_epoch,
                              components=dict(zip(('bond_f1','example_f1_harmonic','subset_acc'), components(metrics))))
