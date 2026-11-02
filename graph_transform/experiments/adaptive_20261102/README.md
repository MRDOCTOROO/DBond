# Adaptive DBond research, authorized 2026-11-02

This branch leaves main and all historical experiments untouched. The initial
training snapshot is **addf42cae71f19c4bd1894249da3623b5f00034c**; later ledger
commits are not a claim that the running snapshot changed.

The first validation-only wave compares full k5/four-state, no theoretical
features, no GCN/GAT messages (bond-edge readout retained), shallow 1GCN+1GAT,
and binary-union supervision with k5 retained. Each uses the five fixed outer
sequence splits and their retained base42 internal split. The 25 initial
training tasks are not a total search cap. The original model/loss production
implementation is unchanged, except the metric API probability-input fix.

Each trajectory saves both strict max validation Bond-F1 and a shadow selector
S = .4 Bond-F1 + .4 harmonic(mean example precision, mean example recall)
+ .2 subset exact-match accuracy. All are on the 0–1 scale. Early stopping
remains max-F1. External-test paths are absent throughout development. The
selection weights are frozen before training, not chosen from external scores.

The probability-input flag of the historical metric API was accepted but
ignored. This branch canonicalizes probability inputs to logits while leaving
the default logit training path unchanged. The first failed preflight is
retained; the second passed all strict assertions, including longest-residue
128-row training steps, finite gradients, strict CPU round trips for each arm,
all five retained baseline weights and sequence-disjoint train/val/test checks.

An independent read-only recomputation of 20 historical s/m/AF/AF-opt runs has
completed. No classification-mean inflation was found relative to the manuscript
table. Bond-F1 means were .7775006111, .7690521832, .7925105241, .7970403864.
This does not establish historical source identity or fair checkpoint selection.
The stored row-level internal split protocol has 408–412 overlapping sequences
per fold. Corrected group-by-sequence baseline retraining is required before
claiming a matched selection comparison. No arbitrary downward correction is
permitted. The old q reevaluation script already applies inverse-logit to its
probability inputs and therefore does not directly exercise the ignored flag.

The worker schedules only idle GPUs and retains all negative results. Training
and metric recomputation are Pod-only. Candidate promotion must use the matching
outer split's own inner validation data; overlapping outer splits and repeated
historical development do not provide a fresh confirmatory test set. Later
waves may tune regularization, simplify architectures, calibrate conditions, or
study label dependencies and controlled combinations, with new frozen commits,
smoke checks and unique launches. Independent seeds and new sequence-isolated
data are needed before a paper-main-model change.

Research targets are deliberately stated as goals, not promised outcomes:
roughly +2 percentage points Bond-F1, or +2 points subset accuracy without more
than .2 points loss in either Bond-F1 or example harmonic F1. Full metrics,
uncertainty, paired fold differences, inference/training resources and provenance
must accompany any improvement claim.

References: [SGC](https://proceedings.mlr.press/v97/wu19e.html),
[AdamW](https://arxiv.org/abs/1711.05101),
[nested selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
[subset loss](https://proceedings.mlr.press/v37/gasse15.html),
[conditional Bernoulli mixtures](https://proceedings.mlr.press/v48/lij16.html).

Data, raw predictions and model weights are not added to this repository.
