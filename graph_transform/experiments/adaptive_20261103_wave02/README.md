# Regularization wave 2

The running training snapshot is **3566c14d461dd5e9d4d2a51a53929dc13da52efd**.
Later commits on this branch add audit tools and the ledger only; they do not
change the frozen Pod training source. Main and historical results stay intact.

Wave 1 completed 25/25 training jobs and 50 policy checkpoints. The independent
CPU review, saved-prediction recomputation, valid-label mask/source/seed checks,
training/target-sidecar SHA binding and declared-change checks all passed. The
exact completion event was acknowledged. These are inner-validation results,
not new outer-test paper results. Relative to the same-wave full max-F1 model:

| Ablation | Bond-F1 change (pp) | Example harmonic F1 change (pp) | Subset change (pp) |
|---|---:|---:|---:|
| No theory | +0.1566 | +0.3494 | -0.0198 |
| No GCN/GAT messages | -1.6919 | -1.7542 | -1.5096 |
| 1GCN + 1GAT | -0.3793 | -0.4999 | -0.7386 |
| Binary head + union BCE, k5 retained | -0.4123 | -0.3860 | -0.0984 |

Twenty-one of 25 max-F1/composite pairs had identical model tensors. Those pairs
use one canonical prediction pass for comparison; replay noise is not a gain.
None achieved the desired large joint improvement. The best paper model is not
replaced. Independent wave-1 result SHA256:
`63f2d27b286450215a4f16d5278947a80a7610ebf72b5cb796c29d277c0b5f02`.

Wave 2 was uniquely submitted at 2026-11-02T20:55:43.389193Z. It tests four
single-factor changes on the original four-state graph route: learning rate
3e-4, dropout .3, AdamW weight decay .001, and hidden width 128 with k5 retained.
Each uses five fixed base42 sequence splits, for 20 new training tasks. Width
128 has 611,948 parameters; the other variants retain 2,002,284. No attention
heads, losses, data rules or feature routes are additionally changed.

Full-reference checkpoints/predictions from wave 1 are reused by exact SHA;
there are **zero new full-reference trainings**. The source revisions differ,
but model/trainer/loss/metrics/selector production file SHAs match exactly.
Both max-F1 and the frozen .4/.4/.2 composite checkpoint policy are retained;
F1 early stopping and p>.5 remain unchanged. External test paths are absent.
Future selection must use each outer split's own inner validation data.

Smoke SHA:
`de8b177476f14b7ea211e8688cf3894ea326b542bdf7515a8801b7362bf2be67`.
Reference-reuse receipt SHA:
`9431a06fa4217f5ab7a3ac9bd49f48dfe7f09694f693ee4b0a90cd0869c71475`.

The paired terminal review explicitly distinguishes the old reference root and
revision from new candidate artifacts. It is prepared and syntax-checked, not
yet run on unfinished candidates. It preserves raw reports, checks identical
states and recomputes all saved valid-bond metrics. The supplement checks only
the declared single-factor changes and binds data/targets/terminal sources.

Archived external baseline classification means were independently verified;
the old internal row split had sequence overlap, so grouped baseline retraining
remains required for a matched model-selection comparison. No metrics are
arbitrarily lowered to improve the apparent DBond-GT advantage.
