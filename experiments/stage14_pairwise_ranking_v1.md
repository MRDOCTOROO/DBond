# Stage 14: minimal train-only pairwise ranking test

The revised authorized budget trains **only** the original single-global-node,
single-scale Conv1D `k=5`, H256 four-state model with pairwise-logistic loss
weight `lambda=0.05`, temperature `tau=0.1`, and minimum target gap `0.05`.
The five fixed sequence-isolated outer folds use training base seed 42 and
internal split base seed 42. This is five new models, one per fold. Lambda
0.01 and 0.10 are cancelled; neither is trained or used for selection.

The pairwise term acts on the existing union bond logit. Folded outer-training
rows supply soft-q labels. Within an ordinary shuffled minibatch, each
condition is averaged first, then conditions are averaged within a sequence;
only distinct-sequence pairs whose mean soft-q gap exceeds 0.05 contribute to
the logistic objective. The training DataLoader retains its original random
row shuffle, batch size 128, drop-last behavior and group-uniform folded
training weights. The algorithm does **not** introduce a sequence-aware
sampler. Batch-local pairs are a stochastic training proxy for the complete
sequence-level `q_seq` metric; they are not full-dataset pair enumeration.

The reviewed Stage 11 single-global-node `k=5`, seed42 weights are the
candidate comparison baseline **only after** the pod verifies model, data,
training, optimizer, checkpoint-selection, threshold and padding rules. The
new launcher checks the historical configuration against a zero-weight
control configuration after excluding only artifact locations and the added
ranking fields. Pod smoke constructs the zero-weight model in memory and
verifies exact four-state CE/gradient, identical initialization, pair
availability and absence of RNG consumption. The zero-weight model is not
trained. Stage 11 result, review and checkpoint registry hashes, all five
train/validation/test/target hashes, and source lineage are frozen into the
Stage 14 launch and manifest. If these checks fail, stop; do not claim the
historical baseline is matched. A new five-fold lambda-zero control is
permitted only if an actual sampling/protocol mismatch is identified and
explicitly recorded under the user's conditional baseline rule.

No external test data select the epoch, threshold, lambda or seed. The fixed
threshold is 0.5; checkpoint selection uses internal valid-bond F1 on the
complete original-spectrum validation source. After five models finish, the
frozen worker registers actual checkpoint/config/prediction/log/source SHA
and terminal status. Independent pod audit strictly restores each checkpoint
and recomputes same-pass padding-free metrics. Because the candidate-only
batch has no new lambda-zero arm, the formal batch audit must be followed by
a separately SHA-bound pod comparison to the reviewed historical Stage 11
single-k5 seed42 outputs; a self-comparison of lambda0.05 is not an efficacy
result. Report all five paired differences and sample SD, including any
negative bond-F1/spectrum-F1/calibration outcomes and ranking gains. Apply
the previously declared Stage 14 screening gates without changing them.

Stage 14 starts only after the revised minimal Stage 13 has an independent
audited receipt at
`/mnt/pvc/graphtrans/dbond_s13_minimal_c64_20261026_ops/independent_audit/audit_receipt.json`.
The dedicated Stage 14 ops path is
`/mnt/pvc/graphtrans/dbond_s14_minimal_pairwise_20261026_ops` and its run root
is under `.../stage14_minimal_pairwise_batch/<timestamp>`. The source must be
frozen to a new exact Git commit, deployed in a separate pod checkout,
smoked once, and launched once on genuinely idle GPUs. Old four-arm source
commit `1eb4570` must not be launched. Five overlapping folds and one seed
are exploratory evidence; passing the engineering gate permits at most one
predeclared seed43 five-fold replication, not automatic manuscript promotion.
