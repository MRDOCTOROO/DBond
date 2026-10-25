# Stage 13: independent multiscale Conv1D round

This is a new Stage 13 namespace. Its three trainable arms use the original
single-global-node, H256, 3 GCN + 2 GAT, FiLM, static-edge, late-theory,
four-state **concat MLP** model, alpha=1, union-Brier training weight=0. The
five repeated external sequence splits are 1222, 2252, 3514, 6072, 9075.
The seed42 training and internal split rules remain paired across arms.

The four requested rows appear in `experiment_matrix.csv`. The existing k5
module has 192 output channels. Its parameter count already equals the
k3/k5/k7 model with 64 channels per branch. Therefore S13-D is an identity
alias of S13-A, not another training run or another result. There are fifteen
first-round trainings: A/B/C across five splits.

The only new model route is `multiscale_357_c32`. The existing c64 and k5
modules keep their weights, shapes, masking, projection initialization, and
checkpoint markers. Every branch convolves the full H256 residue encoding
before global-node packing and FiLM. Invalid padded features are cleared
before convolution; invalid branch outputs and final residual outputs are
cleared afterward. The 3/5/7 branches use same padding, SiLU, concatenation,
and a zero-initialized projection back to H256.

All three configs set `evaluation.export_from_evaluate_cache: true`. The
prediction CSV is exported from the same valid-bond logits used by the metric
calculation, with nine significant digits for float32 round trips. The
snapshot records `evaluation_id`, the selected checkpoint path, source SHA,
config SHA, metric CSV SHA, and raw prediction CSV SHA for each fold.

The pod entry point is `graph_transform/scripts/stage13_multiscale_ops.py`.
An orchestrator first runs its `smoke` action in a fresh
`/mnt/pvc/graphtrans/dbond_s13_multiscale_conv_20261026_ops` directory, then runs `launch` only
after receiving an audited Stage 12 receipt and checking GPU availability.
The required receipt is
`/mnt/pvc/graphtrans/dbond_s12_gated_fusion_20261026_ops/independent_audit/audit_receipt.json`;
its audited status, ops path, and run root must agree with the Stage 12 launch.
`launch` records the source commit, config hashes, worker PID, and independent
batch root under `/mnt/pvc/graphtrans/DBond/checkpoints/graph_transform/`.
The worker writes `terminal.json` once on completion or failure.
The completed batch writes `registry.json` and `audit_manifest.json`; it is
**not** considered audited until the independent auditor checks all fifteen
weights, source bindings, raw predictions, and metrics. D is recorded only as
an alias. Stage 14 may start only after that audit passes.

The `audit_manifest.json` interface is `{stage, ops, root, source_commit, arms}`.
Each arm/fold record includes seed, prediction CSV, test CSV, metric CSV,
checkpoint, runtime config, internal train/validation CSVs, and SHA-256 hashes.
The worker rechecks folded train, test, full-row validation, target manifest,
and internal split hashes after training, then strictly restores every selected
checkpoint with its mode and seed/split metadata before releasing the manifest.
It asks the independent auditor to recalculate Bond-F1/Acc, ROC/PR AUC, MCC,
spectrum P/R/F1, q-Brier, ECE, q_seq, Top10, and Top20. Resource records from
the pod smoke and training batch are separate from the metrics. Formal fold
records also retain the best-epoch train/validation F1 gap, each epoch wall
time, final test inference time, total training time, and peak allocated GPU
memory from the original fold log.

No pod smoke, training, or metric calculation has been run from this local
development snapshot. The original source snapshot was archived from
`3f4ca4d8ed0358bb3aa8b6820cb2e022ad1700dd`; this isolated checkout has
its own local Git history and must never be pushed to an unknown remote.
