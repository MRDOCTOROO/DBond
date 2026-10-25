# Stage 13 development record

Status: code prepared; pod structural preflight, real-batch GPU smoke,
same-round five-split training, seed replication, and independent metric audit
are pending. No model gain is claimed.

## Read-only source audit

The source snapshot comes from `3f4ca4d8ed0358bb3aa8b6820cb2e022ad1700dd`
on `exp/parallel-global-cross-k5` with a clean worktree. The existing
`NodeContextConv` already implements masked full-width H256 residue
convolution before global-node packing and FiLM. `single_k5` uses
Conv1d(256,192,k5,pad2) and Linear(192,256); `multiscale_357` uses three
Conv1d(256,64,k3/k5/k7,pad1/2/3) branches, SiLU, concatenation, and the
same Linear(192,256) residual. The original four-state loss and concat head
remain selected by the frozen k5 config. `pre_synthesis` resolves the source
state mask to charge/mass without intensity and the environment mask to NCE
without scan number.

The k5 and c64 multiscale branches have equal trainable parameter formulas:

`256×192×5 + 192 + 192×256 + 256`

`256×64×(3+5+7) + 3×64 + 192×256 + 256`

The formulas are algebraically equal. A separately widened k5 with the same
budget would be the original 192-channel S13-A again. S13-D therefore
references A's weights, predictions, and metrics and is excluded from the
independent-run count. B has three 32-channel branches and its own checkpoint
mode marker, so an A/C checkpoint cannot silently load as B.

## Pending pod gates

1. Verify all three configs against the historical k5 protocol after
   `pre_synthesis` resolution, and independently check branch shape, residual,
   same padding, short sequences, poisoned padding, real two-step gradients,
   four-state output, global/FiLM order, strict checkpoint mode rejection,
   and exact A=C parameter count.
2. Audit data availability, five sequence splits and source/target SHA, then
   run real train128 and validation512 GPU smoke. Record peak allocated and
   reserved memory. The smoke calculates no outer-test metrics.
3. After Stage 12 is audited, train A/B/C across all five splits with the
   same seed42 fold rule, validation split, optimizer, scheduler, early stop,
   checkpoint selection, and threshold. Record best epoch, train/validation
   gap, per-epoch time, inference time, and resource use. The Stage 12 receipt
   must be under the new `dbond_s12_gated_fusion_20261026_ops` namespace.
4. Run independent raw-prediction audit. Compare B−A and C−A on the same
   five splits. Apply the predeclared Bond-F1 +0.20 percentage-point,
   4/5-win, AUC and q-Brier limits; report every metric even if the gate fails.
5. Only one qualifying candidate may proceed to paired seed43/45 checks.
   Existing historical c64 experiments are context, not results of this round.

The five outer splits overlap and were used in earlier model exploration, so
their results remain developmental evidence rather than an independent final
test. Any paper recommendation awaits the complete audited round.
