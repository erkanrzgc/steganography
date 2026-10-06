# Frozen train-only tiny-subset SRNet sanity — 2026-10-07

Freeze before selecting real rows or fitting. This is an in-sample learning
control after failed complete two-source validation, not accuracy evidence.
Original models, protocols, data and failures remain unchanged. No validation
pixels/scores are loaded; no threshold tuning, deployment or model download.

Use unchanged manifest SHA
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`,
train cache SHA `828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028`
and unrounded float32 Y/256 preprocessing. Select by train metadata only:
lexicographically first four ALASKA2 lineages (declared quality unknown),
and first two BOSSbase-1.01 lineages with complete Q75 and Q95 groups.
Retain cover/JUNIWARD/UERD for every selected quality group. Exactly 24 rows,
six original declared scenes/eight quality groups, 16 pairs/eight four-row
updates per epoch. Quality variants are not new independent scenes.

Seed 20261012, 50 complete epochs, unchanged two-source pair/batch sampler,
400 updates total. Fresh seeded network, no resume, augmentation, input scaling,
BN repair or validation-based stopping. CPU two threads, Adamax .001/decay
.0001/betas [.9,.999]/epsilon 1e-8/foreach false, constant LR. Optimizer deadline
1,800 seconds, external hard wall 1,920 seconds plus 5-second kill grace,
8 GiB address space, CPU 3,660/3,661 seconds, file 32 MiB, core dumps disabled.
Timeout/incomplete fit fails; no partial model or budget extension.

Record all epoch pair/batch order hashes and losses, original selected SHA-256
and source/lineage/quality/method metadata. Check all model BN counters equal
400. Evaluate only the selected train rows in ordinary stored-BN all-stage eval,
one image at a time at fixed .5 threshold (ties stego). No batch-stat inference.
Preserve caller RNG/threads; no extracted artifacts execute and outputs are
fresh/non-symlink numeric model plus portable report, no host paths/secrets.

Preregistered sanity objectives: final training-batch mean cross-entropy <= .35,
at least 25% loss reduction from first epoch, and final stored-BN training-set
balanced accuracy >= .90. Report each objective separately, including failure.
These are deliberately in-sample sanity gates, not detector/release gates.
Do not infer generalization even if all pass; failed sanity motivates train-only
optimization/gradient/input investigation before longer full-corpus compute.
Metrics/numerical export/untouched-source qualification remain separate.
