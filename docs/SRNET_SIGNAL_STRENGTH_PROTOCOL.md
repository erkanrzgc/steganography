# Frozen train-only real-delta amplification control — 2026-10-07

Freeze before tensor preparation/fitting. Two fixed arms, factors 1 and 32,
in that order, no reruns selected by results. Use the unchanged tiny-sanity
metadata-only 24-row selection, original manifest SHA
0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14
and train cache SHA
828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028.
No validation/test pixels, scores, model selection or detector changes.

Keep cover tensors unchanged. For each matched positive row compute in float64
cover + factor*(positive-cover), then cast once to little-endian float32.
No clipping, rounding, normalization, augmentation or encoded JPEG generation.
Factor 32 creates artificial inputs, not valid embedding/payload-rate evidence.
Original JPEG SHA-256 remains the sampler identity to keep both arms' batch
orders identical; separately hash every resulting float32 tensor's bytes.
Never relabel modified tensors as actual JUNIWARD/UERD samples or new sources.

Each arm starts from the same fresh seed 20261012, unchanged SRNet/four-row
sampler and Adamax .001/.0001/betas [.9,.999]/epsilon 1e-8/foreach false.
Twenty complete epochs, eight updates/epoch, 160 updates per arm; two Torch
threads, no resume/BN repair/early stopping. All BN counters must equal 160.
Each fit has the existing 1,800-second cooperative optimizer limit, total
post-preparation job limit 3,720s; external wall 3,840s plus five-second kill
grace, CPU 7,600/7,601s, address space 8 GiB, file 32 MiB, core dumps disabled.
Both complete arms are required; an interrupted arm is never a complete job.

Ordinary stored-BN singleton evaluation at fixed .5 (ties positive) on each
arm's own 24 training inputs AND all 24 original unamplified training tensors.
These are the same correlated training scenes, not held-out/generalization
tests. Preserve original train/corpus/model artifacts, caller RNG and threads.
Record complete pair/batch schedules, losses, logits, metrics, original row
identities, derived tensor hashes, source/protocol/model hashes and timings.

Preregister learning goals per arm separately: final batch cross-entropy <=.35,
relative loss reduction >=25% from first epoch (zero initial loss means zero
reduction), and own-input stored-BN training balanced accuracy >=.90. Keep each
failure visible; do not tune factors/optimizer/checkpoints from outcomes.
Original-input metrics are descriptive, not additional optimization targets.

Save fresh non-symlink bounded numeric snapshots locally only. Independently
reload each snapshot, replay all own/original singleton logits and nine
metadata-first NumPy float64 oracles (first row per declared source/Q/label/
method cell), with existing logit/score/decision tolerances. Numerical failure
is separate from learning failure. No weights/tensors distributed or deployed.

Passing amplification would show ability to learn artificially stronger paired
signals; failing would retain uncertainty about optimization/content/context.
Neither outcome identifies a unique cause or improves real held-out accuracy.
Any further optimization/curriculum experiment needs a separate frozen plan.
