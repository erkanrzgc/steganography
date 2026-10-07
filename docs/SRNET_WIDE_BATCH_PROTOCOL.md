# Frozen eight-row train-only batch-context control — 2026-10-07

Freeze before fitting. Two fixed arms, factors 1 and 32 in that order, no
result-selected reruns. Keep the signal-strength control's metadata-only
24-row selection, manifest SHA
0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14
and train cache SHA
828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028.
Baseline report SHA
a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf.
No validation/test pixels, model selection, downloads or detector changes.

Use unchanged float64 cover + factor*(stego-cover), cast once to float32,
no clipping/rounding. Factor 32 is artificial, not actual payload evidence.
Original JPEG hashes remain sampler identities; derived tensor hashes are
reported separately and must match the frozen baseline.

For each epoch, start from the unchanged eight four-row groups. Each contains
one matched pair from each declared source. Pair these groups into four
eight-row batches with four distinct (source, lineage) keys, two per source.
Use deterministic exhaustive matching: smallest remaining group first,
partner indices ascending, first complete matching. Only eight groups are
allowed (bounded search); fail closed if no complete matching exists.
Keep within-group pair order; no new rows, dropping, padding or repetition.
Report base-group indices, exact ordered batch identity hashes and exposures.
Camera/perceptual source independence remains unverified.

Each arm starts fresh from seed 20261012 and unchanged SRNet/Adamax
.001/.0001/betas [.9,.999]/epsilon 1e-8/foreach false, two Torch threads.
Twenty complete epochs, four updates/epoch, 80 updates and 640 presented rows
per arm; all 26 BN counters must equal 80. Historical four-row baseline has
160 updates and the same 640 presented rows. This matches row exposure, NOT
optimizer steps or batch order: these confounds prevent a unique BN-cause claim.
No resumed weights, BN repair, early stopping, augmentation or optimizer tuning.
Each fit retains its 1,800-second cooperative limit; post-preparation job
limit 3,720s, external wall 3,840s plus five-second kill grace, CPU 7,600/7,601s,
address 8 GiB, file 32 MiB and core dumps disabled. Both arms must complete.

Ordinary stored-BN singleton evaluation at fixed .5 (ties positive) on own
24 training rows and all 24 original rows. These correlated training scenes
are not held-out/generalization data. Preserve old artifacts, caller RNG,
threads, tensors and metadata. Save fresh nonsymlink numeric models locally.
Replay all 48 singleton logits/metrics after reload and nine metadata-first
NumPy float64 forward oracles per model with existing tolerances. Numerical
failure is separate from learning failure. Publish source/protocol/data/model
hashes, every epoch loss/schedule, logits, exposure counts and all failures.

Preregister per-arm goals: final batch loss <= .35, relative loss reduction
>= 25% (zero initial loss means zero reduction), own singleton train balanced
accuracy >= .90. Original-input metrics are descriptive, not tuning targets.
No weights/tensors distributed or installed; no accuracy/support qualification.
Even a pass would be an in-sample learning check, not improved real accuracy.
