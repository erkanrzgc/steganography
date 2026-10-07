# Frozen optimizer/exposure-matched microbatch control — 2026-10-07

Freeze before fitting. Two fixed arms, real-delta factors 1 and 32 in that
order. No result-selected rerun. Baseline is the complete eight-row control
`benchmarks/srnet-wide-batch-20261007.json`, SHA-256
ae828ea18337fcff82f40506717c7fe5f33d664e380537632a4ac1ca4dcf9427.
Use the unchanged metadata-first 24 training rows and original float tensors,
manifest SHA
0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14
and train cache SHA
828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028.
No validation/test pixels, downloads, tuning or detector changes.

Use the exact existing eight-row batch schedule, original JPEG identities,
derived tensor hashes and first-complete matching. For each eight-row group,
clear gradients once, forward its first four rows then last four rows in
unchanged order. Each microbatch contains two matched source/lineage pairs.
Backpropagate half its mean cross-entropy, accumulating both gradients;
check finite logits, loss and accumulated gradients after each microbatch.
Only then perform one Adamax update and validate all model state arrays.
Report mean of both unscaled microbatch losses for that optimizer group.
No gradient clipping, loss rescaling beyond division by two, or mixed precision.

Both arms start fresh from seed 20261012, unchanged SRNet and Adamax
.001/.0001/betas [.9,.999]/epsilon 1e-8/foreach false, two Torch threads.
Twenty epochs/four updates each: 80 optimizer steps and 640 presented rows,
exactly matching the historical eight-row model. Microbatch size four and
160 BN forward updates differ from the baseline's size eight/80 BN updates.
All 26 BN counters must equal 160. No resume, BN repair, early stopping,
augmentation or changed pair order. This controls exposure/optimizer order
but does not equate BN statistics or prove a unique learning-failure cause.

Each fit has the existing 1,800-second cooperative limit; job limit 3,720s
after data preparation; external wall 3,840s plus five-second kill grace,
CPU 7,600/7,601s, address 8 GiB, file 32 MiB, core dumps disabled. Both arms
must finish for a complete job. Save bounded numeric snapshots locally only,
fresh nonsymlink output, never overwrite historical artifacts.

Evaluate ordinary stored-BN singleton inference on all 24 own training rows
and all 24 original rows at .5, ties positive. Reload and replay all 48 logits
and metrics per model, plus nine metadata-first NumPy float64 forward oracles
per arm using existing tolerances. Record losses, exact schedules/exposures,
source/protocol/baseline/cache/model hashes, logits and separate numerical gates.
Preregister goals: final mean microbatch loss <= .35, relative loss reduction
>=25% (zero initial loss means zero reduction), own singleton BA >= .90.
Original-input metrics are descriptive, not tuning targets. Preserve failures.

24 rows from six scenes are a learning sanity subset, not enough to qualify
generalization; source/camera independence is unproven. Factor 32 is artificial,
not actual embedding/payload evidence. Larger training data may matter, but
this control cannot establish dataset size as the cause of prior failures.
Even passing is not detector qualification. No weights/tensors distributed,
model installation or primary detection changes.
