# Physical generated epoch-resume protocol — frozen before execution

This is a new engineering experiment, not a repeat of or replacement for the
failed five-epoch single-job timing experiments. Bind this file's SHA-256 and
all 16 executed sources before the explicit WSL job. Use the actual RTX 5060
Laptop, Torch FP32 IEEE, deterministic CUDA/cuDNN, no autocast/TF32/fallback,
host cgroup <=8 GiB, allocator <=4 GiB and a 180-second whole-probe deadline.

The fixed wave/checker 4x1x256x256 input, seed 20261008, unchanged Adamax
settings and two epochs of 32 identical four-row updates are fixed. Compare
one uninterrupted two-epoch fit with a one-epoch fit saved to checksum-bound
numeric NPZ and restored into a newly constructed model/optimizer for epoch
two. Total 128 generated updates. Require exact model/BN, Adamax moments/steps,
CPU/CUDA RNG bytes and cumulative loss records, not approximate logits alone.
Both paths are in the same isolated process, but reconstruct model/optimizer
and restore disk state; this does not assert cross-version/device portability.

Time the 63 inter-fetch intervals of the uninterrupted fit; discard the first
16, retain exactly 47, compute NumPy linear p95. For each prospective real
whole-epoch job use estimate 120s + 2 * p95 * scheduled optimizer updates.
Only attempt an epoch if its estimate fits the unchanged 1800-second job cap.
The new job segmentation is explicit, not a claim that five epochs fit one
job, and every actual job still has independent hard limits and a deadline.

Reject incomplete/nonfinite/mismatched results without selecting another seed
or dropping arrays. Preserve failures. Source changes invalidate the probe.
No real data, deployment, accuracy estimate or corpus/model redistribution.
