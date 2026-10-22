# PCM pair-energy candidate — non-executable design, 2026-10-22

**Draft only. No training, GPU attempt, download, role assignment, checkpoint
or deployment is authorized by this document.** The previous full-control
attempt remains unavailable and must not be restarted. This is a different
candidate family, not an amendment that makes the old v3 experiment pass.

## Evidence motivating the candidate

The [previous control](WAV_FULL_CONTROL_RESULTS_20261021.md) retains one full
v3 seed: training AUC 0.58510078125 and development AUC
0.557425/0.549725 for sequential/scattered 40%-plane-0. The plane-1 values
0.733100/0.748575 do not satisfy the plane-0 target. The second seed has only
797 confirmed updates, no finished fit and no score receipt. There is no
second-seed AUC to impute or recover from a checkpoint.

The first fit took 2,118.488136818 seconds for 1,140 updates. Doubling that
observed fit time gives 4,236.976273636 seconds, already beyond the old
3,600-second total budget before generation and inference. This is a simple
projection, not a measurement of the second seed or proof of a hardware
bottleneck. Independent systemd monotonic time agrees with the supervisor;
changing the clock basis is not a performance remedy. Existing observations
do not isolate decoder, oracle, host-launch or CUDA-kernel costs.

The historical reduced diagnostic already defines the plane-0 pair statistic
in `scripts/audit-wav-positive-control.py:pov_baseline`. Its historical AUC
cannot be transferred to this corpus or candidate. The proposed global
plane-0 feature is that statistic plus one, preserving its score direction.
The other plane and local features are additional hypotheses. No old
training, threshold or source-data selection is repeated here.

## Implemented untrained representation

`core.wav_pair_candidate` accepts bytes through the existing bounded native
mono PCM16 decoder: 160–220,500 frames, accepted native sample rate/duration,
no resampling. It keeps the complete signed 16-bit alphabet. It does not
open paths, read manifests, consume source IDs or infer labels.

For bit plane p, pair bins that differ only in bit p and keep all other bits
unchanged. Let a,b be their counts and N the window sample count:

`equalization = 1 - sum((a-b)^2 / (a+b)) / N`

Empty pairs contribute zero. Compute in float64; only floating-point
round-off is clipped to [0,1]. No quantization of amplitude, learned scaling,
variance normalization or payload/oracle metadata enters the vector.

There are exactly ten values in this immutable order:

| Window | Positions | Features |
| --- | --- | --- |
| global | [0,N) | plane 0, plane 1 |
| q0 | [0,floor(N/4)) | plane 0, plane 1 |
| q1 | [floor(N/4),floor(2N/4)) | plane 0, plane 1 |
| q2 | [floor(2N/4),floor(3N/4)) | plane 0, plane 1 |
| q3 | [floor(3N/4),N) | plane 0, plane 1 |

Each histogram has exactly 65,536 bins. Histogram scratch and feature width
are bounded independently of filenames and corpus size. Inputs are FP32,
shape (1,10), domain `audio-pcm16-pair-energy-mlp-draft-v1`. An optional
10→32→2 ReLU head has **418 FP32 parameters**, no BatchNorm or Dropout.
Torch absence is explicit `unavailable`. Constructing the head neither
fetches weights nor fits them. No optimizer, training recipe, CLI adapter,
production analyzer or deployment registration is added.

## Implemented checks versus missing evidence

Generated-only tests compare every feature against an independent dictionary
scalar oracle, including signed extremes, sparse bins, uneven quarter tails,
both planes, a one-LSB change and the maximum frame count. Exact balanced
pairs give one; unmatched pairs give zero. Other-bit pairs must remain
separate. Invalid RIFF extents, unsupported stereo/width/rate and frame limits
are rejected through the shared decoder. A Path is not silently opened.

A CPU generated-vector forward/backward check requires finite gradients,
identical train/eval logits and unchanged parameters, with zero optimizer
updates. This verifies computation only. No candidate AUC, accuracy,
throughput, native generalization or production qualification exists yet.

## Required separate protocol before a fit

1. Freeze the complete new source/data/acceptance closure, optimizer and
   regularization, class-balance loss, initialization and two seeds. Retain
   the full 20-epoch /72,000-presentation /1,140-update schedules per seed
   unless a separately justified scientific intervention explicitly changes
   them before execution. The old recipe-bound verifier cannot accept a
   different domain.
2. Freeze any feature cache and row metadata contract before building it.
   Keep original train/development source families grouped; keys bind domain,
   full source hash, recipe and row order. Features contain no IDs or labels.
   Hash/opaque-payload validation stays mandatory. The 5,400×10 FP32 values
   alone would occupy 216,000 bytes; manifests, licenses, temporary files and
   logs are additional and require their own bounded accounting.
3. Specify a separate zero-update, full-length resource check. Include
   generation, all feature extraction/oracles, both full fits, all eight
   cell scores and receipt finalization in an explicit wall/storage debit.
   A small head is not throughput evidence. Do not silently substitute CPU
   work into the old CUDA-only contract or add JPEG time.
4. Freeze learning acceptance before looking at candidate results: training
   AUC >=0.60 and both 40%-plane-0 development cells >=0.70 separately in
   both complete seeds; retain all eight cells and resource evidence. These
   are proposed retained targets, not executable permission or new results.
   No threshold search, best-cell selection or missing-seed extrapolation.
5. Keep full per-seed fit and score receipts plus bounded progress even when
   a later stage fails. Missing required exposure/resources stays unavailable.
   No automatic retry. A generated pass is still not native accuracy or
   permission to fit real AudioMNIST.

Changing both representation and head makes this a family comparison; it
cannot isolate which component causes any future gain. The old v3 full seed
and every failed/incomplete receipt remain unchanged. Existing production
features, analyzers, model domains and all-format dataset roles are preserved.
