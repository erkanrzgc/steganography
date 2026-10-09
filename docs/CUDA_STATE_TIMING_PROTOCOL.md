# On-device numeric-state CUDA timing — frozen before physical execution

Separate measurement of changed execution. Preserve both earlier protocols
and original reports. The engine still checks the entire state after every
optimizer step: exact keys, shapes, float32/int64 dtypes, finite float values,
nonnegative BN variances and integer counters in [0, 10000000]. Tensor layout
must be dense strided and all state tensors share CPU or primary CUDA device.
Explicit CUDA performs these checks on device with one host boolean decision;
final CPU model conversion is followed by the original NumPy validator.
Legacy CPU validation and all floating-point optimizer arithmetic remain.

Before physical execution require NumPy/tensor acceptance parity for malformed
and boundary states, no input-state mutation or per-step CPU copies, generated
exact update/loss/state equivalence, and full regression checks. These CPU
checks are not GPU kernel equivalence, independent mathematics or timing.

Use existing isolated `research_cuda_profile` with fresh output and all 16
before/after source hashes. Same wave/checker four-row FP32 inputs, seed
20261008, 64 updates, discard 16 of 63 inter-fetch intervals and retain all 47.
TF32/AMP/fallback off, strict deterministic CUDA, kernel resident RAM <=8 GiB,
allocator <=4 GiB, internal 180s/parent 210s/CPU 360s/output 1 MiB/core zero.
Temporary concatenations are bounded by the fixed state shapes; record actual
allocator peaks. No corpus, validation inference, exported/deployed model.

Unchanged eligibility: five epochs *3288 =16440 updates;
estimate =120 +2 * steady-interval p95 *16440, eligible only if <=1800s.
No post-result change to epochs/safety factor/ceiling. Eligibility does not
authorize a real fit before a separately frozen and verified full-source
stored-normalization train-only learning/evaluation workflow.

Single-run comparisons are descriptive: thermal/power state, Windows GPU users
and concurrent Kali work are unmatched. Do not attribute a causal speedup,
long-run training guarantee, real learning or accuracy to these measurements.
Preserve every original measurement and failed gate.
