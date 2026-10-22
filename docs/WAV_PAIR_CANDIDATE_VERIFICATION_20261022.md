# Untrained PCM pair candidate checks — 2026-10-22

**Computation checks only. No fitting, GPU attempt, native-media access,
checkpoint or candidate accuracy result.** The
[design snapshot](../benchmarks/wav-pair-candidate-design-20261022.json) has
zero allowed fit/GPU attempts. Its
[non-executable draft](WAV_PAIR_CANDIDATE_DRAFT_20261022.md) requires a new
complete contract before another experiment.

## Computation

Twenty-six generated-only tests pass. Independent dictionary counts match
the vectorized full-alphabet statistic over both planes and every quarter,
including signed extremes, uneven tails and the maximum 220,500 samples.
Invalid native domains and corrupt containers are rejected by the shared
bounded decoder. The new module has 100% statement coverage.

A generated-vector CPU forward/backward check exercises the 418-parameter
head in FP32: gradients are finite, train/eval outputs agree and parameters
remain unchanged. No optimizer is constructed or stepped for this candidate.
Torch absence produces an explicit unavailable error.

## Important representation limit

The single-LSB unit fixture demonstrates one constructed change reaching the
head; it is not a guarantee for every waveform. A separate zero-update
counterexample confirms this exactly:

- Start with 161 alternating samples 0,1, ending in an extra zero.
- The last quarter has 21 zeros /20 ones; global counts are 81/80.
- Flip only the final sample's bit 0: those pairs become 20/21 and 80/81.
- Squared pair imbalance is invariant under swapping the two counts. Both
  the float64 features and FP32 head inputs remain identical.

The statistic also discards ordering inside each quarter. This candidate is
a lossy statistical representation, with no raw-PCM equivalence or universal
single-bit preservation claim. The complete raw-PCM/difference skeletons and
existing production features stay unchanged. No detection accuracy can be
inferred from a computational gradient test or this counterexample.

## Evidence preserved

The old 28-source v3 closure still matches its frozen runner protocol. A
local read-only replay of the cached old result remains **unavailable**. The
completed old seed still fails its plane-0 learning rule; no missing second
seed, threshold, score or CUDA peak was reconstructed. Its human wall-clock
discrepancy does not justify a timer change or another fit.

The doubled first-fit duration is an arithmetic projection, not a measured
new-candidate runtime. The head's smaller parameter count is likewise not
throughput evidence. Full-length feature/oracle and model profiling needs
its own explicit zero-update resource contract before a learning attempt.

## Future confirmation must account for adaptive reuse

The previous control's development metrics informed this candidate design.
Reusing those same generated development sources is an adaptive comparison,
not independent confirmation. Before any candidate fit, a separate contract
must disclose reuse and freeze a fresh generated confirmation seed/split and
its accounting, or explicitly limit the claim to development exploration.
Existing real reserved media stays closed and historical roles are not
reassigned. The old v3 weights were not retained, so its cached scores cannot
be described as a matched fresh-corpus comparison. Any new reference fit also
needs separate authorization and a frozen budget.

## Full software gate

The full unchanged gate passes **3,554 tests /two explicit CUDA skips**,
**96.939193806% total coverage** and **100% candidate-module coverage**.
Ruff, 219-file mypy and diff checks pass. The local CPU host's actual-CUDA
skips remain skips, never GPU passes. The Python 3.12–3.14 CI matrix was not
run locally and remains required. See the
[verification receipt](../benchmarks/wav-pair-candidate-verification-20261022.json).
