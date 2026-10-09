# Physical generated CUDA throughput — 2026-10-09

The [original timing report](../benchmarks/cuda-wsl-generated-profile-20261009.json)
was fetched over authenticated, pinned-host local SSH. Its exact SHA-256 is
`3446a184cf126d4a617b44b896883f9387d7edff5a323dafd824d34f4c24f29b`.
All 16 execution-source hashes match commit `e152f8d`; the
[frozen protocol](CUDA_THROUGHPUT_PROTOCOL.md), SHA-256
`1a2b108cc5f1c24a2a067fa034684b0105bc3c2fc5b049c5c88de630dc6a45a8`,
was committed/pushed before physical execution. No measurements selected the
epoch count, percentile, safety factor or ceiling.

## Measurement and prospective decision

WSL2/Python 3.12.3, Torch 2.14.0+cu130, CUDA runtime 13.0, RTX 5060 Laptop,
compute capability 12.0. Exactly 64 generated four-row FP32 optimizer updates,
unchanged production engine, seed 20261008. Finite-loss/gradient and full-state
checks synchronize before the next fetch; intervals include those costs, not
only queued kernels. Discard 16 initial intervals; retain all 47 steady ones.

- Steady-interval p95: **0.11254113879931538 seconds**.
- Fixed candidate: five epochs, 3,288 updates each, 16,440 total.
- Conservative estimate: `120 + 2 * p95 * 16440` = **3820.3526437214896s**
  (63.67 minutes), greater than the unchanged **1800s** ceiling.
- Candidate decision: **ineligible to attempt**. No real fit was started;
  no epoch reduction, cap extension or safety-factor adjustment.
- Generated job elapsed: 25.013722224999583s including initialization. Actual
  allocator peaks: allocated 1,057,371,648 bytes, reserved 1,438,646,272 bytes;
  4 GiB allocator and 8 GiB kernel resident-RAM bounds, no TF32/AMP/fallback.

This is an estimate, not an observed full-fit duration, thermal endurance test
or identification of the bottleneck. Identical generated images exclude real
I/O/content variation and Windows workload changes. No real data, validation
inference, real trained model, numerical independent oracle, deployed weights
or accuracy gain. Earlier failed real detector scores remain unchanged.

## Next bounded step

Investigate shared-engine synchronization/state-check costs using separately
specified, verified execution optimizations. Preserve every numerical and
resource guard, CPU behavior and the full original report. Freeze a separate
protocol before timing changed execution; do not reuse the old protocol as if
the engine were unchanged. Full-source stored-normalization train-only learning
and independent held-out qualification are still required before deployment.

## Verification

Before timing, Kali/Python 3.11.14: **1627 passed, one actual-CUDA test explicitly
skipped**, 32 warnings, 303.32s, total coverage **95.47%**. The 27 generated/
control-flow profiler tests cover its two files **99.21%**; they are not live
hardware or accuracy tests. Ruff, mypy (140 files), diff checks and wheel/sdist
builds pass; 150/298 members, no raw data, numeric weights, SSH secrets or
provider-specific memory. WSL evidence is actual timing, not a full-suite pass.
Recorded-evidence tests additionally pin report/protocol bytes, source hashes
and independently recompute the percentile/budget; these are static regressions.

After adding recorded-evidence tests: **1629 passed, one actual-CUDA test
explicitly skipped**, 32 warnings, 247.74s, total coverage **95.47%** on Kali.
Ruff, mypy (140 files), diff checks and fresh wheel/sdist builds pass;
150/299 members, the same model/data/credential exclusions verified. Python
3.13–3.14 and full Docker remain unverified; no absent hardware/data gate is
reported as passed.
