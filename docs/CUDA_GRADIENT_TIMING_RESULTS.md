# Consolidated-gradient physical CUDA timing — 2026-10-09

The [original report](../benchmarks/cuda-wsl-gradient-profile-20261009.json)
was fetched over authenticated pinned-host local SSH. Exact SHA-256:
`cb1c999ffc5d8b8b3ba1fcecf230ce2edf27d3d5be1edc345492881250ea66ae`.
All 16 execution-source hashes match `d972a00`; only `core/srnet_training.py`
differs from the earlier measured snapshot. The separately frozen
[protocol](CUDA_GRADIENT_TIMING_PROTOCOL.md), SHA-256
`bf015399391f2a3c23f9cc3edd8bc25ac3ef630bb2b342f5b51c5bbeacd4eb78`,
was committed/pushed before execution. The original protocol/result remain
unchanged; no retrospective exposure/safety-factor/ceiling adjustment.

## Result

Same WSL2/Python 3.12.3, Torch 2.14.0+cu130, CUDA 13.0, RTX 5060 Laptop
compute capability 12.0. Exactly 64 generated four-row FP32 updates, 16 initial
intervals discarded and all 47 steady intervals retained. All gradients and
full per-update numeric state remain checked; CUDA scalar gradient decisions
are consolidated, not optimizer arithmetic. No TF32/AMP/CPU fallback, kernel
RAM <=8 GiB, allocator <=4 GiB, unchanged process/deadline/file limits.

- Steady-interval p95: **0.09584681120086316s**, versus the first run's
  0.11254113879931538s. These are descriptive single-run measurements, not a
  controlled causal speedup estimate or a bottleneck diagnosis.
- Fixed five-epoch estimate: `120 + 2 * p95 * 16440` =
  **3271.4431522843806s** (54.52 minutes), versus the earlier 3820.35s.
- Decision remains **ineligible** against the unchanged **1800s** ceiling.
  No real fit started, epoch reduction or safety-factor/cap change.
- Generated job elapsed: 23.194207513000947s. Allocator peaks: allocated
  1,057,371,648 bytes, reserved 1,438,646,272 bytes.

Windows workload, thermal/power state and concurrent Kali work are not matched.
Short generated timing excludes real input I/O/content and cannot establish
long-run training time, CPU/GPU optimizer equivalence, independent mathematics,
learning or detection accuracy. No real corpus or validation inference, real
trained/exported/deployed model or new accuracy result.

## Verification and next step

Before physical timing: **1643 tests pass, one live-CUDA test explicitly skipped**
on Kali/Python 3.11.14; 32 warnings, 235.96s, total coverage **95.47%**.
Fourteen new generated tests cover missing/nonfinite gradients, unchanged
gradient values, scalar-read counts and two actual CPU optimizer updates with
exact loss/model-state equality. These are not live GPU parity tests.
`core/srnet_training.py`: **100%** line coverage. Ruff, mypy (140 files),
diff checks and wheel/sdist pass (150/300 members, no corpus/numeric weights/
credentials/provider memory). WSL timing is not a Python 3.12 full-suite pass;
Python 3.13–3.14 and full Docker remain unverified.

Next investigate numeric-state transfer/check overhead without weakening any
guard or changing arithmetic; its contribution has not yet been isolated.
Any further optimization needs verification and separately frozen timing.
Full-source stored-normalization train-only learning and independent-source
qualification remain pending; previous failed detector scores stay published.
