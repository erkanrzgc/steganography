# On-device state-check CUDA timing — original report recovered 2026-10-10

The existing [original report](../benchmarks/cuda-wsl-state-profile-20261009.json)
was fetched after the user restored local SSH; the measurement was not rerun.
Exact SHA-256:
`58c21efd8a93c60f8a83550ee29b7e635cc6320376e03cb5955dc981b1394bfa`.
All 16 source hashes match `c9a6f94`; only `core/srnet_model.py` and
`core/srnet_training.py` differ from the prior gradient-check snapshot.
The [separate protocol](CUDA_STATE_TIMING_PROTOCOL.md), SHA-256
`78798edde84b0aeb39c0d7e0e281cce9241ab74bd44b043768224dd226ac8727`,
was committed/pushed before execution. Earlier results remain unchanged.

## Verified outcome

Same WSL2/Python 3.12.3, Torch 2.14.0+cu130, CUDA 13.0, RTX 5060 Laptop,
compute capability 12.0. Exactly 64 generated four-row FP32 optimizer updates;
16 initial intervals discarded, all 47 steady intervals retained. Full state
checks run after every update, with final original NumPy verification after
CPU conversion. No TF32/AMP/fallback; unchanged 8 GiB kernel RAM, 4 GiB
allocator, internal 180s/parent 210s/CPU 360s/output 1 MiB/core-zero limits.

- Steady-interval p95: **0.09155291089991806s**, versus the prior 0.0958468112s.
- Fixed five-epoch estimate: `120 + 2 * p95 * 16440` =
  **3130.2597103893054s** (52.17 minutes), versus the prior 3271.44s.
- Decision remains **ineligible** against the unchanged **1800s** ceiling.
  No real fitting, epoch reduction, safety-factor change or cap extension.
- Generated job elapsed: 22.86691301800056s. Allocator peaks: allocated
  1,057,371,648 bytes, reserved 1,438,646,272 bytes.

Comparisons are descriptive single runs, not controlled speedups or isolated
bottleneck attribution. Windows load and thermal/power state are unmatched;
generated data excludes real input I/O/content and long-run endurance. No
independent mathematics, CUDA optimizer parity, real learning or accuracy gate
is established. No real corpus, validation predictions or deployed model.

The historical retrieval failure in [verification notes](CUDA_STATE_VERIFICATION.md)
is preserved as history, not an absent measurement relabeled as success.
Restored access allowed fetching and independently verifying existing bytes;
the exact reason for the earlier remote SSH reset was not established.

## Verification and next step

Before execution: 1663 tests passed, one live-CUDA test explicitly skipped on
Kali; total coverage 95.48%, changed training/model files 100% line coverage.
Ruff, mypy (140 files), diff checks and wheel/sdist passed; details in the
verification notes. Recorded evidence tests pin both optimized reports and
protocols, independently recompute interval percentiles/budgets and assert
ineligible/no-learning status; they are not live hardware tests.

After adding evidence regressions: **1665 passed, one live-CUDA test explicitly
skipped**, 32 warnings, 264.62s, total coverage **95.48%** on Kali/Python 3.11.14.
Ruff, mypy (140 files), diff checks and fresh wheel/sdist pass (150/301 members);
no raw corpus, numeric weights, SSH credentials or provider memory packaged.

Next investigate remaining execution costs or design a separately preregistered
bounded resumable training workflow. Do not repeatedly retime until a favorable
run appears, weaken guards or claim accuracy from generated timing. Adequate
full-source exposure, stored-normalization learning and independent-source
qualification remain required. Python 3.12 full-suite, Python 3.13–3.14 and
full Docker remain unverified; previous failed real detector scores remain.
