# Five-epoch real GPU learning — 2026-10-10

**Detection failed.** This is same-source development evaluation, not
cross-source qualification. No weights are installed or deployed.

The protocol was frozen before execution at `3e242a7`:
`JPEG_EPOCH_LEARNING_PROTOCOL.md`, SHA-256
`5a371c2a2a8192835ac194f2e57e21f583cbc34b3f490772e88e63f4a05d76c2`.
The original physical disk-resume probe passed. Five separate bounded jobs
trained on all 7,380 ALASKA+BOSS rows (1,638 train lineages), with no validation
pixels during fitting. Final validation used all 1,611 lineage-disjoint rows
once, unchanged stored BN statistics and the fixed 0.5 decision threshold.

## Actual optimization, not accuracy

RTX 5060 Laptop GPU, WSL2, Torch 2.14.0+cu130, CUDA 13.0, IEEE float32;
deterministic algorithms, no TF32, mixed precision or CPU fallback.
The unchanged per-job limits were 1,800 seconds, 8 GiB host cgroup and
4 GiB Torch GPU allocator. These do not constrain all driver/desktop GPU usage.

| Epoch | Updates | Mean paired train loss | Fit seconds |
| --- | ---: | ---: | ---: |
| 1 | 3,288 | 0.69356697 | 533.00 |
| 2 | 3,288 | 0.68222512 | 575.20 |
| 3 | 3,288 | 0.65967868 | 581.56 |
| 4 | 3,288 | 0.64392754 | 592.52 |
| 5 | 3,288 | 0.62864730 | 606.30 |

16,440 total updates; approximately 48 minutes fitting. Training loss fell
9.36%, but this is not evidence of improved detection. Earlier pilots used
different exposure/data, so this is not a matched causal improvement study.

## Fixed final development scores

| Source / JPEG quality | Method | Cover / stego | ROC-AUC | Balanced accuracy | Recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ALASKA2 / unknown | JUNIWARD | 181 / 181 | .513675 | .513812 | .475138 | .447514 |
| ALASKA2 / unknown | UERD | 181 / 181 | .520115 | .511050 | .469613 | .447514 |
| BOSS / 75 | JUNIWARD | 178 / 178 | .505113 | .502809 | .264045 | .258427 |
| BOSS / 75 | UERD | 178 / 178 | .518732 | .519663 | .297753 | .258427 |
| BOSS / 95 | JUNIWARD | 178 / 178 | .503582 | .500000 | .129213 | .129213 |
| BOSS / 95 | UERD | 178 / 178 | .505728 | .500000 | .129213 | .129213 |

Covers are shared across method cells; the table is not 2,148 unique images.
All six cells remain experimental, below the sample-size and detection gates.
No bootstrap confidence intervals were computed, and neither source is an
unseen training origin. BOSS stego uses the recorded embedding simulation,
not independently recovered messages. Dataset licenses and sources remain
in `DATASET_CATALOG.md`; raw/licensed corpus and model/checkpoint files stay
outside the repository.

Nine independent float64 NumPy forward audits passed (maximum logit error
approximately 1.37e-7; score error 3.98e-8). This establishes numerical replay,
not successful learning or a sole causal explanation for the failed ranking.
Validation took 294.12 seconds on CPU; fitting actually used the GPU.
Do not refresh BN, tune thresholds, choose epochs or repeatedly evaluate
these rows to manufacture a pass. Any follow-up intervention needs a new
train-only protocol and separate evaluation policy.

## Immutable portable evidence

All paths below are relative to `benchmarks/`. Original bytes are copied,
not regenerated or rounded. `completed` denotes execution, not detection pass.

| Evidence | SHA-256 |
| --- | --- |
| jpeg-gpu-epoch-plan-20261010.json | 13ea2f8d97e382e5a4c1ed2df8d84aeebe70c656948219dfed2407adeaf0ed90 |
| jpeg-gpu-epoch-000-20261010.json | e9ca5d1beecfacd5de74f88a079bc211b739f1a63c998deba56d557a3eb3fdaf |
| jpeg-gpu-epoch-001-20261010.json | 904fd7fbf15777b09e4bc8073624cb83c2305860f85b48a6052a7f06880893d3 |
| jpeg-gpu-epoch-002-20261010.json | 8142a05f362715198c0a3a122e57440951775ee4c2442632cebd396d7c69c4e2 |
| jpeg-gpu-epoch-003-20261010.json | cf3bbf41f41662da05e4678ac9195ffa1483207be293963059463bd412edd824 |
| jpeg-gpu-epoch-004-20261010.json | b48686d96f9302b95380a830763c550f238ba9f378878ac4ccf5c6ad2b1e22c7 |
| jpeg-epoch-development-validation-20261010.json | 652fdf708beceb70ef4fb864817fe8fd5dca6cb043cc67e21b4d48bad847e657 |

Implementation verification and the original CPU test timeout are recorded
in `EPOCH_PILOT_VERIFICATION.md`; the fresh full regression passed 1,793 tests
with 95.61% coverage. This does not override any failed detector gate.
