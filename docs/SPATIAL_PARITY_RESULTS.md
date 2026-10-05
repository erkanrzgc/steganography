# Spatial parity/residual results — 2026-10-05

**Ranking improved, but weak-payload recovery sensitivity is not solved.**
The new model is not deployed, calibrated or cross-source qualified. This is
iterative development on already inspected validation, not a fresh blind test.
Installed native detector scores and frozen upstream-method baselines are unchanged.

## One fixed comparison

The [protocol](SPATIAL_PARITY_PROTOCOL.md) and code were committed as `86d1ed0`
before real extraction, training and scoring. One completed real run reused
the exact 7,000-file audited corpus and unchanged lineage split: 5,712 train
(816 originals), 1,288 validation (184 originals), zero test rows. Original
report/prediction hashes and ordered validation identities bind the reference.

The new 684-dimensional descriptor retains all 468 full/top co-occurrences
and adds 216 joint center-bit/neighbor-difference probabilities. Four
directions, three bit planes and clipped signed residuals are pooled across
channels; replicated grayscale does not create an RGB label shortcut. Only
this new feature version rounds probabilities to 1e-8. Its train artifact is
43,464,354 bytes, validation 9,809,697 bytes, below the existing 64 MiB cap.

The linear model uses the same seed 20261005, Adam .01, 300 epochs, balanced
loss and train-only standardization floor 1e-4 as the reference. Training is
float32; inference explicitly uses stable float64 accumulation and float32
I/O. Threshold .5 is fixed; no hyperparameter/threshold/calibration search or
best-run selection occurred. Generic gray-plane LSB replacements are not
Steghide/OpenStego or naturally occurring mobile-app examples.

## Every validation cell

Each row uses the same 184 covers and 184 stegos; covers are shared across
cells. Reference numbers are from the completed 468-feature model, not from
the chance-level native detector or a different subset.

| Method / rate | Reference AUC → new AUC | Reference recall → new recall | New balanced accuracy |
| --- | --- | --- | --- |
| Sequential 5% | 0.918449 → 0.935078 | 66.30% → 38.04% | 68.21% |
| Sequential 20% | 0.995776 → 0.996987 | 97.28% → 96.20% | 97.28% |
| Sequential 40% | 0.997460 → 0.998789 | 97.83% → 97.83% | 98.10% |
| Scattered 5% | 0.716801 → 0.906486 | 19.57% → 14.67% | 56.52% |
| Scattered 20% | 0.956965 → 0.995894 | 76.09% → 97.28% | 97.83% |
| Scattered 40% | 0.992320 → 0.999114 | 95.11% → 99.46% | 98.91% |

Shared cover false alarms improve from **9/184 (4.89%) to 3/184 (1.63%)** on
the identical cohort. However, both low-rate cells lose recall at .5; their
balanced accuracy also decreases. The target scattered-5% AUC rises markedly,
with paired bootstrap 95% interval **0.876154–0.934193**, but recall is just
**27/184 (14.67%)**, interval **9.24–20.11%**. AUC is not an exact recovery
rate, accuracy or proof of good deployed sensitivity.

New FPR 95% interval is **0–3.80%**, still extending above the <=3% support
target. ECE worsens at low rates: sequential-5% 0.133499 → 0.255661 and
scattered-5% 0.205829 → 0.275259. Other cells' new ECE is 0.094376–0.148128,
still above .05. Scores are not calibrated probabilities. No cell passes the
full support gates: only one source, 184 pairs and unknown camera/device IDs;
>=1,000 pairs, independent sources and calibrated deployment remain absent.

The [portable record](../benchmarks/spatial-parity-development-20261005.json)
contains all confusions, metric deltas, and 200-replicate paired-lineage
intervals. It publishes regressions alongside gains. Conditional bootstrap
intervals cannot establish unseen-camera generalization or repair selection
bias from iterative inspection of this validation set.

## Verification

- A separate scalar audit recomputed all new joint histograms over four actual
  full-size validation images: PNG/BMP cover and scattered-5% examples. It
  checked the 468-feature prefix against legacy extraction before rounding;
  independent unit oracles additionally cover all directions/planes/RGB pooling.
- Train-only means/scales, all 1,288 validation row identities, logits and
  prediction scores were independently reproduced. Float64 NumPy and CPU
  ONNX logits matched exactly after declared float32 output rounding; pairwise
  AUC and confusion counts were recomputed independently for all six cells.
- All validation rows pass the original absolute 1e-6/relative-zero PyTorch/
  ONNX gate, with zero logit/probability differences and matching decisions.
- Complete feature extraction, training, export and metrics took **120.74 s**
  on the existing Linux/Python 3.11 CPU environment, four feature workers,
  BLAS/OMP/MKL thread limits one, no CUDA. This is dataset research runtime,
  not a challenge-latency benchmark.
- Python 3.11: **529 tests passed, 93.45% total coverage**; descriptor 100%,
  comparison runner 97.33%. Ruff, mypy (82 application files), diff checks pass.
  Eighteen warnings are ONNX export deprecations. Python 3.12–3.14, a new full
  Docker build, other runtime/hardware and accelerator providers were not tested.

Protocol SHA-256:
`e0d2600aab71ac2b324b4794a6cffec7aba20138c82cd5411625f91875bcd6e1`.
Raw report SHA-256:
`0381ae94413e7c9050859aadc08d67208f43f3ce577ea090237301b10f6cd0bf`.
Other artifact and manifest hashes are in the portable record. Raw data,
feature matrices, checkpoints and models remain local under the source's
unverified redistribution conditions; no automatic installation occurred.

## Next work

Develop a properly separated calibration/operating point and stronger
low-payload training objective; report ranking and sensitivity separately.
Do not select a flattering threshold on all validation and call it blind
evidence. Freeze the model and policy before newly untouched independent-source
measurement, and separately evaluate named upstream embedding methods.
