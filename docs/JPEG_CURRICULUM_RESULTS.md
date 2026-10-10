# Q75 two-source curriculum — 2026-10-10

Implementation/protocol checkpoint `bb45059`, frozen before preparation or
fitting. Both stages completed on the own RTX 5060 in 1,400.77s (23m20.77s).
Two of ten development cells move from 50% to 56.25% balanced accuracy; eight
remain at 50%. This is one additional correct prediction in each 16-row cell,
not statistically reliable or broad improvement. No deployment/support gate
has passed. Training loss improves, but useful generalization remains unproven.

## Prepared data

48 BOSS/BOWS Q75 fit originals, 24 per source, exactly the preregistered first
24 historical fit lineages each. Real covers with upstream JUNIWARD simulation
at .4 bpnzAC; low-payload adaptation retains original .2 kit rows. No new dataset
downloads or redistribution of private images.

All 96 new cover/stego crops pass independent SciPy IDCT math checks (maximum
float32 pixel difference zero); quantization tables match and each changed
coefficient differs by at most one. Minimum 740 coefficient changes per stego;
no identical selected crop and no outcome-based filtering. Native parsing uses
the same pinned jpeglib backend; it is not an independent parser oracle.

[Original preparation metadata](../benchmarks/jpeg-q75-curriculum-high-audit-20261010.json)
is byte-identical to the private high-cache manifest, SHA-256
`8e551b93a69eab1e4499a6863a8dac8bf35f5e55f4a4f5cc167b6fc42e9d3c16`.
The 24 MiB tensor and 48 generated JPEGs remain private; portable metadata
contains hashes and provenance only, with all 34 source/protocol hashes.

`probe_pixels_read: 0` counts probe row fetches/exposure during preparation.
The inherited source reader still hashes and finite-checks the entire cache
for integrity, including probe bytes; none are used to generate or select fit
examples. Do not interpret this field as zero filesystem bytes read.

## Completed physical fit

Own RTX 5060, paired D4 augmentation, .4 warm-up / .2 adaptation, 40 epochs
each, 7,680 total updates. The final evaluation retains all ten previously
inspected train-role development contexts, including off-fit qualities/methods.
These are original-disjoint from fit, but not a blind/cross-camera benchmark.
No epoch/threshold selection, cloud resources, saved weights or deployment.
All 80 epochs /7,680 updates completed. Warm-up epoch-average train loss drops
from .715796 to .534693; adaptation drops from .715296 to .599193. These are
train-mode losses, not unseen-image accuracy or calibration evidence.

| Source | Q | Method | BA | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- |
| ALASKA2 | unknown | JUNIWARD | 50% | .4922 | 50% |
| ALASKA2 | unknown | UERD | 50% | .4688 | 50% |
| BOSS | 75 | JUNIWARD | 50% | .5313 | 50% |
| BOSS | 75 | UERD | 50% | .5156 | 50% |
| BOSS | 95 | JUNIWARD | 50% | .4844 | 25% |
| BOSS | 95 | UERD | 56.25% | .4688 | 25% |
| BOWS | 75 | JUNIWARD | 56.25% | .5938 | 25% |
| BOWS | 75 | UERD | 50% | .5625 | 25% |
| BOWS | 95 | JUNIWARD | 50% | .5313 | 12.5% |
| BOWS | 95 | UERD | 50% | .5156 | 12.5% |

Eight cover/eight stego rows per cell; cover rows are shared across methods.
Only Q75/JUNIWARD BOSS/BOWS were fitted. Every other context is off-fit transfer,
not a supported method or an independent successful source gate. All cells fail
support qualification, with FPR well above 3%. Do not advertise 56.25% as the
tool's real-world accuracy or select just the best cell for README marketing.

Final singleton/four-row parity passes at atol/rtol .0001; unchanged source
identities and both caches re-verified. Fixed FP32, no AMP/TF32/fallback, host
8 GiB /allocator 4 GiB; no cap relaxation. Peak GPU reserved 1,440,743,424 bytes;
no weights persisted. VM CPU regression ran concurrently, so the wall time is
not a dedicated hardware performance benchmark.

[Original training report](../benchmarks/jpeg-q75-curriculum-fit-20261010.json),
[independent scalar/rank audit](../benchmarks/jpeg-q75-curriculum-independent-metrics-20261010.json)
and [portable replay tests](../tests/test_curriculum_evidence.py) preserve all
results. Python logsumexp verifies loss; average tied ranks independently verify
ROC-AUC rather than using the engine's pairwise AUC helper. Audit also verifies
fixed metadata selection, all 120 raw probe logits, 34 current source hashes,
report/cache SHA-256 values and warm-up/adaptation weight continuity. Historical
BN/GN metrics are independently replayed as descriptive comparisons, not a
matched causal trial. This is scalar accounting, not whole-model/embedding parity.

## Next gate

Do not blindly extend this small fit or deploy its discarded model. Next freeze
an own-GPU larger-batch feasibility/learnability control and a substantially
larger original-diverse Q75 corpus before renewed fitting. Retain paired D4 and
source/quality/method context; measure untouched original/source groups once.
The authors' batch/exposure/pretraining recipes remain substantially different.
More data, a larger batch, or another architecture is a hypothesis, not a promised
accuracy gain. A small two-cell change does not justify paid cloud resources.

Verification: full disk-backed regression 2,144 passed /one actual-CUDA skip on
the CPU VM, 32 existing ONNX warnings, plus three later portable-evidence tests
separately. Total coverage 95.95%, new core 98% /frontend 97%. Lint, types,
dependencies and fresh model-free wheel/sdist checks pass. Physical GPU fitting
is separately evidenced, never inferred from the local skipped hardware test.
Earlier learnability passes and failed historical broad probe scores stay public.
