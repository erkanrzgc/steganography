# Numerical repair and operating-point results — 2026-10-05

**Opt-in stable inference passes the frozen numerical gates. Detection support
is still unqualified, and no model/threshold is deployed.** Original failed
exports, checkpoints and reports are retained unchanged.

## Numerical repair

The [precision protocol](INFERENCE_PRECISION_PROTOCOL.md) and code were committed
as `6dcf4fc` before executing the real audits. The three existing development
models were derived without retraining or changing weights/train-only scaling.
The derived preprocessing explicitly requests float64 normalization and
accumulation, with a single cast back to float32 output; input remains float32.
Legacy checkpoints retain their float32 arithmetic automatically.

| Model | Validation rows | Original-to-derived maximum probability drift | PyTorch/ONNX maximum logit and probability difference | Gate |
| --- | --- | --- | --- | --- |
| Spatial 12-feature reference | 1,288 | 5.96e-8 | 0 / 0 | Passed |
| Spatial 468-feature co-occurrences | 1,288 | 4.47e-7 | 0 / 0 | Passed |
| JPEG 968-feature summary | 820 | 7.75e-7 | 0 / 0 | Passed |

Every validation row was compared using batch sizes 1, 17 and the full batch.
Absolute tolerance remained **1e-6, relative zero**, for logits and probabilities.
All original-to-derived and PyTorch-to-ONNX threshold-.5 decisions match.
This is engineering verification on inspected development data, not a new
accuracy score. Rows across models are not distinct independent observations.

A separate read-only audit rehashed original checkpoints, confirmed unchanged
weights and training provenance, inspected ONNX float32 I/O and internal double
casts, and independently computed all validation logits with float64 NumPy.
Its maximum difference against ONNX was zero after the declared float32 output
rounding for all three models. This is verified only on this CPU environment;
other hardware/runtime versions and accelerator providers remain unverified.

The [portable precision record](../benchmarks/inference-precision-20261005.json)
binds protocol, original/derived checkpoint, ONNX, features, manifest and raw
audit hashes. Files stay local, with no model/data redistribution or automatic
installation. Loader preflight bounds checkpoint and expanded ZIP bytes,
member count and loaded model dimensions; weights-only loading is **not** native
process isolation or a hard memory/CPU sandbox. Import trusted research
checkpoints only; fully isolated malformed-framework parser handling remains work.

## Cover-only operating-point experiment

The separate [operating-point protocol](OPERATING_POINT_PROTOCOL.md) was
committed as `b1e5916` before threshold selection. Whole-lineage salted SHA
assignment divides the already inspected spatial validation into **94 fitting
and 90 assessment covers**; each cover's six derivatives follow it. These
assessment images are not blind, newly acquired, or cross-source test data.

The threshold, selected using fitting covers only, is
**0.5585215699254572**. It produces 2/94 fitting false alarms (allowed floor of
3% = 2). The remaining 90 assessment covers yield **2/90 false alarms (2.22%)**,
versus **3/90 (3.33%)** at .5 on those exact same images. This is a one-file
change, not statistical proof of a general deployed FPR improvement. Its
paired bootstrap 95% FPR interval is **0–5.56%**, still above the 3% target.

| Assessment cell | Recall at .5 | Recall at selected threshold | Balanced accuracy before → after |
| --- | --- | --- | --- |
| Sequential 5% | 68.89% | 55.56% | 82.78% → 76.67% |
| Sequential 20% | 97.78% | 96.67% | 97.22% → 97.22% |
| Sequential 40% | 98.89% | 97.78% | 97.78% → 97.78% |
| Scattered 5% | 16.67% | 12.22% | 56.67% → 55.00% |
| Scattered 20% | 76.67% | 68.89% | 86.67% → 83.33% |
| Scattered 40% | 95.56% | 86.67% | 96.11% → 92.22% |

AUC/ranking and ECE are unchanged. This is **not probability calibration**;
weak sensitivity is not repaired by raising a threshold. All before/after
confusions and paired uncertainty are in the
[portable operating-point record](../benchmarks/spatial-operating-point-20261005.json).
A separate audit recomputed role hashes, cover-only selection, all twelve
confusion matrices, ranking/ECE invariance and artifact integrity. No image bytes
were accessed and no installed detector threshold changed.

This subset's 3.33% → 2.22% FPR must not be presented as the previous whole
184-lineage experiment's 4.89% → 2.22% improvement: denominators are different.
All named upstream methods, cross-source >=1,000-pair cells, calibration and
blind CTF qualification remain open. Next detector work should target genuinely
stronger low-payload features rather than claiming a threshold alone fixes it.

## Local verification

Python 3.11: 524 tests passed, 93.38% total coverage; feature reconstruction
100%, checkpoint/features 99.38%, precision audit 96.39%, operating-point service
97.40%. Ruff, mypy (80 application files) and diff checks pass. Twelve warnings
are existing ONNX export deprecations. Python 3.12–3.14, full Docker rebuild and
non-CPU inference providers were not exercised in this slice.
