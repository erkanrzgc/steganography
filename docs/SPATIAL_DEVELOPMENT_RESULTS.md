# Spatial development results — 2026-10-05

The custom residual model substantially improves this **same-source controlled
LSB validation**, but is **not deployed or qualified for real-world detection**.
It is not an OpenStego/Steghide measurement, a blind CTF score, or an improvement
to the installed primary detector. The old frozen results remain unchanged.

## Fixed experiment

The [protocol](SPATIAL_DEVELOPMENT_PROTOCOL.md) and runnable implementation were
committed as `a0466e3` before real generation/features/training/prediction.
One complete run produced 7,000 unique PNG/BMP files from the 1,000 newly
acquired BOSSbase originals: 5,712 train (816 lineages), 1,288 validation
(184 lineages), no test samples. The source and eight reserved manifests were
bound by checksums; old test image bytes were not accessed.

Each original supplies a cover and sequential/scattered gray-plane LSB
replacements at 5%, 20%, 40%. Payloads are marker-free deterministic random
bytes. PNG or BMP is assigned per original independently of its label, and
all variants retain that format and split. Grayscale values are preserved;
there is no independently randomized RGB-channel shortcut.

Both linear models use the same train/validation rows, train-only normalization,
class-balanced BCE, seed 20261005, Adam .01 and 300 full-batch epochs. The score
threshold is fixed at .5; no threshold/hyperparameter search occurred. Results
below use 184 covers and 184 stegos per cell; covers are shared, not six
independent sets. The 468-dimensional descriptor is custom experimental
co-occurrence histograms, **not a full SRM/SPAM implementation**.

## All primary validation cells

| Method / changed sample rate | 12-feature reference AUC | 468-feature AUC (paired 95% CI) | New balanced accuracy | New recall |
| --- | --- | --- | --- | --- |
| Sequential 5% | 0.518254 | 0.918449 (0.892002–0.941071) | 80.71% | 66.30% |
| Sequential 20% | 0.580015 | 0.995776 (0.991575–0.998672) | 96.20% | 97.28% |
| Sequential 40% | 0.648423 | 0.997460 (0.994621–0.999204) | 96.47% | 97.83% |
| Scattered 5% | 0.516422 | 0.716801 (0.681818–0.749798) | 57.34% | 19.57% |
| Scattered 20% | 0.568585 | 0.956965 (0.939074–0.973986) | 85.60% | 76.09% |
| Scattered 40% | 0.645262 | 0.992320 (0.985936–0.996723) | 95.11% | 95.11% |

Shared cover false positives: reference **119/184 (64.67%)**, richer model
**9/184 (4.89%)**. The richer model therefore misses the <=3% FPR target, and
low-rate scattered recall remains poor. Its per-cell ECE is 0.109806–0.205829,
above the <=.05 target; outputs are uncalibrated model scores. No cell passes
all support requirements: only one source, 184 pairs/cell and unknown camera/
device identities, not the required >=1,000 pairs and independent sources.

The [portable record](../benchmarks/spatial-development-20261005.json) includes
every confusion matrix, metric and 200-replicate paired-lineage bootstrap
interval. Per-format breakdowns are secondary descriptive results, not separate
qualification: validation has just 87 PNG and 97 BMP cover lineages. Bootstrap
intervals do not establish unseen-camera or cross-source generalization.

## Audit and export

- A separate read-only audit reread all 7,000 generated files, recomputed
  SHA-256/sizes, verified original ancestry/splits and lossless grayscale
  decoding, extracted every one of the 6,000 payloads independently, and
  verified selected-only changes with absolute pixel distortion <=1.
  Corpus size is 1,397,481,768 bytes, below the frozen 2 GiB bound.
- Both models' train-only means/scales and all validation row identities were
  checked independently. Float64 NumPy inference reproduced all threshold
  decisions; maximum probability differences were 7.08e-8 and 4.22e-7. This
  audit is not a replacement for the stricter ONNX gate.
- All 1,288 validation rows were compared in CPU ONNX Runtime and PyTorch.
  Reference maximum logit/probability differences: 2.98e-7 / 8.94e-8, passed.
  Richer model: **7.63e-6 / 5.96e-7**, strict absolute 1e-6/relative-zero
  combined gate **failed** because of logits. All decisions matched; neither
  the tolerance nor failed status was changed after inspection.

Generation, feature extraction, both trainings, predictions, ONNX exports and
intervals took 379.75 seconds on this Linux Python 3.11 CPU environment
(8 exposed logical CPUs, VMware guest / Ryzen 9 8945HX host); four feature
workers and BLAS/OMP/MKL thread limits of one, no CUDA. This is dataset research
runtime, not an individual-challenge CTF latency measurement.

Protocol SHA-256:
`73307b70c6004698086c98a53e8c5f32437b41ce7feaa4165d3fae7192c73c3f`.
Experiment manifest SHA-256:
`707db7debce75b2a241e1ea3b1a5e3cb9b89c287d7396f95abad5c3364e5eda7`.
Original report SHA-256:
`1e8f51c9576b39cee00b5554ee33c7a8969495f222965090f8970476b3c35b75`.
The portable record additionally binds checkpoints, features and ONNX files.
Subsequent hardening commit `74d5864` adds revalidation against source changes after
preflight and more adversarial/scalar tests; it did not change feature/model
definitions or regenerate/select the reported experiment.

## Next gates

Improve low-payload scattered sensitivity and false alarms using development
data; freeze a model and calibration before scoring a newly untouched source.
Resolve numerical export parity without relaxing the published tolerance.
Separately evaluate named upstream embedding tools: generic LSB replacement
is not evidence of recovery/detection of every spatial algorithm.

Dataset [source and restrictions](SPATIAL_DEVELOPMENT_ACQUISITION.md) apply:
no redistribution permission was verified. Raw originals/derivatives, features,
checkpoints and models stay local; only protocols/code/hashes/metrics are in Git.
