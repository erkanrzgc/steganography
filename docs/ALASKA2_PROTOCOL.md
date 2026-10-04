# ALASKA2 native-score baseline — preregistered 2026-10-04

This protocol is written before scoring any of the selected 4,000 images.
Acquisition and label ambiguity are documented in `ALASKA2_ACQUISITION.md`.
No detector, training, calibration or threshold adjustment is part of this run.

## Frozen inputs and detector

- Detector application revision: `f929429a1603e5819156c8462d124ba1fe84c36a`.
  The evaluator and a metrics option disabling threshold search are added
  separately; they do not change analyzer code or score aggregation.
  Detector source fingerprint (relative paths + SHA-256 over `core/`,
  `modules/`, `registry.py`, `config.py`):
  `da7a3a47c2d79357c20c721c9c149fa6e83f1b573c08ddaec4c6424a870bd020`.
- Source manifest SHA-256:
  `a75c1237d9d67194374062e62baba69b9c84e9283b2e2680cceaf04869bc6844`.
- Selection SHA-256:
  `a83b1924e5cadeb64ec169f87ba80cebb4592f7cf28cfcd675f4c369e8998244`.
- Original JPEG bytes, no evaluator resize/recompression. The existing DCT
  detector's internal calibration remains unchanged. `AnalysisService` uses
  balanced profile, fixed threshold **70**, `ai_provider=None`, no ML models.
- Host-native environment: Python 3.11, jpeglib 1.0.2, NumPy 2.4.6,
  Pillow 12.2.0. Record actual versions/tool availability in run provenance.
  ExifTool 12.76 is present; Stegseek and zsteg are absent. No tool installation
  or environment adjustment in response to test outcomes.

## Denominators, uncertainty and missing coverage

Score every file once. Each method is compared with the same 1,000 covers:
2,000 observations per method, not 6,000 independent images or three sources.
Do not publish a pooled accuracy that hides the 1:3 class imbalance.
Keep all three byte-identical UERD/cover pairs under their original source
labels in primary metrics. Preregister one sensitivity calculation excluding
both members of those three UERD pairs (997 pairs). Do not replace samples.

Publish confusion counts, ROC-AUC, balanced accuracy, recall and FPR at 70.
Use 200 bootstrap draws, seed 20261004, percentile 95% intervals, resampling
whole original-cover lineages with their paired stegos. Do not optimize or
recommend a test-set threshold. Average precision is omitted (the legacy
implementation is order-dependent on tied scores); ECE is **unavailable**
because heuristic scores are not calibrated probabilities.

Required native components for this JPEG baseline are `image_jpeg`,
`image_jpeg_dct`, `filestruct_appended`, `filestruct_exif`, `file_structure`,
and `signatures`. All must return `ok` for every member of a method cell.
Any failed sample, missing required component, or analyzer error makes that
cell's primary metrics unavailable; do not drop rows or count failures as
negative predictions. Retain raw available scores, counts and coverage.
Optional missing tools are reported explicitly and do not imply clean files.
An absent/unsupported analyzer for another format is not required JPEG coverage.

These are **available native-score metrics**, not full-stack verdict accuracy.
The legacy pipeline currently has no declared per-format required-component
policy; its low-score `no_indicators` verdict must not be interpreted as a
coverage-complete clean decision. The evaluator does not implement a replacement
detector or use that verdict as ground truth. Extraction/confirmed-payload
accuracy, calibrated ECE, and cross-source support remain unavailable.

## Execution and preservation

Use the shared service in four worker processes, CPU math thread counts one,
maximum 4,000 samples, 2 MiB per JPEG, 8 MiB manifest, 512×512 dimensions.
Reject changed hashes, unsafe/symlink paths, incomplete four-way lineages,
non-test splits, duplicate paths, and preexisting output directories. Write
per-file JSONL as results arrive, then final aggregate JSON only on completion.
Output contains relative sample names, hashes and structured signal/status
codes, never raw plugin error messages, absolute host paths or credentials.
No downloaded content is executed or redistributed.

Run under an external 1,800-second process-group timeout (10-second kill grace).
The existing DCT worker has a 15-second deadline and external analyzers a
20-second deadline. Native parser isolation is still incomplete; this is a
fixed, integrity-verified local research corpus, not an OS-sandbox claim.
Interrupted/failed attempts retain their partial evidence and do not pass.
Any rerun must be identified and use a new output directory, not replace results.

Publish environment, evaluator/protocol/manifest/score-record hashes, elapsed
time and median/p95 analysis latency. Analysis latency is not CTF recovery
latency. A single ALASKA2 source with unknown camera/device/payload-rate metadata
cannot pass the cross-source or method/payload support gate, regardless of scores.
Keep these images permanently excluded from training and threshold selection.

Reproduction (the output directory must not exist):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --kill-after=10s 1800s venv/bin/python -m steganography.benchmarking.alaska2 \
  --source .benchmark/alaska2-holdout-20261004 \
  --out .benchmark/alaska2-baseline-20261004 \
  --source-sha256 a75c1237d9d67194374062e62baba69b9c84e9283b2e2680cceaf04869bc6844 \
  --protocol docs/ALASKA2_PROTOCOL.md --workers 4
```
