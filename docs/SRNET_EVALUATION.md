# SRNet complete-validation replay

`research srnet-evaluate --config FILE --out FRESH_JSON` explicitly evaluates a
completed research fit. No downloads, calibration, installation or primary
verdict changes occur. This is not a qualified detector or a blind benchmark.

Configuration has exactly these string fields:

```json
{
  "manifest": "manifest.json",
  "manifest_sha256": "<sha256>",
  "cache": "train/cache.json",
  "cache_sha256": "<sha256>",
  "plan": "training-plan.json",
  "plan_sha256": "<sha256>",
  "validation_cache": "validation/cache.json",
  "validation_cache_sha256": "<sha256>",
  "model_dir": "fit",
  "card_sha256": "<sha256>"
}
```

The complete training cache is verified and released before loading validation.
Shared fit verification reconstructs source exclusions and every epoch's order
and count. Split integrity is checked by the cache loader. Rehashed incompatible
cards, changed decoders, wrong optimizer contracts and incomplete epochs fail.
Numeric model hashes and every BN counter must match training update accounting.
Checksums bind supplied records; they cannot independently prove that someone
actually trained a model or correctly labeled/licensed a dataset.

All validation rows get native CPU inference in batches of at most four, without
normalization. Before inference select the first manifest row per declared
source/quality/label/method cell. More than 24 cells fail explicitly, never get
silently truncated. Independent NumPy float64 replay uses the unchanged gates
in `SRNET_NUMERICAL_READINESS.md`. Any failed comparison retains audit evidence
but produces `failed_numerical_gate`, CLI exit 2 and an empty predictions list.
Exceptions, nonfinite results and deadlines never publish partial predictions.

Successful reports contain every validation row's SHA, lineage, method, opaque
source ID, declared quality and uncalibrated class-one softmax score. Threshold
is fixed at 0.5 (ties stego), never tuned on validation. Reports exclude host
paths and remain experimental/non-deployed. Validation is explicitly reused,
not untouched. ROC-AUC/recall/FPR/ECE/sample-size/external-source gates are
separate; this command does not claim they passed. Full ONNX replay remains
explicitly unavailable in this service.

Service and CLI currently have a cooperative 1,800-second wall check covering
preparation/inference/replay. Torch threads are restored on failure. Unlike
fitting, evaluation does not yet have a hard-kill worker/resource sandbox;
blocking native work cannot be interrupted by this check. Freeze real fit
parameters and add actual trained-model/ONNX replay and independent-source
benchmark evidence before deployment. Generated regressions are not accuracy.

Local verification: Python 3.11.14, 1,093 tests passed, total coverage 94.77%;
evaluation/shared fit 149/149 statements covered. Ruff, mypy (112 files),
whitespace and model/data-free wheel/sdist checks passed. Actual generated
paired fitting/native/NumPy replay and distinct-row ordering are included.
Python 3.12–3.14 and fresh full Docker were not rerun locally.
