# Models and research

Models are optional. The base package contains no weights. `models install`
accepts a local JSON manifest and adjacent ONNX file; the manifest must include
`id`, `version`, `domain`, `file`, `sha256`, `license`, `model_card`,
`benchmark_summary` and a base64 Ed25519 `signature` over canonical JSON without
the signature field.

`models catalog` lists only release-gated model packs. `models install
MODEL@VERSION --accept-license` is the sole network download path: it requires
an explicit license acknowledgement, HTTPS, a bounded download, the catalog
SHA-256, and an Ed25519 signature. There is no automatic install or update. An
empty catalog honestly means no project model has passed the real-data gates.

Only manifests declaring `"calibrated": true` may raise the hybrid ensemble
score. Uncalibrated outputs remain visible model diagnostics and deterministic
signals retain precedence.

Supported domains are `spatial-srnet-v1` and `jpeg-srnet-v1`. Unsupported file
formats or distributions remain `inconclusive`. The selected ONNX Runtime
execution provider is included in status/report data; CPU is the fallback.

`research import` creates a deterministic, hash-isolated 70/15/15 manifest from
a user-provided dataset directory. Dataset licenses and weights remain the
user's responsibility and are never copied into this repository.

`research benchmark` accepts that manifest and a JSON prediction map keyed by
sample SHA-256 (or relative path). It re-hashes every source file, evaluates only
labeled `test` records, detects missing scores and applies per-cell release
gates: at least 1,000 covers plus 1,000 stegos from two source groups, ROC-AUC
>= 0.90, balanced accuracy >= 0.85, recall >= 0.80, deployed FPR <= 0.03, and
ECE <= 0.05. Deterministic bootstrap 95% confidence intervals are published.
The report records both input hashes, all metric values and every failed gate so
a smoke run cannot be mistaken for a supported-model claim.

Example prediction input:

```json
{
  "predictions": {
    "<sample-sha256>": 0.97,
    "<another-sample-sha256>": 0.03
  }
}
```

`research calibrate` fits scalar temperature on only a train/validation split.
`research train --config` consumes explicitly configured NumPy features and
labels using the opt-in PyTorch extra. `research export` emits an ONNX file and
adjacent model-card/preprocessing contract; exported files remain uncalibrated
and unsigned until the release pipeline separately calibrates, benchmarks, and
signs them.
