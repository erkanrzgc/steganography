# Models and research

Models are optional. The base package contains no weights. `models install`
accepts a local JSON manifest and adjacent ONNX file; the manifest must include
`id`, `version`, `domain`, `file`, `sha256`, `license`, `model_card`,
`benchmark_summary` and a base64 Ed25519 `signature` over canonical JSON without
the signature field.

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
labeled `test` records, detects missing scores and enforces the release gates:
5,000 held-out samples, at least two source groups, ROC-AUC >= 0.80 and FPR <=
0.05 at the high-confidence threshold. The report records both input hashes,
all metric values and every failed gate so a smoke run cannot be mistaken for a
supported-model claim.

Example prediction input:

```json
{
  "predictions": {
    "<sample-sha256>": 0.97,
    "<another-sample-sha256>": 0.03
  }
}
```
