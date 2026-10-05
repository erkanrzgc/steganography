# Low-payload weighting protocol — 2026-10-05

Freeze before training/scoring. This is iterative development on already
inspected, same-source BOSSbase validation; not a blind test or deployment.

- Reuse the exact 7,000-file corpus and cached 684-feature parity artifacts.
  Manifest SHA: `707db7debce75b2a241e1ea3b1a5e3cb9b89c287d7396f95abad5c3364e5eda7`.
  Train feature SHA: `370bae54bde822104af64e4fc16142bdff3e0a1476440d2ea717afd145c7069d`.
  Validation feature SHA: `dc58c420733bed3cd5250c7169513432c975f13f6ff6ed5544a6e8385d96fbf0`.
  Reference report SHA: `0381ae94413e7c9050859aadc08d67208f43f3ce577ea090237301b10f6cd0bf`.
- One fixed recipe `spatial-low-payload-x4-v1`: weight both sequential and
  scattered 5% training positives by 4; covers and 20/40% positives by 1.
  Complete seven-variant training lineages are mandatory. Validation/test
  metadata never supplies weights. No adaptive or score-dependent weights.
- BCE logits use positive class weight = weighted negative mass / weighted
  positive mass (816/9792 = 1/12), reducing weighted loss by row mean. All other
  settings stay unchanged: CPU linear model, seed 20261005, Adam .01, 300 epochs,
  float32 training, train-only normalization with std floor 1e-4, float64
  inference accumulation with float32 I/O. Fixed threshold .5.
- Bind cached features, manifest, previous predictions and ordered validation
  identities by SHA. No feature regeneration, downloads or held-out test access.
- Publish all six cells: AUC, balanced accuracy, recall, FPR, ECE, confusion
  counts and paired bootstrap 95% intervals (200 replicates, seed 20261005).
  Preserve the earlier run. No threshold/hyperparameter search or best-run pick.
- Export and check every validation logit/probability against CPU ONNX at
  absolute 1e-6, relative zero, and identical threshold decisions.
- Remain experimental, uncalibrated and undeployed regardless of a local gain;
  independent-source qualification and low-payload sensitivity remain gates.

Reproduce with the provisioned research environment (exclusive new output):

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 venv/bin/python -m steganography.research_weighted_comparison \
  .benchmark/spatial-parity-20261005 .benchmark/spatial-weighted-20261005 \
  --reference-sha256 0381ae94413e7c9050859aadc08d67208f43f3ce577ea090237301b10f6cd0bf
```

The cached vectors were previously audited against the source images. This
run rechecks their hashes/contracts, not all original image bytes. It performs
no corpus generation or download. The reference's original train/validation
config paths must point to that same manifest and cached artifacts.
