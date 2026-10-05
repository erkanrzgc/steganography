# Measured JPEG context development results — 2026-10-05

**The new model failed; it is not deployed.** Adding a second declared origin
and measured compression/content interactions did not improve detection.
See the [frozen protocol](JPEG_CONTEXT_PROTOCOL.md), committed with code in
`6bef8c0` before generation/training, and [portable evidence](../benchmarks/jpeg-context-development-20261005.json).
No native baseline, held-out images, installed model or primary score changed.

## Data and scope

The fixed experiment uses 3,750 JPEG files: 2,985 train, 765 validation.
ALASKA2 contributes 994 original development lineages (789 train / 205
validation), covers and JUNIWARD/UERD. BOSS contributes 128 previously acquired
development originals (103 train / 25 validation), each encoded at quality
75 and 95 with a cover and two upstream simulated stegos: 768 new JPEGs.
Source/class-balanced training uses 1,098 byte-derived DCT/context features.
All quality/method derivatives retain original ancestry and split roles.

Both training and validation share **both** origins; unseen validation sources
remain **zero**. These BOSS originals were already spatial development data.
Two acquisition origins do not prove independent scenes/cameras or eliminate
processing shortcuts (grayscale BOSS versus color ALASKA). Source identities
are not model features. All camera/device/app metadata remains unknown;
ALASKA quality/payload rates remain unknown. ALASKA's JMiPOD rows are excluded
because conseal 2025.11 has no matching simulator. That method was not improved.
No data or model redistribution is authorized; everything stays local.

## Fixed .5 validation: every origin and quality cohort

Covers are shared between method comparisons, not additional independent data.
BOSS has 50 JPEG covers but only **25 original validation lineages**; the two
quality derivatives are correlated. The JSON retains every cell's paired-lineage
200-resample 95% intervals, including pooled/format cells and unknown-quality
unavailable cells. These intervals are development diagnostics, not support.

| Origin / cohort | Method | Cover/stego files each | AUC | Balanced accuracy | Recall | FP / covers (FPR) | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALASKA2 | JUNIWARD | 205 | .566234 | .539024 | .580488 | 103 / 205 (.502439) | .010574 |
| ALASKA2 | UERD | 205 | .560440 | .521951 | .546341 | 103 / 205 (.502439) | .018915 |
| BOSS, both qualities | JUNIWARD | 50 | .516400 | .530000 | .540000 | 24 / 50 (.480000) | .025444 |
| BOSS, both qualities | UERD | 50 | .512800 | .510000 | .500000 | 24 / 50 (.480000) | .032962 |
| BOSS Q75 | JUNIWARD | 25 | .502400 | .500000 | .480000 | 12 / 25 (.480000) | .037619 |
| BOSS Q75 | UERD | 25 | .507200 | .480000 | .440000 | 12 / 25 (.480000) | .057992 |
| BOSS Q95 | JUNIWARD | 25 | .532800 | .560000 | .600000 | 12 / 25 (.480000) | .062104 |
| BOSS Q95 | UERD | 25 | .512000 | .540000 | .560000 | 12 / 25 (.480000) | .041037 |

On the same ALASKA validation rows, the previous single-origin linear model
had JUNIWARD/UERD AUC .592481/.585306 and FPR .414634. New AUC regresses to
.566234/.560440 and FPR rises to .502439. Training origin, objective, representation
and seed change together; this is not an isolated causal feature ablation.
Low ECE around near-chance predictions does **not** rescue failed ranking,
balanced accuracy, recall or FPR. No threshold/temperature search was performed.
Every measurable cohort fails detection targets, and no cohort qualifies for
the independent-source/sample-size gate. No signed model pack is published.

## Independent checks

- Rehashed all 1,000 original BOSS files: unchanged. Checked all 768 generated
  JPEG hashes, sizes, grayscale decode and DCT; all 512 stegos have nonzero unit
  coefficient changes and unchanged quantization. Upstream simulations also
  modify DC coefficients (115,211 changes total); no AC-only embedding claim.
  Simulated changes are not encoded payloads or extraction success.
- An independent count-based/scalar feature oracle matches all 1,098 cached
  values on 12 actual JPEGs (two lineages × two qualities × three variants),
  maximum difference zero. Rehashed all 3,750 combined copied files.
- Independently reproduced confusion counts, pairwise AUC, BA, recall, FPR and
  ECE for all 12 measurable diagnostic cells. NumPy train mean/scale deviations
  are 7.14e-8 / 1.47e-8; NumPy64/native logits agree exactly and saved sigmoid
  probabilities differ by at most 7.72e-8. CPU ONNX batches 1, 17 and all 765
  validation rows have zero logit difference and identical .5 decisions at the
  unchanged 1e-6 absolute tolerance, zero relative tolerance.
- Python 3.11: 631 tests, total coverage 93.73%; new context/corpus/preparation/
  weighting coverage 100% / 95.60% / 98.10% / 100%. Ruff, mypy (88 files), diff
  checks pass. Other Python versions and a fresh Docker rebuild were not run.

## Reproduction

After the explicit acquisitions documented elsewhere, install optional research
dependencies (`.[research,jpeg-sim]`). Use **fresh** output directories, bound
input hashes from the protocol, and all eight reserved manifests listed in the
portable record. The public service sequence is:

```python
from steganography.research_jpeg_corpus import generate_boss
from steganography.research_jpeg_multisource import prepare_dataset
from steganography.research import train_model, export_onnx
from steganography.research_features import predict_validation
from steganography.research_spatial import onnx_parity
from steganography.research_generalization import diagnose_validation

# Explicit local input/output Paths and expected SHA-256 values are required.
generate_boss(boss_originals, generated, source_sha256=boss_source_hash,
              reserved_manifests=reserved, count=128)
prepare_dataset(alaska_features, alaska_originals, generated, experiment,
                alaska_manifest_sha256=alaska_manifest_hash,
                boss_manifest_sha256=generated_manifest_hash)
train_model(experiment / "train-config.json", experiment / "context.pt")
predict_validation(experiment / "validation-config.json",
                   experiment / "context.pt", experiment / "predictions.json")
export_onnx(experiment / "context.pt", experiment / "context.onnx")
onnx_parity(experiment / "validation-config.json", experiment / "context.pt",
            experiment / "context.onnx")
diagnose_validation(experiment / "manifest.json", experiment / "predictions.json",
                    experiment / "context.onnx.model-card.json", diagnostics,
                    predictions_sha256=prediction_hash, model_card_sha256=card_hash)
```

The placeholders are not a standalone download command. The local run used two
simulation workers and two CPU math threads; merger/training/prediction/export/
diagnostics took 11.22 seconds, **excluding** simulation and later audits. This
is not a tool/inference/CTF latency claim. Artifact hashes and library versions
are in the portable record. Originals and historical outputs are never overwritten.

## Next gate

The coarse linear summary failed again; increasing source count alone is not
sufficient. Next freeze a richer spatial/DCT-residual representation or suitable
trained architecture, matched compression/preprocessing controls and a train-only
comparison; retain this failed experiment. Obtain a genuinely untouched,
licensed same-format source with camera/scene ancestry for final qualification.
Do not fit a source-name rule or tune this model on the frozen ALASKA baseline.
