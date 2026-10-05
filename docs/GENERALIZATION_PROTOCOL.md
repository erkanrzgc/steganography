# Source/context diagnostics protocol — 2026-10-05

This slice audits existing validation predictions, not a new model experiment.
It neither claims generalization nor turns inspected validation into a blind
test. Freeze its grouping and threshold before running on the cached artifacts.

- Input manifests, prediction documents and exported model cards are bounded,
  nonsymlink documents. Bind prediction/card SHA-256 explicitly and verify
  the manifest hash, feature/domain contract, complete ordered validation
  identities, training-only provenance and finite scores. No image bytes or
  model executables are opened, and output creation is exclusive.
- Compare declared training/validation source sets. A shared source is not
  independent-source evidence. Missing source metadata also prevents that
  claim. Disjoint source declarations are only `candidate_not_qualified`;
  they cannot prove provenance, perceptual independence or blind evaluation.
- Fixed threshold .5, no score/threshold/model changes. Group each method and
  payload rate by source, decoded-format declaration and declared JPEG quality
  factor, plus pooled results. Shared covers are counted again per method,
  never described as extra independent covers. PNG/BMP and JPEG contexts do
  not qualify each other's method/format cells.
- Source/method identifiers use opaque SHA-based categories; only recognized
  method names and allowlisted formats are shown literally. Raw host paths,
  camera/device/app values and arbitrary metadata strings are not published.
- Unknown quality is unknown, not an inferred quality-factor estimate. Entirely
  missing quality metadata skips that context dimension. Partial unknown or
  single-label contexts are unavailable, not good zero-error results.
- Publish all cells, including small cohorts, AUC, BA, recall, FPR, ECE,
  confusion counts and paired lineage bootstrap (200, seed 20261005).
  At most 10,000 manifest rows and 256 context/family cells. Underpowered
  cohorts remain experimental; no qualification or calibration is performed.
- Initial audits: existing weighted PNG/BMP development predictions and JPEG
  development predictions, using their existing cards and manifests. No
  retraining, new acquisition, held-out baseline reuse or best-run selection.

Command (`HASH` values come from the retained artifacts):

```sh
steganography research diagnose --manifest MANIFEST --predictions PREDICTIONS \
  --predictions-sha256 PREDICTION_HASH --model-card MODEL_CARD \
  --model-card-sha256 CARD_HASH --out NEW_REPORT.json
```

This is a first domain-shift diagnostic, not a completed context-aware model.
JPEG features already include quantization tables; spatial features include
local residual/parity structure. Neither feature contract includes filenames,
source identities or dataset labels as inputs. These facts do not eliminate
dataset shortcuts: compression/noise/content statistics can still proxy source.
Next qualification needs untouched sources, known device/processing metadata,
compression/content strata, paired cover provenance and preregistered models.
