# Frozen first pixel-residual CNN training — 2026-10-06

Commit implementation and protocol before any real-corpus fitting/scoring.
This is the first small custom raw-pixel baseline, not SRNet or a deployed
context-invariant detector. Preparation was audited in
`PIXEL_RESIDUAL_PREPARATION_RESULTS.md`; all previous failed benchmarks remain.

## Frozen data and settings

Same 3,750 original JPEGs, untouched train/validation row order and lineages;
manifest SHA-256:
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
Use existing 128×128 uint8 center crops, no new decode, image regeneration,
download, augmentation, exclusions, test access or calibration. Descriptor SHA:
train `7fa5ab60df47483eedb4e1306dff5a0586b9b8cc2cba431d2d38ec0eb5ad011f`,
validation `a333104b64139f385f1d1a35a034593917ef5613ba9ed1b29a4f1a21b4b07a21`.
Raw tensor hashes are bound by descriptors and the published preparation record.

Fit exactly three models: all sources (2,985 rows), ALASKA-only (2,367),
BOSS-only (618 quality variants / 103 original scenes). No validation rows enter
fitting or normalization. Require distinct declared origin/license records,
complete paired JUNIWARD/UERD families and training-scope identity hashes.
Training sizes and optimization-step counts differ; not a causal source ablation.

`jpeg-center128-residual-cnn8-v1` stays the same custom three-filter/three-conv
network: fixed filters and 3,729 learned parameters. Explicit CPU float32
construction overrides ambient dtype/device defaults, without changing the
normal preprocessing/network. Raw /255 normalization only; no fitted pixelwise
standardization, batch norm, dropout, shape search or pretrained weights.

Seed 20261011, 10 epochs, batch 32, Adam LR .001, weight decay .0001, two
math threads. Fixed CPU shuffle generator, every row once per epoch, short final
batch retained. Source/class row weights `n/(2*sources*source_label_count)` on
selected train rows; source-only fits use one source and both labels. Weighted
elementwise BCE mean over current batch; no additional positive-class multiplier.
Epoch loss records are row-count-weighted means, not validation metrics.
No early stopping, best-run selection, threshold search or repeated candidate fits.
Publish failed/timeout runs rather than changing settings to make them pass.

## Safety, persistence and inference

Only bounded raw caches load; source IDs/labels/names/declared quality are not
model inputs. CPU thread/RNG state is restored after fitting; process remains a
single research job, not an API background trainer. Per-fit deadline 1,800 s,
checked before each batch and after the last epoch. A running batch can finish
before the deadline is observed. Explicit service rejects epochs >50, batch >64,
threads >2, nonfinite/out-of-range optimizer settings and invalid source scope.
The fixed experiment is more restrictive than these service bounds.

Numeric model ≤1 MiB NPZ, exactly nine float32 arrays with allowlisted names,
fixed shapes and fixed high-pass values. Preflight bounded NPY 1/2 headers before
NumPy allocation, expanded-size limits and SHA-256; no objects/pickle, arbitrary
code, symlinks or overwrite. Weights/filters must be finite. Checksum-bound card
persists manifest/cache/raw-data SHA, source scope, decoder, settings/weights,
losses, Torch version and timing. No artifact is installed/deployed.

Validation checks card hash/domain/decoder/scope/settings before model loading.
Eval inference streamed in batches ≤64, CPU, sigmoid logits with cutoff .5
inclusive. Scores uncalibrated; no extraction/confirmed verdict implications.
Generic feature-model APIs and existing installed detectors remain unchanged.
Use `research pixel-cnn --config CONFIG --out FRESH`, stage `train` or `predict`.
Config requires `manifest`, `manifest_sha256`, `cache`, `cache_sha256`; training
may explicitly choose opaque `training_source_id`; prediction additionally
requires `model_dir`, `card_sha256`. No automatic download or model loading.

## Reporting and qualification

Report all 12 fit × validation-origin × family cells, same ordered validation
rows versus the all-source JRM reference, whose prediction SHA is
`62b0b34b2386f2f34a0d0bb346b613ee6f65fad7980f480737ac7b22bbfe870e`.
Include complete confusion/AUC/BA/recall/FPR/ECE and original-lineage uncertainty;
retain both Q variants together, unknown metadata and every regression.
Representation, crop, classifier and objective change together; not isolated
causal evidence that CNN architecture is better than JRM. Numeric audits and
ONNX export are distinct from detection gates and must not be inferred passed.

Qualification remains unavailable: reused inspected validation, insufficient
independent scene counts, unknown camera/device independence and ALASKA
quality/payload, simulated BOSS positives, JMiPOD excluded. No untouched-source,
blind CTF, supported method or global accuracy claim. First fitted-model scores
are development evidence only; require independent numeric/metric replay before
publication. Raw weights/cache/data remain local. Base wheel model-free.
