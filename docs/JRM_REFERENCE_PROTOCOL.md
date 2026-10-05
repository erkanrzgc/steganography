# Frozen JRM + FLD local reference — 2026-10-05

Freeze implementation/protocol before real-corpus extraction, training or scoring.
This is a **research reference**, not installed/deployed model inference or
proof of context invariance. No upstream model/code is vendored. Optional
`jrm-reference` extra pins sealwatch 2024.12 (pure Python JRM/FLD, without
later torchvision imports), jpeglib and setuptools <81 for upstream pkg_resources.
Latest upstream is not substituted. Wheel SHA-256:
`f76f5a1a03d3608e9883530f59ce6e51fbd118ab91ebe46226038815234d3533`.

Upstream [sealwatch](https://github.com/uibk-uncover/sealwatch) is MPL-2.0;
the JRM implementation also carries the DDE Lab educational/research/non-profit
notice. Restrict this experiment to local research, preserve notices and review
all component terms before redistribution/commercial use. This is not a license
grant for weights/raw datasets. See `THIRD_PARTY_NOTICES.md`.

## Data and controls

Reuse all 3,750 files, original lineage/quality splits and row order from
`JPEG_NONLINEAR_PROTOCOL.md`: 2,985 train / 765 validation, both origins,
JUNIWARD/UERD only. No acquisition, regeneration, exclusions, quality guesses,
test access or calibration. Manifest SHA-256:
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
Reference publication: `benchmarks/jpeg-nonlinear-development-20261005.json`.
Reference publication SHA-256:
`839ebbba9f5138f89469585c8422fc9c01e8f9a26e6c05be38a593cef73180c9`.
Bind its SHA before running. Both training/validation sources overlap;
previously inspected validation is iterative development, not blind evidence.
BOSS simulations retain correlated Q75/Q95 variants and only 25 original
validation scenes. Missing camera/device/app and ALASKA quality/payload stay unknown.

## Bounded representation

`sealwatch-jrm-2024.12-v1`: uncalibrated luminance JRM, 11,255 dimensions.
Validated signed coefficients converted to int32 before upstream absolute
values. Flatten ordered submodels in C order, then little-endian float32,
no extra rounding. Exact submodel names/shapes layout SHA-256:
`c3e65542745f259383b18d85574a24f0701e72879f5a2df3cef21d8b44681549`.
No labels/source IDs/filenames/reference covers/declared QF are prediction inputs.
Do not call this descriptor our prior custom residual features or CC-JRM.

Fixed allowlisted worker, up to eight length-framed JPEGs: each input ≤2 MiB,
≤4 million pixels, minimum eight blocks per dimension. Total stdin ≤16 MiB
plus fixed headers; output exactly 45,020 bytes per image, maximum 360,160 bytes.
15-second wall / 15–16-second CPU, 1 GiB address space, 2 MiB file and zero-core
limits. Four workers maximum, queue only one batch per worker. Extraction split
deadline 1,800 seconds (at most one pending 15-second batch group drains).
Native failures/absence are not success. Incomplete output never gets a descriptor.

Separate checksum-bound raw float32 cache: ≤4,000 rows / ≤192 MiB per split,
small JSON descriptor binds all row identities, split, manifest, upstream version,
layout and data SHA. Exact size, no symlinks/pickle/overwrite. Original generic
64 MiB feature-document and 4,096-dimension model limits remain unchanged.

## Fixed training/evaluation

Train one mixed-method binary FLD ensemble using upstream FldEnsembleTrainer:
31 learners, subspace 256, seeds 20261008/20261009/20261010, fixed settings
without OOB grid search/early stopping/best-run selection. Use all 995 training
covers paired separately with each corresponding JUNIWARD/UERD stego: 1,990
paired cover rows and 1,990 stego rows. Duplicate covers are not new scenes.
Distinct origin/license records and matched method sets remain mandatory.

**Unweighted paired training**, not the previous source/class-weighted Adam
objective: upstream FLD does not accept those sample weights. This benchmark
changes representation/classifier/objective together, not an isolated causal
ablation; explicitly retain ALASKA/BOSS source imbalance and all regressions.
No learned normalization, no validation-driven parameter search.

Save only bounded numeric subspaces/weights/biases (≤1 MiB NPZ, exactly three
members, allow_pickle=False) and provenance card. Independent NumPy inference
must match upstream training vote fractions exactly before persistence.
Score `(mean sign(margin) + 1)/2` is a vote fraction, **not calibrated probability**.
Threshold .5 inclusive; do not use upstream randomized tie-label resolution.
Unknown method does not select a family model: one shared score for every file.

Publish every source, pooled/format and declared-quality × method cell against
the exact earlier nonlinear predictions, including missing/failed cells and
paired original-lineage 200-resample 95% change intervals. ECE on vote fractions
is diagnostic, not a confidence guarantee. Independently verify image hashes,
upstream JRM flatten/layout parity, scalar FLD margins/votes, confusion/pairwise
AUC/ECE and numeric reload at batch sizes 1/17/765. JRM upstream parity is not
an independent proof of the algorithm or its MATLAB equivalence. No ONNX or
signed model publication is claimed in this reference slice. No deployment or
new supported methods; qualification still needs untouched licensed sources.

## Explicit interface

`steganography research jrm-reference --config CONFIG.json --out FRESH_OUTPUT`
uses shared research services. Config `stage` is `features`, `train`, or `predict`:

- features: `manifest`, `source`, `split` (train/validation), `workers` (1–4).
- train: `manifest`, `cache` (cache.json), `cache_sha256`.
- predict: `manifest`, `cache`, `cache_sha256`, `model_dir`, `card_sha256`.

All acquisition/optional installation is explicit; no automatic downloads or
detector calls are added to normal analyze/CTF/API/TUI flows.
