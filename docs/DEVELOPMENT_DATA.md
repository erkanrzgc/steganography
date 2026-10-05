# Preparing the next detector experiment

`JPEG_RESIDUAL_RESULTS.md` now records the fixed same-file residual/parity
follow-up. Prefix integrity and numerical export pass, detection does not;
retain this failure when moving beyond coarse linear summaries. No acquisition,
new original lineage, calibration or untouched-source evidence was added.

`JPEG_CONTEXT_PROTOCOL.md` freezes the next local two-origin JPEG development
experiment before generation/training: no new download, original split/lineage
preservation, optional pinned simulation and measured quantization/content
features. JMiPOD is explicitly excluded from both-source comparison, not falsely
claimed as simulated. Raw BOSS data and derivatives remain local research only.
The completed run (`JPEG_CONTEXT_RESULTS.md`) fails detection and is not installed;
both validation source groups remain shared with training, not independent tests.

Source/context diagnosis is fixed in `GENERALIZATION_PROTOCOL.md` and exposed
as `research diagnose`. It detects source overlap and stratifies existing
scores; it does not learn source-specific rules or fit score corrections.
Byte-derived descriptors can still encode dataset shortcuts, so unchanged
source-free model inputs are not evidence of cross-source generalization.
The actual audits (`GENERALIZATION_RESULTS.md`) confirm zero unseen validation
sources in both current models and missing declared quality/device metadata.

The completed spatial objective experiment reuses existing audited parity vectors;
no new download or test-image access is needed. Complete controlled training
lineages receive fixed weights, never validation-derived weights. See
`SPATIAL_WEIGHTING_PROTOCOL.md` / `SPATIAL_WEIGHTING_RESULTS.md`: low-rate
recall improves, but false alarms increase and no model is qualified/deployed.

The published pilot remains frozen. Its images, covers and derivatives are not
development or validation data. This workflow prepares a new experiment; it
does not train a detector or establish an accuracy claim.
Current acquisition checks and the next required user-supplied input are recorded
in `DATA_ACCESS_STATUS.md`; an accessible landing page is not dataset access or
a verified license grant.

The first separate JPEG development experiment has now completed; see
`JPEG_DEVELOPMENT_RESULTS.md`. Its poor validation/FPR result is not deployed.
The partitioning guidance below still applies to subsequent spatial and audio
work; `WAV_RESULTS.md` now records the failed fixed-threshold FSDD baseline,
with training/validation speakers reserved separately from the inspected test.

The first spatial comparison also completed on the new BOSSbase development
data; see `SPATIAL_DEVELOPMENT_RESULTS.md`. Its richer descriptor improves
same-source validation discrimination, but does not pass support/export gates
and is not installed. These validation results may inform development, never
be relabeled a fresh blind or independent-source test.

The fixed parity/residual follow-up (`SPATIAL_PARITY_RESULTS.md`) improves
same-row ranking/FPR but loses low-rate recall at the fixed threshold and is
not calibrated. All future validation reuse remains iterative development,
including calibration fitting; qualification needs genuinely untouched sources.

The 4,000-file ALASKA2 holdout has now been acquired and scored
(`ALASKA2_RESULTS.md`), with failed native detection targets. Its manifest
`.benchmark/alaska2-holdout-20261004/source.json` is another mandatory reserved
identity/lineage source for future development; do not train, calibrate or
choose thresholds on it. This baseline run did not create training data or a
validated JPEG feature/model workflow.

## Required inputs

Supply a verified research manifest (`schema_version: "1.0"`) for new, locally
available images. Each sample needs a relative path, SHA-256, size, cover/stego
label, explicit source group and cover lineage. Prefer the original cover's
SHA-256 as lineage, retained unchanged in every transformation/embedding.
Provide camera and device IDs where known; leave unknown values null rather
than inventing them. Review usage conditions before acquiring any new corpus.

Filename-based `research import` inference is only a starting point: review
and correct ancestry/source metadata before partitioning. Hashes cannot discover
undocumented transformations or perceptual duplicates. Metadata declarations
and independence still require human/source documentation.

## Command

Activate the provisioned Python environment first (`source venv/bin/activate`).
The following paths are placeholders except the existing frozen pilot manifest:

```sh
steganography research partition \
  --manifest new-corpus.json --source /path/to/new-corpus \
  --reserved-manifest .benchmark/pilot-v1/manifest.json \
  --test-source untouched-source \
  --out next-experiment.json
```

Repeat `--reserved-manifest` for other published/frozen corpora and
`--test-source` for additional untouched sources. No network access, training,
calibration or embedding occurs. Existing output files are never overwritten.

The command:

1. Verifies input file sizes/hashes and rejects symlinks.
2. Rejects any sample identity or original-cover hash overlapping reserved data.
3. Groups all connected lineages, cameras and devices within each source. A
   content-hash lineage is global, even if source/camera metadata changes.
4. Assigns entire selected sources to test; rejects related samples bridging
   the test/development source boundary. Remaining groups receive a deterministic
   80/20 train/validation assignment (seed 20260918), not a guaranteed count ratio.
5. Requires both labels in every split. Too few independent groups fail; the
   command does not silently split a camera or rebalance related images.
6. Records input/reserved-manifest hashes, grouping policy, counts and metadata
   completeness. Subsequent research manifest verification checks the recorded
   held-out-source and camera/device separation policy again.

Source names, not source paths, identify a source group. Human-readable lineage
IDs are scoped to their source; only SHA-256 lineage IDs link across renamed
sources. Renaming arbitrary lineage IDs can defeat ancestry checks and must
never be used to make data appear independent.

## Manifest-bound feature extraction and training

```sh
steganography research features --manifest next-experiment.json \
  --split train --out train-features.json
steganography research features --manifest next-experiment.json \
  --split validation --out validation-features.json
```

The command opens image bytes only for the selected split, never test images.
It checks sizes/hashes, refuses symlinks and existing output, and limits inputs
to 16 MiB / 4 million pixels and 10,000 rows. Documents are limited to 64 MiB.
Only PNG/BMP are accepted. It performs RGB conversion without resizing.
This paragraph describes the default spatial contract. JPEG now has a separate
opt-in `--feature-version jpeg-dct-summary-v1`; it reads original luminance DCT
coefficients with optional `jpeglib`, not a lossy conversion to PNG. Use
`--workers 1..4` for bounded parallel extraction. The 968 features and initial
CPU experiment are specified in `JPEG_DEVELOPMENT_PROTOCOL.md`.

`spatial-summary-v1` contains twelve simple exploratory features: per-channel
mean absolute adjacent difference and difference standard deviation (divided
by 255), LSB one-ratio and adjacent LSB agreement. Horizontal and vertical
differences are pooled. This is neither SRM nor SRNet, and no accuracy claim
is attached to these features. Conversion and replicated grayscale channels
are potential confounders.

The output records manifest hash, ordered feature contract and each row's
sample hash, lineage and label. The command prints the artifact SHA-256.
Use that exact hash in a training configuration:

```json
{
  "manifest": "next-experiment.json",
  "features": "train-features.json",
  "features_sha256": "COPY_THE_SHA256_PRINTED_BY_FEATURE_EXTRACTION",
  "seed": 20260918,
  "epochs": 10,
  "learning_rate": 0.001
}
```

With the optional research dependencies installed, run
`steganography research train --config train.json --out baseline.pt`.
Paths in this configuration are relative to the working directory. Training
rejects legacy unbound NPZ inputs, incomplete/reordered/mislabeled rows,
validation/test artifacts, changed manifest/artifact hashes, wrong feature
contracts and nonfinite/out-of-range values. The model is a linear binary
baseline (`spatial-summary-linear-v1`), not a neural image steganalyzer. Its
checkpoint and exported model card retain training provenance; it is not
automatically installed in the analysis service and remains uncalibrated.

These checks detect inconsistency and accidental misuse, not deliberate fraud:
a caller who rewrites both data and the declared hashes can fabricate provenance.
Keep the reviewed partition manifest and its reserved corpus records immutable.

## Remaining work before a new accuracy measurement

- Acquire/document a second source; none has yet been added by this change.
- Measure useful features on the separate development corpus; the simple
  versioned features are a reproducible baseline, not demonstrated improvement.
- Train on train only; choose thresholds/calibration on validation only.
- Freeze preprocessing/model/threshold before accessing the new test source.
- Publish failures and per-cell results using the release benchmark protocol.

Cloud AI cannot alter the analysis service's aggregate score or the pipeline's
primary findings/verdict. Its original output remains available as triage only.
The pilot had AI disabled, so this isolation fix does not improve or invalidate
its failed detector baseline.

## Explicit JPEG development workflow

`scripts/fetch-alaska2-pilot.py --purpose development --seed 20261005` excludes
reserved ALASKA2 basenames before selection and all reserved hashes during
download. Supply every prior manifest with `--reserved-manifest`. Selection
and resume provenance bind purpose, seed and exclusions; old evaluation
manifests are refused by the development importer.

`python -m steganography.research_jpeg run-development --source DIR --out NEWDIR
--source-sha256 HASH --reserved-manifest FILE` runs the fixed protocol:
verified import, whole-lineage train/validation separation, explicit quarantine
of contradictory byte-identical labels, features, 300-epoch class-balanced
linear training, validation-only prediction and ONNX export. It does not
download data, install a model, access old test images or change verdicts.
Training statistics are preserved inside the exported graph. An unsuccessful
experiment is reported, not deployed or retuned using old test results.
