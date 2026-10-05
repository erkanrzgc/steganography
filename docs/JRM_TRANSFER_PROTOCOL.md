# Frozen JPEG training-source exclusion diagnostic — 2026-10-05

Commit implementation/protocol before real cached-corpus fitting or scoring.
This diagnostic changes only the training-source inclusion relative to the
completed [JRM reference](JRM_REFERENCE_RESULTS.md). It is not untouched-source
qualification: both original validation origins and outcomes were inspected in
earlier development runs. No new download, scenes, calibration or deployment.

## Fixed inputs and fits

Same original 3,750 JPEGs, split and row order. Manifest SHA-256:
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
Reuse checksum-bound 11,255-dimensional JRM caches from the completed reference;
do not regenerate JPEGs or extract a changed descriptor. Bind the published
reference prediction/cache hashes before fitting. Preserve all previous results.

Frozen descriptor hashes: train
`bf60c5ee18a9c759039370897b08b38c6c2c2b80c357689dccd494014e794ca1`, validation
`90e616cb6098d903024951f1d91b8eac31725d9954350df1cf4c50a70cc404eb`.
All-source reference predictions SHA-256:
`62b0b34b2386f2f34a0d0bb346b613ee6f65fad7980f480737ac7b22bbfe870e`.

Fit exactly two ensembles: one using only the 2,367 ALASKA training rows
(789 covers, 1,578 stegos); one using only the 618 BOSS training rows (206 cover
variants, 412 stegos, 103 original scenes). Keep original validation completely
out of fitting: 615 ALASKA rows and 150 BOSS rows (25 original BOSS scenes,
Q75/Q95 kept together). No source swapping of train/validation samples.

Training settings stay sealwatch 2024.12, 31 learners, subspace 256, seeds
20261008/09/10, complete cover pairs for both JUNIWARD/UERD families, unweighted,
no normalization, OOB search, early stopping, best-run selection or tuning.
Reference marker/card/schema stays compatible; additive training-scope provenance
binds selected/excluded opaque source IDs and ordered training-row SHA-256.
Scope metadata must validate against the manifest before numeric model loading.
Legacy all-source cards without this additive field remain loadable.

## Reporting and audit

Publish all eight train-origin × validation-origin × family cells, including
within-training-origin controls and cross-origin regressions. Shared .5 inclusive
cutoff, vote fractions not calibrated probabilities; ECE diagnostic only.
Compare against the previous all-source JRM predictions on the identical rows.
Show AUC, BA, recall, FPR, ECE, confusion counts and original-lineage 200-resample
95% intervals (seed 20261005). Retain correlated Q variants together. Never call
this training-source ablation an independent confirmatory test or supported cell.

Before publication, independently recompute confusion, pairwise AUC and ECE,
scalar FLD votes for all 765 rows per model; rehash original input files and
compare prediction identities. Record every model/cache/card/prediction hash,
execution commit, timing and optional dependency versions. Retain failed or
unavailable cells; interrupted output has no complete report. No threshold search.

Qualification remains unavailable: source origins are not verified camera/device
or perceptual independence, sample counts are insufficient, reused validation
is inspected, ALASKA quality/payload provenance is unknown, BOSS stegos are
simulations and JMiPOD is excluded. The two training sizes differ; attribute
differences to training-source exclusion plus sample-size shift, not source
identity causally. No new method, signed model, ONNX or extraction claim.

## Interface and safety

`research jrm-reference --config CONFIG --out FRESH_DIR`, with config stage
`source-transfer`; required keys: `manifest`, `manifest_sha256`, `train_cache`,
`train_cache_sha256`, `validation_cache`, `validation_cache_sha256`,
`reference_predictions`, `reference_predictions_sha256`. All use existing bounded,
non-symlink JSON/raw/numeric loaders. Exactly two declared matched-family origins
must occur in each split. Source IDs are opaque report metadata, never features.
Independent `train` stage additionally accepts explicit `training_source_id`.
Default remains all sources. Base wheel stays model-free, upstream dependencies
optional, research/non-profit terms retained in `THIRD_PARTY_NOTICES.md`.

Stronger raw-residual learning and a licensed untouched evaluation source remain
the subsequent experiment, not replaced or qualified by this diagnostic.
