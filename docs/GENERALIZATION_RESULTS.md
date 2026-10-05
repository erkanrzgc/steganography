# Source/context diagnostic results — 2026-10-05

Fixed grouping protocol `55a1889`, compatibility adapter `2886997`. Existing
validation scores were audited, not retrained or calibrated. No `analyze` /
`detect` score changes, model installation, acquisition or image access.
See [protocol](GENERALIZATION_PROTOCOL.md) and
[portable record](../benchmarks/generalization-development-20261005.json).

## What the models do and don't use

JPEG uses byte-derived luminance DCT statistics and quantization tables.
Spatial models use pixel residual/parity statistics. Neither contract includes
filenames, source IDs, lineage or cover/stego labels as prediction inputs.
Labels are training targets; metadata is used for split/diagnostic controls.

**This does not establish generalization.** Compression, camera processing,
texture and noise distributions can implicitly identify a dataset. JPEG
training remains ALASKA2-only; spatial training uses a different BOSSbase
corpus. Different carrier formats/models are not two-source qualification for
either model. A fully context-aware detector has not been delivered here.

## Source overlap and missing context

| Model/corpus | Training sources | Validation sources | Shared | Unseen validation sources | Result |
| --- | --- | --- | --- | --- | --- |
| Weighted spatial / controlled BOSSbase LSB | 1 | 1 | 1 | 0 | Independent-source evidence unavailable |
| JPEG / ALASKA2 development | 1 | 1 | 1 | 0 | Independent-source evidence unavailable |

All 1,288 spatial and 820 JPEG validation rows lack declared quality-factor,
camera, device and app metadata. These dimensions are not treated as verified
context. Quantization-table features do not imply that an original JPEG
quality factor is known, particularly for custom tables. Source declarations
are available but cannot themselves prove device/perceptual independence.

## Spatial format cohorts: all low-rate cells

At unchanged threshold .5; previous weighted model scores, not native detector
scores. Covers are shared across method comparisons, not extra independent
observations. All higher-rate cells/intervals are retained in the JSON record.

| Format / method at 5% | Pairs | AUC | Recall | FP / covers | FPR |
| --- | --- | --- | --- | --- | --- |
| PNG / sequential | 87 | .943718 | .689655 | 2 / 87 | .022989 |
| BMP / sequential | 97 | .954724 | .670103 | 4 / 97 | .041237 |
| PNG / scattered | 87 | .921390 | .287356 | 2 / 87 | .022989 |
| BMP / scattered | 97 | .928792 | .309278 | 4 / 97 | .041237 |

BMP cohort false alarms exceed the 3% target; low-rate recall fails in both
formats. These are **different original-lineage cohorts**, assigned formats
by a label-independent hash rule, not paired re-encodings of the same images.
Do not attribute the gap causally to BMP encoding: decoded spatial descriptors
agree for identical PNG/BMP pixels. Small cohorts and overlapping uncertainty
prevent a robust format-generalization claim.

## JPEG: unchanged failed development baseline

| Family | Pairs | AUC | Recall | FPR |
| --- | --- | --- | --- | --- |
| JMiPOD | 205 | .648804 | .678049 | .414634 |
| JUNIWARD | 205 | .592481 | .526829 | .414634 |
| UERD | 205 | .585306 | .536585 | .414634 |

These values are not new model improvements. They reproduce the existing
same-source development failure. The original JPEG export numerical failure
and separate precision repair remain documented in their original results;
this cached-score audit does not qualify an export or deployed detector.

The first JPEG diagnostic preflight rejected its historical v1 predictions,
which lacked additive `format` / `rate_percent` fields. A version/domain-
restricted adapter now obtains missing fields from the already SHA-bound
manifest after full identity validation (1,640 fields across 820 rows). No
input documents, hashes, probabilities, threshold or model weights changed.
Unrecognized legacy schemas and conflicting existing fields are still rejected.

## Verification and next implementation gate

The 24 spatial and 9 JPEG diagnostic cells were independently checked with
pairwise AUC and confusion calculations against original predictions/manifests.
All checks agree. Grouping, threshold and lineage bootstrap are unchanged.
Raw report hashes and model-card/prediction bindings are in the portable record.

Next work must distinguish measured file context from dataset identity:

1. Add independent **same-format** sources with reviewed licenses, complete
   cover ancestry and device/processing metadata where available. Freeze whole
   lineage/device/source separation before experiments; preserve old test sets.
2. Diagnose measured compression and image-content/noise strata. Unknown
   camera/quality values stay unknown; custom quantization tables are physical
   context, not an invented quality-factor label or dataset-specific rule.
3. Preregister any context-aware representation/calibration and compare all
   methods/rates, including out-of-domain and clean edited/noisy files. Missing
   validated coverage is uncertainty, not a clean-file certificate.
4. Only publish qualified independent-source metrics as generalization in the
   README. Keep inspected development, frozen native baselines and controlled
   CTF recovery in separate tables, with cohort size, version, FPR and intervals.

No cell passes support qualification. This slice improves diagnostic honesty
and controls, not learned sensitivity or real-world detection accuracy.
