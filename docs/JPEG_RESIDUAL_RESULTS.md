# JPEG block residual/parity results — 2026-10-05

**Small development changes, still failed detection; no deployment.** The
custom descriptor adds signed block-DCT residual/parity and DC statistics to
the previous measured-context prefix. It is not DCTR/JRM/SRNet or a deep model.
Code and [protocol](JPEG_RESIDUAL_PROTOCOL.md) were frozen in `abc3cb5` before
actual feature extraction/training. See [portable evidence](../benchmarks/jpeg-residual-development-20261005.json).

## Same files, objective and threshold

Exactly 3,750 existing JPEGs, 2,985 train / 765 validation, 2,066 features.
The previous 1,098 context values match exactly on every file. Seed 20261006,
300 Adam epochs, .01 learning rate, train-only scaling and source/class-balanced
objective remain unchanged. Threshold stays .5, no calibration/threshold search.
Increasing representation dimension changes linear capacity/initialization
distribution; this is not an isolated causal explanation of individual features.

Both ALASKA2 and BOSS origins are shared between train/validation: **zero unseen
sources**. BOSS has 25 original validation lineages, two correlated quality
derivatives each; they are not 50 independent scenes. Camera/device/app and ALASKA
quality/payload metadata remain unknown. JMiPOD remains excluded. BOSS stegos
are upstream simulations, not encoded-message or payload-recovery evidence.

## Every source and declared-quality cohort

Each method reuses its cohort's covers. FP/TN therefore repeat across methods,
not extra independent covers. All pooled/format/unknown-quality cells, confusion
counts, paired-lineage 200-resample 95% intervals and same-resample reference/new
delta intervals are retained in JSON. Unknown quality is unavailable, not guessed.

| Cohort | Method | Cover/stego files each | AUC | BA | Recall | FP / covers (FPR) | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALASKA2 | JUNIWARD | 205 | .572945 | .543902 | .551220 | 95 / 205 (.463415) | .021644 |
| ALASKA2 | UERD | 205 | .570137 | .546341 | .556098 | 95 / 205 (.463415) | .025980 |
| BOSS, both qualities | JUNIWARD | 50 | .525600 | .520000 | .440000 | 20 / 50 (.400000) | .024531 |
| BOSS, both qualities | UERD | 50 | .521600 | .520000 | .440000 | 20 / 50 (.400000) | .016406 |
| BOSS Q75 | JUNIWARD | 25 | .502400 | .520000 | .400000 | 9 / 25 (.360000) | .021936 |
| BOSS Q75 | UERD | 25 | .523200 | .540000 | .440000 | 9 / 25 (.360000) | .025418 |
| BOSS Q95 | JUNIWARD | 25 | .550400 | .520000 | .480000 | 11 / 25 (.440000) | .027126 |
| BOSS Q95 | UERD | 25 | .532800 | .500000 | .440000 | 11 / 25 (.440000) | .036862 |

Against the immediate [two-origin context reference](JPEG_CONTEXT_RESULTS.md),
ALASKA FPR falls 50.24% → 46.34%, BOSS 48% → 40%. This is not enough: every
measurable cohort fails AUC/BA/recall/FPR targets, no independent support cell
qualifies. ALASKA AUC .572945/.570137 and FPR 46.34% are still worse than the
**earlier single-origin** model's .592481/.585306 and 41.46% FPR. Do not select
only the more flattering immediate reference or combine scopes into one score.

Trade-offs against the immediate reference stay visible:

- ALASKA JUNIWARD recall falls .580488 → .551220 and ECE worsens .010574 →
  .021644; its AUC delta CI crosses zero. UERD recall rises .546341 → .556098
  but ECE also worsens. UERD's small AUC delta is .009697, development bootstrap
  interval [.001580, .018360], not a generalization qualification.
- BOSS JUNIWARD recall falls .54 → .44 and BA .53 → .52; UERD recall .50 → .44.
  Both BOSS AUC delta intervals cross zero. FPR improvements come with missed
  stegos; lower FPR alone is not success.
- ALASKA FPR delta interval [-.078171, .005000] crosses zero. BOSS interval
  [-.18, -.02] is based on just 25 original validation lineages. These are
  iterative-development, uncorrected diagnostic intervals, not blind evidence.

## Independent audit

Rehashed all 3,750 source files. Verified every old/new context prefix exactly.
An independent scalar count/parity/residual oracle matches 18 actual JPEGs,
including both sources/splits and all selected quality/method variants, with
zero difference. Independently reproduced confusion counts, pairwise AUC, BA,
recall, FPR and ECE in all 12 measurable diagnostic cells.

NumPy train mean/scale deviations are at most 1.21e-7 / 1.48e-8. NumPy64/native
logits agree exactly; saved sigmoid scores differ by at most 5.36e-8. CPU ONNX
batches 1, 17 and all 765 validation rows agree exactly in logits and threshold
decisions at the unchanged absolute 1e-6 tolerance, relative tolerance zero.
This passes numerical export, **not** accuracy or signed-model publication.

An explicitly **post-hoc exploratory** pair-score diagnostic finds stego scores
above their corresponding cover on 137/205 JUNIWARD and 132/205 UERD ALASKA
pairs; BOSS 30/50 and 33/50. This needs a reference cover and cannot be deployed
as detection, counted as payload recovery or relabeled an accuracy gate. It
shows only weak controlled pair separation, not a causal explanation of failures.

Python 3.11: 692 tests, total coverage 93.84%; new residual descriptor 96.30%,
shared worker 98.89%; Ruff, mypy (90 files) and diff checks pass. Other Python
versions and a fresh Docker E2E rebuild were not run. All raw files, feature
caches, model weights and cards remain local; no primary score/verdict change.
Wheel/sdist build passes; the new module is included and no model/corpus is bundled.

## Reproduction and next gate

After explicit prior acquisition/preparation, use shared `research features`
with `--feature-version jpeg-dct-residual-parity-v1`, `--source` pointing at the
existing combined corpus, original bound manifest and a **fresh** output for
each train/validation split. Keep the original weighting/configuration, bind
the returned feature hashes, then run shared training/prediction/export/diagnosis
as in `JPEG_CONTEXT_RESULTS.md`. No dataset or model implicitly downloads.

Four bounded extraction workers and two CPU math threads took 139.31 seconds
including reference preflight, feature extraction, prefix checks, training,
prediction, export and diagnostics; independent audits are excluded. This is
not CTF/inference latency. Versions/hashes are in the portable record.

Further histogram expansion has not delivered useful detection. Next freeze a
materially stronger residual/quantization-conditioned architecture and its
train-only controls; avoid claiming that this handcrafted summary is a solved
context-aware detector. Qualification still needs a licensed untouched source
and adequate original-lineage counts. Preserve every failed experiment and
never tune on the frozen ALASKA native baseline.
