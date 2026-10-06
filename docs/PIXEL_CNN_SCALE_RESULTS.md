# Residual-domain intervention results — 2026-10-06

**All 12 detection cells fail again.** Fixed pixel-unit residuals/clipping did
not rescue the tiny network under the frozen settings. Every validation row
still predicts cover at inclusive .5: BA .50, recall 0, FPR 0. No deployment,
calibration, supported method or primary detector change.

Implementation/protocol frozen in `e1fea41` before fitting;
[protocol](PIXEL_CNN_SCALE_PROTOCOL.md). Exactly three fresh fits, same ten
epochs/seed/optimizer/source weights/data/learned layers as the first trial;
only residual scale and clipping domain changed together. No retries, tuning,
early stopping, warm-start or best-run selection. This hypothesis was chosen
after inspecting the first failure, so it remains exploratory development.

[Complete JSON evidence](../benchmarks/pixel-cnn-scale-development-20261006.json)
binds both reference publications, every prediction/model/card, frozen protocol,
all confusion matrices, lineage bootstrap 95% intervals, paired comparisons
against same-fit first CNN and all-source JRM, and all epoch losses.

## All cells, not the best run

Each BOSS cell: 50 cover + 50 stego variants, only 25 original scenes. ALASKA:
205 + 205 per cell. BA .50, recall/FPR 0% in every row. First CNN is compared
with the same training scope and validation rows, not a different subset.

| Training | Validation | Family | First CNN AUC | Pixel-unit AUC | Pixel-unit ECE |
|---|---|---|---:|---:|---:|
| All | BOSS | JUNIWARD | .502600 | .495200 | .000944 |
| All | BOSS | UERD | .502600 | .503800 | .000944 |
| All | ALASKA | JUNIWARD | .502451 | .501309 | .000942 |
| All | ALASKA | UERD | .503189 | .500155 | .000942 |
| BOSS | BOSS | JUNIWARD | .508000 | .499400 | .002103 |
| BOSS | BOSS | UERD | .509400 | .506800 | .002100 |
| BOSS | ALASKA | JUNIWARD | .503129 | .501499 | .002082 |
| BOSS | ALASKA | UERD | .503058 | .501916 | .002082 |
| ALASKA | BOSS | JUNIWARD | .501800 | .500200 | .001149 |
| ALASKA | BOSS | UERD | .505800 | .509200 | .001148 |
| ALASKA | ALASKA | JUNIWARD | .502546 | .501416 | .001138 |
| ALASKA | ALASKA | UERD | .502903 | .498572 | .001138 |

Ten AUC values regress, two rise slightly; neither small gain qualifies
detection. ECE worsens in all twelve versus the previous CNN. Low absolute ECE
and zero FPR reflect near-.5 scores with all-cover decisions, not useful
calibration or safety. AUC/BA/recall point targets fail in every cell.

## Integrity, independent replay and timing

All 3,750 originals rehashed; unchanged caches, train scopes/settings and
validation identities. Every 765-row fit replayed from numeric weights through
native inference and a separate NumPy forward pass; maximum score differences
below 5.97e-8. ONNX CPU batches 1/17/64 replay every row, maximum error below
5.97e-8; all cutoff decisions agree. Frozen absolute tolerance 1e-6 / zero
relative. Independent confusion/pairwise AUC/ECE agree with production metrics.
Numerical gates pass, detection gates do not.

Training seconds: all 43.59, BOSS 9.77, ALASKA 36.08; fit/prediction orchestration
93.00 s excluding extraction/audit. CPU-only, VMware eight allocated vCPUs on
AMD Ryzen 9 8945HX, two math threads; versions remain recorded in JSON. No
new sources, scenes, crop regeneration, automatic model or dataset download.

Qualification remains unavailable: reused inspected validation, unknown camera/
device/app independence and ALASKA quality/payload, BOSS simulated positives
and correlated Q variants, insufficient independent scenes, JMiPOD excluded.
Different source-only sample sizes/optimization steps remain confounded.
This does not prove normalization is irrelevant to larger architectures,
other crops, optimizers or sufficient data. Stop treating isolated tiny-network
tweaks as a route to deployment; stronger separately frozen learning experiments
and an untouched licensed external source remain necessary.

Raw caches, numeric weights and ONNX stay local/ignored. Audit from repository
root after explicit licensed preparation. Verification: Python 3.11.14, 877
tests pass, 94.31% total coverage; CNN/model/training code 244/244 statements.
Ruff, mypy (99 files), diff checks and model-free wheel/sdist build/inspection
pass. Python 3.12–3.14 and fresh full Docker unverified. Reproduce from repository
root using fresh output/export paths, after explicit licensed preparation:

```sh
venv/bin/python scripts/audit-pixel-cnn.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --corpus .benchmark/jpeg-context-multisource-20261005 \
  --cache-root .benchmark/pixel-residual-preparation-20261005 \
  --experiment .benchmark/jpeg-pixel-cnn-scale-20261006 \
  --reference-record benchmarks/jrm-reference-development-20261005.json \
  --pixel-reference-record benchmarks/pixel-cnn-development-20261006.json \
  --pixel-reference-experiment .benchmark/jpeg-pixel-cnn-20261006 \
  --architecture jpeg-center128-pixel-residual-cnn8-v1 \
  --protocol docs/PIXEL_CNN_SCALE_PROTOCOL.md \
  --exports .benchmark/jpeg-pixel-cnn-scale-20261006/exports-replay \
  --out .benchmark/jpeg-pixel-cnn-scale-20261006/audit-replay.json
```
