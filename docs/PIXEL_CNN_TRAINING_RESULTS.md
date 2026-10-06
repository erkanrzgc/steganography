# First pixel-residual CNN results — 2026-10-06

**Detection failed in every one of the 12 declared cells.** All three fixed
fits predict cover for every validation row at cutoff .5. Balanced accuracy is
.50, recall 0 and FPR 0; low ECE does not imply useful detection or calibration.
No detector, model catalog, supported method or primary verdict changes.

Protocol and implementation were committed before fitting in `ac377b4`;
[frozen protocol](PIXEL_CNN_TRAINING_PROTOCOL.md) SHA-256
`5fb3a1bd2c5ff3538d6fd3295ffcd87cad62c0baa5271355912381f95def6967`.
Exactly one fit per scope, ten fixed epochs, no tuning, rerun, early stopping,
calibration or best-run selection. The same 3,750 originals and audited crops
remain unchanged. [Machine-readable evidence](../benchmarks/pixel-cnn-development-20261006.json)
contains all confusion matrices, scores, original-lineage bootstrap 95% intervals,
paired differences against JRM, model/card/prediction hashes and epoch losses.

## Complete results

Each ALASKA cell has 205 covers + 205 stegos; each BOSS cell has 50 + 50,
representing only 25 original scenes at two quality factors. All rows below
have BA .50, recall 0%, FPR 0%. JRM AUC is the previous all-source reference,
not a same-scope classifier or isolated causal architecture comparison.

| Training scope | Validation origin | Family | CNN AUC | JRM AUC | CNN ECE |
|---|---|---|---:|---:|---:|
| All | BOSS | JUNIWARD | .502600 | .524000 | .000359 |
| All | BOSS | UERD | .502600 | .559400 | .000359 |
| All | ALASKA | JUNIWARD | .502451 | .630208 | .000361 |
| All | ALASKA | UERD | .503189 | .698251 | .000361 |
| BOSS | BOSS | JUNIWARD | .508000 | .524000 | .001001 |
| BOSS | BOSS | UERD | .509400 | .559400 | .001001 |
| BOSS | ALASKA | JUNIWARD | .503129 | .630208 | .001022 |
| BOSS | ALASKA | UERD | .503058 | .698251 | .001022 |
| ALASKA | BOSS | JUNIWARD | .501800 | .524000 | .000757 |
| ALASKA | BOSS | UERD | .505800 | .559400 | .000757 |
| ALASKA | ALASKA | JUNIWARD | .502546 | .630208 | .000760 |
| ALASKA | ALASKA | UERD | .502903 | .698251 | .000760 |

AUC, balanced accuracy and recall fail their point targets in every cell.
FPR/ECE point targets alone pass because scores are near .5 and every decision
is cover; this is not a qualified detector. Source exclusion is training-only
and diagnostic: both origins and this validation were already inspected.
Different training sizes/step counts preclude a causal source-size comparison.

## Independent replay and timing

All 3,750 original file hashes were rechecked. All 765 validation rows per fit
match their declared identities; independent pairwise AUC, confusion and ECE
agree with shared metric code. A separately implemented NumPy forward pass
replays every row from persisted numeric weights; maximum score error is below
`5.86e-8`. Native reload is exact. ONNX CPU replay checks every row in batch
sizes 1, 17 and 64; maximum error `5.97e-8`, every threshold decision identical.
The frozen tolerance was `1e-6` absolute / zero relative. **Numerical correctness
passes; detection does not.** Missing optional dependencies report unavailable.

Training wall times: all-source 38.31 s, BOSS 7.31 s, ALASKA 30.51 s; total
fit/prediction orchestration 79.31 s, excluding preparation and audit. AMD
Ryzen 9 8945HX host, VMware eight allocated vCPUs, CPU-only with two math
threads. Torch 2.14.0+cpu, NumPy 2.4.6, ONNX 1.22.0, ORT 1.30.0; decoder
contract remains Pillow 12.2.0 / reported JPEG API codec 6.2. This API version
does not guarantee binary-identical JPEG implementations on another host.

## Limitations and next decision

Qualification remains **unavailable**: reused development validation, insufficient
independent scenes, unknown camera/device/app independence and ALASKA quality/
payload, simulated BOSS positives, excluded JMiPOD. We cannot attribute the
failure specifically to crop size, residual scale, pooling, optimization or
capacity from this experiment. Do not silently increase epochs or choose a
threshold on these validation results. A stronger representation/architecture
needs a separately frozen experiment and an untouched external source before
release claims; this tiny model is a failed baseline, not SRNet.

Raw data, crops, numeric weights and ONNX remain local and ignored. Publication
verification: Python 3.11.14, 869 tests pass, total coverage 94.30%, new CNN/
numeric persistence/training code 228/228 statements. Ruff, mypy (99 files),
diff check and model-free wheel/sdist build/inspection pass. Python 3.12–3.14
and fresh full Docker were not verified here. Reproduce the audit from the
repository root after explicit licensed dataset preparation:

```sh
venv/bin/python scripts/audit-pixel-cnn.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --corpus .benchmark/jpeg-context-multisource-20261005 \
  --cache-root .benchmark/pixel-residual-preparation-20261005 \
  --experiment .benchmark/jpeg-pixel-cnn-20261006 \
  --reference-record benchmarks/jrm-reference-development-20261005.json \
  --exports .benchmark/jpeg-pixel-cnn-20261006/exports-replay \
  --out .benchmark/jpeg-pixel-cnn-20261006/audit-replay.json
```

Outputs must be fresh; do not overwrite the original audit/exports. Numeric
model loading does not execute pickle; independent audit scripts operate only
on explicitly supplied local research artifacts, not untrusted API requests.
