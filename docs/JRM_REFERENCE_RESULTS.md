# JRM + FLD research-reference results — 2026-10-05

**UERD ranking improves; JUNIWARD recall regresses; detection remains failed.**
Optional sealwatch 2024.12 JRM + fixed FLD is a local research reference, not
a deployed detector, CNN or calibrated model. The [protocol](JRM_REFERENCE_PROTOCOL.md)
and implementation were frozen in `2a9d005` before extraction/training. The
[portable evidence](../benchmarks/jrm-reference-development-20261005.json)
preserves every cell, original-reference metrics, paired intervals and failures.

## Same-file comparison

All 3,750 original JPEGs: 2,985 train / 765 validation, unchanged hashes,
lineages, quality variants and row order. Uncalibrated luminance JRM has 11,255
features. Fixed 31 FLD learners, subspace 256, seeds 20261008/09/10, threshold
.5 inclusive; no parameter search, calibration, exclusions or best-run selection.
All 995 training covers pair with both corresponding JUNIWARD/UERD stegos,
giving 1,990 pairs; duplicate covers are not new scenes. Final training OOB
error .401759 is a development statistic, not independent validation.

Representation, classifier and objective change together: upstream FLD uses
unweighted pairs, whereas the previous MLP used source/class weighting.
ALASKA dominates training. This is not an isolated feature/architecture ablation.
Source IDs, filenames, labels, reference covers and declared quality are not
inference inputs. That does not establish source/context invariance.

| Cohort | Method | Covers/stegos each | AUC | BA | Recall | FP / covers (FPR) | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALASKA | JUNIWARD | 205 | .630208 | .578049 | .473171 | 65 / 205 (.317073) | .077419 |
| ALASKA | UERD | 205 | .698251 | .624390 | .565854 | 65 / 205 (.317073) | .052950 |
| BOSS, both qualities | JUNIWARD | 50 | .524000 | .540000 | .380000 | 15 / 50 (.300000) | .107097 |
| BOSS, both qualities | UERD | 50 | .559400 | .540000 | .380000 | 15 / 50 (.300000) | .100968 |
| BOSS Q75 | JUNIWARD | 25 | .488000 | .540000 | .440000 | 9 / 25 (.360000) | .138065 |
| BOSS Q75 | UERD | 25 | .584000 | .540000 | .440000 | 9 / 25 (.360000) | .082581 |
| BOSS Q95 | JUNIWARD | 25 | .566400 | .540000 | .320000 | 6 / 25 (.240000) | .122581 |
| BOSS Q95 | UERD | 25 | .540000 | .540000 | .320000 | 6 / 25 (.240000) | .119355 |

Covers repeat across families, not independent extra evidence. JSON retains all
14 cells: 12 measurable, two unknown-quality cells unavailable. Format/pooled
cells repeat the same JPEG rows. All measurable cells fail AUC .90, BA .85,
recall .80 and FPR .03 targets; vote-fraction ECE also exceeds .05 everywhere.
Vote fractions are not probabilities; ECE is diagnostic, not calibration proof.

Against the [previous nonlinear feature model](JPEG_NONLINEAR_RESULTS.md):

- ALASKA/UERD AUC .602546 → .698251, paired delta CI [.070356, .128129];
  BA .582927 → .624390, CI [.004817, .080549]. Recall .551220 → .565854,
  change CI crosses zero. Shared FPR .385366 → .317073, CI [-.156341, .019512].
- ALASKA/JUNIWARD AUC .608923 → .630208, change CI crosses zero. Recall
  .590244 → .473171 regresses, delta CI [-.200000, -.034146]; BA also falls.
- BOSS/JUNIWARD AUC .554400 → .524000 and recall .46 → .38 regress;
  UERD AUC .510400 → .559400 rises, recall .40 → .38 falls. Aggregate FPR
  .40 → .30, change CI [-.22, .06]. Both source-level AUC change intervals
  cross zero. ECE worsens for both methods and both quality variants; all those
  paired ECE change intervals are positive.

Intervals use 200 paired resamples of original lineages, seed 20261005; Q75/Q95
stay together. Uncorrected iterative-development intervals, not confirmatory
evidence. Both sources already occur in training; **zero unseen sources**.
BOSS has only 25 original validation scenes, with simulated payloads, not
payload-recovery evidence. ALASKA quality/payload and camera/device/app provenance
are unknown; JMiPOD is excluded. Cross-source qualification is unavailable,
numeric targets fail, and no method becomes supported.

## Audit, limits and reproduction

Rehashed all 3,750 files; bound all caches/cards/models and 765 ordered prediction
identities. Independent scalar FLD margins/votes match every saved score exactly;
numeric reload batches 1/17/765 match scores and decisions exactly. Independently
checked confusion, pairwise AUC, BA, recall, FPR and ten-bin ECE in all 12
measurable cells. Eighteen real JPEGs match direct upstream flattening across
both splits/origins/families/qualities. That oracle shares the upstream JRM
algorithm: it verifies boundary/layout parity, not MATLAB equivalence or an
independent proof of algorithm correctness. No ONNX export is claimed.

Extraction uses four bounded workers, up to eight JPEGs each, native threads 1,
15-second batch timeout and 1,800-second split deadline. Checksummed binary
caches replace giant feature JSON without relaxing the old generic limits.
Model loading preflights exact numeric array headers before allocation; giant
declared shapes, objects, truncation, unsupported versions, symlinks and overwrite
are rejected. No pickle or extracted code execution.

Observed Linux VMware guest: AMD Ryzen 9 8945HX, eight virtual CPUs. Train
extraction 862.12 s, validation 225.82 s, training 3.01 s; whole reference job
1,091.06 s. This is corpus preparation/training timing, **not** per-challenge CTF
median/p95. Execution summary captured final HEAD `a151e68`; original freeze
was `2a9d005`. Intervening commits added replay audit/header guards, not learned
parameters, objective or scoring. Original execution record is preserved.

Python 3.11: 757 tests, total coverage 94.03%, new reference code 289/290
statements covered. Ruff, mypy (93 files), diff checks and wheel/sdist build
pass. Wheel includes reference services, no weights/corpus/vendored sealwatch;
dependencies remain opt-in. Python 3.12–3.14 and fresh full Docker E2E are
unverified locally. Legacy ONNX tests emit 28 deprecation warnings.

Install explicitly with `python -m pip install '.[research,jrm-reference]'`.
CLI stages and bound config are in the frozen protocol. Replay all actual-artifact
audits (fresh output required):

```bash
python scripts/audit-jrm-reference.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --corpus .benchmark/jpeg-context-multisource-20261005 \
  --experiment .benchmark/jpeg-jrm-reference-20261005 \
  --reference-predictions .benchmark/jpeg-nonlinear-20261005/predictions.json \
  --reference-record benchmarks/jpeg-nonlinear-development-20261005.json \
  --out FRESH_AUDIT.json
```

Optional sealwatch MPL-2.0 and DDE JRM educational/research/non-profit terms
remain in [third-party notices](../THIRD_PARTY_NOTICES.md). Local research only;
no upstream code, weights or raw data redistributed. Model installation,
primary scores and calibrated/confirmed verdicts are unchanged. Next work:
preregister stronger raw-residual learning, obtain a licensed untouched source,
then evaluate cross-source behavior and calibrated operating points separately.
