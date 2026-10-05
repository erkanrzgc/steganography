# JPEG training-source transfer diagnostic — 2026-10-05

**Source transfer is weak; no detector is qualified or deployed.** The
[protocol](JRM_TRANSFER_PROTOCOL.md) and implementation were committed as
`f8a4660` before actual fitting. [Portable evidence](../benchmarks/jrm-transfer-development-20261005.json)
retains all eight cells, source-scope cards, confusion counts, intervals and
hashes. This is an inspected-development diagnostic, not blind/new-source proof.

## Fixed two-way fits

Same 11,255-dimensional caches and 3,750 JPEGs as the completed all-source
[JRM/FLD reference](JRM_REFERENCE_RESULTS.md). Only training-source inclusion
changes: ALASKA-only uses 789 covers / 1,578 stegos; BOSS-only uses 206 quality
variants / 412 stegos, representing 103 original training scenes. The .5 cutoff,
31 learners, subspace 256, seeds, pairing and unweighted objective stay fixed.
No validation fitting, calibration, parameter search or best-run selection.

Both models predict every original validation row. BOSS validation represents
25 original scenes with correlated Q75/Q95 variants, not 50 independent scenes.
Training sizes differ markedly: no causal attribution to source identity alone.

| Train → validation | Method | Covers/stegos each | AUC | BA | Recall | FP / covers (FPR) | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BOSS → BOSS | JUNIWARD | 50 | .528600 | .510000 | .440000 | 21 / 50 (.420000) | .085806 |
| BOSS → BOSS | UERD | 50 | .504000 | .480000 | .380000 | 21 / 50 (.420000) | .120000 |
| BOSS → ALASKA | JUNIWARD | 205 | .547960 | .531707 | .395122 | 68 / 205 (.331707) | .054131 |
| BOSS → ALASKA | UERD | 205 | .554182 | .531707 | .395122 | 68 / 205 (.331707) | .057671 |
| ALASKA → BOSS | JUNIWARD | 50 | .533800 | .510000 | .420000 | 20 / 50 (.400000) | .161613 |
| ALASKA → BOSS | UERD | 50 | .522200 | .510000 | .420000 | 20 / 50 (.400000) | .132903 |
| ALASKA → ALASKA | JUNIWARD | 205 | .624485 | .595122 | .478049 | 59 / 205 (.287805) | .071518 |
| ALASKA → ALASKA | UERD | 205 | .690553 | .646341 | .580488 | 59 / 205 (.287805) | .067427 |

Within-origin controls remain visible; shared covers/FP counts are not extra
independent evidence. Every cell fails all five numeric targets (AUC .90,
BA .85, recall .80, FPR .03, ECE .05). Vote fractions are not probabilities;
their ECE is diagnostic. No general accuracy percentage or supported method.

Compared with the all-source reference on identical validation rows:

- BOSS-only → ALASKA JUNIWARD AUC .630208 → .547960, paired difference CI
  [-.109747, -.056991]; UERD .698251 → .554182, CI [-.181331, -.110681].
  Both recall values fall to .395122. The small BOSS training set also explains
  a possible sample-size confound; this is not proof of a specific shortcut.
- ALASKA-only → BOSS JUNIWARD .524000 → .533800, change CI crosses zero;
  UERD .559400 → .522200, CI [-.073415, -.003595]. Shared FPR rises .30 → .40;
  recall .38 → .42, but ECE worsens for both methods.
- ALASKA-only within-origin UERD AUC .690553 is close to the all-source .698251;
  AUC-change CI crosses zero. This cannot establish generalization. BOSS-only
  within-origin UERD regresses to .504000; all controls remain failed.

All new-score and paired-difference 95% intervals are retained in JSON:
200 resamples, original lineage as unit, correlated qualities kept together,
seed 20261005. Uncorrected iterative-development intervals, not confirmatory
statistical support. Both corpora and validation outcomes were previously used;
camera/device/perceptual independence is unknown. ALASKA quality/payload metadata
is unavailable; BOSS stegos are simulations, not extraction evidence. JMiPOD is
excluded. Cross-source release qualification remains unavailable.

## Verification and reproduction

Rehashed all 3,750 original files and bound both caches. Independently reproduced
scalar FLD votes for 765 rows per model, all eight confusion/AUC/BA/recall/FPR/ECE
cells and original-reference metrics. Reload batches 1/17/765 match scores and
decisions exactly for both models. This does not prove the JRM algorithm
independently; upstream boundary parity was measured in the earlier reference.
Training-scope hashes bind only original train rows; tests verify the actual
trainer arrays exclude other sources and preserve complete paired families.
Legacy cards work; invalid/altered scope is rejected before numeric loading.

The cached two-fit/metric job took 2.91 seconds on the previously recorded
eight-vCPU Ryzen 9 8945HX VMware guest, BLAS threads 2. This excludes earlier
feature extraction and independent audit; it is not CTF median/p95 evidence.
No raw data, model weights or upstream code are committed. Base wheel remains
model-free; upstream optional research/non-profit terms remain unchanged.

Python 3.11: 779 tests, total coverage 94.08%; JRM services 255/255 statements
covered. Ruff, mypy (94 application files), diff checks and wheel/sdist build
pass. Other Python versions and fresh full Docker E2E remain unverified.

Use explicit `stage: source-transfer` config through the existing
`research jrm-reference` CLI, as specified in the frozen protocol. Replay without
fitting, with a fresh output path:

```bash
python scripts/audit-jrm-transfer.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --corpus .benchmark/jpeg-context-multisource-20261005 \
  --experiment .benchmark/jpeg-jrm-transfer-20261005 \
  --reference-cache .benchmark/jpeg-jrm-reference-20261005/validation/cache.json \
  --reference-record benchmarks/jrm-reference-development-20261005.json \
  --reference-record-sha256 eb7b69e97c8e43d7c46a6deeb29d04b82bfd9972f5cb61bffb1187fa09f8a9cc \
  --out FRESH_AUDIT.json
```

Next: preregister stronger raw-residual learning with explicit content/quality
preprocessing and source-exclusion controls; then acquire a licensed untouched
evaluation origin. Do not deploy or tune a threshold on these failed validation
cells to manufacture a pass. New sources and blind CTF gates remain separate.
