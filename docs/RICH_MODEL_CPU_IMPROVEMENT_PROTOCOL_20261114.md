# Cached rich-model CPU classifier improvement (2026-11-14)

The user's continuation selects the CPU/rich-model direction. This is a fresh
classifier comparison on existing caches, not a restart of any consumed JPEG,
WAV, native GPU, Colab or collection attempt. No downloads or local GPU work.

The first target is HILL, S-UNIWARD and MiPOD at **0.4 bpp**. Keep their old
test results and all earlier misses. No improvement is claimed from the 0.6
bpp rows. All datasets, learned states and feature caches remain private.

## Data and selection

- Reuse only the 9,000-image training cohort and cached SRM + weighted features
  used by spatial v3. No media is decoded and no features are re-extracted.
- The old 1,000-image BOSS test caches and ALASKA2 holdout are not opened during
  development. Native Commons roles and reserved data remain unchanged.
- Source-image pairs stay together. Identical plain cover-feature vectors are
  grouped before splitting to prevent duplicate cached covers crossing folds.
- Split source groups deterministically, seed `2026111401`: two thirds fit,
  one sixth calibration, remainder audit (6,000 /1,500 /1,500 if all unique).
  Freeze indices and cache/source hashes before fitting.
- Fit statistics use fit rows only. Calibration chooses both the operating
  threshold and candidate. Audit rows are scored only after selection, for
  the selected candidate and the raw-Fisher reference.
- This is a new-model held-out comparison within historical training data.
  Those sources were fitted by older models; it is not an untouched external
  qualification set or a deployment result.

## Frozen candidate grid

All candidates use the same paired fit/calibration/audit groups.

1. Raw standardized ridge Fisher, ridge 0.03 (reference).
2. Signed square-root features + the same Fisher, ridge 0.03.
3. Signed fourth-root features + the same Fisher, ridge 0.03.
4. Paired bootstrap/random-subspace Fisher ensemble: 31 learners, 2,000
   features/learner, ridge 0.03, seed `2026111402`, mean tanh margin.
5. Standardized square-root features + linear SVM, C 0.01.
6. The same linear SVM, C 0.1.

The two SVMs use installed scikit-learn 1.9.1, `LinearSVC`, `dual="auto"`,
`tol=0.001`, `max_iter=2000`, seed `2026111403`. Nonconvergence is recorded;
an unconverged candidate cannot be selected. This adds no package download.
Classifier code, versions, input bindings and this protocol are frozen in a
new private attempt directory after focused generated software checks pass.

Thresholds maximize calibration balanced accuracy (score >= threshold is
stego); ties choose the value closest to zero, then the lower value. Select
the candidate by calibration balanced accuracy, then calibration AUC, then
the fixed order above. The reference gets the same threshold calibration.

Report audit AUC, balanced accuracy, TPR/FPR and paired bootstrap difference
against the reference (2,000 image-group resamples, seed `2026111404`). A
positive result requires at least one percentage point of audit improvement
and a positive lower bound of the per-method 95% paired interval. Report
whether each 0.4 bpp row actually reaches 80%; do not round misses into passes.
Intervals are per method, not adjusted for the three comparisons.

## Execution and reporting

Use a detached single CPU parent with a separately recorded operator-selected
20-minute watchdog per method and 64-MiB output ceiling. Reuse caches in RAM;
do not duplicate them on the almost-full disk. Preserve every intent, model,
score vector, failure and partial result. No automatic retry or threshold
selection from audit/test scores. Ineligible/unavailable runs are not passes.

A later source-frozen evaluation of selected recipes on the historical BOSS
test may be prepared only after this comparison and the complete software
gate. It must remain labelled adaptive because that test has been viewed
before; this protocol does not unlock ALASKA2 or authorize a GPU intervention.

References: the original authors' [rich-model paper](https://ws2.binghamton.edu/fridrich/Research/TIFS2012-SRM.pdf)
and [ensemble description](https://dde.binghamton.edu/download/ensemble/) support
the feature/ensemble direction. The power maps and this candidate grid are
local experiments, not claimed reproductions of those papers.
