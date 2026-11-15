# BOSS pixel-role correction for mixed LSB (2026-11-15)

The first fresh mixed-LSB worker correctly exits 1 at its source-role check,
before calibration scoring, model choice or test inference. The original
random split partitions file names without grouping equal pixel hashes.
One training file (training index 3,044) and one test file (test index 503)
share pixel content; their cached cover-feature rows independently match.
Training/validation and validation/test share no pixel hash.

Preserve `.benchmark/floor-mixed-lsb-20261115/` unchanged, including source,
freeze, generated receipt, failure log and exit. A source-corrected completion
uses the same existing models and 500 validation pairs, but quarantines
that test source by metadata before calibration. It evaluates the remaining
999 pairs. No original file, cache, split, model, threshold or old score is
rewritten. This is a source-leakage correction, not an accuracy-selected
exclusion; its index is fixed independently of labels/model performance.

The separate read-only audit in `benchmarks/boss-pixel-role-audit-20261115.json`
retains the original 1,000-source results alongside corrected source-clean
counts. HILL/S-UNIWARD/MiPOD retain point estimates above 80%. WOW's exact
corrected metric is unavailable without saved score vectors; conservative
aggregate bounds remain above 80%. No consumed fit is rerun to reconstruct
missing evidence. All BOSS historical results remain adaptive, without
untouched qualification.

The exclusive mixed-LSB completion directory is
`.benchmark/floor-mixed-lsb-completion-20261115/`. Freeze corrected source,
this amendment, original protocol and failed-source/role-audit bindings.
Charge ten conservative seconds for the old 1.004-second command, parent
overhead and read-only diagnosis; the original 120-second allowance is not
reset, leaving 110 seconds for preparation, calibration, inference and
independent closing. Keep the original two-GiB sampled RSS /one-MiB output
limits, numeric checks and no-refitting rule. No guard is relaxed and no
original sample/payload is changed to force an 80% score.
