# Paired score diagnostics

`core/jpeg_score_diagnostics.py` descriptively compares existing probabilities
for original-disjoint development cover/stego pairs. Inputs are bounded to
1,024 rows, finite probabilities, ordered matched pairs and balanced declared
BOSS/BOWS source groups. No pixels, model files or reserved inputs are opened.

For each source report the number of positive, tied and negative within-pair
score changes; mean/median signed change; cover/stego score means; population
dispersion of pair midpoints; and midpoint dispersion divided by mean absolute
pair change. A zero/subnormal-unrepresentable denominator yields a null ratio,
never infinite evidence. Paths and original identities are not emitted.

This calculation requires a known cover. Even every stego scoring higher than
its own cover does not imply high standalone ROC-AUC or accuracy when comparing
unrelated images. Larger between-pair dispersion is a descriptive observation,
not proof of a causal content, BatchNorm, overfitting or preprocessing defect.
Use paired direction and existing standalone metrics together to formulate a
preregistered train-only follow-up, not to select development thresholds.

No score adjustment, threshold optimization, extra model update, inference
change or deployment is authorized by this diagnostic. Current running study
sources/controller bindings stay frozen: this module is a separate CPU-only
research aid. Qualification remains unavailable; acceptance gates and existing
primary verdicts/calibration are unchanged. Independent arithmetic, overflow,
ancestry, bounds and privacy tests include a fixture with every pair positive
but cross-image AUC only 0.75, illustrating the different questions measured.
