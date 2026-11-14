# Predeclared nonlinear extension (2026-11-14)

This is a separate, bounded CPU comparison under the user's current rich-model
continuation. It extends the [linear protocol](RICH_MODEL_CPU_IMPROVEMENT_PROTOCOL_20261114.md)
without changing or relaunching that attempt. This document is written before
opening any of its audit results. No choice below depends on an audit metric.
All old BOSS test features and ALASKA2 remain closed.

Use the identical frozen 5,999 fit /1,500 calibration /1,501 audit source pairs
and 8,997 exact-cover-feature groups. Fit a signed-square-root StandardScaler
on fit rows only. Compare six Gaussian-kernel SVMs, in this order:

- gamma = 0.1 /9,464, C = 1 /10 /100;
- gamma = 1 /9,464, C = 1 /10 /100.

Use installed scikit-learn 1.9.1 SVC with precomputed float64 RBF kernels,
`tol=0.001`, `max_iter=100000`, `probability=False`, cache 128 MiB. Generate
kernels in memory only. Record convergence and exclude nonconverged models.
The [official SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
defines the precomputed-kernel and numeric decision-function contracts.
Generated fixtures must verify manual support/dual prediction against both
precomputed and direct RBF SVC before any real-data fit.

Select the final recipe from the union of both six-candidate grids using
calibration balanced accuracy, then AUC, then original grid order followed
by the six nonlinear candidates. Freeze this union choice **before** reading
or computing its audit scores. Original-grid selection files may be read;
its audit JSON/scores must stay unopened until this final choice is written.
The original attempt may already have computed its own audit; those outputs
cannot influence this extension's fitting, selection or thresholds.

Only the union winner and the raw-Fisher reference support the final audit
comparison. Apply the original threshold rule, 2,000 paired group bootstrap
replicates, seed 2026111404 and >=1-point /positive-lower-95%-bound improvement
rule. Preserve the original grid's separate results and report both selection
stages. Intervals are per method, not adjusted for three methods or families;
this remains an internal audit within previously used historical training.

Wait for the original parent to close, then run one fresh detached parent,
with a 20-minute watchdog per method and a separate 64-MiB total output cap.
Freeze new sources, this document, existing cache bindings and generated
fixture evidence first. No automatic retry; preserve partial and unavailable
results. Save scaler/dual/support indices plus immutable training-cache
references, not redundant support-vector feature copies. These are private,
source-bound research states, not standalone released or deployed models.
No package installation, media decode, download, GPU or old-test scoring.
