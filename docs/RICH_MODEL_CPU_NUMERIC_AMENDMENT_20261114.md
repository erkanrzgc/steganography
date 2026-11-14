# Numeric replay correction and bounded completion (2026-11-14)

The first [nonlinear comparison](RICH_MODEL_CPU_NONLINEAR_PROTOCOL_20261114.md)
exits 1 after 48.043474 seconds on HILL. C=1 at gamma .1/9,464 completes;
the following C=10 fit trips the 1e-10 absolute/relative replay assertion
before saving its state. Preserve that unavailable attempt, its successful
C=1 state/calibration, source freeze and failed log. No union audit was opened.

Generated 12,000-row near-constant kernels reproduce cancellation between
positive and negative dual terms: direct libsvm and matrix-vector predictions
differ by 5.356e-10 at C=10 and 6.187e-9 at C=100. Both generated fits converge.
The original tiny fixture passed but did not cover this accumulation scale.
The first real failure did not retain its exact error magnitude; do not claim
that unavailable value was measured by the generated controls.

A separately frozen software-corrected completion uses 1e-7 absolute/relative
prediction parity and records the actual maximum error for every fresh fit.
This compares equivalent floating-point sums; it changes no model family,
gamma/C, source partition, fit statistics, calibration threshold rule,
selection rule or audit acceptance criterion. No audit metric motivates it.
The generated controls must pass the revised guard first.

Keep the failed parent closed; do not relaunch it. Use a fresh output directory
and parent. Reuse the completed first C=1 state and calibration byte-for-byte;
all unfinished candidates retain their fixed recipes. Charge the first
48.043474 seconds to HILL's original 1,200-second allowance: the completion
watchdog is at most 1,151 seconds. S-UNIWARD and MiPOD retain their unspent
1,200 seconds each. The combined output of both nonlinear directories stays
below the original 64-MiB ceiling. This is a manually diagnosed source
correction with cumulative debit, not an automatic failure retry or reset.
No further completion or guard relaxation follows automatically.

Original linear/ensemble results stay separate. The union winner is still
frozen from calibration before any audit scores are opened. Preserve all
sources, failed evidence, generated controls and both attempts' exit records.
No installed dependency, public application source, dataset, old test,
reserved pixel, GPU or download changes in this correction.
