# Fresh MiPOD kernel comparison (2026-11-15)

The user's explicit continuation permits this fresh CPU comparison at **0.4
bpp**, using existing BOSS caches only. Previous parents/states/test outcomes
and allowances remain closed. No GPU, new media/model download, raw decoding,
native Commons change, reserved pixels or ALASKA2 access.

## Data and fixed candidates

Retain 7,500 fit source pairs /7,498 groups and the same 1,500 calibration
pairs /1,499 groups. The former internal audit stays training data; there is
no new internal audit accuracy. Keep pairs/duplicate feature groups together.
The predecessor is the frozen square-root standardized C=100 Gaussian model,
with calibration BA .798 and historical test BA .779.

Freeze the following 34 fresh candidates before fitting, in this order:

1. Four kernel-ridge classifiers on the predecessor's full standardized
   square-root Gaussian kernel, gamma 1/9,464; diagonal ridge .01/.03/.1/.3,
   targets -1/+1 and no fitted intercept. Solve by float64 Cholesky. Require
   a maximum training-equation residual <=1e-6.
2. Fifteen SVCs on an additive family kernel, signed power .5: plain weight
   0/.25/.5/.75/1, each with C=30/100/300. The complementary weight belongs
   to the selection-channel-aware family. Both halves have 4,732 features;
   each Gaussian uses gamma 1/4,732 after fit-only standardization.
3. The same fifteen SVCs with signed fourth-root power .25.

A family kernel is the weighted sum of its two Gaussian kernels, not the
predecessor's Gaussian over concatenated features. No image features are
extracted or changed. Fit scalers use training rows only. SVC uses installed
scikit-learn 1.9.1, tol=.001, max_iter=1,000,000, cache 128 MiB and no
probability estimation. Record all convergence warnings and make any
nonconverged state ineligible. Numeric SVC prediction parity is 1e-7 absolute/
relative. Unreplayable states are ineligible; no retry or guard relaxation.

Compare all eligible candidates plus the predecessor by exact-count
calibration BA, then calibration AUC, then original order (old state first).
Thresholds use the same exact-count/closest-to-zero rule. No choice uses a
test score. Persist a global choice/model/scaler/source freeze first. If the
predecessor wins, retain its recorded test result without another inference.
Otherwise perform one inference of the selected state on the historical
1,000 BOSS pairs, then independently replay saved metrics and sampled states.
This is a further adaptive historical test, not untouched qualification.
Report the 80% point-estimate pass/miss, TPR/FPR and exploratory paired-group
95% gain interval (2,000 resamples, seed 2026111501). No further grid follows.

## Software, execution and storage

Application/test code is unchanged: verify all 570 inputs of the passed
4,646-test software gate. Reuse that actual passed gate; do not claim a new
full run or generate another large QA fixture tree. Generated in-memory
checks independently verify scalers, additive Gaussian kernels, Cholesky
solutions and exported SVC/ridge scores before any actual fit. Freeze all
new private sources, actual versions, cache hashes and this protocol.

Use one fresh exclusive parent/attempt marker, one BLAS/OpenMP thread and a
new operator cap of 1,200 seconds including preparation, comparison, optional
inference and closing reserve. This is not reused time from an old allowance.
The parent samples RSS against 10 GiB and retains wait4 peaks; this observer
is not a cgroup resident-RAM/zero-swap guarantee. Preserve every failure or
unavailable attempt, with no automatic restart.

All Gram matrices stay in RAM. Mixture updates use bounded row blocks to
avoid a third full-sized matrix. Shared normalizers/indices and compressed
numeric support/dual or ridge vectors keep total output <=8 MiB. No kernel
or feature-cache copy is written. Models remain private and are not deployed.
