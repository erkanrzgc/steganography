# MiPOD Hellinger geometry protocol (2026-11-15)

This separate CPU intervention tests a representation issue under the current
user request: previous Gaussian kernels use per-column standardization, which
changes the geometry of square-root histograms. The cited normalization paper
also considers exponential Hellinger features. We test the unwhitened square-
root geometry directly, without claiming an exact reproduction of that paper.
No historical outcome supplies a new threshold, bandwidth or candidate score.
Prior normalization/content attempts remain independently closed, with all
successes and failures retained and no allowance reset or restart.

## Fixed geometry and candidates

Use the same 7,500 fit /1,500 calibration pairs at 0.4 bpp, existing caches
and duplicate-group separation. No new media, raw decoding, GPU, native
Commons role change or ALASKA2 access. The global normalization reference is
`rc8-sqrt-ridge0.03` (79.70% historical BA); the selected content-system
reference (78.40%) is also eligible. Those historical numbers are context;
all choices below use calibration only. Comparisons use the global reference
as the controlled global-classifier ablation, retaining both reference rows.

Freeze two representations in order: plain square-root concatenated features,
then the fixed RC8 square-root representation from the normalization study.
Do not standardize individual columns or fit a replacement scaler. Transform
float32 inputs, then use float64 squared Euclidean distance between square-
root vectors. This corresponds to Hellinger-type histogram geometry; RC8
uses its conditional groups rather than the original block normalization.

For each representation, draw the same fixed 256 fit-cover positions without
replacement (seed 2026111507), and find the median positive off-diagonal
squared distance (values below 1e-12 are rounded to zero to suppress
floating cancellation for identical rows). Freeze Gaussian bandwidth gamma = factor / median, with
factors .25/1/4 in order. The median uses fit covers only. It is not estimated
from calibration/test. Share Gram matrices in RAM, raising the preceding
kernel to the fourth power for successive bandwidths; no matrix is persisted.

For each of six kernels, fit ridge .01/.03/.1 (targets -1/+1, no fitted
intercept, float64 Cholesky, equation residual <=1e-6), then C=100 SVC
(tol=.001, max_iter=1,000,000, cache 128 MiB, no probability estimation).
This is exactly 24 fresh models. Retain failed convergence/score replay as
ineligible, with fixed 1e-7 absolute/relative prediction guard and no retry.

Selection considers the global reference first, content reference second,
then the 24 fresh models, using exact-count calibration BA, AUC, original
order. Thresholds use the existing exact-count closest-to-zero tie rule.
Freeze choice, state, source and bandwidth bindings before any historical
feature access. If a reference wins, reuse its saved outcome without new
inference. Otherwise make one selected inference on the historical 1,000
pairs. Report all misses and TPR/FPR, with paired-group 95% gain interval
against the global reference (2,000 resamples, seed 2026111508). All outcomes
are further adaptive historical evidence, not untouched qualification.

## Execution and closure

Fresh exclusive parent, **900 seconds** total including preparation, fitting,
choice freeze, optional inference and independent audit. One BLAS/OpenMP
thread, sampled <=10-GiB RSS plus wait4 peak; <=8-MiB private output. This
is observer evidence, not strict cgroup RAM acceptance. All kernel/features
stay in RAM; private states contain only one-dimensional numeric arrays.
No learned state or dataset is published or deployed.

Application/test code remains unchanged; verify the passed full 4,646-test
software gate's 570 input bindings and original Ruff /270-file mypy /97.42%
coverage. Generated checks verify histogram distance, seeded positive median,
in-place bandwidth changes and exported ridge/SVC score parity. Independent
closing replay checks all 26 calibration records, order, test counts/AUC/
bootstrap, every bandwidth and sampled direct-distance/compensated-sum state
predictions. Preserve every failure, log and old attempt without relabelling.
