# MiPOD diagnosis and normalization protocol (2026-11-15)

The user's new explicit instruction requests investigating the remaining
MiPOD miss and attempting at least 80%, while checking other methods honestly.
This authorizes a fresh CPU-only intervention on existing private caches;
it does not reopen old attempts, permit downloads/GPU work, unlock ALASKA2
or change native Commons roles. The 0.4-bpp condition stays fixed.

## Diagnosis and rationale

Before fitting, inspect existing fit/calibration features and saved calibration
scores for constant/rare bins, varying cover content, changed-pixel rates and
error concentration. Use the 7,500 fit /1,500 calibration source pairs and
existing group separation. The old internal audit remains training data.
No test features or saved test-score distribution is used for diagnosis or
candidate selection; earlier published test metrics remain historical context.
Record associations as associations, not causal proof.

The [MiPOD paper](https://ws2.binghamton.edu/fridrich/Research/MiPOD.pdf),
Table I, reports maxSRMd2 error .2038 at 0.4 bpp for its medium-block version:
79.62% complementary accuracy under that paper's different protocol. It is
not a guaranteed ceiling or a directly comparable score. Our extractor is a
compact SRM-style subset with 4,732 plain plus 4,732 weighted values, rather
than a reproduction of the full published model. No result implies a software
bug merely because MiPOD trails WOW or S-UNIWARD.

[Nonlinear Feature Normalization in Steganalysis](https://dde.binghamton.edu/mehdi/pdf/nonlinear-feature-normalization.pdf)
motivates random conditioning and cover-fit empirical CDF transforms. This
experiment adapts those ideas to our compact caches; it does not claim an
exact reproduction of the paper's classifier or selective uniformization.

## Fixed transforms and models

Keep the same 7,500 fit pairs /7,498 groups and 1,500 calibration pairs /1,499
groups. In this order, freeze six representations before fitting:

1. `cdf`: empirical CDF on each raw feature using fit covers only.
2. `cdf-normal`: the same CDF, clipped to [1/(2n), 1-1/(2n)] for n fit covers,
   followed by the inverse standard-normal CDF.
3. `rc8-sqrt`: a single global permutation (seed 2026111502), disjoint groups
   of eight feature indices, per-row group-sum conditioning, then square root.
4. `rc4-sqrt`: the same permutation with groups of four, then square root.
5. `rc8-cdf`: group-eight conditioning followed by cover-fit empirical CDF.
6. `rc4-cdf`: group-four conditioning followed by cover-fit empirical CDF.

Every group divides 9,464 exactly. An all-zero group stays zero. The empirical
CDF is count(reference <= x)/n, with deterministic ties, no interpolation or
jitter. References always come from fit covers after the same conditioning.
No CDF reference is fitted on calibration/test or written as a large cache;
it can be rebuilt in RAM from bound fit indices and original cache hashes.

Each representation uses StandardScaler fitted on both fit classes only,
float32 transform arithmetic with mean/scale cast to the input dtype, then
float64 Gaussian kernels with gamma 1/9,464. Fit four fixed classifiers each,
in order: kernel ridge .01/.03/.1 (targets -1/+1, no fitted intercept,
Cholesky, maximum equation residual 1e-6), then C=100 SVC (tol=.001,
max_iter=1,000,000, cache 128 MiB, no probability estimation). Thus there
are exactly 24 fresh models. Record nonconvergence/score-parity failures as
ineligible without retry. Numeric parity uses absolute/relative 1e-7.

The old ridge .03 state comes first in selection order. Compare all eligible
models by exact-count calibration BA, then AUC, then original order. Freeze
all choices, normalizer/state hashes and sources before opening historical
test features. If the old state wins, reuse 78.75% without a new inference.
Otherwise perform exactly one selected historical-test inference; save all
misses and TPR/FPR without any test-based adjustment. This remains further
adaptive historical evidence, never untouched qualification. Report the 80%
point-estimate target honestly; it cannot be guaranteed beforehand. Paired
source-group 95% gain interval uses 2,000 draws, seed 2026111503.

## Software and bounded execution

Application/test code stays unchanged. Verify all 570 bindings to the passed
4,646-test /two-CUDA-skip gate, preserving 97.42% coverage and its actual Ruff /
270-file mypy. Reuse the actual source-identical full gate without a new large
QA tree. Test generated CDF ties/extremes, exact conditioning, scaler casting,
Gaussian kernels, Cholesky/exported SVC scores before any actual fit. Close
with independent count/pairwise-AUC/selection/bootstrap and direct-distance
numeric-state replay, including all calibration candidates.

Use one exclusive parent/attempt with a fresh cumulative **1,500-second** cap
including diagnosis, preparation, comparison, optional inference and closing
reserve. One BLAS/OpenMP thread, sampled RSS <=10 GiB plus wait4 peak evidence,
and <=8 MiB private output. This observer is not strict cgroup RAM acceptance.
All Gram/CDF-reference matrices stay in RAM; models and scores stay private.
Preserve every failure; no automatic restart, guard relaxation or next grid.
No source/media acquisition, raw decoding, local GPU or deployment follows.
