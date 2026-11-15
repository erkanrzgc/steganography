# Remaining image-floor comparison (2026-11-15)

The user explicitly requests work on the remaining sub-80% conditions.
This fresh CPU-only comparison covers OpenStego .1 bpp and nsF5 .05/.1
bpnzac, with unchanged cached payloads/media. No old parent or fit restarts,
new media/model download, raw decode, GPU, ALASKA2 access, native Commons
role change or deployment. Every historical/adaptive result remains visible.

## Source roles and candidates

OpenStego uses the existing four-crop cache: 800 DIV2K training parents,
2,560 fit crop pairs /640 calibration crop pairs, with index%1000 parent
groups kept together and calibration parents index%5==0. Its 200 test crops
belong to 100 separate validation parents. Manifest order is replayed exactly.
Source-parent roles and exact cover-feature duplicates are checked before
fitting; an unresolved cross-role collision fails the attempt.

nsF5 uses the original first 3,000 BOSS training sources, with row%5==0
as 600 calibration pairs and 2,400 fit pairs; source pixel-hash roles are
checked against the unchanged 1,000 test sources. The wider gray split's
known train/test duplicate is at training row 3,044, outside this cohort.
Each payload remains a separate cell; .2 bpnzac is not substituted for .05/.1.

For OpenStego compare square-root features without column scaling, then
square-root features with fit-only StandardScaler. For each nsF5 cell compare
plain first-8,000 DCTR square roots without scaling, signed square roots of
the complete 16,000 calibrated features without scaling, then the complete
signed square roots with fit-only StandardScaler. Input arithmetic is float32
with scaler statistics cast to that dtype, then float64 kernels.

For each representation choose 256 fixed fit-cover positions, seed
2026111511, and compute the median positive direct squared distance using
SciPy pdist; distances <1e-12 count as zero. Gamma factors .25/1/4 divided
by that median are fixed in order. Share Gram matrices in RAM with in-place
fourth powers for increasing factors. For each kernel fit ridge .01 then .1
(targets -1/+1, no fitted intercept, float64 Cholesky, equation residual
<=1e-6), then C=100 SVC (tol=.001, max_iter=1,000,000, cache 128 MiB,
no probability estimation). Exactly 18 OpenStego and 27 per-nsF5 candidates.
Convergence/score-parity failures are ineligible, with 1e-7 absolute/relative
score guards and no retries or guard relaxation.

Each cell selects by exact-count calibration BA, AUC, original order, with
closest-to-zero/lower-threshold ties. Freeze all three selections, states,
normalizers, bandwidths and source bindings before any test-row inference.
Combined OpenStego cache bytes may be hash-bound before the freeze, but test
rows never participate in fit/calibration. Score each selected cell once.
This is further adaptive historical evaluation, not untouched qualification.
Report all misses, TPR/FPR and source-group BA intervals (2,000 resamples,
seed 2026111512). Prior result rows remain separate; the previous detectors
used different fit/calibration policies and their saved scores are unavailable,
so no controlled paired gain interval is invented.

## Software and resources

One exclusive parent, new **1,800-second** cumulative cap for preparation,
three comparisons, global freeze, inference and independent closing audit.
One BLAS/OpenMP thread; sampled RSS <=10 GiB plus wait4 peaks, <=8 MiB
private output. This observer is not strict cgroup enforcement. All Gram
and raw-feature matrices stay in RAM. Numeric learned states remain private.

Verify all 570 bindings to the unchanged passed 4,646-test full software
gate (two CUDA skips, 97.42% coverage, Ruff /270-file mypy). Do not claim
another full run or regenerate large QA scratch. Generated controls check
signed square roots, dtype casting, stable bandwidths and ridge/SVC exports.
Independent closeout checks every calibration threshold/count/pairwise AUC,
selection, all test counts/AUC/intervals and sampled direct-distance numeric
states. Preserve each failure, source freeze, log and consumed marker.
