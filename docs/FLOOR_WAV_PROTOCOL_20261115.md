# Remaining WAV-floor comparison (2026-11-15)

The user explicitly requests work on remaining sub-80% conditions. This new
CPU intervention uses existing v2 WAV feature caches and fixed 5/10/20/40%
replacement/matching samples. No old worker/fit is repeated, no new source,
raw decode, GPU, native role change, reserved speaker or ALASKA2 access.
Previously passing dedicated cells are retained without rerunning them.

## Source-separated roles

Replay exact cached row order from the original v2 dataset loader using file
names/manifest metadata only. ESC-50 folds 1–3 fit, fold 4 calibrates, fold 5
tests; original recording IDs must be disjoint. AudioMNIST uses the original
20 training speakers, with the last five sorted speaker IDs for calibration
and the other fifteen for fitting; the existing ten development speakers are
the adaptive test, never the closed reserved speakers. FSDD uses george,
jackson and nicolas for fitting, lucas for calibration, theo/yweweler for
testing. Nicolas's known upscaled 8-bit source stays fit-only; no role is
chosen from new test scores. Every cover/stego pair shares its source role.

Dedicated cells are ESC-50 replacement .05/.1 and matching at all four rates;
AudioMNIST matching .05/.1; FSDD both types at all four rates. Sixteen cells
in total, preserving the other dedicated outcomes. All three datasets also
get a separate mixed classifier with the same eight fixed conditions.

## Fixed candidate families and selection

For each dedicated cell fit signed square-root features with a fit-only
StandardScaler, float32 arithmetic/statistic casting followed by float64.
Gamma is factor .25/1/4 divided by the median positive direct squared distance
of up to 256 fixed fit covers (seed 2026111515, zero cutoff 1e-12). At each
width fit ridge .01 then .1 (targets -1/+1, no fitted intercept, float64
Cholesky, equation residual <=1e-6), then C=100 SVC (tol=.001,
max_iter=1,000,000, cache 128 MiB, no probability estimation): nine kernels.
The tenth candidate is HistGradientBoosting on untransformed features,
300 iterations /learning rate .05 /15 leaves /l2=1 /seed 0, explicitly
no early stopping and no categorical features. Each query uses its own
features; no paired clean query or condition label enters feature scoring.

Dedicated selection uses exact calibration BA, AUC, original order and one
calibration threshold per model. Nonconvergence or 1e-7 kernel-score parity
failures make a candidate ineligible without retry. Gradient boosting stores
only numeric tree nodes/offsets/baseline, no pickle or binning-cache copy;
manual raw-tree traversal must match predict_proba to 1e-10 absolute/relative.

Each mixed model uses that one fixed gradient-boosting recipe, with one copy
of fit covers weighted eight and all eight stego variants weighted one for
class balance. Compare its zero raw-margin threshold and a single threshold
maximizing the minimum exact-count BA over eight calibration cells. Select
by minimum BA, mean BA, mean AUC, original order. Report every test cell,
including regressions; no average hides a miss. Ties use closest-to-zero,
then lower threshold. All models/thresholds remain method/payload-specific
research except explicitly labeled mixed systems.

Freeze all dataset choices and numeric states before test inference. Score
each selected dedicated/mixed model once on its unchanged historical roles.
All results are further adaptive evaluations, not untouched qualification.
Report TPR/FPR and paired source-group BA intervals using 2,000 resamples,
seed 2026111516. ESC recording groups and AudioMNIST/FSDD speaker groups are
the bootstrap units; FSDD has only two test speakers, limiting the interval.
No old detector scores were retained, so no controlled paired gain is claimed.

## Bounds and software verification

One exclusive parent, fresh **2,400-second** total cap for preparation,
three dataset comparisons, global freeze, inference and independent closeout.
One BLAS/OpenMP thread; sampled <=10-GiB RSS plus wait4 peaks and <=24 MiB
private output. Observer evidence is not strict cgroup RAM enforcement.
No raw feature/Gram/reference matrices persist and all learned states stay
private. Stop on a resource breach; preserve every marker/source/log/failure.

Application/test sources stay unchanged: verify all 570 bindings to the passed
4,646-test full gate, two explicit CUDA skips, 97.42% coverage and original
Ruff /270-file mypy. No fresh full-suite claim or large QA scratch tree.
Generated controls verify roles, stable kernel median/exports, minimax
thresholds and numeric gradient-tree export/traversal. Independent closing
replay verifies all calibration/test counts, pairwise AUCs, choices, source
roles, intervals, selected scores and sampled numeric candidate predictions.
No deployment, automatic old retry or resource-budget reset follows.
