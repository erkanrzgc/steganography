# Cached rich-model CPU comparison (2026-11-14)

This fresh classifier study follows the user's CPU rich-model continuation,
using the existing 0.4-bpp HILL, S-UNIWARD and MiPOD SRM + weighted caches.
It does not rerun the old queue, change a payload or claim that the higher
0.6-bpp results solve the old misses.

The prospective [linear/ensemble protocol](RICH_MODEL_CPU_IMPROVEMENT_PROTOCOL_20261114.md)
and [nonlinear extension](RICH_MODEL_CPU_NONLINEAR_PROTOCOL_20261114.md) fix
both candidate grids and selection before the union audit is opened.
The nonlinear extension was frozen before the first original-grid audit was
completed; its parent waits for the original parent to close. No old BOSS
test-feature cache, ALASKA2 holdout, native Commons reserved data, new media,
GPU or media/model download is used.

## Data and selection

The historical 9,000 training originals form 8,997 exact plain-cover-feature
groups. Identical cached cover vectors stay together, as do their cover/stego
pairs. The deterministic split contains:

| Role | Source pairs | Source groups |
|---|---:|---:|
| Fit | 5,999 | 5,998 |
| Calibration | 1,500 | 1,499 |
| Audit | 1,501 | 1,500 |

The final candidate is selected by calibration balanced accuracy, then AUC,
then fixed grid order. Calibration also sets its operating threshold. The raw
Fisher reference has the same threshold calibration. No audit row contributes
to fit statistics, ridge/C/gamma choice or threshold selection.

**This audit is held out for these newly fitted models, within historical
training data used by older experiments. It is not untouched external
qualification.** It is a different cohort from the old 1,000-image BOSS test;
its scores must not replace those historical rows or imply a direct gain
against their percentages. No model is deployed.

## Audit results

This first audit applies to the twelve-candidate union, whose three winners
are square-root Gaussian SVC, gamma 1/9,464 and C=100. The later boundary
choice is a separate development step and is not rescored on this audit.

| Method, 0.4 bpp | Raw Fisher BA | Selected BA | AUC | Gain (points) | 95% paired interval (points) | TPR /FPR |
|---|---:|---:|---:|---:|---:|---:|
| HILL | 75.52% | 79.28% (miss) | 0.884 | +3.76 | +2.43 to +5.06 | 82.01 /23.45% |
| S-UNIWARD | 79.08% | 82.38% (passes) | 0.910 | +3.30 | +2.16 to +4.49 | 83.88 /19.12% |
| MiPOD | 74.32% | 78.38% (miss) | 0.879 | +4.06 | +2.77 to +5.33 | 85.34 /28.58% |

The improvement rule requires at least one percentage point of paired audit
BA gain and a positive lower bound of a 2,000-resample source-group bootstrap
95% interval. Intervals are per method, without adjustment for the three
methods or two classifier families. Report misses at 0.4 bpp; never round a
sub-80% result into a pass.

## Calibration comparison

The first six candidates are raw, square-root and fourth-root standardized
Fisher (ridge .03), 31 paired bootstrap/random-subspace Fisher learners
(2,000 features, ridge .03), and standardized square-root LinearSVC
(C .01/.1). The second six are standardized square-root Gaussian SVC
(gamma .1/9,464 or 1/9,464, each with C 1/10/100).

All normalizers use fit rows only. Candidate selection is over the union,
using calibration scores only; the original six-candidate selection is
preserved separately. Nonconverged candidates are excluded, with their
warnings and partial states preserved.

The later [boundary protocol](RICH_MODEL_CPU_BOUNDARY_PROTOCOL_20261114.md)
adds four fixed C=300/1,000 candidates after the first development audit
is consumed. Only calibration scores are used; original candidates are
recalibrated with exact-count ties without refitting. Eight of the twelve
new fits do not converge at the fixed 100,000-iteration limit and remain
ineligible. Their states, warnings and scores are retained. No retry or
iteration-limit increase follows. D=9,464. Values below are calibration BA.

| Candidate | HILL | S-UNIWARD | MiPOD |
|---|---:|---:|---:|
| Raw Fisher | 76.63% | 79.13% | 75.73% |
| Square-root Fisher | 76.77% | 79.63% | 76.37% |
| Fourth-root Fisher | 77.37% | 79.80% | 76.27% |
| 31-learner ensemble | 75.93% | 77.63% | 75.80% |
| Linear SVC C=.01 | 77.07% | 81.23% | 76.97% |
| Linear SVC C=.1 | 76.43% | 80.67% | 76.30% |
| RBF gamma=.1/D C=1 | 66.13% | 65.80% | 66.60% |
| RBF gamma=.1/D C=10 | 73.30% | 74.93% | 73.00% |
| RBF gamma=.1/D C=100 | 78.87% | 80.43% | 78.23% |
| RBF gamma=1/D C=1 | 70.43% | 72.30% | 70.33% |
| RBF gamma=1/D C=10 | 78.23% | 79.43% | 76.97% |
| RBF gamma=1/D C=100 | 80.10% | 81.73% | 79.50% |
| RBF gamma=.1/D C=300 | 79.43% | 81.40% | 79.03% |
| RBF gamma=.1/D C=1000 | 79.57% † | 81.90% † | 79.63% † |
| RBF gamma=1/D C=300 | 80.00% † | 82.17% | 78.80% † |
| RBF gamma=1/D C=1000 | 79.70% † | 81.70% † | 78.60% † |

† Ineligible: nonconvergence. Every new candidate passes the unchanged
1e-7 numeric prediction-parity guard. Higher C does not automatically
improve a detector. HILL and MiPOD keep C=100; S-UNIWARD selects C=300,
all at gamma=1/D. These final choices are frozen from calibration only.

## Historical test evaluation

The [final evaluation protocol](RICH_MODEL_CPU_FINAL_EVALUATION_PROTOCOL_20261114.md)
is written after the complete post-correction software gate passes. All
three states, sources and saved calibration thresholds are globally frozen
before opening the unchanged 1,000-pair BOSS test feature caches. This is
**a further adaptive look at the historical test**, not untouched external
qualification. There are zero fitting/optimizer updates and no test-based
threshold or model changes. Final models use 5,999 fit /1,500 calibration
sources; v3 used 9,000 fit sources and an OOF threshold, so the old v3
percentages are historical comparisons rather than a controlled ablation.

| Method, 0.4 bpp | Common-fit raw BA | Final RBF BA | AUC | TPR /FPR | Gain (points) | Exploratory 95% gain interval |
|---|---:|---:|---:|---:|---:|---:|
| HILL | 76.00% | 79.50% (miss) | 0.888 | 83.3 /24.3% | +3.50 | +2.00 to +5.15 points |
| S-UNIWARD | 78.95% | 80.75% (passes) | 0.898 | 85.7 /24.2% | +1.80 | +0.30 to +3.25 points |
| MiPOD | 74.05% | 77.65% (miss) | 0.874 | 83.0 /27.7% | +3.60 | +2.00 to +5.15 points |

**Remaining misses at 0.4 bpp:** HILL has 1,590/2,000 correct decisions,
ten short of 80%; MiPOD has 1,553/2,000, 47 short. S-UNIWARD has
1,615/2,000 and passes at 80.75%. No threshold is adjusted to turn a miss
into a pass. The overall every-method goal remains unmet.

The independent closing audit verifies all confusion counts, pairwise
AUCs, paired group intervals and another 66 numeric-state predictions.
The largest prediction discrepancy is 2.438e-10. Preparation, both earlier
nonlinear stages, this inference and closing verification conservatively
charge 194.567984 /142.542859 /170.848608 seconds to HILL /S-UNIWARD /
MiPOD, each below its unchanged 1,200-second allowance. All cache/source/
state/QA bindings verify, with no remaining parent or model worker.

## Software and evidence

`research/power_fisher.py` adds explicit signed power maps and a Fisher grid
sharing transformed statistics; threshold calibration stays separate.
`research/fld_ensemble.py` scores bounded row batches, avoiding the former
N x learners x subspace temporary while preserving the stored model schema.
The SVM fits are private source-bound experimental recipes using already
installed scikit-learn 1.9.1; no sklearn dependency is added to the base wheel.
Saved RBF states use immutable training-cache references, numeric support
indices, dual coefficients and fit-only scaler statistics, without pickle
or duplicate support-vector feature caches. They are not standalone releases.

The initial fifty-nine focused tests pass with 100% coverage of the two public
numeric modules, including independent closed-form Fisher scores and scalar
ensemble votes. An exact-tie regression brings the corrected focused run to
sixty passes, still with 100% coverage. Separate generated kernel fixtures verify precomputed/direct
RBF and manual support/dual decision parity before real-data fitting.

The first nonlinear attempt exits 1 on a strict replay assertion. The
[separate numeric amendment](RICH_MODEL_CPU_NUMERIC_AMENDMENT_20261114.md)
preserves it, reuses its completed C=1 state byte-for-byte and charges the old
48.043474 seconds to HILL's original allowance. Generated controls reproduce
dual-sum roundoff; fresh fits record their actual errors under the revised
1e-7 guard. The unavailable first real candidate's exact error was not saved.

Independent replay also finds two optimal-count thresholds with incorrect
float tie-breaking: HILL's gamma=.1/D C=10 and MiPOD's linear C=.01. Both are
unselected in the twelve-candidate union. The original six-candidate MiPOD
choice used that threshold; retain its observed audit as historical evidence,
without claiming exact tie-rule compliance. The twelve-candidate winners,
reference thresholds and reported audit metrics remain unchanged. Integer
TN/TP comparisons now implement the declared rule exactly, including unequal
class sizes. The original helper/test snapshots remain private.

The first read-only auditor used a different scaler-rounding assumption;
sklearn 1.9.1 casts mean/scale to input dtype before arithmetic. Its corrected
independent calculation preserves that order and verifies 66 actual selected
predictions through direct squared differences and compensated dual sums,
plus all confusion metrics, pairwise AUCs, group bootstraps and union choices.
Both failed auditor logs are retained. A separate exact-count replay verifies
all sixteen calibration candidates and final selections.

A later complete gate exposes a pre-existing fork/coverage lock in the
optional external-tool adapter's Python preexec callback during the parallel
benchmark. The stuck child is deliberately aborted for its traceback, then
the parent is stopped; this attempt is unavailable, not passed. The adapter
now invokes a trusted isolated Python launcher, which applies resource limits
after startup and replaces itself with the resolved tool, without a Python
preexec callback. The existing memory/CPU/output/file-descriptor limits and
parent timeout remain, core dumps are disabled, and launch failures return
an error rather than a clean analysis. No frozen core source changes.
Twenty-seven focused tests pass; the new launcher has 100% coverage. Twelve
real parallel launches verify limits, literal argv and exit/overflow behavior.
The earlier 42-pass focused command failed its combined 95% coverage assertion
on the pre-existing adapter; its log is retained separately.

The fresh final complete gate passes **4,646 tests /two explicit CUDA skips /
32 warnings**, **97.42%** total statement coverage (27,971/28,713), in
1,674.81 seconds; wrapper time is 1,680.578804 seconds. Ruff and 270-file mypy
pass, including research. All 570 bound Python/notebook/TOML inputs match at
close. Power Fisher, ensemble and the new launcher each have 100% coverage;
the existing external adapter has 98.35%. The unchanged total gate remains
90%, and the new analysis services exceed their 95% target.

The initial pre-tie-fix full command separately passes 4,624 tests /two CUDA
skips /97.40% coverage. The interrupted post-tie-fix command is retained as
unavailable after 118 completed tests and 797.375897 seconds, not a pass.
All historical source snapshots, logs and failed checks remain. Fresh
TMPDIR/basetemp are on disk with passed-test retention cleanup; only a
rebuildable npm download cache is purged for space. No checkpoint, dataset,
failed log, installed package or npm log is deleted. A model-free wheel build
also verifies the new launcher/adapter bytes and excludes private data/states.

All fifteen final README/link/scope checks pass after document bookkeeping;
README has 896 words. No application/test source changes after the complete
gate. Publication is recorded separately in its private receipt.


Private records remain under `.benchmark/`:

- `rich-model-software-qa-20261114/`: focused log, coverage and source contract;
- `rich-model-improvement-20261114/`: source/cache freeze, split, all original
  candidates, numeric states, calibration/audit scores, selection and exits;
- `rich-model-nonlinear-20261114/`: prospective source freeze, generated
  kernel QA, all nonlinear candidates, union selections, scores and exits;
- `rich-model-nonlinear-completion-20261114/`: corrected completion, retained
  auditor failures and independent union audit;
- `rich-model-boundary-20261114/`: exact-count calibration replay and final choices;
- `rich-model-final-evaluation-20261114/`: global recipe/test-input freezes,
  one inference per method, unchanged-input closeout and independent replay;
- `rich-model-full-qa-20261114/`: initial complete gate before the tie regression;
- `rich-model-full-qa-tie-fix-20261114/`: interrupted preexec deadlock and traceback;
- `rich-model-launcher-qa-20261114/`: launcher-focused checks and retained first failure;
- `rich-model-full-qa-final-20261114/`: fresh final complete gate and input hashes.

The original linear comparison has its own exclusive parent and cap. The
nonlinear/completion/boundary/final stages share cumulative 20-minute
per-method debits and one combined 64-MiB output cap. Preserve every attempt and failed/unavailable
result. No further automatic retry, tuning, test look, data acquisition or GPU
work follows from this completed comparison.
