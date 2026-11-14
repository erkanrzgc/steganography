# Fresh CPU sample-count continuation (2026-11-14)

After the user explicitly requests storage cleanup and continuation, the
verified cleanup frees 27.859467 GiB. This is a separate fresh CPU study for
HILL and MiPOD at **0.4 bpp**, not a restart of any consumed parent. All older
states, budgets, test results and failures remain historical. No local GPU,
media/model download, raw-image decode, native Commons change or ALASKA2 access.

## Fixed intervention

Reuse the existing 9,000 historical BOSS training-source feature caches and
original exact-cover-feature groups. Keep all 1,500 calibration source pairs
(1,499 groups) unchanged. Combine the former 5,999 fit and consumed 1,501
internal-audit pairs into **7,500 fit sources /7,498 groups**. Cover/stego
pairs and duplicate groups never cross fit/calibration boundaries.
The former audit is now training data; it supplies no new audit accuracy.

Fit exactly one fresh model for each missing method. Use the previous fixed
recipe: signed-square-root 9,464-dimensional features, fit-only StandardScaler,
Gaussian SVC C=100, gamma=1/9,464, tol=.001, no probability estimation. Set
max_iter=1,000,000 prospectively for this larger fit; do not raise it or retry
if the new fit fails. Freeze scikit-learn 1.9.1, NumPy 2.4.6 and actual source/
cache/partition hashes before fitting. S-UNIWARD's passing recipe is unchanged.

Compare old 5,999-fit and fresh 7,500-fit states only on the same calibration
scores, using exact-count threshold calibration. Choose by calibration BA,
then AUC, then prefer the old model. Preserve both states and all scores.
Nonconvergence or >1e-7 absolute/relative numeric prediction disagreement
makes the fresh candidate ineligible. No model selection uses a test score.

## Evaluation and safeguards

Before any test-feature access, freeze both calibration choices and model/
threshold hashes globally. Only a newly selected fresh model receives one
historical BOSS 1,000-pair test inference. If the old model wins, reuse its
already-saved score record without another test look. Compare a fresh score
with the predecessor's saved predictions, not a refit or rerun. Report BA,
AUC, TPR/FPR, exact 80% pass/miss and a 2,000-source-group paired bootstrap
(seed 2026111411). This is a further adaptive development/test step; it is not
untouched external qualification. No additional grid or test tuning follows.

Use a fresh exclusive parent and per-method attempt markers, with one BLAS/
OpenMP thread. Each method has a fresh operator-selected maximum of 1,200
seconds including fitting, optional scoring and closing reserve. This is a
new intervention's cap, not reclaimed time from an old allowance. The parent
samples RSS against 10 GiB; this is an observer guard, not a cgroup resident-
RAM/zero-swap guarantee. Preserve kernel wait4 evidence and every failure.
No automatic retry or guard relaxation.

**Disk output is capped at 8 MiB.** Kernel matrices live in RAM only: one
in-place training Gram matrix and one calibration matrix, with batched final
inference. No `.npy` kernel, feature duplicate or full QA fixture tree is
persisted. New states use numeric support indices and immutable cache refs.
Independent generated kernel/scaler/dual checks precede fitting; independent
score arithmetic and selected numeric-state replay close any actual result.

Application/test sources have not changed: verify all 570 inputs from the
passed 4,646-test full gate and retain its logs/coverage. No new full gate is
claimed from that reuse. Prospective sources, cleanup receipts and software
checks are frozen separately. Scientific results alone authorize no model
release, deployment, download or GPU work.
