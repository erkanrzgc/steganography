# Fresh cross-representation fusion intervention (2026-11-15)

The user's explicit retry request permits this distinct CPU intervention on
existing admitted caches. It never relaunches a consumed parent, refits an old
candidate, changes target payloads, downloads data/models, uses a GPU or deploys
a research model. Preserve every previous score, failure and consumed allowance.
The hypothesis is complementary information between representation families;
this is a new predefined family, not a test-selected replacement of old rows.

## JPEG: joint training instead of independent late scores

Use the unchanged first-3,000 source roles: 2,400 fit /600 calibration pairs,
1,000 pixel-disjoint historical test sources, .05/.1 bpnzac. Join existing
8,000 DCTR and 3,016 coefficient-subset histogram features after independent
float32 square-root maps. No new training raw decode or feature extraction.
The existing [JRM work](https://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf)
motivates diversity/ensemble use; this adaptation is not full JRM or a claim
of reproducing its algorithm or performance.

For each payload, fifteen new fixed candidates: three paired-bootstrap Fisher
ensembles from unchanged `research/fld_ensemble.py` (51 learners, ridge .1,
subspace 500/1,000/2,000, seed 2026111534), plus twelve composite kernels.
Each representation's Gaussian width uses the positive direct squared-distance
median of 256 fixed fit covers (seed 2026111533, zero <1e-12 excluded).
Use DCTR gamma .25/median, coefficient gamma 1/median. At coefficient weights
.25/.5/.75 compare weighted sums of both kernels and products of their powered
kernels, each with ridge .01 or SVC C=100 /tol=.001 /max_iter=1,000,000.
The ensemble uses its existing tanh/spread-normalized score, not author votes.
Cholesky residual <=1e-6; SVC requires convergence and exported-score parity
absolute/relative 1e-7. No Gram matrix is persisted.

Choose the new model by calibration BA/AUC/order. Compare its raw score,
the previous frozen coefficient/DCTR system, and fixed .25/.5/.75 new/reference
mixtures normalized by calibration-cover mean/std. Calibrate thresholds with
exact-count closest-to-zero/lower ties. The old system is a candidate, not a
automatic best-test fallback. Freeze all JPEG/WAV choices before any optional
selected test inference. If the reference wins, retain its old result with no
new inference. Other new conditions require selected calibration BA >=.80;
otherwise `not_evaluated_calibration_gate`, never a pass.

Reuse the verified existing .1 test coefficient cache. Only if a new .05
choice passes its calibration gate may unchanged existing JPEG covers be
read (regular <=2 MiB, 64x64 blocks, same seeds/payload), through the frozen
coefficient worker. Save at most 100 sources per hashed numeric chunk;
verify changed counts against the old cache. This is feature coverage for
the new selected system, not a retry/relabeling of an old gated experiment.
Keep the inherited eight-MiB file guard and all raw bytes unchanged.

## WAV: frozen models with new calibration-only score combinations

For each of sixteen formerly failing dedicated conditions, use the selected
old feature model and selected modular/re-embedding model, both source-bound
and already fitted. Calibration source/index order must match exactly. FSDD
old/new fits use different bit-depth cohorts, already declared; no new role
or model refit. New choice candidates: old raw score, new raw score, three
standardized mixtures with new weights .25/.5/.75, and the elementwise max
and min of the two standardized scores. No query class, payload label or
paired clean source chooses a candidate. Threshold/choice BA/AUC/order is
calibration-only; no new classifier fitting occurs in this WAV branch.

A selected calibration BA below .80 excludes new test inference. If the old
model wins, reuse its saved test result. Otherwise use only that frozen
selected system. New modular test features may then be constructed with the
unchanged bounded worker (PCM16 <=500,000 frames, <=4 MiB, first channel,
original seeds/rates), four file workers and hashed <=100-source chunks.
Do not score unselected alternatives or use test outcomes to switch models.
Prior mixed/pass rows and every old miss remain separate unchanged records.

## Execution and independent closing

One exclusive fresh parent in `.benchmark/retry-fusion-20261115/`, 1,800 seconds
including five source-preparation seconds, summed sampled descendant RSS
<=6 GiB, output <=96 MiB, one BLAS/OpenMP thread per process. At most four
file workers, only for gated feature coverage; disk-backed TMPDIR, no copied
raw media. Every phase respects remaining wall time. Preserve every failure
without automatically retrying actual fitted candidates or resetting the cap.

Generated scalar controls check ensemble export, kernel sum/product formulas,
SVC/ridge numeric prediction and fixed score combinations before actual fitting.
Independent closing replays calibration metrics/thresholds/choices, state and
source hashes, sampled direct predictions, selected test counts/AUC and paired
source/group intervals (2,000 draws, seed 2026111535). JPEG gains are against
the previous frozen measured system; no unavailable legacy score interval is
invented. Historical test reuse is adaptive, not untouched qualification.
All 570 application/test inputs must match the actual passed 4,646-test gate;
private experiment output does not change product verdicts or install models.
