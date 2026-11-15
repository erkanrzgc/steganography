# Mixed gray LSB floor continuation (2026-11-15)

The user explicitly requests improving the remaining sub-80% conditions.
This fresh bounded CPU study changes only calibration/combination of three
existing numeric classifiers; no old fitting job is repeated. Payloads and
simulated samples remain unchanged. No new data, raw decode, GPU, ALASKA2,
native Commons role change or deployment occurs.

Use the original BOSS 5,497-source training models and separate 500-source
validation cache as calibration. The historical 1,000-source test remains
outside fitting/calibration. Existing mixed, replacement-5% and matching-5%
Fisher models are source-bound and never refitted. Source pairs stay grouped.

In fixed order compare: (1) mixed score at its old zero threshold;
(2) mixed score with a newly calibrated threshold; (3) mean, (4) maximum,
(5) minimum of the two 5%-model standardized scores; (6) median, (7) mean
of all three standardized scores. Scalar means/stds use calibration covers
only; std is floored at 1e-12. Every file receives the same fixed combination;
no query class, payload, paired cover or test label selects a component.

For candidates 2–7 choose one threshold maximizing the minimum exact-count
balanced accuracy over all eight kind/rate cells, ties closest to zero then
lower threshold. Select candidates by minimum BA, then mean BA, mean AUC,
then original order. This directly targets the requested floor. All eight
cells, including regressions, must be reported; no averaging hides a miss.

Freeze sources, scalar anchors, choice and threshold before one historical
inference over all eight unchanged conditions. These are further adaptive
historical tests, not untouched qualification. Independent closing replay
checks thresholds/counts, pairwise AUCs, selection, and direct numeric model
scores. Preserve all previous outcomes without test-based adjustment.

One exclusive parent, fresh 120-second cumulative cap, one BLAS/OpenMP thread,
sampled RSS <=2 GiB and <=1 MiB private output. No feature matrix is persisted.
Verify the unchanged 570-source closure of the passed 4,646-test gate and
run generated threshold/combination controls before calibration. The old
actual full gate is reused, not relabelled as a new run. Keep every failure
and marker; no automatic retry or old allowance reuse follows.
