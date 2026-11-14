# Calibration boundary extension (2026-11-14)

The first twelve-candidate comparison is closed and independently replayed.
All three calibration winners are Gaussian SVC at the grid's upper C=100,
gamma 1/9,464 boundary. A bounded additional calibration comparison checks
C=300 and C=1,000, each at the same two gammas .1/9,464 and 1/9,464, in that
order. No new feature, source split, normalizer, kernel or embedding changes.

**The earlier development audit has now been viewed and is consumed.** This
is a later adaptive development step, not another unbiased audit. Fit/calibrate
on the same 5,999 /1,500 source pairs; do not score the 1,501-source audit for
these new candidates. Do not open historical BOSS or ALASKA2 test features.
No further boundary expansion follows automatically, even if C=1,000 wins.

The independent replay found two unselected candidates whose optimal-count
thresholds did not obey the closest-to-zero tie rule because of float
roundoff. A generated three-pair fixture reproduces it. The public helper now
uses common-denominator integer TN/TP counts, including unequal class sizes.
The original selected models, reference thresholds, choices and audit metrics
are unchanged. Preserve the original snapshots, two failed auditor logs and
the successful independent replay. The auditor also now matches sklearn
1.9.1's explicit casting of scaler statistics to input dtype before arithmetic.

For all sixteen candidates, recompute calibration threshold/BA from saved
calibration scores using exact counts, without refitting original candidates.
Retain the original records and record each changed threshold explicitly.
Select by exact calibration BA, AUC, then the old twelve plus new four order.
Persist the final choice and numeric-state hash before any later evaluation.
New candidates retain the 1e-7 replay guard; save numeric evidence and mark
any nonconverged or unreplayable candidate ineligible, without retry/relaxation.

Use one new exclusive parent. Debit both previous nonlinear attempts from
its original 1,200-second per-method allowance: HILL 114.13438497599419,
S-UNIWARD 65.11538829900383, MiPOD 94.41648174700094 seconds. Keep at least
one second of closing reserve; no budget reset. The combined first nonlinear,
corrected completion and boundary directories stay under the original 64-MiB
output cap. Preserve all failures, states and cache/source bindings.

The first complete software gate passes 4,624 tests /two CUDA skips /97.40%
coverage. The exact-tie correction adds a regression: sixty focused tests
pass with 100% coverage of both public numeric modules. A separate complete
post-correction gate is running; publication or later test evaluation requires
that fresh gate to pass. No old-test evaluation is prepared by this protocol.
No download, GPU, raw-media decoding, native Commons or reserved access.
