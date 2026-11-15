# MiPOD content-expert protocol (2026-11-15)

This distinct intervention implements the content-heterogeneity hypothesis
identified from fit/calibration diagnosis under the user's current instruction
to investigate and improve MiPOD. The normalization study remains closed;
its 79.70% historical result is adaptive context, not selection data or a
source of usable leftover allowance. No old fit is relaunched or extended.

## Data, gates and fixed candidates

Keep the same 7,500 fit /1,500 calibration pairs and duplicate-group separation,
fixed 0.4 bpp and existing caches. The global reference is the independently
closed `rc8-sqrt-ridge0.03` state. Its calibration BA is .8056666666666666.
No test score distribution is inspected to design gates or thresholds.

For every analyzed file independently, residual activity is one minus the
mean plain-histogram all-zero-pattern frequency across the 28 blocks. Fit
cover activity alone defines equal-quantile partitions of size two and four.
Train pairs share their cover's gate; both members remain together. Query
cover/stego gates each use that query's own features, with searchsorted
side=right. There is no access to a query stego's cover, class label or pair.

Reuse the fixed eight-bin random conditioning and square-root representation
from the normalization study. Each expert fits its own StandardScaler on
both training classes in its bucket only, then float64 Gaussian kernel gamma
1/9,464. Train three ridge values .01/.03/.1 per expert, targets -1/+1,
no fitted intercept, Cholesky and maximum equation residual 1e-6. All experts
within a system use the same ridge, avoiding a separate per-bucket search.
The two partitions contain six experts and eighteen ridge solves in total.

For partitions 2 then 4, and ridge .01/.03/.1, compare two fixed system
outputs in order: .5 local + .5 global score, then local score alone. Thus
there are twelve fresh candidates plus the global reference first. Direct
score mixing uses the common -1/+1 target convention, with no test-fitted
rescaling. Calibrate one global exact-count BA threshold for each system;
select by calibration BA, AUC, then original order. Any failed expert makes
its whole system ineligible; no retries or numeric guard relaxation.

Freeze all gate, scaler, expert-state, global-reference and selection hashes
before test access. If the reference wins, retain its existing 79.70% without
another inference. Otherwise perform one inference of the chosen system on
the same historical 1,000 pairs. A mixed system reuses saved global-reference
scores rather than rerunning that model. All tests remain further adaptive
historical evidence. Keep every miss, TPR/FPR and paired-group 95% gain
interval (2,000 resamples, seed 2026111505); no test-based adjustment.

## Execution and software closure

Use a fresh exclusive parent with **600 seconds** cumulative preparation,
fitting, choice freeze, optional inference and independent closing audit.
One BLAS/OpenMP thread; sampled RSS <=10 GiB plus wait4 peaks; private output
<=8 MiB. This is observer evidence rather than strict cgroup RAM enforcement.
All feature/Gram matrices remain in RAM; only numeric expert/scaler states,
score vectors and metadata persist. No media/model download, raw decode,
local GPU, ALASKA2 access, native Commons role change or deployment occurs.

Application/test code is unchanged. Verify the passed 4,646-test gate's 570
source bindings, with original Ruff /270-file mypy and 97.42% coverage.
Generated checks cover per-file gates, paired membership, conditioning,
local standardizers and ridge score export. Independent closeout replays
all thirteen calibration records, choice, saved test counts/pairwise AUC and
bootstrap, and sampled direct-distance/compensated-sum expert predictions.
Preserve every failure and all prior source/state/ledger evidence. No old
allowance or test score licenses another automatic retry or grid extension.
