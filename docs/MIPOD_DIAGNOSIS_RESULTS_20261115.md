# MiPOD diagnosis and CPU follow-up results (2026-11-15)

MiPOD reaches **80.70% balanced accuracy at 0.4 bpp**, 1,614/2,000 correct,
on the reused historical BOSS test. The four academic spatial point estimates
now pass 80%; other payload-specific misses remain. Three bounded CPU
studies use existing private caches, with unchanged 0.4-bpp payload and
7,500 fit /1,500 calibration pairs. The old internal audit remains training.
Every model/threshold choice is calibration-only and frozen before its
optional selected test inference. The historical BOSS test has been viewed
repeatedly: all following numbers are adaptive research evidence, with no
untouched cross-source qualification, deployed model or guaranteed minimum.

## What the diagnosis establishes

The old ridge errors strongly associate with residual activity. Quartile
boundaries come from fit covers only; calibration scores use the original
frozen global threshold. This is an association, not a causal decomposition.

| Fit-defined activity quartile | Calibration sources | BA | TPR /FPR |
|---|---:|---:|---:|
| 1 | 378 | 96.16% | 98.15 /5.82% |
| 2 | 376 | 85.90% | 90.16 /18.35% |
| 3 | 396 | 74.75% | 78.79 /29.29% |
| 4 | 350 | 63.29% | 72.57 /46.00% |

The highest-activity quarter is much harder. Mean changed-pixel fractions
are 8.3427% in fit and 8.3608% in calibration, offering no evidence of a
large payload mismatch between those roles. The compact representation has
9,464 values: 4,732 plain /4,732 weighted, with 272 constant fit-cover bins.
This is an SRM-style subset, not a reproduction of full published maxSRMd2.
That is a plausible information limitation, not proof that dimensionality
alone causes the miss. Raising payload would change the task and is not done.

The installed conseal 2025.11 source uses 9x9 /degree-nine variance estimation,
7x7 Fisher smoothing and the corrected wet-boundary implementation. The
[MiPOD paper](https://ws2.binghamton.edu/fridrich/Research/MiPOD.pdf), Table I,
reports .2038 error for unsmoothed medium blocks at 0.4 bpp; Table II reports
.2243 after smoothing (complementary accuracy 77.57%). The prospective
normalization protocol cited Table I as context before the installed-source
audit established this variant distinction. Different protocols prevent
direct score comparison; neither figure is a ceiling or a guarantee.

## Three distinct hypothesis tests

The [normalization protocol](MIPOD_NORMALIZATION_PROTOCOL_20261115.md) tests
six representations and four classifiers each. Its fixed random conditioning
and cover-fit empirical CDF ideas come from
[Nonlinear Feature Normalization in Steganalysis](https://dde.binghamton.edu/mehdi/pdf/nonlinear-feature-normalization.pdf).
Our compact-feature adaptation and classifiers differ from that paper.
All 24 models are eligible. Calibration selects RC8 plus square root, Gaussian
gamma 1/9,464, ridge .03: BA 80.5667%. One historical inference is 79.70%.

The [content protocol](MIPOD_CONTENT_PROTOCOL_20261115.md) tests two/four
local experts and fixed .5/1 local-global weights. Training pairs stay
grouped; every query gate uses that file alone. Eighteen local ridge solves
produce twelve systems. Calibration chooses two experts /ridge .1 /.5 mix,
BA 80.60%: only one extra correct calibration decision. Historical BA falls
from 79.70 to 78.40%, with a paired 95% change interval of -2.05 to
-0.54875 points. **The content-expert hypothesis is not validated.** We do
not replace its outcome with an unselected system or hide the negative row.

The [geometry protocol](MIPOD_HELLINGER_PROTOCOL_20261115.md) tests
unwhitened square-root and RC8 square-root Gaussian distances. Bandwidths
come only from 256 fixed fit-cover vectors, with three fixed factors and
four classifiers each. Both prior references remain eligible. This changes
the histogram geometry/bandwidth together; it is not a clean attribution
of any gain solely to removing column scaling.

The selected geometry recipe is `rc8-sqrt-factor1-ridge0.01`, with calibration BA
81.8667% and AUC 0.907768889.
Its gamma is .006107876271462043, ridge .01 and calibration threshold
-.010646304581314325; no individual-column standardization is applied.

## Historical test outcomes, 0.4 bpp

| Experiment | BA | AUC | TPR /FPR | Correct decisions |
|---|---:|---:|---:|---:|
| Previous global ridge | 78.75% | 0.885946 | 84.6 /27.1% | 1575/2,000 |
| RC8 normalization | 79.70% | 0.887769 | 81.4 /22.0% | 1594/2,000 |
| Content experts /global mix | 78.40% | 0.885336 | 81.2 /24.4% | 1568/2,000 |
| Selected Hellinger geometry | **80.70%** | 0.900443 | 82.5 /21.1% | 1614/2,000 |

The geometry model passes the 80% point-estimate target
with 1614/2,000 correct. Its change versus the global RC8 reference is
+1.00 points, exploratory paired 95% interval -0.15000 to +2.15000
points (2,000 source-group draws, seed 2026111508). Repeated model selection
and test access limit that interval; it is not untouched qualification.
Every original row, including the negative content-expert outcome, remains.

## Other methods and the 80% claim

HILL 80.05%, S-UNIWARD 80.75% and WOW v2 80.30% remain their previous
0.4-bpp historical point estimates, without reruns. HILL passes by one
decision. **Not every method/payload is above 80%**: nsF5 at .05/.1 bpnzac
is 60.70 /75.15%; OpenStego at .1 bpp is 79.8%; unknown-payload LSB at 5%
is 72.1%; FSDD WAV misses at every reported payload. EBS is unavailable.
See the [complete overview](DETECTION_EXPANSION_20261111.md) for all
payload-specific cells. Higher-payload successes do not repair lower-payload
misses, and software test coverage is not detector accuracy.

## Verification, preserved failure and resource use

The first geometry preparation fails generated-only positive-distance median
QA before real fitting or a source freeze. Dot-product cancellation makes
an identical pair spuriously positive. Its source/log/exit and diagnosis
are retained. The separately bound
[numeric amendment](MIPOD_HELLINGER_NUMERIC_AMENDMENT_20261115.md) uses
direct-difference SciPy pdist for the small bandwidth sample. No recipe or
guard changes. A conservative ten seconds debit the old preparation and
diagnosis; the original 900-second allowance is not reset. No real fit is
repeated and no failed command is called passing.

| Completed study | Cumulative seconds /cap | Peak wait4 RSS | Parent-close output |
|---|---:|---:|---:|
| normalization | 842.469969 /1500 | 5.982 GiB | 5,420,617 bytes |
| content | 99.080078 /600 | 2.645 GiB | 3,220,932 bytes |
| hellinger | 495.513698 /900 | 5.976 GiB | 4,391,195 bytes |

All studies close with generated controls, source/data hash replay, exact
calibration thresholds/confusion counts, independent pairwise AUCs, selection
order, paired bootstrap and direct-distance compensated numeric predictions.
Their independent prediction checks total 198; all pass the fixed guard.
Activity gates and all content-system calibration scores replay as well.
Each output stays below eight MiB; feature/Gram/CDF-reference matrices stay
in RAM. Models remain private. About 30 GiB of guest space stays available.
No new media, raw decode, local GPU, native role change or ALASKA2 access.

Application/test sources are unchanged: all 570 inputs match the passed
4,646-test /two-CUDA-skip /97.42%-coverage full gate, with original Ruff and
270-file mypy. No new full run or large QA scratch tree is claimed.
All fifteen documentation checks pass separately; README has 899 words.

Public numeric evidence is `benchmarks/mipod-diagnosis-20261115.json`;
learned states, sources, score vectors and observations stay in the private
`mipod-normalization-20261115`, `mipod-content-20261115`, failed
`mipod-hellinger-20261115` and corrected `mipod-hellinger-completion-20261115`
directories under `.benchmark/`. All parents are closed. No old attempt,
allowance, score or result permits an automatic retry, source acquisition,
GPU intervention or deployment.
