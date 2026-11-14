# MiPOD CPU kernel comparison results (2026-11-15)

The fresh [fixed protocol](MIPOD_KERNEL_PROTOCOL_20261115.md) completes. On
**0.4 bpp MiPOD**, calibration selects kernel ridge with diagonal ridge .03.
The selected model scores **78.75% balanced accuracy** on the repeatedly
viewed historical BOSS test, versus 77.90% for its predecessor: **+0.85 points**.
It still misses 80%. The exploratory paired 95% gain interval is **-0.25 to
+1.90 points**, so a reliable gain is not established. No qualification or
deployment follows; no test-selected adjustment is made.

## Fixed data and all calibration candidates

The unchanged caches supply 7,500 fit source pairs /7,498 groups and 1,500
calibration pairs /1,499 groups. Fit-only scaling and paired/duplicate-group
separation are preserved. The former internal audit remains training data,
not fresh held-out evidence. No new media, raw decoding, GPU, ALASKA2 access,
native Commons role change or old attempt restart occurs.

All 34 fresh candidates are eligible: four ridge solutions meet the 1e-6
equation-residual bound; all thirty SVCs converge and pass the fixed 1e-7
prediction-parity guard. Selection compares them with the old state using
exact calibration BA, then AUC, then frozen order. The old calibration BA is
79.80%. No candidate is selected using historical test scores.

| Full square-root Gaussian ridge | Calibration BA |
|---|---:|
| 0.01 | 80.1333% |
| 0.03 | 80.2667% |
| 0.1 | 79.4667% |
| 0.3 | 78.2000% |

Additive-family SVC calibration BA follows. Plain-family weight is alpha;
weighted-family weight is 1-alpha, with gamma 1/4,732 per family. These are
calibration-selection scores, not evaluation accuracies.

| Power | Plain weight | C=30 | C=100 | C=300 |
|---:|---:|---:|---:|---:|
| 0.5 | 0 | 77.9000% | 77.9333% | 77.4000% |
| 0.5 | 0.25 | 79.4000% | 79.6000% | 79.0333% |
| 0.5 | 0.5 | 79.8000% | 79.4333% | 79.1667% |
| 0.5 | 0.75 | 79.2667% | 79.3000% | 79.0000% |
| 0.5 | 1 | 77.0000% | 77.3333% | 76.4000% |
| 0.25 | 0 | 77.9333% | 78.0667% | 77.2333% |
| 0.25 | 0.25 | 79.2333% | 79.3000% | 79.1667% |
| 0.25 | 0.5 | 79.5000% | 79.3667% | 79.1667% |
| 0.25 | 0.75 | 79.3000% | 79.4000% | 79.3000% |
| 0.25 | 1 | 76.9333% | 77.0000% | 76.0667% |

The winner uses the full concatenated square-root Gaussian kernel, gamma
1/9,464, targets -1/+1, no fitted intercept and ridge .03. Its calibration
BA is 80.2667%, AUC .894679111, and threshold -0.05769335119293828.
Global model/source choice and both numeric scaler bindings precede test
access. The same 1,000 historical test pairs receive one selected inference.
All prior test rows remain preserved.

## Historical test outcome

| MiPOD, 0.4 bpp | BA | AUC | TPR /FPR | Correct decisions |
|---|---:|---:|---:|---:|
| Previous 7,500-fit SVC | 77.90% | .876463 | 77.7 /21.9% | 1,558/2,000 |
| Selected ridge .03 | **78.75% (miss)** | .885946 | 84.6 /27.1% | 1,575/2,000 |

The gain is 17 decisions, with **25 still missing** to reach 1,600/2,000.
True positives rise from 777 to 846, while true negatives fall from 781 to
729: greater sensitivity comes with more false alarms. The 2,000-resample
paired-source-group interval uses seed 2026111501 and contains zero. This
additional adaptive historical look is not untouched external confirmation.
HILL stays 80.05%, S-UNIWARD 80.75% and WOW v2 80.30%, without reruns.
The every-method goal remains unmet, including MiPOD at 0.4 bpp and the
other payload-specific misses in the [overview](DETECTION_EXPANSION_20261111.md).

## Verification and footprint

Generated checks precede real fitting and independently verify scaler casting,
family Gaussian kernels, Cholesky solves and exported scores. Closing audits
replay all 35 calibration thresholds/counts/pairwise AUCs, selection order,
full historical test metrics and paired bootstrap. Ninety predictions replay
from all 34 numeric states and the selected historical test through direct
squared distances and compensated sums; maximum discrepancy is 2.154e-9.
Both scaler file hashes bind before global choice and verify again at close.

Parent execution and closing audit take **398.727003 seconds**;
the supplemental binding audit brings conservative debit to **400.727003 /1,200
seconds**. Kernel wait4 peak RSS is **6.54 GiB**, below the sampled 10-GiB
observer cap; this is not strict cgroup RAM acceptance. All Gram matrices
remain in RAM. Parent-close output is **4,225,621 bytes**, about four MiB,
below the eight-MiB cap. Only one-dimensional numeric states/score vectors
are persisted; no copied feature or kernel matrix. Models stay private.
About 30 GiB remains free on the guest.

Application/test inputs are unchanged: all **570 hashes** match the passed
**4,646-test /two CUDA-skip /97.42% coverage** full gate. Ruff /270-file mypy
are reused from that actual source-identical gate; no fresh full run is
claimed and no large QA fixture tree is regenerated. All fifteen documentation
checks pass separately after bookkeeping; README has 897 words.

Public numeric metadata is in `benchmarks/mipod-kernel-20261115.json`.
Private sources, freezes, all candidate states/calibration scores, observations
and closing receipts stay in `.benchmark/mipod-kernel-20261115/`. All workers
are closed. No further grid, retry, test look, download or GPU run follows
automatically. The [previous sample-count step](RICH_MODEL_CPU_SAMPLE_COUNT_RESULTS_20261114.md)
remains an unchanged historical record.
