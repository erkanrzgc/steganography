# CPU sample-count continuation results (2026-11-14)

The user's storage-cleanup/continuation request first leads to
[27.859467 GiB of verified cleanup](STORAGE_CLEANUP_20261114.md), then the
fresh [sample-count protocol](RICH_MODEL_CPU_SAMPLE_COUNT_PROTOCOL_20261114.md).
All earlier parents, outcomes, allowances and failed evidence remain preserved.
No old run is restarted; no new media, GPU, source role or ALASKA2 access.

## Fixed data intervention and selection

Existing 9,000-source feature caches are reused unchanged. The former 5,999
fit and consumed 1,501 internal-audit pairs become 7,500 fit sources across
7,498 groups. The same 1,500 calibration sources /1,499 groups stay separate.
Pairs and duplicate cover-feature groups never cross the boundary. The former
audit is now training, not new held-out evidence.

Fit one fresh model each for HILL and MiPOD, preserving square-root features,
fit-only scaling, Gaussian C=100 and gamma=1/9,464. The prospectively declared
1,000,000-iteration cap is not reached: HILL converges at 81,653 iterations,
MiPOD at 78,409. Both numeric prediction-parity checks pass. Calibration BA
rises from 80.10 to 81.47% for HILL and 79.50 to 79.80% for MiPOD; both fresh
states win the fixed calibration BA/AUC comparison against their predecessors.
S-UNIWARD remains at its previous frozen C=300 state, with no new fit or test.

A global choice freeze precedes all historical test-feature access. Each newly
selected model receives exactly one inference on the same 1,000 BOSS pairs;
the predecessor predictions are reused from their saved files without rerunning
or refitting the older models. This is another **adaptive historical test**,
not untouched external qualification. No test-based threshold/model adjustment.

| Method, 0.4 bpp | Previous 5,999-fit BA | New 7,500-fit BA | AUC | TPR /FPR | Change | Exploratory 95% change interval |
|---|---:|---:|---:|---:|---:|---:|
| HILL | 79.50% | **80.05%** | .890597 | 83.6 /23.5% | +0.55 points | -0.45 to +1.55 points |
| MiPOD | 77.65% | **77.90% (miss)** | .876463 | 77.7 /21.9% | +0.25 points | -0.90 to +1.30 points |

HILL has 1,601/2,000 correct decisions: only one above the 80% boundary.
MiPOD has 1,558/2,000, still 42 short. S-UNIWARD's earlier **80.75%** remains
unchanged. Both gain intervals include zero: **the extra training sources do
not establish a statistically reliable gain on this reused test**. This is a
small measured point-estimate change, with no qualification or deployment.
The overall every-method goal remains unmet, including MiPOD at 0.4 bpp.

## Verification and footprint

The independent closing audit verifies both calibration choices, exact-count
thresholds, full confusion metrics, pairwise AUCs and 2,000-group bootstrap
intervals (seed 2026111411). Forty-four sampled predictions replay from numeric
states using direct squared differences and compensated dual sums. The largest
difference is 9.195e-11. Original feature-cache hashes, versions, groups and all
570 application/test source bindings match their retained records.

All matrices remain in RAM; no `.npy` kernel or duplicated feature cache is
written. Kernel wait4 peak RSS is about 4.44 GiB, below the 10-GiB observer
cap. Fresh fit processes take about 50 seconds each. Inference takes about nine
seconds each. Preparation, fitting, choice freeze, inference and both closing
audits conservatively charge **73.373759 /72.885059 seconds** to HILL /MiPOD,
below the separately declared new 1,200-second per-method caps. No old unused
time is reclaimed or reset. Persistent output remains below **two MiB**, well
under the eight-MiB cap; about 30 GiB remains free on the guest filesystem.

Application/test code is unchanged. The passed **4,646-test /two CUDA-skip**
full gate, Ruff /270-file mypy and 97.42% coverage are reused only after all
570 source hashes verify; no new full-suite run is claimed. Generated in-memory
kernel/scaler/dual checks precede fitting, and independent actual-state replay
closes it. Documentation checks run separately after bookkeeping.

Private source freezes, states, score vectors, wait4 observations and closing
receipts are in `.benchmark/rich-model-samplecount-20261114/`. Earlier results
are in the [initial CPU comparison](RICH_MODEL_CPU_IMPROVEMENT_RESULTS_20261114.md).
All new parents are closed. No further tuning, retry, test look, data download,
GPU work or model release follows automatically.
