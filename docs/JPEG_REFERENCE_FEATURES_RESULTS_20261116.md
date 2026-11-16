# Reference DCTR/JRM features for nsF5 (2026-11-16)

nsF5 at an unchanged **.1 bpnzac** reaches **80.85%** balanced accuracy on
the reused 1,000-pair historical BOSS test. The model is a calibration-selected
51-learner Fisher-ensemble on corrected DCTR plus full JRM. This is a point
estimate only 17 decisions above the 80% boundary: its paired-source 95%
interval is **79.35–82.40%**. The test was viewed by earlier studies, so this
is adaptive development evidence, not untouched qualification or deployment.
nsF5 .05 stays below its calibration gate and has no new test inference.

The [numeric receipt](../benchmarks/jpeg-reference-features-20261116.json)
contains every candidate, both choice freezes, the selected test cell, source
and QA bindings, failures and resource closeouts. Definitions are bound by the
[protocol](JPEG_REFERENCE_FEATURES_PROTOCOL_20261116.md), its
[completion amendment](JPEG_REFERENCE_FEATURES_COMPLETION_20261116.md) and
[path correction](JPEG_REFERENCE_FEATURES_PATH_CORRECTION_20261116.md).

## Reference parity correction

The [read-only parity audit](REFERENCE_PARITY_PROTOCOL_20261116.md) compares
the project's DCTR with installed sealwatch 2024.12. On two existing fit
covers, 213 /209 of 8,000 bins differ (maximum .137 /.139). Changing only
nearest-even rounding to reference half-away rounding removes every difference
above 1e-6 (maximum 2.972e-8); dequantized IDCT already agrees within 2.6e-13.
`research/dctr_features.py` now defaults to reference rounding and keeps an
explicit `rounding="legacy"` path for reproducing historical caches and states.
Old caches are not rewritten or mixed with corrected features.

The fresh full CPU gate for that change passes **4,651 tests** /two explicit
CUDA skips /32 warnings, **97.416884%** statement coverage (27,983/28,725).
Ruff, research-inclusive mypy and `git diff --check` pass; all 570 bound inputs
match. Parity alone is not claimed as the cause of the accuracy change below.

## Models and calibration

Same 2,400 fit /600 calibration sources from the existing 3,000 Q75 training
JPEGs, the original 1,000 pixel-disjoint test sources, unchanged payloads and
embedding seeds. Features are corrected DCTR (8,000) plus all 11,255 JRM
features (19,255 joint), replacing the earlier 2,512-dimensional subset.
Eight fixed candidates per payload; calibration BA chooses, then AUC, then
declared order. All sixteen candidates are eligible.

| Candidate | .05 cal BA | .05 AUC | .1 cal BA | .1 AUC |
|---|---:|---:|---:|---:|
| DCTR Gaussian ridge .01 | 62.50% | .657886 | 75.17% | .822414 |
| DCTR Gaussian SVC C=100 | 58.25% | .605461 | 72.67% | .792417 |
| JRM Gaussian ridge .01 | 63.33% | .675158 | 77.08% | .849306 |
| JRM Gaussian SVC C=100 | 63.83% | .671108 | 77.17% | .847300 |
| Joint Gaussian ridge .01 | **65.75%** | .695869 | 80.33% | .880839 |
| Joint Gaussian SVC C=100 | 63.92% | .685111 | 79.67% | .876900 |
| Reference FLD ensemble, d_sub 1,000 | 64.92% | .709599 | **82.50%** | .904847 |
| Reference FLD ensemble, d_sub 2,000 | 65.33% | .708821 | 80.42% | .883079 |

Both choices were frozen globally before any new test feature was built.
**.05** selects joint ridge at 65.75%: below the predefined .80 gate, so it is
`not_evaluated_calibration_gate`. Its earlier measured misses stay unchanged.
**.1** selects the d_sub-1,000 ensemble (OOB error .1875) at 82.50%, threshold
7/51 of the sign-vote confidence. Confidences are odd multiples of 1/51, so
the frozen threshold has no tie ambiguity.

## Selected historical test (.1 bpnzac)

| Measure | Previous fusion mixture | Reference-feature ensemble |
|---|---:|---:|
| Balanced accuracy | 77.85% | **80.85%** |
| Correct decisions | 1,557/2,000 | **1,617/2,000** |
| ROC-AUC | .864762 | .894063 |
| TPR | 70.0% | 76.0% |
| FPR | 14.3% | 14.3% |

The paired-source change is **+3.00 points**, exploratory 95% interval
**+1.50 to +4.50** (2,000 resamples, seed 2026111604). Features and classifier
change together, so this is not a single-factor attribution to parity or to
JRM. The 80.85% point estimate passes the stated floor; the interval's lower
end does not, and repeated historical use limits independent confirmation.
An independent replay of the retained score vectors reproduces BA, counts,
both intervals and the previous 77.85% baseline exactly.

## Failures, resources and closing

All earlier failures stay preserved: the original data-only parent was
explicitly stopped after 480 feature rows; an extra AST helper and a relative
path reader failed during preparation; the first completion failed on an
incorrect document filename. None fitted a classifier. The path-corrected
completion reuses the 480 verified rows by reference and extracts the rest.

| Stage | Wall time | Sampled summed RSS peak |
|---|---:|---:|
| Training/calibration features | 3,236.61 s | 2.05 GiB |
| .05 /.1 candidate fits | 86.89 /86.93 s | 3.11 /2.67 GiB |
| Test features (.1 only) | 776.51 s | 2.02 GiB |
| Independent audit | 17.60 s | 1.94 GiB |

Completion wall time is 4,219.138375 seconds; with 810 prior seconds the
cumulative debit is **5,029.138375 /7,200**. Combined output across all three
parents and the cost probe is 646,354,568 bytes, below 768 MiB, with no Gram
matrix or raw media persisted. Sampled RSS and per-process wait4 peaks are
observer records, not a cgroup memory guarantee.

The independent audit replays 16 calibration records, 118 direct numeric
predictions (maximum error 1.164e-10), one selected test cell, the .05 gate
exclusion, two actual calibration descriptors and all source/QA/cache
bindings. The parent is closed; unused allowance grants no retry or new grid.
No download, GPU, locked-data access, native-role change or deployment occurs.

nsF5 .05, the WAV misses and unavailable EBS remain, so **the overall
80%-per-method goal is still not met**.
