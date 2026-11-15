# Remaining 80% floor studies and source corrections (2026-11-15)

The unknown-payload gray LSB detector now exceeds 80% in all eight tested
conditions. OpenStego .1 bpp reaches **87.75%**. The remaining low-payload
JPEG and WAV conditions do not all pass.
Payloads and original data are unchanged. These are private CPU research
models; the application detector is unchanged and no model is deployed.

The [numeric receipt](../benchmarks/remaining-floor-20261115.json) retains
selected models, calibration records, every measured test cell, independent
replays, source bindings and resource closeouts. Historical results remain
in the [detection report](DETECTION_EXPANSION_20261111.md). Every further use
of these historical evaluation sets is adaptive, not untouched qualification.
Balanced accuracy (BA) averages specificity and sensitivity; intervals below
are exploratory paired-source bootstraps, not certification of an 80% floor.

A [later cause investigation](LOW_SIGNAL_DIAGNOSIS_RESULTS_20261115.md)
adds measured quantizer/source diagnostics and fresh coefficient/modular
features. nsF5 .1 bpnzac improves to 77.70%, still below 80%; seventeen
other new conditions stop at their predefined calibration gate. Earlier
metrics below remain unchanged historical records, not overwritten results.

## Source-role corrections

A new preflight detects one real BOSSbase split error: training row 3,044 and
test row 503 share a pixel hash, and independently cached cover-feature rows
are equal. The original loader shuffled names without grouping pixel hashes.
No new fit or scoring occurred in the failed preflight. Keep the original
split, data, scores and failure; the following corrected evaluations exclude
the duplicate test pair, leaving 999 sources /1,998 decisions. Earlier blanket
claims that all 1,000 original test sources were pixel-disjoint are superseded.

Saved-score replay with unchanged states and thresholds gives:

| Method at 0.4 bpp | Original 1,000-source BA | Clean 999-source BA | Clean correct decisions | Clean AUC |
|---|---:|---:|---:|---:|
| HILL | 80.05% | **80.030030%** | 1,599/1,998 | .890383877 |
| S-UNIWARD | 80.75% | **80.730731%** | 1,613/1,998 | .897463029 |
| MiPOD | 80.70% | **80.680681%** | 1,612/1,998 | .900253607 |
| WOW v2 | 80.30% | **80.280280–80.380380% bounds** | 1,604–1,606/1,998 | unavailable |

WOW's old per-source score vectors were not retained, so its exact clean BA,
AUC and confusion counts are unavailable. The interval shown is an aggregate
count bound, **not a confidence interval**: deleting two decisions from the
old 1,606 correct decisions leaves between 1,604 and 1,606. Both bounds pass.
No consumed WOW fit/inference is rerun to manufacture missing evidence.
The JPEG study uses only the first 3,000 training sources, excluding row
3,044, and its 1,000-source test remains pixel-disjoint from that cohort.
See the [BOSS audit](../benchmarks/boss-pixel-role-audit-20261115.json).

WAV preflight separately finds four original ESC-50 recording IDs crossing
filename folds. These are different clips, not exact waveform duplicates.
The already-admitted `esc-original-group-max-fold-v1` metadata assigns each
whole recording group to its maximum fold. New roles therefore contain
1,200 fit /398 calibration /402 test clips across 919/295/310 recording
groups. Legacy 400-test metrics remain separate and are not directly paired
improvements. A first source correction mistakenly used the constant dataset
label `source_group='ESC-50'`; a second uses the actual recording `group_key`
and agrees with `upstream_recording_id`. Both pre-fit failures are preserved.
See the [ESC audit](../benchmarks/esc50-role-audit-20261115.json).

## Unknown-payload gray LSB: all eight cells pass

Reuse the existing mixed, replacement-5% and matching-5% numeric Fisher states;
no refitting. Standardize each score with its calibration-cover mean/std,
then select one combination and threshold on 500 calibration sources to
maximize minimum BA across all eight kind/rate conditions. The selected
combination is their median, threshold **.812125764584136**. A query's true
class, payload and paired cover never choose a state or threshold.

| Payload, fraction of pixels | Replacement BA | Matching BA | Replacement AUC | Matching AUC |
|---|---:|---:|---:|---:|
| 5% | **85.485485%** | **86.036036%** | .945464 | .946236 |
| 10% | **88.488488%** | **88.038038%** | .978261 | .977965 |
| 20% | **88.538539%** | **88.588589%** | .987215 | .986285 |
| 40% | **88.838839%** | **88.738739%** | .992509 | .990345 |

All cells use the clean 999 sources and the same **22.022022% false-positive
rate**. At 5%, replacement/matching recall is 92.992993 /94.094094%.
The lowest BA is 85.485485%, but stronger-payload accuracy regresses versus
the historical mixed detector's roughly 94% maximum. This tradeoff remains
visible; neither the false alarms nor the regressions are removed.

All 56 calibration-cell records, eight test metrics, exact threshold choice
and eighteen direct numeric predictions replay independently. No scientific
model is refitted or automatically installed. See the [protocol](FLOOR_MIXED_LSB_PROTOCOL_20261115.md)
and [source-role amendment](FLOOR_MIXED_LSB_ROLE_AMENDMENT_20261115.md).

## Fixed low-payload image kernels: negative results retained

Use unchanged cached features and source-separated fit/calibration roles.
OpenStego keeps all four crops of each DIV2K training parent together:
2,560 fit /640 calibration crop pairs across 640/160 parents. Its 200 test
crop pairs represent **100 validation parents**, not 200 independent sources.
The nsF5 studies use 2,400 fit /600 calibration pairs and 1,000 test sources.

Calibration chooses among square-root or standardized square-root kernels,
three fit-cover-median width factors and ridge .01/.1 or SVC C=100. nsF5 also
compares plain 8,000-dimensional DCTR with signed calibrated 16,000-dimensional
features. All 72 candidates converge/pass numeric guards, then choices freeze
globally before selected inference. There is no test-based replacement.

| Condition | BA | Correct decisions | AUC | TPR /FPR | Exploratory source-group BA 95% interval |
|---|---:|---:|---:|---:|---:|
| OpenStego .1 bpp | 79.75% **miss** | 319/400 | .870475 | 82.0 /22.5% | 75.75–83.75% |
| nsF5 .05 bpnzac | 59.35% **miss** | 1,187/2,000 | .633729 | 52.7 /34.0% | 58.20–60.60% |
| nsF5 .1 bpnzac | 70.75% **miss** | 1,415/2,000 | .784427 | 64.9 /23.4% | 69.34875–72.30% |

OpenStego selects standardized square-root /factor .25 /ridge .01; nsF5
selects plain-DCTR square-root /factor .25 /ridge .01 at both payloads.
The new nsF5 rows are worse than historical **60.70 /75.15%**; those old
rows remain, without a post-test switch presented as prospective success.
Independent replay verifies all calibration metrics and 210 direct-distance
predictions, maximum error 6.888e-10. See the [protocol](FLOOR_IMAGE_PROTOCOL_20261115.md).

## OpenStego neighbor/parity augmentation

A distinct intervention computes 114 parity, neighbor-prediction and
weighted-stego statistics from each existing 512x512 RGB PNG independently,
with bounded decoding and no password, key or paired-cover lookup. Fit and
calibration keep the same 640/160 parents; the test remains 200 crop pairs
from 100 separate validation parents. No payload or file is modified.

All ten fixed WS classifiers pass their numeric guards: three Fisher ridges,
four histogram trees, three Gaussian ridges. Calibration selects the
15-leaf /l2=1 tree (BA 88.75%), then a fixed **75% WS /25% SRM** mixture,
with scalar standardization from calibration covers only (BA 90.703125%).
The threshold is frozen at **1.061469973523839** before test features.

One selected historical inference reaches **87.75% BA**, **351/400** correct,
AUC **.954325**, TPR **88.0%** and FPR **12.5%** (TN 175, TP 176).
The source-parent BA 95% interval is **84.5–91.0%**. Against the unchanged
kernel reference's 79.75%, the paired gain is **+8.00 percentage points**,
with exploratory 95% interval **+3.75 to +12.00 points**. This closes the
reported .1-bpp point-estimate miss on this reused historical cohort;
it does not establish new cross-source qualification or deploy a detector.

The first data-only feature phase hit its 600-second watchdog before any
actual model fit. Preserve that failed source/freeze/log, not a retried model.
A separately source-bound completion uses two ordered independent-file
workers with unchanged descriptor bytes and a summed descendant-RSS guard.
It charges 650 old seconds without resetting the original 1,500-second cap.
See the [feature/model protocol](FLOOR_OPENSTEGO_WS_PROTOCOL_20261115.md) and
[data-only execution amendment](FLOOR_OPENSTEGO_WS_EXECUTION_AMENDMENT_20261115.md).

All ten model and five system calibration records, selected test metrics,
mixture identities and parent bootstraps replay independently. Forty-two
sampled predictions agree within 1.464e-11; a separate scalar implementation
reproduces all 114 descriptors on two actual calibration images exactly.
The feature phase's summed sampled RSS peak is 496,332,800 bytes; its wait4
peak is per-process metadata, not a simultaneous aggregate peak. All sources
still match the 570-file full-gate binding.

## WAV: the failing conditions still miss

All work uses the existing nine-condition, 5,433-feature PCM16 caches, without
raw audio decoding. Keep every previously passing dedicated row in the old
report; sixteen formerly failing dedicated cells are compared here. AudioMNIST
uses 300 fit /100 calibration /200 test clips from 15/5/10 speakers. FSDD
uses 1,500/500/1,000 clips from three/one/two speakers, with lucas reserved for
calibration and theo/yweweler for this historical test. No reserved AudioMNIST
speaker or native Commons/ALASKA2 source is accessed.

Each dedicated cell compares nine signed-square-root standardized Gaussian
kernel candidates and one fixed 300-iteration numeric-export histogram tree.
Two full-5,433-feature mixed trees hit their phase watchdogs, ESC-50 at
680.654 seconds and AudioMNIST at 240.549 seconds. They are **unavailable**,
not evaluated models. Completed dedicated states are reused byte-identically,
with their original debit, never refitted. A separate source amendment compares
new 256-feature trees for all three mixed detectors and the eight previously
unattempted FSDD dedicated tree candidates. These are distinct alternatives;
the unavailable full-feature references retain their status.

Feature selection uses only fit covers/stegos: rank the squared difference
in means divided by summed within-class variances plus 1e-12; use stable
feature-index ties and retain 256 columns. Mixed selection pools all eight
stego conditions equally. Calibration selects dedicated BA/AUC/order and a
single mixed threshold maximizing minimum BA across all conditions. State,
column indices and threshold freeze globally before inference.

Dedicated selected results, every failing payload shown:

| Dataset | Kind | 5% BA | 10% BA | 20% BA | 40% BA |
|---|---|---:|---:|---:|---:|
| ESC-50, corrected 402 clips | replacement | 73.1343% **miss** | 78.3582% **miss** | historical pass, not rerun | historical pass, not rerun |
| ESC-50, corrected 402 clips | matching | 65.2985% **miss** | 65.1741% **miss** | 66.0448% **miss** | 69.4030% **miss** |
| AudioMNIST | matching | 56.50% **miss** | 59.50% **miss** | historical pass, not rerun | historical pass, not rerun |
| FSDD | replacement | 50.00% **miss** | 49.70% **miss** | 49.95% **miss** | 51.10% **miss** |
| FSDD | matching | 50.35% **miss** | 50.00% **miss** | 49.45% **miss** | 50.85% **miss** |

The new mixed detector's BA per condition:

| Dataset | Kind | 5% BA | 10% BA | 20% BA | 40% BA | Shared FPR |
|---|---|---:|---:|---:|---:|---:|
| ESC-50 | replacement | 65.1741% | 66.7910% | 66.7910% | 67.7861% | 32.3383% |
| ESC-50 | matching | 65.1741% | 67.0398% | 66.7910% | 67.7861% | 32.3383% |
| AudioMNIST | replacement | 71.00% | **85.50%** | **86.00%** | **86.25%** | 27.50% |
| AudioMNIST | matching | 56.50% | 65.00% | 79.50% | **85.25%** | 27.50% |
| FSDD | replacement | 49.85% | 50.00% | 49.90% | 49.70% | 99.50% |
| FSDD | matching | 50.00% | 49.80% | 49.80% | 49.85% | 99.50% |

Every nonbold mixed cell misses 80%. In particular, AudioMNIST matching at
20% is **79.50%, not an 80% pass**. The extreme FSDD mixed false-alarm rate
is reported explicitly. FSDD's near-chance AUC (.495–.517 dedicated) and the
other low-payload results provide no evidence that these cached features and
fixed model families meet the target. This does not prove an impossibility
bound for better features or different data. The two-speaker FSDD bootstrap
is especially limited; degenerate intervals do not establish generalization.

Independent closing verifies **166 calibration records, 40 measured test
cells and 726 direct numeric predictions**, maximum error 1.475e-10. All
source hashes and model/tree exports verify. Full confusion counts, AUCs,
thresholds, selected candidates and intervals are in the numeric receipt.
The sequence is bound by the [WAV protocol](FLOOR_WAV_PROTOCOL_20261115.md),
[source-role amendment](FLOOR_WAV_ROLE_AMENDMENT_20261115.md),
[group-key correction](FLOOR_WAV_GROUP_KEY_CORRECTION_20261115.md),
[partial-state continuation](FLOOR_WAV_PARTIAL_CONTINUATION_20261115.md) and
[feature-selection continuation](FLOOR_WAV_FEATURE_SELECTION_CONTINUATION_20261115.md).

## Resource and software closeout

| Study | Cumulative parent debit | Declared cap | Parent-close output |
|---|---:|---:|---:|
| Mixed gray LSB | 13.392694 s, including 10 s old debit | 120 s | 455,465 bytes |
| Image kernels | 317.921871 s | 1,800 s | 4,666,651 bytes |
| WAV, all source/partial continuations | 1,170.574979 s, including 980 s prior debit | 2,400 s | 18,379,294 combined logical bytes |
| OpenStego WS, both attempts | 1,058.041839 s, including 650 s prior debit | 1,500 s | 6,655,173 combined logical bytes |

WAV's combined logical-byte count deliberately counts retained hard links in
each evidence directory. Its final successful continuation alone is 190.574979
seconds and 10,058,670 bytes. The failed preparations and timed-out mixed
states remain; no old allowance was reset. New files are small numeric
features/states and receipts, with no raw-media copies or Gram matrices.
Source-role audit time is retained separately; unused closed allowances grant
no historical retry or new data acquisition. All numerical states stay private.

Application/test sources remain identical to all **570 bindings** of the
passed full CPU gate: **4,646 tests**, two explicit CUDA skips, **97.415805%**
statement coverage, Ruff and mypy over 270 files including `research`.
This is reuse of the actual 2026-11-14 gate, not a fresh full-suite claim.
Fresh Ruff and 270-file mypy checks pass; all fifteen README/scope checks
pass with disk-backed temporary storage. Publication is recorded in git.
No new GPU work, media/model download, source-role rewrite of native media,
ALASKA2 access or product deployment occurs.
