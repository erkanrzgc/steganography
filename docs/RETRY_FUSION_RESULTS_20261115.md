# Cross-representation retry and larger-source JPEG fit (2026-11-15)

The user's new retry request produced two distinct closed CPU studies. Neither
achieves the 80% floor: nsF5 .1 bpnzac reaches **77.85%** on the reused test,
only +.15 percentage points over 77.70%; a separate 7,496-source fit reaches
75.333333% on calibration and is not tested. WAV score fusion does not solve
the existing misses. Payloads, previous results and original data are retained.

The [numeric receipt](../benchmarks/retry-fusion-20261115.json) contains all
choices, metrics, gate exclusions, source/control failure, model hashes,
independent replays and resource closeouts. [Earlier cause measurements](LOW_SIGNAL_DIAGNOSIS_RESULTS_20261115.md)
remain unchanged. These are private research models, with no deployment,
download, GPU, native-role change or locked ALASKA2 access.

## New JPEG models trained across both representations

The [fusion protocol](RETRY_FUSION_PROTOCOL_20261115.md) joins cached square-root
DCTR and coefficient/histogram features on the same 2,400 fit /600 calibration
sources. Each unchanged .05/.1 payload compares fifteen fresh models: twelve
sum/product Gaussian kernel ridge/SVC classifiers and three 51-learner paired-
bootstrap Fisher ensembles (subspaces 500/1,000/2,000). No old candidate is
refitted. This is a compact adaptation, not full published JRM or its exact
ensemble routine. Fixed score mixtures with the frozen predecessor are also
selected only on calibration, before a global JPEG/WAV choice freeze.

At .05, the selected new model calibrates at **66.25%**, AUC .702419. It fails
the predefined .80 gate, so no new .05 test feature or inference is made.
The earlier measured test misses stay unchanged.

At .1, calibration selects a 75% new /25% predecessor score mixture:
BA **81.583333%**, AUC .889186, threshold **1.0578880601691112**. The new
component is product-kernel ridge .01 with coefficient weight .25. One frozen
historical inference gives:

| Measure | Previous coefficient/DCTR mixture | Fresh joint-kernel mixture |
|---|---:|---:|
| Balanced accuracy | 77.70% | **77.85%, miss** |
| Correct decisions | 1,554/2,000 | 1,557/2,000 |
| ROC-AUC | .853915 | .864762 |
| TPR | 74.5% | 70.0% |
| FPR | 19.1% | 14.3% |

Three additional correct decisions leave **43 short** of the 80% boundary.
The paired-source BA 95% interval is **76.30–79.35125%**. The +.15-point
paired change interval is **-1.05125 to +1.45 points**, containing zero;
there is no established reliable gain. Fewer false alarms trade against
lower recall. No unselected model or old test-based fallback replaces it.
Repeated historical evaluation remains adaptive, not untouched qualification.

## WAV: frozen score combinations

For each of sixteen formerly failing dedicated conditions, seven new fixed
systems compare old/new raw scores, three standardized mixtures and their
max/min. The component classifiers are reused byte-identically, without
refitting. Calibration sources/order match; the previously documented FSDD
old/new bit-depth training cohorts remain distinct.

| Dataset /condition | Selected calibration system | Calibration BA | New historical test |
|---|---|---:|---|
| ESC replacement 5% | .50 new mixture | 75.376884% | gated |
| ESC replacement 10% | old reference | 80.904523% | old 78.358209% retained; no new inference |
| ESC matching 5/10% | .25 new mixtures | 65.703518 /67.713568% | gated |
| ESC matching 20/40% | old references | 71.105528 /71.984925% | gated |
| AudioMNIST matching 5% | old reference | 61.00% | gated |
| AudioMNIST matching 10% | .75 new mixture | 72.00% | gated |
| FSDD, all eight conditions | fixed mixtures/min/old | 52.0–54.3% | gated |

All **fifteen new WAV exclusions** remain `not_evaluated_calibration_gate`,
not passed. The one retained ESC reference is still an old measured test
miss. **No new WAV test features or scores** are constructed; the raw-worker
coverage branch is unexercised, with no general executor qualification claim.
Every payload's exact calibration record and selected system is in the receipt.

Fusion closing independently verifies **152 calibration records**, one new
selected test cell and **114 direct numeric predictions** (maximum error
4.762e-10), with all 570 software hashes unchanged.

## A separate larger-source nsF5 .1 fit

The [sample-count protocol](NSF5_SAMPLE_COUNT_PROTOCOL_20261115.md) tests a
separate hypothesis: the old 3,000-source JPEG cohort may be too small.
A bounded inventory of the existing 10,000 BOSS originals uses known hashes
and decodes only 3,003 previously excluded non-test images for missing pixel
hashes. No current test pixels are decoded and no media is downloaded.
Removing test pixel groups and duplicate groups leaves **8,996 unique**
fit/calibration sources. Preserve the old 3,000 order and 600 calibration
roles; fixed additional selection gives **7,496 fit /1,500 calibration**.
All original manifests/roles remain; this is a new experimental overlay.

Reuse the old 3,000 cached .1 feature pairs and derive 5,996 added-source pairs
from bounded transient Q75 encoding. Exact old DCT and two feature vectors
replay before fitting. All pixel and exact cover-feature hash intersections
across fit/calibration/test are zero. Added JPEGs are not persisted as a corpus.

One fixed square-root coefficient-only Gaussian ridge .01 model uses a
fit-cover-median width. There is no new grid, DCTR extraction or payload
increase. Training and calibration cohorts change together, so this is not
a single-factor causal attribution against the previous fusion model.

Calibration is **75.333333%**, AUC **.831252444**, TPR **79.133333%**,
FPR **28.466667%**, 2,260/3,000 correct. It fails the prospectively declared
.80 gate: **no new test inference**, not a test pass or a test accuracy.
The historical 77.85% test result stays as context only.

A read-only split of the retained calibration scores, at the same unchanged
threshold -.0420271442453668, gives:

| Calibration subset | Sources | BA | AUC |
|---|---:|---:|---:|
| Preserved old subset | 600 | 76.833333% | .848905556 |
| Added subset | 900 | 74.333333% | .819381481 |

The added subset scores lower for this fixed model, while the old subset
also does not improve. Source composition and repeated model development on
the old 600 cannot be separated here; this is not an estimate of intrinsic
source difficulty. These are fixed-threshold diagnostics, not new model
selection or proof of an irreducible accuracy ceiling. The larger-source
recipe did not solve the target.

Independent closing replays the one calibration record, landmark median,
22 direct calibration predictions (max error 1.196e-10), eight sampled
training equations (max residual 3.743e-11), and two actual added-source
feature vectors. The test gate is independently verified.

## Failure, resources and software verification

The first fusion generated control mistakenly used float32 kernels while
the real fit path already used float64. Generated residual 3.268e-6 fails
the unchanged 1e-6 guard; float64 diagnostic residual is <=8.483e-14.
The original parent was incorrectly launched after failed preflight and
then exits on its missing success receipt after 1.503 seconds, before any
actual model. Preserve both failures. A [separate numeric amendment](RETRY_FUSION_NUMERIC_AMENDMENT_20261115.md)
corrects only the generated dtype and enforces passing preflight before launch,
with ten old seconds debited inside the original 1,800-second allowance.
No actual candidate is retried and no tolerance is relaxed.

| Closed study | Successful parent wall time | Conservative cumulative debit /cap | Parent-close output |
|---|---:|---:|---:|
| Fusion, including the preserved first parent | 166.483623 s | 176.483623 /1,800 s | 7,258,804 bytes combined |
| Larger-source fit | 235.421302 s | 265.421302 /1,800 s | 66,902,300 bytes |

Both parents are closed; unused consumed allowances do not grant a retry.
About 70.73 MiB of combined logical evidence is retained. Fusion stays below
six-GiB summed sampled RSS; the larger fit's per-process wait4 peak is about
4.647 GiB, below its ten-GiB observer cap. These are observer/kernel records,
not a cgroup resident-memory guarantee. No Gram/raw image copy is persisted,
and no dataset/model/failure evidence is deleted.

All 570 application/test inputs remain byte-identical to the actual passed
4,646-test /two-explicit-CUDA-skip /97.415805% full CPU gate. Fresh Ruff,
research-inclusive mypy over 270 files and all fifteen disk-backed README/scope
checks pass; README has 898 words and four badges. This is
source-identical full-gate reuse, not a new full-suite claim. The overall
80%-per-method target remains unmet; nothing here qualifies or deploys a model.
