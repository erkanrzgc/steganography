# Low WAV/nsF5 scores: measured causes and fresh interventions (2026-11-15)

The earlier classifier-only studies did not solve low-payload nsF5 and WAV.
A fresh fit/calibration diagnosis now identifies three concrete issues:
weak embedding perturbations, heavily saturated audio descriptors, and an
upscaled-eight-bit training shortcut. These findings motivate new features;
they do not prove a universal data accuracy ceiling. No target payload was
raised and no result was rounded into an 80% pass.

The [numeric receipt](../benchmarks/low-signal-diagnosis-20261115.json) retains
source hashes, every calibration candidate/choice, actual test outcomes or
explicit calibration-gate exclusions, failures, independent audits and resource
closeouts. [Earlier results](REMAINING_FLOOR_RESULTS_20261115.md) remain intact.
All work uses existing private media and CPU; no download, GPU, native-role
assignment, locked ALASKA2 access or detector deployment occurs.

## What the local measurements establish

The [prospective diagnosis](LOW_SIGNAL_DIAGNOSIS_PROTOCOL_20261115.md) opens
only fixed fit/calibration sources: sixteen ESC-50, forty AudioMNIST, eight
FSDD clips, and twenty-four first-3,000-source BOSS JPEGs. It never opens a
historical test source. Eleven complete feature rows replay exactly against
the old WAV/DCTR caches; all sampled nsF5 changed counts replay too. There
is no evidence here of swapped feature rows, failed embeddings or a numeric
prediction-export error. Selection-specific statistics below describe this
fixed sample, not all clips or an independently sampled population.

| Sampled source | Clip samples, median | Old quiet first-difference saturation, median | Quiet second-difference roughness, median in LSB |
|---|---:|---:|---:|
| ESC-50 fit/calibration | 220,500 | **93.5435%** | 53.9259 |
| AudioMNIST fit/calibration | 31,515.5 | 5.4450% | 1.26883 |
| FSDD fit/calibration | 4,219 | **89.2962%** | 113.1740 |

The old first-difference descriptor rounds in one-LSB units and clamps values
to +/-3. Most measured ESC/FSDD differences therefore share endpoint bins,
losing their magnitude distinctions. Some higher-order/quiet descriptors and
WS statistics still retain other information; saturation of this one family
does **not** mean the entire old vector is constant. Quiet roughness includes
speech/transients and is not a direct sensor-noise variance measurement.

FSDD's source mismatch is separately verified across all 1,500 old fit clips,
using only the cached cover/5%-stego occupancies, with no test access:

| Fit speaker | Clips with all lowest eight bits zero | Median even cover samples | Median even replacement-5% stego samples |
|---|---:|---:|---:|
| george | 0/500 | 49.9638% | 49.9649% |
| jackson | 0/500 | 49.9897% | 50.0272% |
| nicolas | **500/500** | **100.0000%** | **97.4764%** |

The nicolas signal is easy to recognize after embedding, while genuine
16-bit cover parity remains near 50%. This supports a source-shortcut
hypothesis; it does not establish which exact old tree split caused an error.
Many clips from one speaker also do not supply many independent recording
domains. The fresh matched-bit-depth fit excludes nicolas without deleting
or moving any original file/role. It changes cohort and features together,
so its result cannot isolate either as a sole cause.

JPEG nsF5 uses near-optimal coding: the earlier measured mean changes are
only **.563909% /1.302061%** of nonzero AC coefficients at **.05/.1 bpnzac**.
The [authors' simulator explanation](https://dde.binghamton.edu/download/nsf5simulator/)
expressly distinguishes optimal coding from less efficient older coding.
Thus matching nominal payload numbers across the real F5 encoder and nsF5
does not establish equal numbers of modifications or equal detectability.
Our source replay confirms decrements, unchanged DC/preexisting zero modes,
and exact old sample change counts; it does not justify raising payload.

## What the papers support, and what they do not

The [WAV matching paper](https://www.atlantis-press.com/article/4689.pdf),
Table I, reports **62.25% at one second /10% embedding**, versus **81.37% at
ten seconds /10%**. Its abstract's above-80% claim is not a duration-independent
promise for short FSDD recordings. These figures use that paper's experiment,
not our speaker-separated corpus; they are context, not direct benchmarks.

[JRM](https://dde.binghamton.edu/kodovsky/pdf/SPIE2012_Kodovsky_Steganalysis_of_JPEG_Images_Using_Rich_Models_paper.pdf)
models relationships between DCT modes and neighboring blocks. That motivates
changing JPEG representation instead of another DCTR classifier grid. Our
compact adaptation uses only its absolute-coefficient subset plus targeted
histograms/estimates; it is not full JRM, an exact reproduction, or a claim
that a published score transfers to this split. [Calibration research](https://ws2.binghamton.edu/fridrich/Research/calibration-color.pdf)
also describes cases where calibration hurts, so improvement is measured,
not assumed. The installed sealwatch version is source-bound at 2024.12;
no library or pretrained weights were downloaded.

Even a perfectly implemented detector cannot universally detect replacement
in an exactly fair, independent low-bit source: replacing those bits with
new independent fair bits leaves the distribution unchanged. That is a
mathematical limiting example, **not proof that these real WAVs meet those
assumptions**. Low measured parity bias, short clips and weak new calibration
scores can motivate caution, but cannot prove an irreducible 50% ceiling.

## Fresh nsF5 coefficient-domain intervention

The [frozen protocol](NSF5_CALIBRATED_RELATIONS_PROTOCOL_20261115.md) retains
.05/.1 payloads and 2,400 fit /600 calibration /1,000 pixel-disjoint historical
test sources. New descriptors combine 2,512 absolute JRM-subset values,
504 per-mode histogram values, their four-edge-cropped libjpeg calibration
reference, and 64 signed decrement projections: 6,096 values total. This
reference avoids padding a 508-pixel one-sided crop to the old block extent.

Ten fixed new classifiers per payload plus five fixed systems/mixtures use
calibration only. Feature indices, states, mixtures and exact BA thresholds
freeze globally before any new test features. A selected calibration BA
below .80 stops that new condition before test decoding/inference; the old
measured miss remains visible. Such a gate exclusion is **not an 80% pass**.

Calibration selects a 75% new /25% frozen-DCTR mixture at .05 bpnzac,
BA **65.416667%**, AUC .694908. It fails the predefined calibration gate:
**no new .05 test features or inference**. Earlier measured 60.70% legacy
and 59.35% later-kernel test misses remain separate unchanged records.

At .1 bpnzac, calibration selects a **50% new /50% frozen-DCTR** mixture,
BA **81.00%**, with the new plain-square-root coefficient kernel (factor 1,
ridge .01) and threshold **.7342438804846576**. One frozen selected historical
inference yields **77.70% BA**, AUC **.853915**, TPR **74.5%**, FPR **19.1%**,
TN 809 /TP 745, **1,554/2,000** correct. It remains an **80% miss**: 46 more
correct decisions would be needed. The exploratory paired-source BA 95%
interval is **76.15–79.15%**.

Against this study's frozen fair-fit DCTR kernel, 70.75%, the paired gain is
**+6.95 points**, exploratory 95% interval **+5.30 to +8.65 points**. Against
the best earlier reported legacy 75.15%, the point difference is only
**+2.55 points**; no paired interval is invented for unavailable old score
vectors. That old model fitted 3,000 sources, versus this study's 2,400 fit /
600 calibration split. Calibration-to-test loss from 81.00 to 77.70% also
shows why a calibration pass is insufficient. All thirty calibration records,
one measured test cell, three real calibration feature vectors and sixty-two
direct numeric predictions replay independently (max error 1.018e-10).
These results support a representation gain on this reused cohort, not a
single-factor explanation, untouched qualification or an 80% success.

## Fresh WAV modular/re-embedding intervention

The [frozen protocol](WAV_MODULAR_CALIBRATION_PROTOCOL_20261115.md) replaces
saturated low-range statistics with modulo-eight single/pair residual
frequencies, wider residual histograms and variance-conditioned predictor
statistics. A fixed independent full-rate matching reference supplies base
minus reference descriptors, inspired by [audio re-embedding calibration](https://arxiv.org/abs/1701.05614).
This reference is a declared analysis transform, not a change to the target
payload or a lookup of the original embedding seed/key. Each file supplies
its own reference; no paired clean image/audio is available to inference.

ESC keeps the corrected 1,200/398/402 recording-group split; AudioMNIST keeps
15/5/10 speakers. The new FSDD training overlay uses 1,000 george/jackson
clips, 500 lucas calibration clips and unchanged 1,000 theo/yweweler test
clips. All original roles/files and previous 8-bit-source results stay.
Eight candidates per formerly failing dedicated condition, 128 total, use
fit-only feature selection and calibration-only model/threshold choice.
Choices freeze before optional selected inference, with the same .80
calibration gate. Prior passing dedicated and mixed WAV rows are not rerun.

All sixteen selected calibration scores remain below .80, so the
predeclared gate excludes **every new WAV test condition**. No new WAV test
features or selected predictions are constructed; old measured misses remain
unchanged. The following are **calibration metrics**, not test accuracy:

| Dataset | Condition | Selected calibration BA | Calibration AUC | New test status |
|---|---|---:|---:|---|
| esc50 | replacement-0.05 | 71.733668% | 0.802032 | not evaluated: calibration gate |
| esc50 | replacement-0.1 | 77.135678% | 0.849852 | not evaluated: calibration gate |
| esc50 | matching-0.05 | 61.934673% | 0.689534 | not evaluated: calibration gate |
| esc50 | matching-0.1 | 64.572864% | 0.706207 | not evaluated: calibration gate |
| esc50 | matching-0.2 | 66.080402% | 0.717861 | not evaluated: calibration gate |
| esc50 | matching-0.4 | 65.452261% | 0.714774 | not evaluated: calibration gate |
| audiomnist | matching-0.05 | 59.000000% | 0.579200 | not evaluated: calibration gate |
| audiomnist | matching-0.1 | 70.000000% | 0.722900 | not evaluated: calibration gate |
| fsdd | replacement-0.05 | 52.100000% | 0.514880 | not evaluated: calibration gate |
| fsdd | replacement-0.1 | 52.300000% | 0.513272 | not evaluated: calibration gate |
| fsdd | replacement-0.2 | 51.600000% | 0.500512 | not evaluated: calibration gate |
| fsdd | replacement-0.4 | 52.100000% | 0.507892 | not evaluated: calibration gate |
| fsdd | matching-0.05 | 51.700000% | 0.503376 | not evaluated: calibration gate |
| fsdd | matching-0.1 | 51.800000% | 0.503744 | not evaluated: calibration gate |
| fsdd | matching-0.2 | 52.300000% | 0.506644 | not evaluated: calibration gate |
| fsdd | matching-0.4 | 51.700000% | 0.504592 | not evaluated: calibration gate |

All 128 new candidates are included in the numeric receipt. FSDD calibration
stays **51.6–52.3%** after removing the upscaled-eight-bit speaker and adding
unsaturated modular/re-embedding features. This intervention does not support
the hypothesis that either old quantizer saturation or that speaker alone
explains the miss. It also does not prove that no future detector can improve.
AudioMNIST matching 5/10% remains 59.0 /70.0% on its separate calibration
speakers; ESC's maximum is 77.135678% at replacement 10%.

Independent replay verifies all 128 calibration records, 256 direct numeric
predictions (max error 2.602e-11), all sixteen exclusions and three real
calibration sources' complete descriptors. No test score is fabricated from
calibration, and a gate exclusion is never counted as a test pass.

## Preserved source failures and bounded cache correction

The first nsF5 preparation passes a string to a reused Path-only JSON reader
and fails before real decoding/fitting. The [source correction](NSF5_CALIBRATED_RELATIONS_SOURCE_AMENDMENT_20261115.md)
preserves that failure. Its subsequent feature computation finishes, but
export fails at the legacy helper's eight-MiB hard per-file limit after
254.107 seconds. Keep the truncated archive as unavailable, not usable data.
The first WAV data-only phase is explicitly stopped with SIGTERM before the
same known export fault, with its actual -15 exit and external-stop receipt
retained. It is not relabelled as a watchdog result or successful preparation.
No real classifier was fitted in any of these three partial attempts.

A [separate cache amendment](LOW_SIGNAL_CHUNKED_CACHE_AMENDMENT_20261115.md)
preserves the hard file guard: at most 100 source rows per compressed part,
per-part hashes, and a complete manifest verified on ordered concatenation.
Generated scalar and exact chunk roundtrips pass before real fits. The fresh
parents bind every earlier source/log/partial. nsF5 charges 280 old seconds
inside its original 2,400 cap; WAV charges 150 inside 1,800. Four independent
file workers each use one BLAS thread, summed sampled RSS <=6 GiB and combined
output <=256 MiB per study. No old parent/model is restarted or budget reset.

| Closed study | Successful completion wall time | Conservative cumulative debit /cap | Combined parent-close output |
|---|---:|---:|---:|
| Fit/calibration diagnosis | 6.011487 s | 6.011487 /240 s | see numeric receipt |
| nsF5 chunk completion and preserved prior attempts | 452.055094 s | **732.055094 /2,400 s** | **140,801,014 bytes** |
| WAV chunk completion and stopped parent | 461.113286 s | **611.113286 /1,800 s** | **120,403,931 bytes** |

The two fresh studies together retain about 249.10 MiB of logical evidence,
including prior partial output, far below the available roughly 30 GiB.
Combined output counts the preserved earlier directories, not only the last
successful one. No source/checkpoint/log deletion occurs. Every parent is
closed; unused consumed allowances grant no retry. Remaining work is a new
feature/data hypothesis, not rerunning these fixed candidates.

Application/test code remains identical to all 570 bindings of the actual
passed full CPU gate: 4,646 passes, two explicit CUDA skips and 97.415805%
coverage. Fresh Ruff, research-inclusive mypy over 270 files and all fifteen
README/scope tests pass, with fresh disk-backed temporary storage. This
source-identical reuse is not a new full-suite claim. Models, feature
matrices and source media stay private. Every miss or calibration-gated test
must remain visible; no historical result or failed evidence is deleted.
