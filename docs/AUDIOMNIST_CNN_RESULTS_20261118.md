# AudioMNIST LSB matching 5/10% with a raw-waveform CNN (2026-11-18)

**A raw-waveform CNN lifts AudioMNIST matching 10% from 59.5% to 78.43% test
balanced accuracy, but still misses 80%.** Calibration BA was 83.36%, which
passed the .80 gate, and the test was looked at once. The pair interval is
77.77–79.13% and the speaker-cluster interval is 74.00–83.09%. Matching 5%
reaches 69.52% calibration BA, below the gate, so it was not tested.
Both cells remain in-scope misses.
[Numeric receipt](../benchmarks/audiomnist-cnn-20261118.json).

## Setup

- Code: `research/wav_cnn_audiomnist.py`, run through
  `notebooks/audiomnist_wav_cnn_20261118.ipynb` pinned to commit `e441704`.
  At the user's request, the assistant operated the Colab run in the
  browser on a Tesla T4 (torch 2.14.1, numpy 2.4.6).
- Data: AudioMNIST commit `630d7da` (MIT), downloaded inside Colab. Only the
  30 opened speakers were extracted, and all 15,000 files matched their
  recorded SHA-256. The 30 reserved speakers were not read.
- Roles: the unchanged speaker-grouped layout, with 15 fit (7,500 clips),
  5 calibration (2,500) and 10 test speakers (5,000).
- Fixed stegos: calibration and test use the project embedding with the
  earlier studies' seeds. The four calibration stego sets matched their
  recorded digests in Colab. The 5/10% calibration pairs are therefore the
  same pairs on which the CPU features reached 63.88/75.54%.
- Model: four fixed difference filters, truncation at ±8, a masked 1-D
  convolutional stack, and mean and standard-deviation pooling (466,817
  trainable parameters). Padding is masked, so a clip's score does not
  depend on its batch.
- Training: random 16,384-sample crops of the fit clips, with a fresh ±1
  matching payload at every presentation. The curriculum runs 40%
  (12 epochs), 20% (12), 10% (24) and 5% (24). Each stage starts from the
  previous stage's best calibration checkpoint. The 40% stage is a positive
  control. Each epoch took about 66 s; the full run took 79 minutes.
- Selection: for each target rate, the epoch with the best calibration BA,
  then AUC, then the earlier epoch, together with its calibration threshold.
  Both choices were frozen before the test. Gate .80.

## Calibration

| Stage | Best epoch | Calibration BA | AUC | TPR / FPR | CPU features (same pairs) |
|---|---:|---:|---:|---:|---:|
| 40% (positive control) | 6 | 99.26% | .9987 | 99.7 / 1.2% | — |
| 20% | 11 | 95.00% | .9840 | 95.2 / 5.2% | — |
| **10%** | 18 | **83.36%** | .9045 | 86.4 / 19.6% | 75.54% |
| **5%** | 10 | **69.52%** | .7495 | 77.0 / 37.9% | 63.88% |

The 10% stage stayed between 79.7% and 83.4% from its second epoch on.
The 5% stage stayed between 66.5% and 69.5% for all 24 epochs.

## Test (once)

| Rate | Test BA | Pair 95% | Speaker 95% | AUC | TPR / FPR | Earlier test |
|---|---:|---:|---:|---:|---:|---:|
| 10% | **78.43%** (miss) | 77.77–79.13% | 74.00–83.09% | .8954 | 64.8 / 7.9% | 59.5% |
| 5% | not evaluated (calibration gate) | — | — | — | — | 56.5% |

Test AUC (.895) is close to calibration AUC (.904), so the ranking
transfers to the test speakers. The frozen calibration threshold does not:
calibration gave 86.4% TPR at 19.6% FPR, and test gives 64.8% TPR at 7.9%
FPR. The miss comes from a speaker-dependent shift of the score scale, not
from lost ranking. The threshold is not re-tuned on the test.

## Speaker-robust threshold (2026-11-19, adaptive second look)

The frozen 10% model was rescored on CPU. The calibration BA (83.36%) and
the first test BA (78.43%) replay exactly. Five threshold rules were
compared by leave-one-calibration-speaker-out BA, using calibration data
only:

| Rule | LOSO BA | Worst speaker |
|---|---:|---:|
| pooled BA-optimal (original) | 82.76% | 80.10% |
| **mean of per-speaker thresholds** | **83.26%** | 81.60% |
| median of per-speaker thresholds | 83.20% | 81.40% |
| minimax speaker BA | 82.78% | 79.90% |
| midpoint of cover/stego medians | 83.02% | 81.40% |

The selected rule raises the threshold from 0.265 to 0.315. The test
speakers needed a lower threshold, so the second look gives **78.00%**
(pair 95% 77.32–78.68%, speaker 73.31–82.89%; TPR 63.4 / FPR 7.4%). That
is slightly worse than the first look. The five calibration speakers do not
contain the test speakers' score shift. Threshold rules chosen on these
speakers cannot fix it, and further test looks would only tune on the test.

## Eight calibration speakers (v2, 2026-11-19, calibration only)

To test whether the five calibration speakers were too few, three seeded fit
speakers moved to calibration (12 fit / 8 calibration / 10 test; layout
`notebooks/audiomnist_wav_cnn_layout_v2_20261119.json`, code `ccaaff5`).
Same model and curriculum on a Colab T4; the 40% control passed. The
threshold rule was chosen by leave-one-calibration-speaker-out (LOSO), and
the .80 gate was applied to the LOSO BA.

| Rate | Pooled calibration BA | AUC | Selected rule | LOSO BA | Worst speaker | Test |
|---|---:|---:|---|---:|---:|---|
| 10% | 79.71% | .874 | pooled BA-optimal | **79.08%** | 67.10% | not evaluated |
| 5% | 66.86% | .727 | median of per-speaker | **66.05%** | 55.90% | not evaluated |

Both rates miss the gate, so no test look was taken. With more calibration
speakers the 10% estimate falls from 83.36% to about 79%, which matches the
78.0–78.4% test looks: the five-speaker calibration was optimistic, and the
model itself generalises to new speakers at about 79%. Moving speakers out
of fit (15 → 12) may also cost a little accuracy. Receipt:
`benchmarks/audiomnist-cnn-20261118.json` (`v2_eight_calibration_speakers_20261119`).

## Limits and next steps

- 200 of the 5,000 test clips were scored by earlier AudioMNIST studies.
  This is the first CNN look at the test speakers, with one frozen choice
  per rate.
- Fresh training payloads use Bernoulli positions; calibration and test use
  the exact project embedding.
- One seed and one architecture; there was no hyper-parameter search.
- Threshold rules and more calibration speakers are both exhausted (v2:
  ~79% held-out-speaker BA). The remaining lever is a model that generalises
  better across speakers, for example more training speakers (only 30 of
  60 are opened; the 30 reserved speakers need the user's approval) or
  speaker-invariant inputs. It must be chosen on fit/calibration data only.
  Any new test look would be adaptive evidence.
