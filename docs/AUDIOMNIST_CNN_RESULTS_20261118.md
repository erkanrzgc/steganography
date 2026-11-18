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

## Limits and next steps

- 200 of the 5,000 test clips were scored by earlier AudioMNIST studies.
  This is the first CNN look at the test speakers, with one frozen choice
  per rate.
- Fresh training payloads use Bernoulli positions; calibration and test use
  the exact project embedding.
- One seed and one architecture; there was no hyper-parameter search.
- The likely lever for 10% is score calibration that is robust across
  speakers, for example per-clip normalisation or more calibration
  speakers. It must be chosen on fit/calibration data only. Any new test
  look would be adaptive evidence.
