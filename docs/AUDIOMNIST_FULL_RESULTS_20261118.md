# AudioMNIST LSB matching 5/10% with the full opened-speaker corpus (2026-11-18)

**Twenty-five times more training data does not lift AudioMNIST matching
5/10% to 80%.** The best calibration results are 63.88% and 75.54%, both below
the .80 gate, so no test features were built and no test accuracy is claimed.
The earlier "data-limited" diagnosis is therefore wrong. With the current
residual, spectral and noise-Markov features, these two cells are
signal-limited. [Numeric receipt](../benchmarks/audiomnist-full-20261118.json).

## Data and roles

- The user approved the download on 2026-11-18. Source: AudioMNIST commit
  `630d7dab4c040882834de6fa21baf9a60372accd` (MIT), the same commit as the 2026-10-17 pilot.
  Archive SHA-256 `f0420069…c591f9c573`, 996,621,372 bytes.
- Only the 30 speakers that earlier studies had already opened were extracted: 15,000 clips, 500
  per speaker. The 30 reserved speakers were not extracted, decoded or
  analysed. The archive CRC check read their compressed bytes; nothing else did.
- All 600 previously acquired files match their recorded SHA-256 exactly.
  Every extracted clip is mono 48 kHz PCM16; the raw files are unique.
- Roles keep the unchanged speaker-grouped modular layout: 15 fit speakers
  (7,500 clips), 5 calibration speakers (2,500) and 10 test speakers (5,000).
  The old 300/100/200 clips are subsets of these roles.
- Clips from the old layout keep their original embedding seeds. New clips
  use seed `[2026111801, condition, speaker, digit, repetition]`. All 400
  old fit/calibration rows replay the compact features of the 2026-11-17
  continuation exactly, for covers and both stegos.

## Fixed candidates and calibration

Each condition had three candidates, fixed before fitting:
- `compact-hgb`: the 16 compact LPC/weighted-stego/spectral features with the
  earlier 300-iteration, 15-leaf gradient-boosted trees.
- `mom-fld`: the square root of the 4,716 conditional noise-Markov features
  with a 51-learner FLD ensemble (d_sub 1000).
- `joint-hgb`: both feature sets together with the same trees.

Each candidate fits on the 7,500 fit pairs. Its threshold and the selection
come only from the 2,500 calibration pairs (BA, then AUC, then declared order).

| Condition | compact-hgb | mom-fld | joint-hgb | Selected | Gate |
|---|---:|---:|---:|---|---|
| matching 5% | 62.68% (AUC .675) | 51.50% (.517) | **63.88%** (.686) | joint-hgb | below .80 |
| matching 10% | 74.14% (.817) | 56.76% (.586) | **75.54%** (.827) | joint-hgb | below .80 |

The diagnostic learning curve below refits the chosen recipe on random
subsets of the fit pairs and scores them on calibration:

| Fit pairs | matching 5% BA | AUC | matching 10% BA | AUC |
|---:|---:|---:|---:|---:|
| 300 | 61.04% | .645 | 74.04% | .810 |
| 1,500 | 62.80% | .669 | 75.30% | .821 |
| 3,750 | 63.10% | .678 | 75.76% | .830 |
| 7,500 | 63.88% | .686 | 75.54% | .827 |

Going from 300 to 7,500 pairs adds under three BA points; the 10% curve is
flat after 1,500 pairs. The high-dimensional Markov features stay near chance
even with 7,500 pairs, so their earlier failure was not caused by too few
samples. The historical AudioMNIST test results stay unchanged at 56.5/59.5%.
`test-feature-gate.json` records that no test features were constructed.

## Verification and resources

- Four pure-function study checks pass (seed mapping, speaker-disjoint roles,
  brute-force threshold parity, feature views). The extraction validates every
  member path, type, size and WAV header.
- Feature extraction for 10,000 clips × 3 signals took 119.3 seconds on 6
  workers. The six candidate fits took 856 seconds in total; CPU only, no GPU.
- Raw audio, features and models stay private under
  `.benchmark/audiomnist-full-20261118/` (gitignored).

## What remains

AudioMNIST matching 5/10% remain in-scope misses. More data from these
speakers is exhausted. The remaining routes are a different signal model
(for example a Colab-run raw-waveform CNN, whose earlier versions failed
their positive control) or a user decision about the target scope. The 30
reserved speakers stay closed for a future clean evaluation.
