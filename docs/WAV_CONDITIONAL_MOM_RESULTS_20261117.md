# Conditional noise Markov WAV study (2026-11-17)

The fresh [conditional noise Markov protocol](WAV_CONDITIONAL_MOM_PROTOCOL_20261116.md)
does **not** solve any of the sixteen failing dedicated WAV conditions. Every
calibration-selected model stays below the predefined .80 gate, so no new
historical test feature or inference is made. All sixteen cells are
`not_evaluated_calibration_gate`, never test passes. Earlier measured test
misses and passing rows are unchanged.

The [numeric receipt](../benchmarks/wav-conditional-mom-20261117.json) contains
every selected candidate, the gate exclusions, the independent audit and the
resource closing. Models and features stay private.

## Features and candidates

Each unchanged mono PCM16 clip yields two noise sequences: adjacent sample
differences and the residual of a three-level reversible 5/3 lifting denoiser.
At q=1/2/4, pair and triple windows whose quantized samples all lie in [-3,3]
give joint frequencies, row-conditional probabilities and valid-window
fractions: 4,716 features. Eight fixed candidates per condition (two Fisher
512-feature linear models, square-root Gaussian ridge/SVC with and without a
fit-only scaler, two 51-learner FLD ensembles) are selected on calibration
counts, then AUC, then declared order. Roles reuse the corrected ESC recording
groups, AudioMNIST speakers and the declared FSDD matched-bit-depth overlay.

## Calibration results

| Dataset /condition | Selected candidate | Calibration BA | AUC |
|---|---|---:|---:|
| ESC replacement 5% | ensemble, d_sub 500 | 64.45% | .721 |
| ESC replacement 10% | Fisher ridge .03 | 64.70% | .707 |
| ESC matching 5/10% | ensemble, d_sub 500 | 63.19 /64.32% | .707 /.720 |
| ESC matching 20/40% | ensemble, d_sub 500 | 66.21 /68.22% | .748 /.768 |
| AudioMNIST matching 5/10% | standardized SVC /ridge | 59.00 /62.00% | .563 /.604 |
| FSDD, all eight conditions | mixed | 51.4–54.0% | .504–.535 |

ESC calibration is lower than the earlier modular study at replacement 5/10%
(71.73 /77.14%) and similar at matching. FSDD stays near chance; a private
two-fit-clip probe found zero valid first-difference pairs at q=1 on probed
FSDD clips, so this masked small-noise family carries little FSDD signal.
Source and feature changes are joint; no single-factor cause is claimed.

## Verification and resources

Before launch, the preserved drafts gained a matching auditor and parent;
generated feature/numeric preflight, a generated end-to-end rehearsal of both
measured and gated paths and a write-intercepted real-source preparation
(5,670 input hashes) pass. The one exclusive parent then runs once.

The independent audit replays all 128 calibration records, every selection
and gate, 768 direct predictions (maximum error 4.366e-11), both bandwidth
medians per cell and three actual calibration descriptors; all 570 software
hashes still match the passed 4,651-test full gate. Wall time is 408.74 s,
438.74 /3,600 s with 30 prior seconds. Output is 116,304,461 bytes /512 MiB;
the largest summed sampled RSS is 0.80 GiB. No Gram matrix, raw media copy,
download, GPU or deployment. The parent is closed; no retry follows.

The WAV misses remain, so **the overall 80%-per-method goal is still not met**.
