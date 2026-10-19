# Read-only WAV diagnosis results — 2026-10-16

Execution: **completed**. Zero optimizer updates; production qualification remains **unavailable**.

All training feature/score contrasts and cached development comparisons are in [portable results](../benchmarks/wav-readonly-results-20261016.json).

| Source | Seed | Frozen threshold | Train AUC min–max | Cached development AUC min–max | Train BA min–max |
| --- | ---: | ---: | ---: | ---: | ---: |
| ESC-50 | 20261016 | 0.43 | 0.513–0.739 | 0.500–0.547 | 0.500–0.578 |
| ESC-50 | 20261017 | 0.83 | 0.520–0.812 | 0.484–0.531 | 0.500–0.641 |
| FSDD-v1.0.10 | 20261016 | 1.00 | 0.688–0.902 | 0.422–0.594 | 0.500–0.500 |
| FSDD-v1.0.10 | 20261017 | 0.81 | 0.774–0.915 | 0.359–0.656 | 0.500–0.734 |

Only training features were opened; development metrics were copied at their original thresholds. No validation/test media or validation feature artifact was opened. Every model digest is unchanged. Missing OOF scores/fold snapshots remain unavailable; the diagnostic did not refit them.

## Measured contrasts and limits

Each source contains 32 training carrier files and eight matched stego cells,
but ESC-50 has 31 independent recording groups and FSDD only two speaker groups.
No cover/stego pair has an exactly identical 760-feature vector. The encoder
therefore changes measured features; this is not evidence of useful detection.
The signed mean feature shift is small relative to cover variation:

| Source | Cover variation RMS | Paired feature-difference RMS min–max | Mean-shift /cover-variation min–max |
| --- | ---: | ---: | ---: |
| ESC-50 | 0.07069 | 0.00129–0.02907 | 0.00338–0.20764 |
| FSDD-v1.0.10 | 0.08175 | 0.00530–0.04285 | 0.04303–0.35716 |

These descriptive ratios use unstandardized feature coordinates and do not
measure significance or prove that content variation caused the failure.
No single feature has a strictly same-sign delta across all 32 pairs in any
cell. Normalization and final weighted training BCE independently replay the
stored values to absolute tolerance 1e-12.

FSDD training AUC reaches 0.915 while cached development AUC reaches at most
0.656; ESC-50 development AUC stays between 0.484 and 0.547. This supports a
generalization limitation in this small pilot, without identifying its cause.
The FSDD seed with frozen threshold 1.00 predicts no positives even on training
rows. That explains its operating-point behavior, but weak cached development
AUC means threshold replacement is not a demonstrated solution. Missing OOF
scores prevent reconstructing why that threshold was selected. No sweep,
feature selection, new validation access or fitting was performed.

## Verification and resource use

The unchanged full gate passes **3,273 tests**, with **two actual-CUDA skips**,
**96.78%** total coverage and **100%** for both new diagnosis modules. Ruff and
208-file mypy pass. Forty focused tests cover independent scalar calculations,
closed-role FIFOs, source/hash/domain mismatches, model replay, symlinks,
malformed numeric archives, overwrite, cancellation and resource limits.
Local verification uses Python 3.11; the 3.12–3.14 matrix remains unverified here.
See the [verification receipt](../benchmarks/wav-readonly-verification-20261016.json).

Actual read-only inference takes **0.268 seconds including imports**, with
**61,660 KiB peak resident memory**, under its separate 300-second /512-MiB
one-CPU, zero-swap limits. User/system CPU consumption is 0.2831/0.01595 seconds.
It adds **zero optimizer updates** and opens only two training feature artifacts
and their bound manifests. All 14 model files are hashed before and after;
only four audio models are decoded. No checkpoint is written.

Full verification takes 1,264.60 seconds and peaks at **136,824,862 bytes** of
tracked task outputs. The native four-GiB allocation and JPEG six-GiB reserve
remain enforced. Both owned user services exited successfully; no job remains
running or queued for an automatic retry.
The final [integrity receipt](../benchmarks/wav-readonly-integrity-20261016.json)
records 842,326 bytes of diagnosis output and a 680,747,864-byte current tracked
native footprint across the five task roots. All four frozen source closures
(18/27 native, 35 JPEG, 30 diagnostic files) and all 14 model hashes are
unchanged. Source review records 82,859 metadata bytes and zero new media bytes;
the legacy cumulative raw-transfer attestation remains unavailable.

## Concrete next design and missing sources

The [next experiment draft](WAV_NEXT_PROTOCOL_DRAFT.md) proposes a speaker-grouped
AudioMNIST pilot, 20 train /10 development /30 reserved speakers and a matched
two-seed linear versus raw/difference PCM CNN comparison. It is **not executable**:
file headers/hashes, accepted file-level provenance, transfer allowance, trainer
and whole-update resource preflight remain unavailable. Only pinned upstream
metadata/code was reviewed; recording cuts are grouped by their entire speaker.
No new media was acquired. The older cumulative one-GiB transfer ledger is
incomplete and was not reset. Native RGB, stereo and MP3 pilot gaps persist;
see the [source audit](../benchmarks/native-source-gaps-20261016.json).

Do not infer causality, select features or retune a threshold from this post-hoc report. A new experiment needs a separately frozen model/data/exposure protocol. No missing licensed native-source pilot was completed by this diagnostic.
