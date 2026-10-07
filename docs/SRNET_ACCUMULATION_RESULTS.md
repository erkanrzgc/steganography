# Matched-step/matched-exposure microbatch control — 2026-10-07

**Completed; both arms fail the complete learning goals. Original-input train
balanced accuracy remains .50. No model deployed, no validation pixels loaded,
no claim that corpus size or BN is the unique cause of failed learning.**

Frozen protocol `SRNET_ACCUMULATION_PROTOCOL.md`, commit `47217aa`, SHA-256
`e521568b94919a5225bf5c13fa2b9e2bc4f04e72852a58a1074dc68bcb085d11`.
Executed implementation commit `e63e0d3`; all source hashes are captured.
Baseline `benchmarks/srnet-wide-batch-20261007.json`, SHA-256
`ae828ea18337fcff82f40506717c7fe5f33d664e380537632a4ac1ca4dcf9427`.
No result-selected rerun, checkpoint, clipping, BN repair or optimizer tuning.

Exact same 24 training rows, own/original tensors and derived hashes, fresh
seed 20261012, Adamax and 20 epochs as the eight-row control. **Both sides:
80 optimizer steps, 640 presentations and identical ordered eight-row groups.**
The new arm splits each group into two four-row forwards, clears gradients
once, backpropagates mean CE/2 for each, checks accumulated finite gradients,
then steps once. Average unscaled microbatch CE is reported per group.
BN forward counts differ intentionally: 160 versus 80. Independently reloaded
numeric models have all 26 counters exactly 160; old baseline models retain
their original checksums. No weights/tensors are distributed.

Independent unit verification manually performs the two half-loss backwards
and one Adamax step on a separate full SRNet from the same seed: all 183 state
arrays match exactly. Second-microbatch invalid logits, gradients, deadline or
exception causes zero optimizer steps, and caller RNG/threads are restored.
Two/four/eight-row legacy caller tests continue to pass.

Ordinary stored-BN singleton inference on **24 unique training rows**:

| Measurement | Factor 1 | Factor 32, artificial |
| --- | ---: | ---: |
| First mean microbatch CE | 1.039674 | .783455 |
| Last mean microbatch CE | .693072 | .175415 |
| Relative loss reduction | 33.34% | 77.61% |
| Own-input singleton CE | .697088 | 1.219837 |
| Own-input singleton BA | .50 | .59375 |
| Own-input recall / FPR | .625 / .625 | .6875 / .50 |
| Original-input singleton CE | .697088 | 2.328027 |
| Original-input singleton BA | .50 | .50 |
| Final loss <= .35 | Fail | Pass |
| Loss reduction >= 25% | Pass | Pass |
| Own-input singleton BA >= .90 | Fail | Fail |

Compared with the matched eight-row baseline, own CE worsens from .694339
to .697088 / .464312 to 1.219837. Artificial factor-32 own BA worsens from
.6875 to .59375, and original-input CE rises from 2.250566 to 2.328027.
Neither setting learns to separate the original weak training inputs (BA .50).
Lower factor-32 microbatch training loss is not better singleton discrimination
or real detection accuracy. This comparison controls optimizer-step count,
row exposure and group order but changes training BN context/statistics;
it does not identify one unique failure mechanism.

All 48 own/original singleton logits/metrics replay per model and all 18
independent NumPy float64 forward oracles pass, unchanged tolerances.
Maximum logit/score differences: factor 1, 1.117834e-7 / 2.116105e-8;
factor 32, 2.924287e-6 / 1.444501e-7. CLI exit 2 reflects failed learning
goals in a complete run, not an unavailable or interrupted job.

Portable raw evidence `benchmarks/srnet-accumulation-20261007.json`, SHA-256
`91cd7442d3f235fd962c7fbece67c660ba17804f32262cceb832346f7b9b2588`.
Local models: factor 1 SHA-256
`45b92cffcc96059f2b97f35420463b68fdf9bcbe800d917e2467128f3a0209fe`;
factor 32 `c81cfd003e4b1848322dc547d78431adbae8b7abd875c1d1a46cc7c92dc71e37`.
Complete post-preparation runtime 459.05s, Linux 8 vCPU/15 GiB, two Torch
threads/OpenBLAS one, with concurrent QA. This is engineering timing, not a
controlled performance comparison or challenge-latency benchmark.

`JPEG_TRAINING_DATA_SCOPE.md` distinguishes the six-scene/24-row sanity subset
from the prepared **2,985 training rows / 892 declared original scene lineages**.
The earlier full one-epoch fit also failed; neither that nor this tiny control
establishes insufficient data as the sole cause. Next freeze a data/exposure
scale study over additional training-only lineages with adequate training
budget and independent qualification data, rather than claiming tiny controls
improve accuracy. No previously inspected development validation may select
settings or be relabeled blind; source/camera independence stays unverified.

Explicit checkout-only replay, optional research dependencies, fresh output:

```text
OPENBLAS_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 3840s venv/bin/python -m steganography.research_srnet_accumulation --config .benchmark/srnet-tiny-sanity-20261007/config.json --out .benchmark/accumulation-fresh
```

CLI enforces CPU/address/file/core limits; external wall timeout is required.
Direct service callers supply process isolation. Interrupted jobs may retain
earlier-arm artifacts but never a complete two-arm report. No download occurs.

Verification on Python 3.11.14: 1,326 tests pass, total coverage 95.22%.
The four affected trainer/context/service modules cover 288/290 statements
(99.31%); both core modules have 100% statement coverage. Ruff, full mypy
(130 source files), and `git diff --check` pass. Local `--no-isolation`
wheel/sdist builds contain the shared trainer and new CLI module, no model
or corpus artifacts. Python 3.12–3.14 and full Docker are unverified locally.
