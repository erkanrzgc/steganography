# Real-delta signal-strength control — 2026-10-07

Protocol committed as `09e28da` before tensor preparation/fitting. Two complete
arms, factors 1/32, fresh identical seed, identical sample/pair/batch orders,
20 epochs and 160 optimizer updates each. Only the frozen 24 training rows
were used; no validation, threshold search, resume or model deployment.
Factor 32 inputs are artificial tensors, not actual embedded/encoded JPEGs.

| Training-only measurement | Factor 1 | Factor 32 |
| --- | ---: | ---: |
| First epoch batch loss | 1.020972 | .644712 |
| Final epoch batch loss | .693140 | .128262 |
| Relative loss decrease | 32.1097% | 80.1055% |
| Own-input stored-BN singleton cross-entropy | .711237 | 2.833542 |
| Own-input training balanced accuracy | .50 | .6875 |
| Own-input recall / FPR | .625 / .625 | .625 / .25 |
| Original-input training balanced accuracy | .50 | .50 |
| Original-input recall / FPR | .625 / .625 | .25 / .25 |

Frozen objectives: factor 1 fails final-loss and own-input BA goals; factor 32
passes final-loss/decrease goals but fails the own-input BA >=.90 goal.
**Neither arm passes all learning objectives.** Factor 32's original-input
singleton cross-entropy is 4.776955. A smaller training-batch loss cannot hide
poor stored-BN inference or claim a usable real detector.

Factor 1's entire 20-epoch loss/update record exactly matches the prefix of
the prior frozen 50-epoch tiny sanity, supporting unchanged input/sampling/
optimization wiring. This is reproducibility, not new independent evidence.

Both numeric snapshots reload successfully; all BN counters are 160. All 24
own and 24 original singleton logits/metrics replay per model. All 18 independent
metadata-first NumPy float64 forward oracles pass. Maximum logit/score errors:
factor 1, 5.075934e-7 / 9.142459e-8; factor 32, 5.583214e-6 / 2.682047e-8.
Numerical gates stay separate from failed learning goals. The completed module
exits 2 because goals fail, not because the fit is incomplete or unavailable.

Local timing: factor 1 training/eval 369.22s, factor 32 538.18s; complete job
including reload/oracles 940.54s. Linux, 8 vCPU/15 GiB, two Torch threads and
OpenBLAS one thread; focused/full QA overlapped, so timings are engineering
observations, NOT controlled performance comparisons or CTF latency evidence.
No budget extensions or result-selected reruns occurred.

Evidence: `benchmarks/srnet-signal-strength-20261007.json`, SHA-256
`a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf`.
Model checksums: factor 1
`14dd74b8d27ac046438563376ef70a3b7d27b65e1f87bb1acd960d5e85d88413`;
factor 32 `5bfcf4a9b319a4e33ae22a562e9763bd15c06a186b6a4fa8aa3b006e7ac1895b`.
Original JPEG identities and derived tensor-byte hashes are distinct fields.
Weights/tensors/corpus remain local; prior failures and primary verdicts remain
unchanged. No artifacts execute and no host paths/secrets enter the report.

Interpretation: stronger paired input differences produce lower batch-stat
training loss here, but stored-BN singleton behavior remains inadequate and
the amplified model does not discriminate the original weak training inputs.
This does not establish generalization, a unique BN cause or an accuracy gain.
Next freeze a train-only batch-context/stored-normalization control before
further real fitting; no batch-stat diagnostic inference is deployable.

Explicit checkout-only replay with fresh outputs and optional research deps:

```text
OPENBLAS_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 3840s venv/bin/python -m steganography.research_srnet_signal --config .benchmark/srnet-tiny-sanity-20261007/config.json --out .benchmark/signal-fresh
```

The explicit module applies CPU/address/file/core limits; external wall timeout
is mandatory. Incomplete jobs may leave completed earlier-arm snapshots, but
never a complete two-arm control report. Direct service calls require caller
process isolation. Failed gates remain visible and no model is installed.

Verification: Python 3.11.14, 1,245 tests pass; total coverage 95.07%, new
signal/service code 135/136 statements covered (99.26%). Ruff, mypy (125
source files), diff checks and source/model checksum preservation pass. Torch
2.14.0+cpu and NumPy 2.4.6 were installed for the experiment. Wheel/sdist
corpus/model-free checks used installed setuptools 79.0.1 and wheel 0.48.0
with `--no-isolation`: isolated dependency installation stalled, so a clean
isolated build is not claimed. Python 3.12–3.14 and fresh full-Docker checks
were not run. The interrupted transient final-test session was rerun completely;
training was not rerun and the original experiment evidence was not changed.
