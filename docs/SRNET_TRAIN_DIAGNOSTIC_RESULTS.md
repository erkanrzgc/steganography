# Train-only BatchNorm contrast — 2026-10-07

Frozen protocol `SRNET_TRAIN_DIAGNOSTIC_PROTOCOL.md`, commit `2cbca3a` before
probes; [portable aggregate evidence](../benchmarks/srnet-train-diagnostic-20261007.json).
This targeted diagnosis follows the failed pilot, not a new independent test.

## Observation

Selected the first eight distinct stegos per BOSS source/quality/method cell
in the frozen epoch-zero training order: 32 matched pairs, 64 forward inputs
per mode, all 26 BN layers. Only the train cache was loaded. The paired
contrast took 23.11 seconds; weights, BN buffers/counters, module flags and
original numeric artifact stayed unchanged. Disk SHA still matches the pilot.

| Training cell | Native paired CE | Batch-stat paired CE | Native saturated scores | Batch-stat saturated scores |
| --- | --- | --- | --- | --- |
| Q75 JUNIWARD | 13.7578 | 0.5323 | 14/16 | 0/16 |
| Q75 UERD | 19.2381 | 0.4663 | 12/16 | 0/16 |
| Q95 JUNIWARD | 51.3349 | 0.6748 | 14/16 | 0/16 |
| Q95 UERD | 46.2417 | 0.6566 | 16/16 | 0/16 |

Native eval gives the same decision to both members of all 32 training pairs:
in-sample label accuracy 50%. The BN batch-stat contrast gives different
decisions to both members of all 32 pairs, with in-sample label accuracy 100%
in Q75 and 75% in Q95. **These are selected training examples, not held-out
accuracy, and matched pairs supply joint information unavailable for an
arbitrary single suspect file. Do not advertise these figures as detection
performance or enable batch-stat inference in the primary analyzer.**

The last BN layer's median standardized mean shift is 0.6915 in native eval
versus 0.0354 with batch statistics; median variance ratios 0.4567 versus
1.0688. The first layer's inputs are identical in both modes, as expected.
All per-layer aggregates are retained. This establishes substantial sensitivity
to normalization mode on this training subset and a train/eval mismatch. It
does not establish BN as the sole cause, or that changing stored statistics
alone would solve it. Paired-batch context, learning duration and unseen-source
generalization remain unresolved. Original six failed validation cells stand.

## Implementation and safeguards

`research srnet-diagnose --config FILE --out FRESH_JSON` uses the same complete
training/fit/card/cache bindings as fitting. Config is the fit input's six
path/hash fields plus `model_dir` and `card_sha256`; validation arguments are
rejected. Training scope and saved BN update counters are checked. Probe
selection uses metadata and fixed sampler order, never observed scores.

The core contrast disables gradients and BN tracking, snapshots/restores the
exact numeric state, preserves flags and removes its hooks in `finally`.
Nonfinite input/statistics/output, unexpected layer coverage, attempted state
mutation, wrong scope/hash/counters or a deadline fail without completed output.
Math threads are restored. A mixed/train model is not accepted as an ordinary
eval model. Diagnostic probabilities are uncalibrated, not primary scores.

The direct service has a cooperative 180-second probe deadline. Actual CLI
execution used `timeout --signal=TERM --kill-after=5s 300s`, OPENBLAS one thread
and Torch two threads. This is not a filesystem/network sandbox. The separate
`scripts/audit-srnet-train-diagnostics.py` binds the frozen protocol/identities,
verifies unique complete cells and aggregates local raw probes without copying
paths, samples or weights into public reports. No downloads or model updates.

## Next controlled experiment

Do not switch evaluation to paired BN to manufacture a better result. Freeze
a separate combined-source training control with multiple independent image
lineages per minibatch and unchanged single-file eval semantics. Compare
train-only BN distributions/learning curves before reusing development
validation. Any normalization/statistic fitting must use training only and
produce a separately versioned experimental card; retain the original model,
protocol and failed results. Full ONNX and genuinely independent sources are
still needed before qualification/deployment.

Local verification: Python 3.11.14, 1,139 tests pass, total coverage 94.86%;
new core/service 125/125 statements covered. Ruff, mypy (115 files), whitespace
and model/data-free wheel/sdist checks pass. Generated mutation/exception/
nonfinite/deadline and train-only binding tests accompany the real probes.
Other supported Python versions and fresh full Docker were not rerun locally.
