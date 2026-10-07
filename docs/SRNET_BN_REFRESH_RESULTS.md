# Train-only cumulative BN refresh/context control — 2026-10-07

Frozen protocol commit `71f3569` precedes execution. Both original factor 1/32
models were replayed against checksum-pinned signal-strength evidence before
any clone refresh. No optimizer step, learned-parameter change, validation
loading, checkpoint selection or deployment. All previous failures remain.

| Unique 24-row stored-BN training metrics | Factor 1 | Factor 32 (artificial) |
| --- | ---: | ---: |
| Baseline own-input BA | .50 | .6875 |
| Refreshed own-input BA | .50 | .625 |
| Baseline own-input CE | .711237 | 2.833542 |
| Refreshed own-input CE | .711511 | 3.102798 |
| Refreshed original-input BA | .50 | .50 |
| Refreshed original-input recall / FPR | .50 / .50 | .25 / .25 |

Both frozen goals fail in both arms: own-input BA >=.90 and CE below baseline.
The simple cumulative refresh does not repair this case; it worsens amplified
own-input discrimination. Original weak training-input BA remains chance-level.
Refreshed original-input CE: .711511 / 5.095433. Neither clone is installed.

Read-only final-epoch four-row context contrast, **32 presented training rows**:

| Diagnostic batch context | Factor 1 own/original | Factor 32 own | Factor 32 original |
| --- | ---: | ---: | ---: |
| Stored-BN presented-row BA | .50 | .6875 | .50 |
| Batch-stat diagnostic BA | .50 | .96875 | .53125 |
| Stored-BN presented-row CE | .697893 | 3.654200 | 5.111760 |
| Batch-stat diagnostic CE | .693138 | .121750 | 1.164749 |

These rows repeat matched covers; they are correlated and not a held-out test.
Presented-row loss is not the same weighting as unique-row loss in the first
table. Batch-stat inference depends on neighboring labeled training examples
and is **not deployed or a generalization result**. Its .96875 amplified score
cannot hide .50 original-input BA or failed ordinary singleton goals.

Refresh resets only 26 BN layers' running mean/variance/counters on a clone,
then performs exactly one eight-batch final-epoch pass on its own inputs with
momentum None. PyTorch documents cumulative averaging for this setting and
separate training versus running variance estimators. See the
[PyTorch 2.14 BatchNorm2d reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html).
Averaged within-batch variances do not include between-batch mean variance;
this is not an exact population estimator. No alternate pass/batch/settings
were tried based on results.

Original models/parameters/flags/RNG/threads/inputs remain unchanged. Each
clone changes exactly 78 BN buffers, source counters remain 160, clone counters
are 8, optimizer steps zero. A separate read-only serialized-source/clone
comparison confirms every non-BN-stat array bit-identical after reload.
Existing gradient and hook preservation and adversarial mutation/exception
restoration are tested. All refreshed own/original singleton logits/metrics
reload, and all 18 independent NumPy float64 forward oracles pass. Maximum
logit/score errors: factor 1, 5.963728e-7 / 1.915244e-7;
factor 32, 6.256478e-6 / 5.511948e-7.

Local complete runtime 96.33s, Linux 8 vCPU/15 GiB, two Torch forward threads,
OpenBLAS one thread. This is engineering timing, not a controlled performance
benchmark. CLI exits 2 because complete in-sample objectives fail; numerical
gates pass and the job is not incomplete/unavailable.

Evidence: `benchmarks/srnet-bn-refresh-20261007.json`, SHA-256
`db9d0be00d87b9a2d6bd230fa41e1ebfd8eca62aab33afc63e0ab0d329cf4a73`.
Local clone SHA-256: factor 1
`dc14f74dba9ed2f23e30e1a17a703200f78f0ac9e13abc955eb037840c65ec9f`;
factor 32 `0324ad97964a56812ae7daae628db47d10ed4c33dc3ef49799505c823f4173c4`.
No tensors, weights, host paths or secrets are published. Source and baseline
hashes bind the experiment. No automatic download or model registration occurs.

Interpretation: amplified representations show sensitivity to batch context,
but this fixed refresh is insufficient. This does not identify a unique BN
cause, prove weak-signal learnability, or improve real held-out accuracy.
Next preregister larger/more diverse training-batch context with explicit
compute/exposure accounting; do not replace primary inference with batch stats.

Explicit checkout-only replay, fresh output and optional research dependencies:

```text
OPENBLAS_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 420s venv/bin/python -m steganography.research_srnet_bn_refresh --config .benchmark/srnet-tiny-sanity-20261007/config.json --model-dir .benchmark/srnet-signal-strength-20261007 --out .benchmark/bn-refresh-fresh
```

The module applies CPU/address/file/core limits; external wall timeout is required.
Direct shared-service callers supply process isolation. Incomplete jobs may
retain earlier-arm artifacts but never a complete two-arm report.

Verification on Python 3.11.14: 1,268 tests pass, total coverage 95.15%; the
two new implementation modules cover 205/206 statements (99.51%). Ruff,
full mypy checking, and `git diff --check` pass. Local wheel and sdist builds
with `--no-isolation` include both research modules and contain no model or
corpus artifacts. Python 3.12–3.14 and full Docker were not verified locally.
