# Train-only gradient/null-control diagnosis — 2026-10-07

Preregistered in `SRNET_GRADIENT_PROTOCOL.md`, commit `7c365fc`, before probes.
No training, optimizer updates, saved model, validation loading or primary
detector changes. This diagnoses training behavior, not detector accuracy.

The unchanged metadata-only 24-row train subset yields four deduplicated
epoch-zero four-row batches covering all six declared source/Q/method cells.
Each is probed with a fresh seeded SRNet and the failed 400-update tiny model;
one generated strong-signal batch is additionally probed with the fresh model.
Nine complete true/null probes. BN uses training batch statistics with tracking
disabled, never deployed inference; every parameter/buffer/flag is restored.
Existing gradients, caller RNG/threads and inputs are preserved by tests.

| Descriptive diagnostic | Fresh real batches | Failed-tiny real batches | Generated strong batch |
| --- | ---: | ---: | ---: |
| True cross-entropy | .692750–.693466 | .692996–.693142 | .617195 |
| Global parameter gradient L2 | .058901–.148959 | .007001–.025378 | 7.185203 |
| Front-block gradient L2 | .048536–.093582 | .001519–.006842 | 3.659247 |
| Input positive/cover difference RMS* | .050253–.429909 | same | 36.878178 |

*RMS averages all four rows, including the two zero cover differences; it is
not the RMS over only positive rows. All rows are raw unrounded float32 Y for
real data; generated tensors are not JPEG or steganography. These correlated
training probes are not independent samples or significance tests.

Real positive/cover tensors differ; true gradients and input gradients are
nonzero, including in early layers. Null means positive rows replaced by their
matched covers with unchanged [0,1,0,1] targets, a deliberately contradictory
diagnostic only. Real/null gradients differ, but this does not prove useful
learnability. Classifier analytic versus central-difference directional checks
pass 9/9, max absolute derivative error 3.634503e-5. This checks only one
classifier parameter direction per batch, not all-layer or input derivatives.

The failed trained model's front gradients are smaller than the fresh model's
on these four batches. This motivates early-layer/optimization and signal-scale
investigation; it does not identify a unique cause, prove vanishing gradients
or justify automatic learning-rate/BN/preprocessing changes. The generated
comparison is content-confounded and cannot estimate real detection accuracy.
Previous real-data balanced accuracy .50 and all failed held-out gates remain.

Completed in 25.10 seconds locally (Linux, 8 vCPU/15 GiB, two Torch threads,
OpenBLAS one thread). A second independently launched complete run reproduced
every probe, selection, schedule and provenance field exactly, excluding elapsed
time. No original model/data changed. Public evidence:
`benchmarks/srnet-gradient-diagnostic-20261007.json`, SHA-256
`1921a57a86c506ff1254842d2efe75f9db1b0649b6f45a736420a66e736f5486`.
No raw tensors/images or weights are published. Source hashes are recorded.

Explicit checkout-only replay, fresh report path and optional research deps:

```text
OPENBLAS_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 300s venv/bin/python -m steganography.research_srnet_gradients --config .benchmark/srnet-tiny-sanity-20261007/config.json --model .benchmark/srnet-tiny-sanity-20261007/result/model.npz --out .benchmark/gradient-fresh.json
```

Configuration is the exact four-field tiny-sanity manifest/train-cache contract;
the model checksum and protocol are fixed. No validation or optimizer settings
are accepted. Hard CPU/address/file/core bounds are applied by the module;
direct shared-service invocation requires caller-imposed process isolation.
Incomplete or unavailable work exits 2 and is not usable as a complete result.

Next freeze a train-only signal-strength/early-layer learning control before
more real-corpus fitting. Do not select settings using inspected validation or
claim accuracy gains from gradient magnitudes or generated training success.

Verification: Python 3.11.14, 1,218 tests pass, total coverage 95.02%; new
gradient/service code 173/176 statements covered (98.30%). Ruff, mypy
(123 source files), diff checks and corpus/model-free wheel/sdist checks pass.
Python 3.12–3.14 and a fresh full-Docker run were not verified for this slice.
