# Native RGB: learning trajectory from existing checkpoints — 2026-11-08

Status: **completed read-only CPU investigation**. Every saved train endpoint
remains near chance in global AUC. One seed develops stronger paired sensitivity,
but neither demonstrates successful training-set separation. A high-train/
low-development overfitting pattern is not observed at these checkpoints.

This extends the [initial diagnosis](PNG_BMP_NATIVE_RGB_DIAGNOSIS_20261107.md)
without changing the original `failed_learning_gate` result. The
[aggregate results](../benchmarks/png-bmp-native-checkpoint-trajectory-results-20261108.json)
bind all eight original states and private score arrays. The
[independent replay](../benchmarks/png-bmp-native-checkpoint-trajectory-verification-20261108.json)
checks 32,768 scalar AUC comparisons without Torch, native pixel reads or model
execution in the verifier.

## Measured train endpoints

| Model seed | Epoch | Updates | Global train AUC | Correct matched pairs | Source SD /pair-difference RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026102301 | 5 | 80 | .506592 | 55/64 | 180.00 |
| 2026102301 | 10 | 160 | .509766 | 56/64 | 43.93 |
| 2026102301 | 15 | 240 | .506836 | 47/64 | 48.63 |
| 2026102301 | 20 | 320 | .505615 | 51/64 | 105.48 |
| 2026102302 | 5 | 80 | .503418 | 43/64 | 261.90 |
| 2026102302 | 10 | 160 | .507568 | 57/64 | 127.41 |
| 2026102302 | 15 | 240 | .508789 | 60/64 | 74.98 |
| 2026102302 | 20 | 320 | .514648 | 61/64 | 23.78 |

Both seeds remain weak at the absolute classification/ranking task throughout
the saved trajectory. Seed 2026102302's within-original response becomes more
consistent while image-specific offsets become less dominant relative to that
response. Seed 2026102301 is not monotonic. These observations do not prove
that more steps will solve the task or that additional images will not help.
Matched-pair ordering needs the known counterpart and remains a diagnostic,
not standalone detector accuracy.

This historical trajectory varies training time but holds the dataset fixed.
It cannot independently estimate the effect of data size. The
[new metadata-only comparison](PNG_BMP_NATIVE_DATA_UPDATE_STUDY_20261108.md)
therefore fixes nested eight/64-original sets and compares 320/1,280 updates
at both sizes. That prospective proposal remains non-executable; it does not
relaunch any historical run or create a fitting permission.

## Scope and validation

The native numeric loader validates exact stored source/data/environment/order
bindings and checkpoint hashes. CPU model construction copies numeric arrays
for inference only; environment-bound resume is not called. Local Python
3.11.14 /NumPy 2.4.6 /Torch 2.14.0+cpu is explicitly different from historical
Colab. The two final-state GPU score arrays replay within 1.79e-7 /3.58e-7
maximum absolute difference and reproduce both AUCs exactly. Earlier endpoints
are newly measured portable CPU inferences, not relabelled historical GPU
observations.

The investigation reads the 64 already fitted train originals once to construct
and independently recover their deterministic control payloads, then keeps
the pairs in memory. Across eight states it performs **1,024 inference rows**,
with no backward call, optimizer creation/update, GPU query, network request
or checkpoint write. All model parameters and buffers stay unchanged. It reads
no development, reserved RGB or L pixels. The prospective small subset was
selected by metadata before these new checkpoint scores were computed.

Elapsed CPU-command service time is **78.357453 seconds**, CPU time 154.002797
seconds using two Torch threads, and kernel process peak RSS **701,538,304
bytes**. The process has 600-second CPU/wall hang guards and an eight-GiB DATA
segment limit. A cgroup resident-RAM limit is unavailable; the process peak is
not a historical GPU peak or a whole-tree measurement.

All eight original numeric hashes, the original closing receipt and all 522
historical software bindings verify after replay. The independent verifier
also checks the aggregate metrics, checkpoint schedule, unchanged-model flags
and absence of optimizer/backward/resume entry points in the private script.
No source role, cache acceptance, historical budget or failed result changes.
Private source/scope/log/scores are retained under
`.benchmark/png-bmp-native-study-preparation-20261108`.

This is document/evidence work over unchanged application/test inputs. Ruff,
252-file mypy and diff checks pass; the existing software gate remains 4,396
passes /two explicit CUDA skips and 97.225442% coverage. It is not reported as
a new full-suite execution. Independent metadata and numerical checks provide
the new evidence for this slice.
