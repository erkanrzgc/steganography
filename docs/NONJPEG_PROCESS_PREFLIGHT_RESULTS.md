# Native process preflight amendment results

Execution: **completed**. Production qualification: **unavailable**; no deployment.

Original failed attempt remains immutable. The amended comparison shares its remaining budget with read-only replay.

See [portable results](../benchmarks/nonjpeg-process-results-20261015.json) and [the frozen amendment](NONJPEG_PROCESS_PREFLIGHT_PROTOCOL.md).

| Source | Features | Seed | Cells | AUC min–max | Recall min–max | FPR min–max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BOWS2 | spatial-parity-residual-v1 | 20261016 | 16 | 0.641–1.000 | 0.000–0.875 | 0.000–0.000 |
| BOWS2 | spatial-parity-residual-v1 | 20261017 | 16 | 0.609–1.000 | 0.000–1.000 | 0.000–0.000 |
| BOWS2 | spatial-local-parity-v2 | 20261016 | 16 | 0.609–1.000 | 0.125–0.875 | 0.000–0.000 |
| BOWS2 | spatial-local-parity-v2 | 20261017 | 16 | 0.672–0.969 | 0.000–0.750 | 0.000–0.000 |
| DIV2K | spatial-parity-residual-v1 | 20261016 | 64 | 0.547–0.984 | 0.250–1.000 | 0.250–0.250 |
| DIV2K | spatial-parity-residual-v1 | 20261017 | 64 | 0.547–0.812 | 0.750–0.875 | 0.750–0.750 |
| DIV2K | spatial-local-parity-v2 | 20261016 | 64 | 0.578–0.969 | 0.250–1.000 | 0.250–0.250 |
| DIV2K | spatial-local-parity-v2 | 20261017 | 64 | 0.578–0.781 | 0.375–0.875 | 0.375–0.375 |
| DIV2K | spatial-rgb-parity-v2 | 20261016 | 64 | 0.562–1.000 | 0.125–1.000 | 0.125–0.125 |
| DIV2K | spatial-rgb-parity-v2 | 20261017 | 64 | 0.562–0.766 | 0.375–0.875 | 0.375–0.375 |
| ESC-50 | audio-pcm16-parity-residual-v1 | 20261016 | 8 | 0.500–0.547 | 0.750–0.750 | 0.750–0.750 |
| ESC-50 | audio-pcm16-parity-residual-v1 | 20261017 | 8 | 0.484–0.531 | 0.000–0.000 | 0.000–0.000 |
| FSDD-v1.0.10 | audio-pcm16-parity-residual-v1 | 20261016 | 8 | 0.422–0.594 | 0.000–0.000 | 0.000–0.000 |
| FSDD-v1.0.10 | audio-pcm16-parity-residual-v1 | 20261017 | 8 | 0.359–0.656 | 0.000–0.000 | 0.000–0.000 |

Real optimizer updates including grouped CV: 20200.
Read-only replay: passed.

Every interval is underpowered: eight validation originals per source and only two FSDD validation speakers. Source/seed/cell results and paired intervals are preserved individually in JSON. No native deep models were trained. JPEG fitting was not restarted.

## Verification and resource accounting

The unchanged full gate passes **3,233 tests /two actual-CUDA skips**, with **96.75%** total coverage and **100%** for both amended modules. Ruff and 206-file mypy pass. Focused testing passes 22 cases, including two actual kernel-service cases; initial scoped harness setup failures are retained and are not counted as passes. Python 3.11.14 is verified locally; the 3.12–3.14 CI matrix was not executed here.

Fitting and read-only replay used 19.80 seconds; the cumulative charge is 40.80 seconds including the prior 21-second debit. Retained amended outputs total 2,739,143 bytes, or 2,804,679 bytes including the 64-KiB prior reservation. Ten kernel observations recorded a 343,969,792-byte memory peak and no OOM events. CPU usage is a sampled lower bound, not a fabricated final service counter; exited-unit accounting properties are unavailable. See [resource evidence](../benchmarks/nonjpeg-process-resources-20261015.json).

## Interpretation

WAV balanced accuracy is 0.50 in every source/seed/cell, including the grouped train-only threshold; there is no measured detection improvement at that operating point. Spatial extensions have both positive and negative AUC differences against their matched 684-value reference in every source/version/seed arm. High-AUC cells do not establish a stable cross-seed gain. Every interval remains exploratory and underpowered; no feature version is qualified or promoted.

Next work requires a separately frozen model/data/exposure protocol. Native deep models remain generated-only skeletons, and the missing independently licensed native-source pilots remain unavailable. Do not blindly repeat the current fit or retune using these validation cells.
