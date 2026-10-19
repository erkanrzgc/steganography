# Native linear comparison results

Status: **unavailable**. This is an inspected same-source controlled pilot; production qualification is **unavailable**.

All source/seed/cell results and paired cluster intervals are in [portable results](../benchmarks/nonjpeg-linear-results-20261015.json).

Two FSDD validation speakers and eight validation originals per source make every interval underpowered. PNG/BMP and other derivatives do not add independent originals. Native deep models were not trained.

| Source | Feature version | Seed | Cells | AUC min–max | Recall min–max | FPR min–max |
| --- | --- | --- | ---: | ---: | ---: | ---: |

Bounds: 30-minute CPU allowance, two-GiB resident RAM, zero swap, one CPU thread and 32-MiB retained output. JPEG retains its separate frozen allowance. No automatic retry or deployment.

Real optimizer updates including CV: 0.
Read-only replay: unavailable.

See [the frozen protocol](NONJPEG_LINEAR_COMPARISON_PROTOCOL.md) and the unchanged [JPEG amendment](JPEG_REFINE_NUMERICS_PROTOCOL.md).

Read-only diagnosis confirms that the generic regular-file reader rejects the trusted kernel self-process symlink; a numeric own-PID probe passes the actual bounded-service checks. No real media or optimizer update was reached. Correcting that trusted-process path requires a separate source-bound revision; generic carrier symlink rejection remains required. See [diagnosis](../benchmarks/nonjpeg-linear-preflight-diagnosis-20261015.json).
