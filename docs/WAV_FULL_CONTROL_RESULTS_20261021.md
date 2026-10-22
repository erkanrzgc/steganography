# Full-exposure generated WAV control — 2026-10-21

This is the sole generated attempt bound by the v2 producer/evidence amendment.
It uses the unchanged v3 recipe and two original full seed schedules, without
real media, checkpoint, model download, automatic retry or extra JPEG time.
The unexecuted v1 binding, interrupted QA, transport failures and all prior
failed receipts remain preserved. Neither generated success nor incomplete
execution authorizes real AudioMNIST fitting or production qualification.

The full software gate passes 3,528 tests /two explicit CUDA skips, with
96.9345862603777% total coverage. Thirty-one runner tests cover both new
modules completely. Ruff, 218-file mypy and diff checks pass. The two local
CUDA skips remain skips; the owned WSL GPU separately passed an actual
zero-update generated forward/backward check with finite gradients,
unchanged parameters and the expected one-LSB input difference.

The independent corpus seed and exact bound metadata lengths reproduce the
frozen manifest: 600 generated WAVs, 36,335,714 bytes, 4,800 independent
payload checks, 20 train /10 development groups and no raw/decoded overlap.
No original WAV or reserved source is opened. The ephemeral generated corpus
is confined to the attempt's owned directory; no historical artifact is
removed. Original cumulative transfer accounting is not reset.

The controller starts at 22:18:43 +03 in an exclusive detached user unit.
The 3,600-second total ceiling includes startup, generation, both fits and
scoring; the worker reserves ten seconds for final evidence. Hard limits are
eight-GiB host memory, zero swap and four-GiB CUDA allocation, with a separate
15-second outer termination-only watchdog grace. Each seed must finish
20 epochs /1,140 updates /72,000 row presentations. Retained progress is a
confirmed update lower bound, not an exact count of unobserved work.

The immutable acceptance verifier independently replays the final receipt.
The combined learning rule requires training AUC >=0.60 and development AUC
>=0.70 for both sequential/scattered-40%-plane-0 cells, separately in both
seeds. AUC is not classification accuracy; generated AUC is not native
AudioMNIST performance. Missing full exposure is unavailable, not a learning
failure or a pass. No validation threshold is selected.

## Final outcome

**unavailable**. The local read-only acceptance replay matches the controller. The full two-seed acceptance contract is incomplete; this is neither a combined learning failure nor a pass. The final confirmed update lower bound is **1,937**, with **one** complete scored seed. The second seed retains **797/1,140** confirmed updates as a lower bound; it has no completed fit or scoring receipt. The worker reached its bound deadline. The attempt was not restarted. Real fitting and production qualification remain false.

| Measurement | Observed |
| --- | --- |
| Controller wall seconds | 3593.8358776789973 |
| Kernel host peak bytes | 2519113728 |
| CUDA allocator peak bytes | unavailable: no complete worker receipt |
| Swap bytes under hard zero cap | 0 |

Missing CUDA peak evidence is not replaced with a point-in-time `nvidia-smi` reading. The hard allocator/cgroup bounds remain required; partial resource observations cannot make an incomplete run pass.

The independent systemd monotonic interval is **3,593.861967 seconds**,
within 0.026090 seconds of the supervisor's startup-inclusive measurement.
Human-readable start/end wall timestamps differ by 3,418 seconds; that wall
clock discrepancy is not used for exposure or budget accounting, and its
cause is not established. A live read-only clock probe finds no meaningful
BOOTTIME/MONOTONIC offset. There is no established early-stop timer bug;
the worker reached its frozen monotonic deadline. See the
[independent time check](../benchmarks/wav-control-generated-time-check-20261021.json).

The read-only closeout re-verifies all **31 deployed source/protocol/legal
files**, totaling **668,941 bytes**. The nine retained attempt files total
**4,326 bytes**, largest **1,696 bytes**; the owned generated temporary corpus
was removed normally. Final local verification allocation is
**8,720,764,928 bytes**. See the
[closeout receipt](../benchmarks/wav-control-generated-closeout-20261021.json).
No background GPU work remains.

## Seed 20261016

Twenty epochs /1,140 updates /72,000 presentations; fit 2118.488 seconds. Training AUC over all recipes: **0.585100781**. The seed's learning rule **does not pass**.

| Method | Payload | Plane | Train AUC | Development AUC |
| --- | --- | --- | --- | --- |
| sequential | 5% | 0 | 0.511188 | 0.511025 |
| sequential | 5% | 1 | 0.538094 | 0.538575 |
| sequential | 40% | 0 | 0.559538 | 0.557425 |
| sequential | 40% | 1 | 0.736456 | 0.733100 |
| scattered | 5% | 0 | 0.507300 | 0.508150 |
| scattered | 5% | 1 | 0.530681 | 0.531900 |
| scattered | 40% | 0 | 0.551800 | 0.549725 |
| scattered | 40% | 1 | 0.745750 | 0.748575 |

The table rounds for display only; the machine receipt and acceptance comparison retain full precision. These are generated controls, not native AudioMNIST accuracy.

## Evidence and next step

See the [full machine receipt](../benchmarks/wav-control-generated-results-20261021.json), [pre-launch software/runtime binding](../benchmarks/wav-control-generated-launch-20261021-v2.json), [preserved readiness record](../benchmarks/wav-control-runner-readiness-20261021.json), [runner protocol](WAV_CONTROL_RUNNER_PROTOCOL_20261021.md) and [v2 evidence amendment](WAV_CONTROL_RUNNER_EVIDENCE_AMENDMENT_20261021.md). Private per-seed, progress, failure and resource observations remain retained; only hashes and bounded portable metrics are published.

The pre-launch verification directory used 8,716,374,016 allocated bytes, including both preserved software-verification attempts; this is verification scratch, separate from the bounded generated GPU corpus. No old dataset/checkpoint was deleted to make space. No new media was downloaded and historical transfer/time ledgers are not reset.

Further fitting requires a separately frozen intervention. First review throughput and the input/model representation against the fixed plane-0 learning target; do not select a new threshold, reduce exposure, extend this attempt or treat a strong plane-1 cell as a plane-0 pass. A new generated recipe must pass its full control before any separately approved real AudioMNIST fit.
