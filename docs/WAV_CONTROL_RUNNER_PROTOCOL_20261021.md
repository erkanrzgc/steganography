# Generated WAV full-exposure runner — 2026-10-21

This separately binds the producer/supervisor missing from the immutable
[acceptance contract](WAV_FULL_CONTROL_PROTOCOL_20261021.md). It authorizes one
generated-only control after the software checks and source/corpus/runtime
bindings pass. No real AudioMNIST fit, download, reserved-media access,
checkpoint, automatic restart or CPU substitute is authorized. Neither a
learning failure nor resource exhaustion authorizes another attempt.

## Corpus frozen before fitting

The source is only the frame metadata in the original, hash-bound preparation
receipt. Keep its per-role order: 400 train lengths and 200 development lengths,
without random length sampling. The independent corpus seed is `2026102101`,
not either optimizer seed. The quiet speech-like generator is the implementation
in `core.wav_control_runner.speech_like`, using NumPy 2.4.6. It retains the
legacy harmonic/envelope/dither construction, but freezes this new independent
seed and exact full length schedule before any fit.

Twenty train and ten development generated groups contain 20 originals each.
All 600 raw and decoded-PCM hashes must be unique; every generated cover has
all eight existing controlled recipes. Independent opaque-payload recovery
must pass 4,800 checks before fitting. The generated manifest is prepared once
on the owned execution host with zero updates and stored in
`benchmarks/wav-generated-corpus-20261021.json`. Its canonical bytes and every
generated WAV/PCM hash are frozen in the runner protocol. The worker must
regenerate the identical manifest before using the shared trainer. No original
source path is passed to the generator or scorer.

The unchanged v3 recipe, seeds `20261016` and `20261017`, permutations,
class-balanced weights, 20 epochs, 64-row virtual batches and final 16-row batch
come from the shared trainer. Each seed must complete 72,000 presentations and
1,140 updates. Both use the same generated corpus. There is no warm-up,
reduced-exposure trial, validation sweep, ablation or early selection.

Training AUC over all eight recipes must be >=0.60, and both sequential-40%-p0
and scattered-40%-p0 development AUCs >=0.70, separately in each seed. All eight
development cells are retained. AUC uses fixed score direction and half credit
for ties, with full precision and no rounding before the rule. Infer in eval
mode; retain only score digests and numeric metrics, never model weights.

## Runtime and one-shot supervision

Use the owned RTX 5060 Laptop GPU, Torch 2.14.0+cu130, NumPy 2.4.6, FP32,
two CPU threads, deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
AMP and TF32 stay off. The four-GiB CUDA allocator fraction is set before
network allocation. The worker and supervisor both verify the new source
closure; the worker rechecks it after inference. Code/model sources in the
parent acceptance closure remain unchanged.

A detached user systemd unit uses `MemoryMax=8G`, `MemorySwapMax=0`,
`CPUQuota=200%`, `Restart=no`, `KillMode=control-group`,
`RuntimeMaxSec=3615` and `TimeoutStopSec=5`. The final 15 seconds are only an
outer termination/failed-evidence grace, never eligible exposure. The process
supervisor's 3,600-second wall timer includes interpreter startup/imports using
the numeric own-PID kernel start time. It reserves ten of those seconds for
termination, resource reading and receipt assembly: the generated worker's
deadline is process start +3,590 seconds. This changes no exposure count; an
unfinished fit is unavailable, not a reduced-exposure learning result.

The supervisor verifies its own cgroup's <=eight-GiB memory bound and hard
zero-swap cap before creating an attempt. It starts one process group, waits,
and terminates/reaps it at the wall ceiling. TERM escalates to KILL after one
second. No restart/resume occurs. The worker also verifies the cgroup and
creates an exclusive `worker-started.json` seal before initialization, so the
same workspace cannot be reused for another fit.

Generated WAVs total at most 512 MiB and are confined to an owned temporary
directory. JSON/log files have a one-MiB per-file limit; no derivative WAVs,
features or checkpoint are retained. Normal temporary-corpus cleanup never
touches historical datasets or snapshots. Each completed optimizer callback
atomically replaces `progress.json`. Its count is a confirmed lower bound;
a forced interruption can lose the last callback, so never claim an exact
unobserved update total. Keep all progress, stderr, failure and attempt records.

The supervisor attaches kernel `memory.peak`, current swap under the verified
zero-swap cap, CUDA peak, and measured wall time to the worker result. It then
calls the immutable acceptance verifier and saves separate result/decision
files. Every outcome keeps `real_fit_authorized=false` and
`production_qualified=false`. Incomplete evidence is unavailable; a completed
below-threshold result is a learning failure. A synthetic pass is not native
AudioMNIST accuracy or production qualification.

## Deployment and validation

Copy only the enumerated regular source files, parent contract, metadata and
generated manifest to a fresh exclusive source bundle. Never copy real media,
`.env`, checkpoints or an entire historical workspace. Invoke the adapter as
a direct script with `PYTHONPATH` set to that bundle; this avoids importing
the unrelated application package initializer from an older editable install.
Use the existing CUDA virtual environment without downloading anything.

Before the one bounded GPU attempt: Ruff, full mypy, focused generated-only
tests with >=95% new-code coverage, the full unchanged >=90% coverage gate,
bundle imports, exact runtime/device checks, resource-only cgroup probe and
all source/corpus hashes must pass. Record the initial QA fixture failure:
an eight-bit payload in a 160-frame toy can already match the original plane,
so the existing distortion oracle correctly rejected it. Use 640-frame test
fixtures without relaxing that oracle or changing the real metadata lengths.
This QA failure contains no GPU/real-data updates and consumes no control
attempt.

Freeze the launch unit, command, bundle, runtime, verification and protocol
hashes before starting the single attempt. Keep the PC awake while it runs;
the detached unit survives SSH transport closure while WSL itself stays up.
The generated-control budget is separate from completed JPEG experiments and
does not reset their six-hour or storage ledgers.
