# Real-I/O CUDA timing gate

Frozen before execution: `benchmarks/jpeg-real-timing-probe-protocol-20261010.json`
(commit `4aa68ac`). This is a timing-only transient optimization run, not the
normalization intervention, a full learning experiment or an accuracy result.
It never rents hardware, downloads data, uploads a corpus or publishes a model.

`core/jpeg_timing_probe.py` owns the bounded reader and numerical engine adapter.
The kit manifest and independent whole-kit audit are pinned to their existing
SHA-256 identities. Before optimization, verify all 480 JPEG bytes, the 120 MiB
finite/range-bounded tensor and the unchanged three-origin schedule. Keep one
regular-file handle open and reject mutation using file identity/size/timestamp
checks. Fetch exactly four rows (1 MiB), never mmap/load the entire tensor into
host or GPU RAM. Rehash the complete tensor after the run. No validation cache,
scores, thresholds or model-selection feedback enter this service.

Execute 66 real-data updates through the **unchanged** shared SRNet/BatchNorm/
Adamax engine with seed 20261010, strict IEEE FP32, no AMP/TF32 and no CPU
fallback. Use the first 66 four-row batches of the 192-update timing-kit epoch.
Discard the first 18 inter-fetch intervals; retain 47 positive finite intervals.
The engine's finite-gradient, loss and numeric-state checks synchronize prior
optimizer work before the next fetch, so intervals include real file reads,
host/device transfer and those production checks. They are not kernel-only
queued timings. The last update is still completed/checked, but does not supply
an additional interval. Temporary updated weights/optimizer state are discarded.
`real_model_trained: true` on a completed GPU run means transient updates actually
occurred; `production_model_trained: false` and no saved weights make its limited
purpose explicit. No completed GPU report exists yet.

The fixed projection is `p95(47 warmed intervals) * 4932 * 1.25` optimizer seconds
per prospective full-corpus epoch. Report median/p95 and the raw intervals.
There is **no default epoch count or hourly price**. Setup/upload, validation,
checkpoint and disk costs are excluded and must be measured/quoted separately.
The small tensor has been read in full for verification, so OS cache is likely
warm. Full-corpus cold I/O, thermal endurance and full three-source reader/trainer
integration remain unverified. This projection is not a duration guarantee, a
bill or authorization to start a paid full run.

## Explicit command

Run from a checked-out, fully provisioned repository on the chosen GPU host:

```sh
systemd-run --user --scope -p MemoryMax=8G \
  .venv-gpu/bin/python -m steganography.research_jpeg_timing_probe \
  --root .benchmark/jpeg-real-timing-20261010 \
  --audit benchmarks/jpeg-real-timing-independent-audit-20261010.json \
  --out .benchmark/jpeg-real-timing-20261010/gpu-timing-01.json
```

The example assumes that host's GPU environment is named `.venv-gpu`; this
local Kali environment instead uses `venv` with CPU-only Torch. Do not install a
GPU driver inside WSL to make this command work. Transfer/review private data
permissions first; the service itself performs no transfer. On hosts without
systemd, an equivalent real cgroup-v2 8 GiB resident limit is required; an
environment-variable promise is not a RAM limit.

The frontend starts a fixed isolated worker with a sanitized environment, a
180-second job deadline, 210-second process timeout, existing 4 GiB GPU allocator
limit, kernel-enforced 8 GiB host limit, bounded stdout and 1 MiB output file limit.
Bind parent/worker source hashes, returned report hash, dataset/audit identities,
CUDA device, update count and independently recomputed projection. Reject
overwrite/symlink output. Timeout/incomplete output is unusable, not a partial
success. Missing CUDA/RAM prerequisites produce `unavailable` and exit code 2,
without real data access, CPU fallback or a duration/cost estimate.

## Current local evidence

`benchmarks/jpeg-real-timing-reader-readiness-20261010.json`: all 480 real rows
verified, all 66 scheduled batches read through the new shared reader, maximum
fetch 1 MiB and pre/post tensor identity passed. Ordered fetch-byte SHA-256
`36ac51c26668d48d92eb45a6a1319726a6d326bf5ada64bc9fc7f7ace3331672`.
Zero optimizer updates; this is reader readiness, not GPU timing.

`benchmarks/jpeg-real-timing-local-unavailable-20261010.json`: actual isolated
local invocation returned `unavailable`; Torch 2.14.0+cpu and no CUDA visible.
The previously approved laptop reverse-tunnel port 22240 is not listening.
No physical timing, new production fit, deployment, cloud upload/resource or
charge was initiated. Next execution requires restored laptop GPU access or a
fresh quoted RunPod choice with an explicit maximum total spending limit.

## Regression environment failure

The first full test run did not complete: retained generated fixtures filled
the 7.8 GiB RAM-backed temporary filesystem. A targeted diagnostic reproduced
`OSError` errno 28 in an existing SRNet numeric-model fixture. Preserve this as
`benchmarks/jpeg-real-timing-regression-tmpfs-failure-20261010.json`, not a pass.
Inactive task-owned test directories were moved intact into a private disk-backed
rescue directory. New 120 MiB reader fixtures now clean up only their own
reproducible tensor files at teardown. The full retry uses a fresh private
disk-backed pytest base directory, without changing application or parser limits.

## Final verification

The complete disk-backed retry passes: 1,979 tests, one actual-CUDA hardware
skip (not a pass), 32 existing ONNX deprecation warnings, 478.21 seconds. Total
coverage 95.81%; new reader/profile service and isolated frontend each 99%.
All 44 focused reader/worker/portable-evidence tests pass separately. Ruff,
mypy, dependency and whitespace checks pass. Final wheel/sdist builds have
162/329 members, respectively, and contain no private media/tensors/weights.
Verified on Python 3.11.14 only. Physical CUDA timing, other Python versions and
Docker are not verified; no measured GPU cost or accuracy improvement exists.
