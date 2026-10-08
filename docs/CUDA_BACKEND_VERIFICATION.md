# CUDA software verification — actual hardware unavailable

CUDA software/preflight protocol committed before the local unavailable probe
at `d71680b8d5e9095ce91de430752836202ee0e3b5`. The frozen
[generated protocol](CUDA_GENERATED_PROBE_PROTOCOL.md) SHA-256 is
`1dc0a480136f9a2ace68fa18e345e23762973abee5b3ea030d8b958c55f9b561`.
The [portable unavailable record](../benchmarks/cuda-kali-unavailable-20261008.json)
is an exact copy of the local response, SHA-256
`fa312ee53fdbcfa90e8656e8bca2541e1ea74010d596dca67c333f5808e67373`.

The fixed isolated probe returns exit 2 / **unavailable**, not success, on the
Kali/VMware development host. Prerequisites for bounded CUDA execution are not
available there; no complete hardware probe, real dataset access, real model
fitting, CPU fallback or deployed detector change. The parent records hashes
of its 14 source dependencies before the attempt. This record does not verify
Windows WSL, driver compatibility, actual GPU memory/throughput or runtime math.
The user's Windows 11 / RTX 5060 Laptop is reported, not remotely inspected.

Final local verification, Python 3.11.14 / Torch 2.14.0+cpu:

- **1,597 tests pass, one live CUDA test explicitly skipped**, 32 warnings,
  229.26s; total statement coverage **95.43%**. Missing CUDA is not counted as
  a passed hardware test. Python 3.12–3.14, full GPU/Docker E2E remain unverified.
- Focused CPU/generated/emulated controls: 166 pass, one live CUDA skip;
  CUDA policy 98.92%, generated probe 99.02%, streaming orchestration 98.21%.
  API emulation/CPU-redirected transfers prove control flow, not GPU kernels,
  real CUDA allocator limits or CPU/GPU numerical agreement on physical hardware.
- Ruff, mypy (138 files), `git diff --check`, wheel and sdist build pass.
  Archive member inspection finds no raw benchmark/credential corpus or
  `.f32`/`.pt`/`.pth`/`.onnx` tensors/weights. The base wheel stays model-free;
  existing CPU paths, source scheduling, serialization and primary verdicts
  remain unchanged by default. Windows installation was not performed here.

Tests cover explicit no-fallback device selection, unsupported GPU/build/API,
missing deterministic environment, allocator budget/flags/device restoration,
cgroup ancestor bounds (including a root without `memory.max`), unbounded/
malformed/symlink/missing memory constraints, generated checksum/timeout/source/
parity failure paths, unchanged CPU plans, scoped resource policies and explicit
worker unavailability. Previously verified real-data failures stay published.

Next follow the [Windows GPU setup guide](WINDOWS_GPU_TRAINING.md): observe
Windows `wsl --status`/`nvidia-smi` first, then configure a separate WSL2
environment and a kernel-bounded scope, and run the actual generated probe.
Do not bypass guards, overwrite user WSL configuration, install a Linux GPU
driver inside WSL, auto-upload restricted corpora or silently extend fit budgets.
Adequate real learning and independent held-out qualification need a separate
preregistered protocol after successful hardware preflight. No accuracy gain
or support qualification is claimed by this engineering slice.
