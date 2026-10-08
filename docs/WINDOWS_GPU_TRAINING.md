# Windows 11 / RTX 5060 Laptop: local WSL2 training

Status: CUDA software path implemented; **actual GPU execution unverified**
in the current Kali/VMware CPU-only development environment. No new real model
or accuracy improvement. Kali and its working Python environment stay intact.
Run GPU jobs in Ubuntu/WSL2 on the Windows host, not inside the Kali VM.
See [local software verification and explicit CUDA unavailability](CUDA_BACKEND_VERIFICATION.md).

## 1. Check the Windows host first

In **Windows PowerShell**, not the Kali terminal:

```powershell
wsl --status
wsl --list --verbose
nvidia-smi
```

Send the output before changing an existing WSL installation. Do not uninstall
or unregister an existing distribution. Windows 11 supports CUDA workloads in
WSL2; this does not mean the VMware guest currently exposes the host GPU.
[Microsoft GPU guidance](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute).

If WSL/Ubuntu is not installed, the user can explicitly run, in administrator
PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

This installs a distribution, may enable Windows features and require a reboot;
save work first. It is not a read-only diagnostic. Do not run it again over an
existing setup without inspecting the first commands. [Microsoft installation](https://learn.microsoft.com/en-us/windows/wsl/install).

Use the supported NVIDIA **Windows** driver. Do not install an NVIDIA Linux
driver inside WSL: the Windows driver provides that interface.
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Blackwell needs a compatible CUDA-enabled PyTorch build, not our Kali CPU
wheel. CUDA 13.0+ builds need an appropriate Windows driver (the PyTorch 2.12
release specifies 580.88 or newer). Check the actual installed driver first.
[PyTorch compatibility](https://pytorch.org/blog/pytorch-2-12-release-blog/).

## 2. Separate Ubuntu environment — explicit user installation

After WSL2, Ubuntu and the host driver are verified, prepare a **new** checkout
and virtual environment inside Ubuntu. These commands download software;
they do not download a model or dataset and are not run automatically here.
Use a fresh destination or an existing checkout you intentionally select:

```sh
git clone https://github.com/erkanrzgc/steganography.git steganography-gpu
cd steganography-gpu
python3 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv-gpu/bin/python -m pip install -e '.[jpeg-sim]' 'numpy==2.4.6' 'jpeglib==1.0.2'
```

If Git or Python venv support is missing, install the Ubuntu packages only
after checking that error; do not alter Kali's venv. The CUDA 13.0 wheel is
published in the [official PyTorch index](https://download.pytorch.org/whl/cu130/torch/),
not a nightly/custom third-party build. NumPy/jpeglib versions remain bound to
the already audited float-cache decoder contract, rather than upgrading it
silently. The precision API, compiled GPU
architecture, actual driver and runtime are checked before training. WSL
installation and these commands have **not** been verified on this user's host.

## 3. Generated preflight before any real corpus

CUDA reserves virtual address ranges; the CPU worker's 8 GiB `RLIMIT_AS` is
not an appropriate GPU resident-memory bound. The new CUDA worker instead
requires a **kernel cgroup-v2 RAM limit <=8 GiB**, inherited from a scope or
container; it does not silently remove memory limits. Systemd is available
in supported WSL setups. [Microsoft systemd guidance](https://learn.microsoft.com/en-us/windows/wsl/systemd),
[systemd memory control](https://github.com/systemd/systemd/blob/main/man/systemd.resource-control.xml).

If the user's systemd manager has the memory controller delegated, run:

```sh
systemd-run --user --scope --property=MemoryMax=8G \
  .venv-gpu/bin/python -m steganography.research_cuda_probe \
  --out .benchmark/cuda-host-probe-01.json
```

Use a fresh output each time. If the scope/controller is unavailable, stop and
inspect the error; do not disable the guard or retry an unlimited worker.
A separately configured bounded container is an alternative, not an automatic
Docker installation. No global WSL memory configuration is overwritten here.

This fixed isolated worker has a 60s internal deadline, 90s parent hard timeout,
120s CPU/1 MiB file limits and no core dump. It performs **one generated**
four-row FP32 optimizer update with the shared engine, then compares GPU and
CPU single-image logits using exactly the same returned weights: absolute and
relative tolerance 1e-4. It records allocator peaks and duration. This is not
an independent NumPy oracle, real-data fit, accuracy benchmark or speedup claim.

CUDA execution is explicit `cuda:0`, float32 IEEE, TF32/AMP disabled,
deterministic algorithms required. The fixed worker sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Library callers
must supply the same environment and hard parent resource bounds. Thread/RNG,
precision, deterministic, current-device and allocator-fraction policies are
scoped/restored; CPU defaults are unchanged. The allocator budget is
`min(4 GiB, 70% of physical GPU memory)`. This is a **PyTorch allocator** cap,
not a cap on all NVIDIA driver/library allocations or other Windows GPU apps.
OOM/unsupported kernels are failures, never an automatic CPU/AMP fallback.
Generated CPU/emulated tests are not an actual CUDA hardware pass.

## 4. Actual data and learning are a separate next gate

Only after the generated preflight succeeds, copy the already audited private
prepared block directory from Kali to a user-owned local WSL directory and
verify its index/cache hashes. This stays on the user's computer; do not upload
ALASKA/BOSS originals, tensors or derivatives to GitHub/cloud/public datasets.
Keep the complete index and block manifests; train-only streaming opens no
validation pixels. Full prepared data is about 3.15 GB, not the entire 32 GB
competition archive. Existing source licenses and split/reservation rules stay
in force. [Data inventory](DATASET_CATALOG.md), [preparation evidence](JPEG_SCALE_PREPARATION_RESULTS.md).

The explicit streaming config adds `"device": "cuda:0"`; CUDA plan/card schema
v2 binds the GPU/runtime/precision/allocator/RAM policy in addition to exact
source/Q/method schedules and start-of-job source hashes. CPU plan/card v1
shape is preserved, but source changes intentionally require fresh plans;
do not reuse a historical plan against changed execution source hashes.
The old architecture/checkpoint shape identifier remains for serialization
compatibility; the separate execution metadata identifies actual CUDA fitting.

The new GPU path retains the **1800s** job limit, <=four rows/batch and all
finite/gradient/numeric checkpoint checks. It does not silently extend epochs,
budgets, shrink batches, tune against validation or export/install a model.
Freeze adequate train-only exposure and a learning gate before a real fit;
hardware availability alone does not prove meaningful stego discrimination.
Independent trained-model replay, complete held-out evaluation and blind-source
qualification remain required. Prior failed real scores stay published.
