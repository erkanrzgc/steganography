# Local SRNet execution assessment — 2026-10-26

The [2026-10-28 preparation](PNG_BMP_GENERATED_PREPARATION_20261028.md) records
the subsequently restored tunnel and read-only revalidation of the existing
GPU environment. The disconnected state below is the earlier dated snapshot.

**Existing WSL GPU host recovered from project records; no new PNG/BMP run.**
The user confirms Windows 11 and the previously used Ubuntu/WSL installation.
The [physical WSL record](WSL_GPU_READINESS_RESULTS.md) already verifies
Ubuntu-24.04 /WSL2, RTX 5060 Laptop (compute capability 12.0), Python 3.12.3
and Torch 2.14.0+cu130 /CUDA 13.0 on 2026-10-09. The
[epoch/resume probe](CUDA_EPOCH_RESUME_RESULTS.md) and
[completed JPEG controls](JPEG_CONTROL_RESULTS.md) preserve actual earlier
local GPU work. Those attempts are consumed and remain unchanged. The prior
candidate-only wording overlooked these records and is corrected here.

The current workspace is inside VMware. Read-only inventory finds a VMware
SVGA II display adapter, no NVIDIA device node or `nvidia-smi`, and PyTorch
2.14.0+cpu /Python 3.11.9 with zero visible CUDA devices. This describes the
guest, not the physical laptop's GPU capabilities. Installing a CUDA wheel
inside this guest would not establish GPU access. Preserve the CPU CI
environment and the frozen Colab requirements unchanged. The guest inventory
does not imply that the existing WSL host needs installation or lacks CUDA.

The existing `steganography-wsl` SSH alias, dedicated key and pinned known-host
file are present. The alias uses the previous loopback reverse endpoint on
port 22240, with batch mode and strict host-key checking. The current guest
has no listener on that port, so present host connectivity, driver state and
installed packages cannot yet be reverified. Restore only the existing
[connection procedure](WSL_GPU_RECONNECT.md) from the already installed
Ubuntu terminal. Do not rerun the original key/bootstrap setup, reinstall
Ubuntu, replace keys or start an old experiment to test connectivity.

NVIDIA identifies the RTX 5060 Laptop as Blackwell. Choose a PyTorch build
that explicitly supports that architecture; do not assume every CUDA wheel
does. For example, the current PyTorch release matrix lists Blackwell for
its CUDA 13 builds, while its CUDA 12.6 build has an older architecture list.
The earlier Colab Torch 2.11.0+cu128 environment remains historical evidence,
not a blanket local-install instruction. The established local environment
already used the appropriate CUDA 13.0 Torch build. Recheck the existing
environment over the restored connection before proposing package changes.

The existing T4 result suggests that 8 GB is a plausible capacity candidate:
both 256×256 FP32 skeletons completed batch-eight forward/backward, peaking
at 1,608.18 MiB allocated VRAM. That is an inference about feasibility,
not a new PNG/BMP measurement on the RTX 5060 or a complete training-memory
guarantee. The older local JPEG probes measured their own workloads; they
do not supply optimizer/evaluation/I/O costs for this new PNG/BMP domain.
Keep AMP and TF32 disabled under any future comparison.

The reviewed corpus is still 13 L /105 RGB originals, all unassigned. The
draft target lacks 115 L /23 RGB originals and ten L author groups. The
training adapter and generated learning control remain missing. All consumed
transfer/metadata allowances, reserved datasets and prior attempts remain
unchanged. Local hardware does not complete these preparation requirements.

Next restore the existing tunnel and verify host identity/environment with
read-only commands. Then prepare a separate source-bound generated
acceptance/profile contract for the new domain, covering full steps,
evaluation and I/O. The old Colab profile, historical local jobs and completed
collection must not be relaunched as a local test. Real fitting needs a
concrete source/data/domain/exposure/grouping/budget contract; this assessment
provides no new PNG/BMP accuracy or training result.

Sources: [NVIDIA RTX 5060 Laptop announcement](https://nvidianews.nvidia.com/news/nvidia-blackwell-geforce-rtx-arrives-for-every-gamer-starting-at-299),
[PyTorch CUDA architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[NVIDIA CUDA on WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
[measured T4 capacity](PNG_BMP_COLAB_CAPACITY_RESULTS_20261023.md), and
[completed corpus status](PNG_BMP_NATIVE_EXPANSION_RESULTS_20261023.md).
