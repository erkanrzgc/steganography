"""Opt-in FP32 CUDA policy. Absence is unavailable, never a CPU fallback."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

HOST_BYTES = 8 * 1024**3
GPU_BYTES = 4 * 1024**3


class CUDAUnavailable(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("CUDA research backend unavailable: " + reason)


def device(value):
    if not isinstance(value, str) or value not in {"cpu", "cuda:0"}:
        raise ValueError("device must be explicitly cpu or cuda:0")
    return value


def host_bound(*, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    """Read-only cgroup-v2 resident RAM bound, including ancestor constraints."""
    try:
        with proc.open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError("oversized cgroup identity")
        lines = [line[3:] for line in raw.decode().splitlines() if line.startswith("0::")]
        if len(lines) != 1:
            raise ValueError("cgroup-v2 required")
        relative = PurePosixPath(lines[0])
        if not relative.is_absolute() or ".." in relative.parts or len(relative.parts) > 32:
            raise ValueError("invalid cgroup identity")
        folder = root.joinpath(*relative.parts[1:])
        limits = []
        while True:
            path = folder / "memory.max"
            if path.is_symlink() or any(p.is_symlink() for p in (folder, *folder.parents)):
                raise ValueError("cgroup symlink forbidden")
            if not folder.is_dir():
                raise ValueError("cgroup membership no longer exists")
            # The true root has no memory.max on some cgroup-v2 hosts.
            # Any finite ancestor still bounds all descendants.
            if path.exists():
                with path.open("rb") as stream:
                    value = stream.read(33)
            else:
                value = b"max"
            if len(value) > 32:
                raise ValueError("oversized cgroup limit")
            text = value.decode().strip()
            if text != "max":
                if not text.isascii() or not text.isdecimal() or not 0 < int(text) <= 2**63 - 1:
                    raise ValueError("invalid cgroup memory limit")
                limits.append(int(text))
            if folder == root:
                break
            folder = folder.parent
        if not limits or min(limits) > HOST_BYTES:
            raise ValueError("kernel resident RAM bound required")
        return min(limits)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CUDAUnavailable("bounded_cgroup_v2_required") from exc


def inspect():
    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise CUDAUnavailable("cuda_not_visible_or_cpu_torch")
    resident = host_bound()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise CUDAUnavailable("deterministic_cublas_environment_required")
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision") or not hasattr(
        getattr(torch.backends.cudnn, "conv", None), "fp32_precision"
    ):
        raise CUDAUnavailable("modern_fp32_precision_api_required")
    capability = torch.cuda.get_device_capability(0)
    target = "_" + "".join(str(n) for n in capability)
    if not any(arch in {"sm" + target, "compute" + target} for arch in torch.cuda.get_arch_list()):
        raise CUDAUnavailable("gpu_architecture_not_in_torch_build")
    props = torch.cuda.get_device_properties(0)
    budget = min(GPU_BYTES, int(props.total_memory * 0.7))
    if budget < 1024**3:
        raise CUDAUnavailable("gpu_allocator_budget_below_one_gib")
    return {
        "device": "cuda:0",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": props.name,
        "compute_capability": list(capability),
        "total_gpu_bytes": props.total_memory,
        "torch_allocator_limit_bytes": budget,
        "host_memory_max_bytes": resident,
        "precision": "float32-ieee",
        "tf32": False,
        "mixed_precision": False,
        "deterministic_algorithms": True,
        "cpu_fallback": False,
    }


@contextmanager
def policy():
    import torch

    info = inspect()
    if torch.cuda.memory_reserved(0) > info["torch_allocator_limit_bytes"]:
        raise CUDAUnavailable("existing_allocator_reservation_exceeds_budget")
    matmul, conv = torch.backends.cuda.matmul, getattr(torch.backends.cudnn, "conv", None)
    if conv is None:
        raise CUDAUnavailable("modern_fp32_precision_api_required")
    cudnn = torch.backends.cudnn
    previous = (
        matmul.fp32_precision,
        conv.fp32_precision,
        cudnn.benchmark,
        cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.cuda.get_per_process_memory_fraction(0),
    )
    try:
        torch.cuda.set_per_process_memory_fraction(
            info["torch_allocator_limit_bytes"] / info["total_gpu_bytes"], 0
        )
        matmul.fp32_precision = conv.fp32_precision = "ieee"
        cudnn.benchmark, cudnn.deterministic = False, True
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.cuda.device(0), torch.autocast(device_type="cuda", enabled=False):
            yield info
    finally:
        matmul.fp32_precision, conv.fp32_precision = previous[:2]
        cudnn.benchmark, cudnn.deterministic = previous[2:4]
        torch.use_deterministic_algorithms(previous[4], warn_only=previous[5])
        torch.cuda.set_per_process_memory_fraction(previous[6], 0)
