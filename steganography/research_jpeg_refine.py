"""Opt-in adapter for the frozen owned-GPU refinement service."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
from pathlib import Path

from core import jpeg_refine as service
from core import jpeg_refine_preflight as preflight
from core import jpeg_refine_training as training
from core import jpeg_study_audit, srnet_cuda
from core.jpeg_study_process import run as child
from core.srnet_stream import deadline_after, regular_open

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "core/jpeg_refine.py",
    "core/jpeg_refine_schedule.py",
    "core/jpeg_refine_state.py",
    "core/jpeg_refine_training.py",
    "core/jpeg_refine_metrics.py",
    "core/jpeg_refine_preflight.py",
    "core/jpeg_study.py",
    "core/jpeg_study_data.py",
    "core/jpeg_study_state.py",
    "core/jpeg_study_views.py",
    "core/jpeg_study_audit.py",
    "core/jpeg_study_optimizer.py",
    "core/jpeg_study_process.py",
    "core/jpeg_float256.py",
    "core/jpeg_timing.py",
    "core/jpeg_scale.py",
    "core/srnet.py",
    "core/srnet_samepool.py",
    "core/srnet_model.py",
    "core/srnet_training.py",
    "core/srnet_cuda.py",
    "core/srnet_stream.py",
    "core/srnet_checkpoint.py",
    "core/srnet_batch_probe.py",
    "core/srnet_sampling.py",
    "core/srnet_scale_sampling.py",
    "core/dataset.py",
    "core/jpeg_context.py",
    "steganography/benchmarking/metrics.py",
    "steganography/research_jpeg_refine.py",
    "docs/JPEG_REFINE_PROTOCOL.md",
)


def sources():
    result = {}
    for name in NAMES:
        with regular_open(ROOT / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("refine source closure exceeds bounds")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def inspect(expected):
    import torch

    execution = srnet_cuda.inspect()
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if uuid is None:
        raise srnet_cuda.CUDAUnavailable("physical_gpu_identity_unavailable")
    if (
        execution["gpu_name"] != expected["name"]
        or hashlib.sha256(str(uuid).encode()).hexdigest() != expected["uuid_sha256"]
    ):
        raise srnet_cuda.CUDAUnavailable("owned_gpu_identity_mismatch")
    return {
        **execution,
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "jpeglib", "pillow")
        },
        "python_version": platform.python_version(),
    }


def launch(mode, payload, out, timeout, cancelled):
    raw = json.dumps(payload, default=str, allow_nan=False).encode()
    result = json.loads(
        child(
            [
                sys.executable,
                "-m",
                "steganography.research_jpeg_refine",
                "--worker",
                mode,
                "--out",
                str(out),
            ],
            cwd=ROOT,
            timeout=timeout,
            cancelled=cancelled,
            payload=raw,
        )
    )
    return {**result, "exit_status": 0}


def worker(mode, payload, out):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    resource.setrlimit(resource.RLIMIT_FSIZE, (96 * 1024**2, 96 * 1024**2))
    before = sources()
    config = service.configuration(payload["config"], before)
    try:
        execution = inspect(config["expected_gpu"])
        if mode == "audit":
            result = jpeg_study_audit.audit(
                config["root"], config["manifest_sha256"], deadline=deadline_after(1800)
            )
        elif mode == "train":
            parent = payload["parent"]
            if parent:
                parent["path"] = Path(parent["path"])
            result = training.train(
                config,
                out,
                tuple(payload["task"]),
                payload["binding"],
                payload["execution"],
                parent=parent,
            )
        else:
            with srnet_cuda.policy():
                if mode == "preflight":
                    result = preflight.probe()
                elif mode == "evaluate":
                    checkpoint = payload["checkpoint"]
                    checkpoint["path"] = Path(checkpoint["path"])
                    result = training.evaluate(config, checkpoint, deadline=deadline_after(1800))
                else:
                    raise ValueError("refine worker mode invalid")
        if sources() != before or inspect(config["expected_gpu"]) != execution:
            raise ValueError("refine worker source/runtime mutation")
        return {**result, "source_sha256": before, "execution": execution}
    except srnet_cuda.CUDAUnavailable as exc:
        return {"status": "unavailable", "reason": exc.reason}
    except ImportError:
        return {"status": "unavailable", "reason": "optional_dependency_unavailable"}
    except (TimeoutError, InterruptedError):
        return {"status": "unavailable", "reason": "worker_interrupted_or_timeout"}
    except (ValueError, OSError):
        return {"status": "unavailable", "reason": "worker_contract_failed"}
    except RuntimeError as exc:
        import torch

        return {
            "status": "unavailable",
            "reason": "gpu_allocator_out_of_memory"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            else "worker_runtime_failed",
        }


def run(config_path, out):
    config, _ = service.read(config_path)
    return service.execute(
        config, out, sources=sources(), current_sources=sources, inspect=inspect, launch=launch
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--worker", choices=("audit", "preflight", "train", "evaluate"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.worker:
        raw = sys.stdin.buffer.read(1024**2 + 1)
        if len(raw) > 1024**2:
            parser.error("worker input exceeds bounds")
        result = worker(args.worker, json.loads(raw), args.out)
    else:
        if args.config is None:
            parser.error("--config is required")
        result = run(args.config, args.out)
    print(
        json.dumps(
            {"status": result["status"], "reason": result.get("reason")}
            if not args.worker
            else result,
            allow_nan=False,
        )
    )
    return 0 if result["status"] in ("completed", "checkpointed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
