"""Separately bound numerical preflight revision; original sources stay immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from core import jpeg_refine_preflight_v2 as preflight
from core.jpeg_timing import write_document
from core.srnet_stream import deadline_after, regular_open
from steganography import research_jpeg_refine as old

ROOT = old.ROOT
AMENDMENT = "benchmarks/jpeg-refine-numerics-protocol-20261015.json"
NAMES = (*old.NAMES, "core/jpeg_refine_preflight_v2.py",
         "steganography/research_jpeg_refine_numerics.py",
         "docs/JPEG_REFINE_NUMERICS_PROTOCOL.md", AMENDMENT)
ERRORS = {
    "generated optimizer scalar oracle failed": "optimizer_scalar_oracle_failed",
    "generated logits invalid": "generated_logits_invalid",
    "generated objective invalid": "generated_objective_invalid",
    "generated CPU/CUDA forward parity failed": "cpu_cuda_forward_parity_failed",
    "generated CPU/CUDA gradient parity failed": "cpu_cuda_gradient_parity_failed",
    "generated CPU/CUDA pre-optimizer state parity failed": "cpu_cuda_initial_state_parity_failed",
    "generated save/resume model parity failed": "save_resume_model_parity_failed",
    "generated save/resume optimizer parity failed": "save_resume_optimizer_parity_failed",
}


def sources():
    result = {}
    for name in NAMES:
        with regular_open(ROOT / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("numerics source exceeds bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


class ChargedClock:
    """Set the experiment's start earlier by an explicit conservative prior debit."""

    def __init__(self, debit, clock=time.monotonic):
        self.debit, self.clock, self.first = debit, clock, True

    def __call__(self):
        now = self.clock()
        if self.first:
            self.first = False
            return now - self.debit
        return now


def launch(mode, payload, out, timeout, cancelled):
    raw = json.dumps(payload, default=str, allow_nan=False).encode()
    response = old.child(
        [sys.executable, "-m", "steganography.research_jpeg_refine_numerics",
         "--worker", mode, "--out", str(out)], cwd=ROOT, timeout=timeout,
        cancelled=cancelled, payload=raw,
    )
    return {**json.loads(response), "exit_status": 0}


def worker(mode, payload, out):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    resource.setrlimit(resource.RLIMIT_FSIZE, (96 * 1024**2, 96 * 1024**2))
    before = sources()
    config = old.service.configuration(payload["config"], before)
    try:
        execution = old.inspect(config["expected_gpu"])
        if mode == "audit":
            result = old.jpeg_study_audit.audit(
                config["root"], config["manifest_sha256"], deadline=deadline_after(1800)
            )
        elif mode == "train":
            parent = payload["parent"]
            if parent:
                parent["path"] = Path(parent["path"])
            result = old.training.train(config, out, tuple(payload["task"]), payload["binding"],
                                        payload["execution"], parent=parent)
        else:
            with old.srnet_cuda.policy():
                if mode == "preflight":
                    result = preflight.probe()
                elif mode == "evaluate":
                    checkpoint = payload["checkpoint"]
                    checkpoint["path"] = Path(checkpoint["path"])
                    result = old.training.evaluate(
                        config, checkpoint, deadline=deadline_after(1800)
                    )
                else:
                    raise ValueError("invalid worker mode")
        if sources() != before or old.inspect(config["expected_gpu"]) != execution:
            raise ValueError("numerics source/runtime mutation")
        return {**result, "source_sha256": before, "execution": execution}
    except old.srnet_cuda.CUDAUnavailable as exc:
        return {"status": "unavailable", "reason": exc.reason}
    except ImportError:
        return {"status": "unavailable", "reason": "optional_dependency_unavailable"}
    except (TimeoutError, InterruptedError):
        return {"status": "unavailable", "reason": "worker_interrupted_or_timeout"}
    except ValueError as exc:
        return {"status": "unavailable", "reason": ERRORS.get(str(exc), "worker_contract_failed")}
    except OSError:
        return {"status": "unavailable", "reason": "worker_os_failed"}
    except RuntimeError as exc:
        import torch

        return {"status": "unavailable", "reason": "gpu_allocator_out_of_memory"
                if isinstance(exc, torch.cuda.OutOfMemoryError) else "worker_runtime_failed"}


def run(config_path, out):
    config, _ = old.service.read(config_path)
    amendment, _ = old.service.read(ROOT / AMENDMENT)
    before = sources()
    if any(before.get(k) != v for k, v in amendment["original_source_sha256"].items()):
        raise ValueError("original frozen source changed")
    if "resume" in config:
        raise ValueError("numerics revision cannot import or relabel old snapshots")
    accounting = amendment["accounting"]
    if (
        amendment["original_source_sha256"] != old.sources()
        or any(before.get(k) != v for k, v in amendment["revision_source_sha256"].items())
        or accounting["diagnostic_reservations_seconds"] != [190, 190, 190]
        or accounting["prior_controller_seconds"] != 34.309597279
        or accounting["prior_seconds_debit"] != 604.309597279
        or accounting["prior_output_bytes_debit"] != 256 * 1024**2
        or accounting["remaining_seconds"] != 21600 - accounting["prior_seconds_debit"]
    ):
        raise ValueError("numerics accounting/source amendment mismatch")

    def inspect(expected):
        result = old.inspect(expected)
        write_document(out / "external-accounting.json", accounting)
        return result

    def bounded_launch(mode, payload, target, timeout, cancelled):
        if old.service.storage(out) + accounting["prior_output_bytes_debit"] + 97 * 1024**2 > (
            old.service.BYTES
        ):
            raise ValueError("numerics cumulative output budget exceeded")
        return launch(mode, payload, target, timeout, cancelled)

    return old.service.execute(
        config, out, sources=before, current_sources=sources, inspect=inspect,
        launch=bounded_launch, clock=ChargedClock(accounting["prior_seconds_debit"]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", choices=("audit", "preflight", "train", "evaluate"))
    args = parser.parse_args(argv)
    if args.worker:
        raw = sys.stdin.buffer.read(1024**2 + 1)
        if len(raw) > 1024**2:
            parser.error("worker input exceeds bound")
        result = worker(args.worker, json.loads(raw), args.out)
    else:
        if args.config is None:
            parser.error("--config is required")
        result = run(args.config, args.out)
    print(json.dumps(result if args.worker else {
        "status": result["status"], "reason": result.get("reason")}, allow_nan=False))
    # Handled unavailable reports are valid protocol responses, not process failures.
    return 0 if args.worker or result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
