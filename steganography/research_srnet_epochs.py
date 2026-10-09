"""Explicit isolated whole-epoch jobs, bound to a frozen complete training plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from core import (
    srnet_checkpoint,
    srnet_cuda,
    srnet_model,
    srnet_resume_probe,
    srnet_stream_training,
)
from core.srnet_stream import TrainBlocks, deadline_after, document, regular_open
from steganography import research_srnet_stream as legacy
from steganography.research_jpeg import write_json

EXTRA = {"protocol", "protocol_sha256", "resume_probe", "resume_probe_sha256"}


def sources():
    result = legacy.snapshot()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/srnet_checkpoint.py",
        "core/srnet_resume_probe.py",
        "steganography/research_srnet_epochs.py",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("epoch dependency size exceeded")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def file_sha(path, maximum=16 * 1024**2):
    with regular_open(path) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("epoch input size exceeded")
    return hashlib.sha256(raw).hexdigest()


def configuration(config):
    if not isinstance(config, dict) or not config.keys() >= EXTRA:
        raise ValueError("epoch jobs require explicit protocol and resume probe")
    params = legacy.configuration({k: v for k, v in config.items() if k not in EXTRA})
    for name in ("protocol", "resume_probe"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError("epoch prerequisite path missing")
        from core.jpeg_scale import identity

        identity(config[name + "_sha256"])
    if file_sha(Path(config["protocol"])) != config["protocol_sha256"]:
        raise ValueError("frozen epoch protocol mismatch")
    return params


def verified_probe(config, params, bound_sources):
    record = document(Path(config["resume_probe"]), config["resume_probe_sha256"])
    if (
        record.get("schema_version") != "srnet-epoch-resume-probe-v1"
        or record.get("status") != "completed"
        or record.get("source_sha256") != bound_sources
        or record.get("exact_model_bn_optimizer_rng_loss") is not True
        or record.get("real_data_used") is not False
        or record.get("real_model_trained") is not False
        or record.get("execution", {}).get("device") != params.get("device", "cpu")
        or record.get("generated_optimizer_updates") != 128
        or record.get("uninterrupted_updates") != 64
    ):
        raise ValueError("epoch resume probe incomplete or stale")
    import numpy as np

    intervals = record.get("steady_intervals_seconds")
    if (
        not isinstance(intervals, list)
        or len(intervals) != 47
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in intervals)
        or float(np.quantile(intervals, 0.95)) != record.get("steady_interval_p95_seconds")
    ):
        raise ValueError("epoch probe timing invalid")
    return record


def parent_state(folder, checksum, plan, plan_sha, epoch):
    if epoch == 0:
        if folder is not None or checksum is not None:
            raise ValueError("first epoch cannot have a parent")
        return None
    if folder is None or checksum is None:
        raise ValueError("later epoch requires an explicit parent")
    card = document(folder / "epoch.json", checksum)
    if (
        card.get("schema_version") != "srnet-epoch-job-v1"
        or card.get("status") != "completed"
        or card.get("plan_sha256") != plan_sha
        or card.get("source_sha256") != plan["source_sha256"]
        or card.get("next_epoch") != epoch
        or card.get("validation_used") is not False
    ):
        raise ValueError("epoch parent chain mismatch")
    state = srnet_checkpoint.load(folder / "checkpoint.npz", checksum=card["checkpoint_sha256"])
    if (
        state["metadata"]["binding"] != plan_sha
        or state["metadata"]["next_epoch"] != epoch
        or state["metadata"]["epoch_training"] != card.get("epoch_training")
    ):
        raise ValueError("epoch parent numeric state mismatch")
    return state


def execute(
    config,
    out,
    *,
    operation,
    epoch=None,
    plan_path=None,
    plan_sha=None,
    parent=None,
    parent_sha=None,
):
    """Library callers provide OS limits; CLI always isolates each job."""
    legacy.fresh(out)
    if operation not in {"plan", "fit"}:
        raise ValueError("unknown epoch operation")
    params = configuration(config)
    started = time.monotonic()
    deadline = deadline_after(params["max_seconds"])
    before = sources()
    probe = verified_probe(config, params, before)
    if params.get("device") == "cuda:0" and probe["execution"] != srnet_cuda.inspect():
        raise ValueError("epoch hardware/runtime differs from resume probe")
    with TrainBlocks(
        Path(config["root"]),
        index_sha256=config["index_sha256"],
        audit=Path(config["audit"]),
        audit_sha256=config["audit_sha256"],
        deadline=deadline,
    ) as reader:
        plan = legacy.plan_record(reader, config, params, deadline, before)
        estimates = [
            120
            + 2 * probe["steady_interval_p95_seconds"] * e["batch_schedule"]["optimizer_updates"]
            for e in plan["epochs"]
        ]
        if any(v > params["max_seconds"] for v in estimates):
            raise ValueError("whole epoch exceeds unchanged per-job budget")
        plan.update(
            schema_version="srnet-epoch-plan-v1",
            protocol_sha256=config["protocol_sha256"],
            resume_probe_sha256=config["resume_probe_sha256"],
            epoch_estimated_seconds=estimates,
            per_job_max_seconds=params["max_seconds"],
            fixed_overhead_seconds=120,
            timing_safety_factor=2,
        )
        if operation == "plan":
            if epoch is not None or parent is not None or parent_sha is not None:
                raise ValueError("plan cannot select an epoch or parent")
            deadline()
            if sources() != before:
                raise ValueError("epoch sources changed")
            write_json(out, plan)
            return plan
        if (
            type(epoch) is not int
            or not 0 <= epoch < config["epochs"]
            or plan_path is None
            or plan_sha is None
            or document(plan_path, plan_sha) != plan
        ):
            raise ValueError("epoch fit requires the exact full plan")
        resume = parent_state(parent, parent_sha, plan, plan_sha, epoch)
        collected: list[dict[str, Any]] = []
        model, records = srnet_stream_training.fit(
            reader,
            seed=config["seed"],
            schedule=plan["epochs"],
            config=params,
            deadline=deadline,
            segment={
                "binding": plan_sha,
                "stop_epoch": epoch + 1,
                "resume": resume,
                "sink": collected.append,
            },
        )
        if len(collected) != 1 or len(records) != epoch + 1:
            raise ValueError("epoch completion accounting mismatch")
        deadline()
        # A fresh reader verifies all train cache bytes again before publication.
        with TrainBlocks(
            Path(config["root"]),
            index_sha256=config["index_sha256"],
            audit=Path(config["audit"]),
            audit_sha256=config["audit_sha256"],
            deadline=deadline,
        ):
            pass
        if sources() != before or configuration(config) != params:
            raise ValueError("epoch dependencies changed before publication")
        out.mkdir(parents=True, exist_ok=False)
        checkpoint_sha = srnet_checkpoint.save(collected[0], out / "checkpoint.npz")
        model_sha = srnet_model.save_model(model, out / "model.npz")
        report = {
            "schema_version": "srnet-epoch-job-v1",
            "status": "completed",
            "plan_sha256": plan_sha,
            "protocol_sha256": config["protocol_sha256"],
            "parent_report_sha256": parent_sha,
            "next_epoch": epoch + 1,
            "checkpoint_sha256": checkpoint_sha,
            "model_sha256": model_sha,
            "epoch_training": records,
            "source_sha256": before,
            "seconds": time.monotonic() - started,
            "validation_used": False,
            "accuracy_qualification": "unavailable",
            "deployed": False,
        }
        if params.get("device") == "cuda:0":
            import torch

            report.update(
                execution=plan["execution"],
                process_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
                process_peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0),
            )
        deadline()
        if sources() != before:
            raise ValueError("epoch sources changed during publication")
        write_json(out / "epoch.json", report)
        return report


def execute_probe(out, *, device, protocol, protocol_sha):
    legacy.fresh(out)
    if file_sha(protocol) != protocol_sha:
        raise ValueError("resume probe protocol mismatch")
    before = sources()
    with tempfile.TemporaryDirectory(prefix="srnet-resume-") as temporary:
        report = srnet_resume_probe.probe(Path(temporary), binding=protocol_sha, device=device)
    if sources() != before or file_sha(protocol) != protocol_sha:
        raise ValueError("resume probe dependencies changed")
    report.update(
        schema_version="srnet-epoch-resume-probe-v1",
        status="completed",
        source_sha256=before,
        protocol_sha256=protocol_sha,
    )
    write_json(out, report)
    return report


def run_job(args):
    legacy.fresh(args.out)
    before = sources()
    config_sha = None if args.config is None else file_sha(args.config)
    config = None if config_sha is None else document(args.config, config_sha)
    seconds = 180 if args.operation == "probe" else configuration(config)["max_seconds"]
    command = [
        sys.executable,
        "-m",
        "steganography.research_srnet_epochs",
        "--worker",
        "--operation",
        args.operation,
        "--out",
        str(args.out.absolute()),
    ]
    for name, value in vars(args).items():
        if name in {"worker", "operation", "out", "config_sha256"} or value is None:
            continue
        command.extend(["--" + name.replace("_", "-"), str(value)])
    if config_sha is not None:
        command.extend(["--config-sha256", config_sha])
    try:
        with tempfile.TemporaryFile() as response:
            result = subprocess.run(  # noqa: S603 — fixed isolated trusted module, no shell
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=response,
                stderr=subprocess.DEVNULL,
                timeout=seconds + 30,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )
            response.seek(0)
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("epoch worker response oversized")
        reply = json.loads(raw)
        if (
            result.returncode != 0
            or not isinstance(reply, dict)
            or reply.get("status") != "completed"
        ):
            raise ValueError("epoch worker failed/unavailable")
        target = args.out / "epoch.json" if args.operation == "fit" else args.out
        report = document(target, reply["report_sha256"])
        if sources() != before or report.get("source_sha256") != before:
            raise ValueError("epoch parent/worker sources changed")
        if args.config is not None and file_sha(args.config) != config_sha:
            raise ValueError("epoch configuration changed")
        if args.operation == "fit":
            state = srnet_checkpoint.load(
                args.out / "checkpoint.npz", checksum=report["checkpoint_sha256"]
            )
            if state["metadata"]["next_epoch"] != args.epoch + 1:
                raise ValueError("epoch worker accounting mismatch")
            srnet_model.load_model(args.out / "model.npz", checksum=report["model_sha256"])
        return report
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("epoch job failed; incomplete outputs unusable") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("probe", "plan", "fit"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    for name in ("config", "plan", "parent", "protocol"):
        parser.add_argument("--" + name, type=Path)
    for name in ("plan-sha", "parent-sha", "protocol-sha", "config-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            import resource

            config = (
                None if args.operation == "probe" else document(args.config, args.config_sha256)
            )
            params = {"max_seconds": 180} if config is None else configuration(config)
            selected = args.device if config is None else params.get("device", "cpu")
            if selected == "cuda:0":
                srnet_cuda.host_bound()
            else:
                resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (2 * params["max_seconds"] + 30, 2 * params["max_seconds"] + 31),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (srnet_checkpoint.MAX_BYTES, srnet_checkpoint.MAX_BYTES)
            )
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if config is None:
                execute_probe(
                    args.out,
                    device=selected,
                    protocol=args.protocol,
                    protocol_sha=args.protocol_sha,
                )
            else:
                execute(
                    config,
                    args.out,
                    operation=args.operation,
                    epoch=args.epoch,
                    plan_path=args.plan,
                    plan_sha=args.plan_sha,
                    parent=args.parent,
                    parent_sha=args.parent_sha,
                )
            target = args.out / "epoch.json" if args.operation == "fit" else args.out
            print(json.dumps({"status": "completed", "report_sha256": file_sha(target)}))
        else:
            run_job(args)
            print('{"status":"completed"}')
    except (srnet_cuda.CUDAUnavailable, ImportError):
        print('{"status":"unavailable","reason":"bounded_backend_unavailable"}')
        return 2
    except Exception:
        print('{"status":"failed"}')
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
