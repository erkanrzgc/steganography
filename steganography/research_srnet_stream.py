"""Explicit versioned streaming plan/check/fit; never deploys or downloads weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from core import jpeg_float256, srnet, srnet_model, srnet_scale_sampling, srnet_stream_training
from core.jpeg_scale import identity
from core.srnet_stream import TrainBlocks, deadline_after, document, regular_open
from core.srnet_training import settings
from steganography.research_jpeg import write_json

DEPENDENCIES = (
    "core/srnet.py",
    "core/srnet_model.py",
    "core/srnet_sampling.py",
    "core/srnet_multibatch.py",
    "core/srnet_scale_sampling.py",
    "core/srnet_stream.py",
    "core/srnet_training.py",
    "core/srnet_stream_training.py",
    "core/jpeg_float256.py",
    "core/jpeg_scale.py",
    "steganography/research_srnet_stream.py",
    "steganography/research_jpeg.py",
)


def snapshot():
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name in DEPENDENCIES:
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("training dependency size exceeded")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def fresh(path):
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("stream output exists or uses a symlink")


def configuration(config):
    required = {"root", "index_sha256", "audit", "audit_sha256", "epochs", "seed"}
    optional = {"threads", "max_seconds", "learning_rate", "weight_decay"}
    if (
        not isinstance(config, dict)
        or not required <= config.keys()
        or config.keys() - required - optional
        or any(
            not isinstance(config[k], str) or not config[k] for k in required - {"epochs", "seed"}
        )
        or type(config["epochs"]) is not int
        or not 1 <= config["epochs"] <= 50
        or type(config["seed"]) is not int
        or not 0 <= config["seed"] < 2**32
    ):
        raise ValueError("invalid explicit streaming configuration")
    identity(config["index_sha256"])
    identity(config["audit_sha256"])
    return settings(config)


def plan_record(reader, config, params, deadline, sources):
    epochs = []
    for epoch in range(config["epochs"]):
        deadline()
        epochs.append(
            srnet_scale_sampling.epoch_batches(reader.samples, seed=config["seed"], epoch=epoch)[1]
        )
    return {
        "schema_version": "srnet-stream-plan-v1",
        "index_sha256": reader.index_sha256,
        "audit_sha256": reader.audit_sha256,
        "architecture": srnet.ARCHITECTURE,
        "feature_version": jpeg_float256.FEATURE_VERSION,
        "decoder": jpeg_float256.decoder_contract(),
        "seed": config["seed"],
        "settings": params,
        "source_sha256": sources,
        "epochs": epochs,
        "train_rows": len(reader.samples),
        "original_train_lineages": len({s["lineage"] for s in reader.samples}),
        "train_tensor_bytes": reader.train_bytes,
        "batch_size": 4,
        "validation_used": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }


def execute(config, out, *, operation, plan_path=None, plan_sha256=None):
    """Library caller supplies hard OS limits; CLI always isolates this job."""
    fresh(out)
    params = configuration(config)
    if operation not in {"plan", "check", "fit"}:
        raise ValueError("unknown streaming operation")
    if operation == "fit" and (plan_path is None or plan_sha256 is None):
        raise ValueError("stream fit requires an explicit bound plan")
    started = time.monotonic()
    deadline = deadline_after(params["max_seconds"])
    sources = snapshot()  # Before any corpus read, not retrospective end-only hashes.
    with TrainBlocks(
        Path(config["root"]),
        index_sha256=config["index_sha256"],
        audit=Path(config["audit"]),
        audit_sha256=config["audit_sha256"],
        deadline=deadline,
    ) as reader:
        plan = plan_record(reader, config, params, deadline, sources)
        if operation == "fit":
            expected = document(plan_path, plan_sha256)
            if expected != plan:
                raise ValueError("stream fit plan/provenance mismatch")
            model, records = srnet_stream_training.fit(
                reader,
                seed=config["seed"],
                schedule=plan["epochs"],
                config=params,
                deadline=deadline,
            )
            report = {
                "schema_version": "srnet-stream-fit-v1",
                "training": "completed",
                "plan_sha256": plan_sha256,
                "epoch_training": records,
                "validation_used": False,
                "accuracy_qualification": "unavailable",
                "deployed": False,
            }
        elif operation == "check":
            seen: set[int] = set()
            presentations, updates = 0, 0
            data_digest = hashlib.sha256()
            for epoch in range(config["epochs"]):
                batches, _ = srnet_scale_sampling.epoch_batches(
                    reader.samples, seed=config["seed"], epoch=epoch
                )
                for indices in batches:
                    values = reader.batch(indices)
                    data_digest.update(values.tobytes())
                    seen.update(int(i) for i in indices)
                    presentations += len(indices)
                    updates += 1
            if len(seen) != len(reader.samples):
                raise ValueError("stream readiness omitted training rows")
            report = {
                "schema_version": "srnet-stream-readiness-v1",
                "status": "completed",
                "plan": plan,
                "unique_train_rows_read": len(seen),
                "planned_optimizer_updates": updates,
                "row_presentations": presentations,
                "ordered_tensor_sha256": data_digest.hexdigest(),
                "max_batch_tensor_bytes": reader.max_batch_bytes,
                "models_trained": False,
                "detection_measured": False,
                "validation_pixels_opened": False,
                "accuracy_qualification": "unavailable",
            }
        else:
            report = plan
        deadline()
        if snapshot() != sources:
            raise ValueError("training dependencies changed during job")
        if operation == "fit":
            out.mkdir(parents=True, exist_ok=False)
            report.update(
                model_sha256=srnet_model.save_model(model, out / "model.npz"),
                plan=plan,
                source_sha256=sources,
                seconds=time.monotonic() - started,
                max_batch_tensor_bytes=reader.max_batch_bytes,
            )
            deadline()
            if snapshot() != sources:
                raise ValueError("training dependencies changed before publication")
            write_json(out / "model-card.json", report)
        else:
            if operation == "check":
                report.update(source_sha256=sources, seconds=time.monotonic() - started)
            write_json(out, report)
    return report


def run_job(config_path, out, *, operation, plan_path=None, plan_sha256=None):
    fresh(out)
    # Hash and re-read the explicit config in the isolated worker.
    with regular_open(config_path) as stream:
        raw = stream.read(16 * 1024**2 + 1)
    config_sha = hashlib.sha256(raw).hexdigest()
    config = document(config_path, config_sha)
    params = configuration(config)
    root = Path(__file__).resolve().parents[1]
    args = [
        sys.executable,
        "-m",
        "steganography.research_srnet_stream",
        "--worker",
        "--operation",
        operation,
        "--config",
        str(config_path.absolute()),
        "--config-sha256",
        config_sha,
        "--out",
        str(out.absolute()),
    ]
    if plan_path is not None:
        args.extend(["--plan", str(plan_path.absolute()), "--plan-sha256", plan_sha256 or ""])
    try:
        with tempfile.TemporaryFile() as stream:
            result = subprocess.run(  # noqa: S603 — fixed trusted module, no shell or descendants
                args,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.DEVNULL,
                cwd=root,
                timeout=params["max_seconds"] + 30,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(root),
                    "OMP_NUM_THREADS": str(params["threads"]),
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            stream.seek(0)
            raw_response = stream.read(65537)
            if len(raw_response) > 65536:
                raise ValueError("stream worker response oversized")
            response = json.loads(raw_response)
        if (
            result.returncode != 0
            or not isinstance(response, dict)
            or response.get("status") != "completed"
        ):
            raise ValueError("stream worker incomplete/unavailable")
        target = out / "model-card.json" if operation == "fit" else out
        report = document(target, response["report_sha256"])
        if operation == "fit":
            srnet_model.load_model(out / "model.npz", checksum=report["model_sha256"])
        return report
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("stream job failed; incomplete outputs unusable") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("plan", "check", "fit"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            import resource

            config = document(args.config, args.config_sha256)
            params = configuration(config)
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (2 * params["max_seconds"] + 30, 2 * params["max_seconds"] + 31),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (srnet_model.MAX_BYTES, srnet_model.MAX_BYTES)
            )
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            execute(
                config,
                args.out,
                operation=args.operation,
                plan_path=args.plan,
                plan_sha256=args.plan_sha256,
            )
            target = args.out / "model-card.json" if args.operation == "fit" else args.out
            with regular_open(target) as stream:
                checksum = hashlib.sha256(stream.read(16 * 1024**2 + 1)).hexdigest()
            print(json.dumps({"status": "completed", "report_sha256": checksum}))
        else:
            run_job(
                args.config,
                args.out,
                operation=args.operation,
                plan_path=args.plan,
                plan_sha256=args.plan_sha256,
            )
            print("stream job completed; research-only, no accuracy qualification")
    except Exception:
        print(
            '{"status":"failed"}'
            if args.worker
            else "stream job failed/unavailable; partial outputs are unusable"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
