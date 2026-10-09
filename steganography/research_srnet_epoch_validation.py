"""One fixed complete-development validation after the final epoch only."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from core import srnet_block_validation, srnet_model
from core.srnet_stream import deadline_after, document
from steganography import research_srnet_epochs as epochs
from steganography.benchmarking.metrics import classification_metrics
from steganography.research_jpeg import write_json


def sources():
    result = epochs.sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/srnet_block_validation.py",
        "core/srnet_reference.py",
        "steganography/benchmarking/metrics.py",
        "steganography/research_srnet_epoch_validation.py",
    ):
        result[name] = epochs.file_sha(root / name, maximum=1024**2)
    return result


def execute(config, out, *, plan_path, plan_sha, model_dir, card_sha):
    epochs.legacy.fresh(out)
    params = epochs.configuration(config)
    before, started = sources(), time.monotonic()
    deadline = deadline_after(params["max_seconds"])
    plan = document(plan_path, plan_sha)
    # Recreate the exact original plan, not a relaxed evaluation-only contract.
    with tempfile.TemporaryDirectory(prefix="epoch-validation-plan-") as temporary:
        candidate = epochs.execute(config, Path(temporary) / "plan.json", operation="plan")
    if candidate != plan:
        raise ValueError("complete validation requires the original bound epoch plan")
    state = epochs.parent_state(model_dir, card_sha, plan, plan_sha, config["epochs"])
    card = document(model_dir / "epoch.json", card_sha)
    model = srnet_model.load_model(model_dir / "model.npz", checksum=card["model_sha256"])
    if any(
        not np.array_equal(state["arrays"]["model/" + k], v.detach().numpy())
        for k, v in model.state_dict().items()
    ):
        raise ValueError("final model differs from optimizer checkpoint")
    del state
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(params["threads"])
        with srnet_block_validation.ValidationBlocks(
            Path(config["root"]),
            index_sha256=config["index_sha256"],
            audit=Path(config["audit"]),
            audit_sha256=config["audit_sha256"],
            deadline=deadline,
        ) as reader:
            report = srnet_block_validation.evaluate(
                model, reader, deadline, metrics=classification_metrics
            )
        # Before publication, rehash all validation bytes as well as train catalog.
        with srnet_block_validation.ValidationBlocks(
            Path(config["root"]),
            index_sha256=config["index_sha256"],
            audit=Path(config["audit"]),
            audit_sha256=config["audit_sha256"],
            deadline=deadline,
        ):
            pass
    finally:
        torch.set_num_threads(previous)
    deadline()
    if (
        sources() != before
        or epochs.configuration(config) != params
        or document(model_dir / "epoch.json", card_sha) != card
        or epochs.file_sha(model_dir / "model.npz", maximum=srnet_model.MAX_BYTES)
        != card["model_sha256"]
    ):
        raise ValueError("validation dependencies changed before publication")
    report.update(
        schema_version="srnet-epoch-development-validation-v1",
        plan_sha256=plan_sha,
        final_card_sha256=card_sha,
        model_sha256=card["model_sha256"],
        index_sha256=config["index_sha256"],
        audit_sha256=config["audit_sha256"],
        protocol_sha256=config["protocol_sha256"],
        source_sha256=before,
        training_source_sha256=plan["source_sha256"],
        seconds=time.monotonic() - started,
        cross_source_heldout=False,
        qualification="development_only_not_supported",
    )
    deadline()
    write_json(out, report)
    return report


def run_job(args):
    epochs.legacy.fresh(args.out)
    before = sources()
    config_sha = epochs.file_sha(args.config)
    config = document(args.config, config_sha)
    params = epochs.configuration(config)
    command = [sys.executable, "-m", "steganography.research_srnet_epoch_validation", "--worker"]
    for key, value in vars(args).items():
        if key not in {"worker", "config_sha256"}:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(["--config-sha256", config_sha])
    try:
        with tempfile.TemporaryFile() as stream:
            result = subprocess.run(  # noqa: S603 — fixed trusted isolated module, no shell
                command,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.DEVNULL,
                timeout=params["max_seconds"] + 30,
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )
            stream.seek(0)
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("validation worker response oversized")
        response = json.loads(raw)
        if (
            result.returncode != 0
            or not isinstance(response, dict)
            or response.get("status") != "completed"
        ):
            raise ValueError("validation worker incomplete/unavailable")
        report = document(args.out, response["report_sha256"])
        if (
            sources() != before
            or report.get("source_sha256") != before
            or epochs.file_sha(args.config) != config_sha
        ):
            raise ValueError("validation worker provenance changed")
        return report
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("validation job failed; partial outputs unusable") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "out", "plan", "model-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha", "card-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            import resource

            config = document(args.config, args.config_sha256)
            params = epochs.configuration(config)
            # CPU evaluation in a GPU-capable Torch build reserves CUDA VAs;
            # require the same hard resident cgroup bound, not RLIMIT_AS.
            epochs.srnet_cuda.host_bound()
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (2 * params["max_seconds"] + 30, 2 * params["max_seconds"] + 31),
            )
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2,) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            execute(
                config,
                args.out,
                plan_path=args.plan,
                plan_sha=args.plan_sha,
                model_dir=args.model_dir,
                card_sha=args.card_sha,
            )
            print(json.dumps({"status": "completed", "report_sha256": epochs.file_sha(args.out)}))
        else:
            report = run_job(args)
            print(json.dumps({"status": report["status"]}))
            return 0 if report["status"] == "completed" else 2
    except Exception:
        print('{"status":"failed"}')
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
