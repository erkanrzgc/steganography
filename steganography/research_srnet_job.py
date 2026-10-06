"""Fixed SRNet fit subprocess with hard limits; never runs extracted code."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core.srnet_model import MAX_BYTES
from steganography.research_features import read_document
from steganography.research_pixels import regular
from steganography.research_srnet_fit import configuration, train_srnet

OUTPUT_LIMIT = 65536
ADDRESS_LIMIT = 8 * 1024**3


def run_job(config_path: Path, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet job output exists or uses a symlink")
    config, digest = read_document(config_path)
    params = configuration(config)
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": str(params["threads"]),
        "OPENBLAS_NUM_THREADS": "1",
    }
    try:
        with tempfile.TemporaryFile() as stream:
            result = subprocess.run(  # noqa: S603 — fixed trusted module, no shell
                [
                    sys.executable,
                    "-m",
                    "steganography.research_srnet_job",
                    str(config_path.resolve()),
                    str(out.absolute()),
                    digest,
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.DEVNULL,
                cwd=root,
                env=environment,
                timeout=params["max_seconds"] + 120,
                check=False,
            )
            stream.seek(0)
            raw = stream.read(OUTPUT_LIMIT + 1)
        if len(raw) > OUTPUT_LIMIT:
            raise ValueError("oversized worker response")
        response = json.loads(raw)
        if isinstance(response, dict) and response.get("status") == "unavailable":
            raise RuntimeError("SRNet job unavailable: dependency/resource limits unavailable")
        if (
            result.returncode != 0
            or not isinstance(response, dict)
            or response.get("status") != "completed"
        ):
            raise ValueError("incomplete worker response")
        card, checksum = read_document(out / "model-card.json")
        model_path = out / "model.npz"
        regular(model_path)
        if model_path.stat().st_size > MAX_BYTES:
            raise ValueError("oversized model")
        with model_path.open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
        if (
            checksum != response.get("card_sha256")
            or len(data) > MAX_BYTES
            or hashlib.sha256(data).hexdigest() != response.get("model_sha256")
            or card.get("model_sha256") != response.get("model_sha256")
            or card.get("training_plan_sha256") != config["plan_sha256"]
            or card.get("training") != "completed"
            or card.get("deployed") is not False
        ):
            raise ValueError("worker artifact integrity mismatch")
        return card
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "SRNet job hard wall deadline exceeded; incomplete outputs are unusable"
        ) from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "SRNet job failed or artifacts invalid; incomplete outputs are unusable"
        ) from exc


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    response = {"status": "failed"}
    try:
        if len(args) != 3:
            raise ValueError("invalid worker arguments")
        config_path, out, expected = Path(args[0]), Path(args[1]), args[2]
        config, digest = read_document(config_path)
        if digest != expected:
            raise ValueError("config identity changed")
        params = configuration(config)
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_LIMIT, ADDRESS_LIMIT))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (2 * params["max_seconds"] + 60, 2 * params["max_seconds"] + 61),
            )
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            import torch  # noqa: F401 — explicit optional-dependency availability check
        except (ImportError, OSError, ValueError):
            response = {"status": "unavailable"}
        else:
            card = train_srnet(config, out)
            _, card_sha = read_document(out / "model-card.json")
            response = {
                "status": "completed",
                "card_sha256": card_sha,
                "model_sha256": card["model_sha256"],
            }
    except Exception:
        response = {"status": "failed"}
    print(json.dumps(response))
    return 0 if response["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
