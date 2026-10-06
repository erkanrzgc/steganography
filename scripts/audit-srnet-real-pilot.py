#!/usr/bin/env python3
"""Audit the frozen real SRNet pilot; no fitting, downloads or installation."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_srnet_evaluate import evaluate  # noqa: E402
from steganography.research_srnet_results import summarize  # noqa: E402

PROTOCOL_SHA = "60f6595d1d4d696eaa3ed88acf0f9aeaebf261f48736eec9873052a5bce58c84"
MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"
PLAN_SHA = "27634b6576a1428669d3d8c7c490544f86a59a3449c5b7e304fbbd2dac38bdda"
TRAIN_CACHE_SHA = "828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028"
VALIDATION_CACHE_SHA = "1c8c25244f204aaa0cad981d032241c06f7535b31bd03d7c547818dc046ae3a6"


def audit(job: Path, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("pilot audit output must be fresh and non-symlink")
    protocol = ROOT / "docs/SRNET_REAL_PILOT_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("frozen pilot protocol changed")
    manifest = ROOT / ".benchmark/jpeg-context-multisource-20261005/manifest.json"
    cache = ROOT / ".benchmark/srnet-float-preparation-20261006"
    plan, plan_sha = read_document(job / "plan.json")
    card, card_sha = read_document(job / "model/model-card.json")
    fixed_optimizer = {
        "name": "Adamax",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "foreach": False,
        "lr_schedule": "constant",
        "threads": 2,
        "max_seconds": 1800,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    if (
        plan_sha != PLAN_SHA
        or card.get("optimizer") != fixed_optimizer
        or plan.get("settings") != {"epochs": 1, "seed": 20261012}
        or card.get("epoch_training", [{}])[0].get("updates") != 412
    ):
        raise ValueError("frozen pilot plan/optimizer/update contract mismatch")
    config = {
        "manifest": str(manifest),
        "manifest_sha256": MANIFEST_SHA,
        "cache": str(cache / "train/cache.json"),
        "cache_sha256": TRAIN_CACHE_SHA,
        "plan": str(job / "plan.json"),
        "plan_sha256": PLAN_SHA,
        "validation_cache": str(cache / "validation/cache.json"),
        "validation_cache_sha256": VALIDATION_CACHE_SHA,
        "model_dir": str(job / "model"),
        "card_sha256": card_sha,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "evaluation-config.json", config)  # Local only, never portable report.
    predictions = out / "predictions.json"
    evaluation = evaluate(config, predictions)
    prediction_sha = hashlib.sha256(predictions.read_bytes()).hexdigest()
    diagnostics = summarize(
        config, predictions, out / "diagnostics.json", predictions_sha256=prediction_sha
    )
    result = {
        "schema_version": "srnet-real-pilot-evidence-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "ff24d10",
        "fit_evaluation_base_commit": "8a73ffa",
        "diagnostic_service_sha256": hashlib.sha256(
            (ROOT / "steganography/research_srnet_results.py").read_bytes()
        ).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": MANIFEST_SHA,
        "training_plan_sha256": PLAN_SHA,
        "model_card_sha256": card_sha,
        "model_sha256": card["model_sha256"],
        "evaluation_seconds": evaluation["seconds"],
        "fit_card": card,
        "diagnostics": diagnostics,
        "hardware": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "vm_vcpu": 8,
            "vm_memory_gib": 15,
            "fit_math_threads": 2,
        },
        "qualification": "unavailable",
        "deployed": False,
        "raw_data_and_model_published": False,
        "onnx_replay": "unavailable; separate full replay remains pending",
    }
    write_json(out / "evidence.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.job, args.out)
    print(result["diagnostics"]["status"])
    return 0 if result["diagnostics"]["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
