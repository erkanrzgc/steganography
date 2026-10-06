#!/usr/bin/env python3
"""Evaluate the frozen two-source SRNet control, retaining failed metric cells."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_srnet_evaluate import evaluate  # noqa: E402
from steganography.research_srnet_results import summarize  # noqa: E402

PROTOCOL_SHA = "c84a1357195fdc59650679ed9878eef66d1f3c89784a69addd90beec894b031b"
MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"
PLAN_SHA = "1fc01723c4f10d717bf7350e7c982d22f6adf58d32bcbda8be1fd5860a3ecea9"
TRAIN_CACHE_SHA = "828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028"
VALIDATION_CACHE_SHA = "1c8c25244f204aaa0cad981d032241c06f7535b31bd03d7c547818dc046ae3a6"
RECIPE = "two-source-two-lineage-pairs-v1"
FIXED_OPTIMIZER = {
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


def check_frozen(plan, plan_sha, card):
    if (
        plan_sha != PLAN_SHA
        or plan.get("settings") != {"epochs": 1, "seed": 20261012, "batch_recipe": RECIPE}
        or card.get("schema_version") != "srnet-fit-v2"
        or card.get("optimizer") != FIXED_OPTIMIZER
        or card.get("batch_size") != 4
        or card.get("training_scope", {}).get("rows") != 2985
        or card.get("epoch_training", [{}])[0].get("updates") != 1580
    ):
        raise ValueError("frozen multipair plan/optimizer/update contract mismatch")


def comparison(current, previous):
    """Descriptive point changes only; changed training context is confounded."""

    def key(cell):
        return cell["source_id"], cell["quality_factor"], cell["method"]

    before = {key(c): c for c in previous["cells"]}
    after = {key(c): c for c in current["cells"]}
    if len(before) != 6 or len(after) != 6 or before.keys() != after.keys():
        raise ValueError("complete same-context comparison required")
    metrics = (
        "roc_auc",
        "balanced_accuracy",
        "recall",
        "false_positive_rate",
        "expected_calibration_error",
    )
    return [
        {
            "source_id": k[0],
            "quality_factor": k[1],
            "method": k[2],
            "point_change": {m: after[k]["metrics"][m] - before[k]["metrics"][m] for m in metrics},
        }
        for k in sorted(after, key=str)
    ]


def audit(job: Path, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("multipair audit output must be fresh and non-symlink")
    if (
        hashlib.sha256((ROOT / "docs/SRNET_MULTIPAIR_PROTOCOL.md").read_bytes()).hexdigest()
        != PROTOCOL_SHA
    ):
        raise ValueError("frozen multipair protocol changed")
    plan_path = ROOT / ".benchmark/srnet-multipair-accounting-20261007/plan.json"
    plan, plan_sha = read_document(plan_path)
    card, card_sha = read_document(job / "model/model-card.json")
    check_frozen(plan, plan_sha, card)
    cache = ROOT / ".benchmark/srnet-float-preparation-20261006"
    config = {
        "manifest": str(ROOT / ".benchmark/jpeg-context-multisource-20261005/manifest.json"),
        "manifest_sha256": MANIFEST_SHA,
        "cache": str(cache / "train/cache.json"),
        "cache_sha256": TRAIN_CACHE_SHA,
        "plan": str(plan_path),
        "plan_sha256": PLAN_SHA,
        "validation_cache": str(cache / "validation/cache.json"),
        "validation_cache_sha256": VALIDATION_CACHE_SHA,
        "model_dir": str(job / "model"),
        "card_sha256": card_sha,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "evaluation-config.json", config)  # Local, never portable.
    predictions = out / "predictions.json"
    evaluation = evaluate(config, predictions)
    prediction_sha = hashlib.sha256(predictions.read_bytes()).hexdigest()
    diagnostics = summarize(
        config, predictions, out / "diagnostics.json", predictions_sha256=prediction_sha
    )
    prior, prior_sha = read_document(ROOT / "benchmarks/srnet-real-pilot-20261006.json")
    result = {
        "schema_version": "srnet-multipair-real-evidence-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "837cb79",
        "fit_evaluation_base_commit": "1b0c80c",
        "execution_source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "core/srnet_multibatch.py",
                "core/srnet_training.py",
                "steganography/research_srnet_plan.py",
                "steganography/research_srnet_fit.py",
                "steganography/research_srnet_evaluate.py",
                "steganography/research_srnet_results.py",
                "scripts/audit-srnet-multipair-real.py",
            )
        },
        "manifest_sha256": MANIFEST_SHA,
        "training_plan_sha256": PLAN_SHA,
        "model_card_sha256": card_sha,
        "model_sha256": card["model_sha256"],
        "evaluation_seconds": evaluation["seconds"],
        "fit_card": card,
        "diagnostics": diagnostics,
        "prior_evidence_sha256": prior_sha,
        "comparison": comparison(diagnostics, prior["diagnostics"])
        if diagnostics["status"] == "completed"
        else [],
        "comparison_interpretation": (
            "descriptive reused-development points; source scope, BN context and "
            "update counts changed; not causal or blind qualification"
        ),
        "hardware": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
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
