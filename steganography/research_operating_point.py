"""Development-only whole-lineage threshold fitting and separated assessment."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.feature_model import feature_model
from steganography.research import ResearchManifestError
from steganography.research_features import feature_inputs, read_document, read_feature_checkpoint
from steganography.research_jpeg import write_json
from steganography.research_spatial import METHODS, RATES, cell_metrics, paired_intervals


def role(lineage: str) -> str:
    digest = hashlib.sha256(f"operating-point:20261005:{lineage}".encode()).digest()
    return "fit" if int.from_bytes(digest[:8], "big") / 2**64 < 0.5 else "assessment"


def choose_threshold(cover_scores: list[float]) -> float:
    if len(cover_scores) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in cover_scores):
        raise ResearchManifestError("threshold fitting requires finite cover scores")
    covers = np.clip(np.asarray(cover_scores), 1e-12, 1 - 1e-12)
    allowed = math.floor(len(covers) * 0.03)
    return float(np.nextafter(np.sort(covers)[::-1][allowed], np.inf))


def run_assessment(config_path: Path, checkpoint_path: Path, out: Path, *, checkpoint_sha256: str):
    import torch

    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("operating-point output exists or uses a symlink")
    config, _ = read_document(config_path)
    x, _, provenance = feature_inputs(config, split="validation")
    checkpoint = read_feature_checkpoint(checkpoint_path, expected_sha256=checkpoint_sha256)
    if (
        checkpoint["domain"] != "spatial-cooccurrence-linear-v1"
        or checkpoint["preprocessing"].get("inference_arithmetic") != "float64"
        or checkpoint["preprocessing"]["feature_version"] != provenance["feature_version"]
        or checkpoint["preprocessing"]["feature_names"] != provenance["feature_names"]
        or checkpoint["training_provenance"]["manifest_sha256"] != provenance["manifest_sha256"]
    ):
        raise ResearchManifestError("precise spatial checkpoint/validation contract mismatch")
    manifest, _ = read_document(Path(config["manifest"]))
    samples = [s for s in manifest["samples"] if s["split"] == "validation"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(sample["lineage"], []).append(sample)
    expected = {(None, 0), *((method, rate) for method in METHODS for rate in RATES)}
    for group in groups.values():
        if len(group) != 7 or {(s.get("method"), s.get("rate_percent")) for s in group} != expected:
            raise ResearchManifestError("operating point requires complete controlled lineages")
        if any(s["label"] != ("cover" if s["method"] is None else "stego") for s in group):
            raise ResearchManifestError("operating point label mismatch")
    if any(sum(role(lineage) == r for lineage in groups) < 2 for r in ("fit", "assessment")):
        raise ResearchManifestError("both development roles need at least two lineages")
    with torch.no_grad():
        logits = feature_model(checkpoint).eval()(torch.from_numpy(x))
        probabilities = torch.sigmoid(logits.double()).numpy().reshape(-1)
    if not np.isfinite(probabilities).all():
        raise ResearchManifestError("nonfinite operating-point scores")
    rows = [
        {
            "sha256": s["sha256"],
            "lineage": s["lineage"],
            "label": s["label"],
            "method": s["method"],
            "rate_percent": s["rate_percent"],
            "role": role(s["lineage"]),
            "score": float(np.clip(p, 1e-12, 1 - 1e-12)),
        }
        for s, p in zip(samples, probabilities, strict=True)
    ]
    fit_covers = [r["score"] for r in rows if r["role"] == "fit" and r["label"] == "cover"]
    threshold = choose_threshold(fit_covers)
    assessment = [r for r in rows if r["role"] == "assessment"]
    cells = {}
    for method in METHODS:
        for rate in RATES:
            selected = [
                r
                for r in assessment
                if r["label"] == "cover" or (r["method"] == method and r["rate_percent"] == rate)
            ]
            before = cell_metrics(selected, threshold=0.5, score_scale=1)
            after = cell_metrics(selected, threshold=threshold, score_scale=1)
            intervals = paired_intervals(selected, threshold=threshold, score_scale=1)
            cells[f"{method}-{rate}"] = {
                "before": before,
                "after": after,
                "after_bootstrap_95_percent": intervals,
            }
    report = {
        "schema_version": "spatial-operating-point-development-v1",
        "provenance": provenance,
        "checkpoint_sha256": checkpoint_sha256,
        "threshold": threshold,
        "threshold_scale": "probability",
        "fit_cover_count": len(fit_covers),
        "fit_false_positives": sum(s >= threshold for s in fit_covers),
        "fit_allowed_false_positives": math.floor(len(fit_covers) * 0.03),
        "assessment_lineages": sum(role(lineage) == "assessment" for lineage in groups),
        "by_method_rate": cells,
        "roles": {lineage: role(lineage) for lineage in sorted(groups)},
        "deployed": False,
        "probability_calibrated": False,
        "cross_source_gate": "unavailable",
        "scope": "previously inspected same-source validation; development assessment only",
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "scores.json", {"rows": rows})
    report["scores_sha256"] = hashlib.sha256((out / "scores.json").read_bytes()).hexdigest()
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    result = run_assessment(
        args.config, args.checkpoint, args.out, checkpoint_sha256=args.checkpoint_sha256
    )
    print(f"development threshold={result['threshold']}; deployed=False")


if __name__ == "__main__":
    main()
