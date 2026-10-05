"""Explicit ALASKA2 development import and validation-only model prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from core.dataset import identity_keys
from core.jpeg_features import FEATURE_VERSION, MAX_IMAGE_BYTES
from steganography.benchmarking.metrics import classification_metrics
from steganography.research import (
    ResearchManifestError,
    export_onnx,
    train_model,
    verify_dataset_manifest,
)
from steganography.research_features import extract_features, read_document
from steganography.research_features import predict_validation as predict_feature_validation


def write_json(path: Path, document: dict[str, Any]) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ResearchManifestError("symlink research output is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def import_development(
    root: Path,
    out: Path,
    *,
    source_sha256: str,
    reserved_manifests: list[Path],
) -> dict[str, Any]:
    source, actual_hash = read_document(root / "source.json")
    if actual_hash != source_sha256:
        raise ResearchManifestError("development source checksum mismatch")
    if (
        source.get("schema_version") != "alaska2-acquisition-v1"
        or source.get("purpose") != "development"
    ):
        raise ResearchManifestError("only an explicit development acquisition may be imported")
    selection, selection_hash = read_document(root / "selection.json")
    if source.get("selection_sha256") != selection_hash:
        raise ResearchManifestError("development selection checksum mismatch")
    if not reserved_manifests:
        raise ResearchManifestError("frozen reserved manifests are mandatory")
    reserved = set()
    reserved_hashes = []
    for path in reserved_manifests:
        document, digest = read_document(path)
        if not document.get("samples"):
            raise ResearchManifestError("empty reserved manifest")
        for row in document["samples"]:
            if not isinstance(row, dict) or not identity_keys(row):
                raise ResearchManifestError("invalid reserved identity")
            reserved.update(identity_keys(row))
        reserved_hashes.append(digest)
    if set(reserved_hashes) != set(source.get("reserved_manifest_sha256", [])):
        raise ResearchManifestError("reserved acquisition provenance mismatch")
    rows = source.get("samples", [])
    if not 4 <= len(rows) <= 4000:
        raise ResearchManifestError("development sample count outside limits")
    if sorted(r["path"] for r in rows) != sorted(r["path"] for r in selection.get("members", [])):
        raise ResearchManifestError("development selection membership mismatch")
    groups: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        match = re.fullmatch(r"(Cover|JMiPOD|JUNIWARD|UERD)/([0-9]{5}\.jpg)", row["path"])
        if not match or row.get("source_group") != "ALASKA2":
            raise ResearchManifestError("invalid development path/source")
        family, name = match.groups()
        fraction = (
            int.from_bytes(
                hashlib.sha256(f"{source['selection_seed']}:{name}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        if (
            row.get("split") != ("train" if fraction < 0.8 else "validation")
            or row.get("label") != ("cover" if family == "Cover" else "stego")
            or row.get("method") != (None if family == "Cover" else family)
        ):
            raise ResearchManifestError("development labels/split policy mismatch")
        if identity_keys(row) & reserved:
            raise ResearchManifestError("development overlaps frozen identities")
        groups.setdefault(name, []).append(row)
        hashes.setdefault(row["sha256"], []).append(row)
        path = root / row["path"]
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ResearchManifestError("development input is not a regular nonsymlink file")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if (
            len(data) > MAX_IMAGE_BYTES
            or len(data) != row["size"]
            or hashlib.sha256(data).hexdigest() != row["sha256"]
        ):
            raise ResearchManifestError("development image integrity failed")
    for group in groups.values():
        if len(group) != 4 or {r["method"] for r in group} != {None, "JMiPOD", "JUNIWARD", "UERD"}:
            raise ResearchManifestError("incomplete development lineage")
        cover = next(r for r in group if r["label"] == "cover")
        if any(r.get("lineage") != cover["sha256"] for r in group):
            raise ResearchManifestError("development lineage mismatch")
    quarantined = {
        r["lineage"]
        for duplicate in hashes.values()
        if {r["label"] for r in duplicate} == {"cover", "stego"}
        for r in duplicate
    }
    retained = [r for r in rows if r["lineage"] not in quarantined]
    if any(
        not any(r["split"] == split and r["label"] == label for r in retained)
        for split in ("train", "validation")
        for label in ("cover", "stego")
    ):
        raise ResearchManifestError("each development split needs both labels")
    manifest = {
        "schema_version": "1.0",
        "source": ".",
        "samples": retained,
        "catalog": {"license": source["license"], "source_url": source["source_url"]},
        "source_sha256": actual_hash,
        "selection_sha256": selection_hash,
        "partition": {
            "policy": "identity-camera-device-development-v1",
            "test_sources": [],
            "reserved_manifest_sha256": sorted(reserved_hashes),
            "seed": source["selection_seed"],
            "camera_device_metadata_complete": False,
            "support_status": "experimental",
            "cross_source_gate": "unavailable",
        },
        "quarantined_lineages": sorted(quarantined),
        "quarantine_reason": (
            "conflicting source labels on byte-identical files; entire lineage excluded"
        ),
    }
    verify_dataset_manifest(manifest, source=root, verify_files=False)
    write_json(out, manifest)
    return manifest


def predict_validation(config_path: Path, checkpoint_path: Path, out: Path) -> dict[str, Any]:
    """Preserve the JPEG interface/schema through shared feature inference."""
    return predict_feature_validation(
        config_path,
        checkpoint_path,
        out,
        required_domain="jpeg-dct-summary-linear-v1",
        schema_version="jpeg-development-predictions-v1",
    )


def run_development(
    root: Path,
    out: Path,
    *,
    source_sha256: str,
    reserved_manifests: list[Path],
) -> dict[str, Any]:
    """One fixed CPU experiment; outputs never enter the installed model catalog."""
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ResearchManifestError("symlink experiment output is forbidden")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest_path = out / "manifest.json"
    manifest = import_development(
        root, manifest_path, source_sha256=source_sha256, reserved_manifests=reserved_manifests
    )
    feature_summaries = {}
    for split in ("train", "validation"):
        features_path = out / f"{split}-features.json"
        print(f"extracting {split} JPEG features", flush=True)
        feature_summaries[split] = extract_features(
            manifest_path,
            features_path,
            source=root,
            split=split,
            feature_version=FEATURE_VERSION,
            workers=4,
        )
        write_json(
            out / f"{split}-config.json",
            {
                "manifest": str(manifest_path),
                "features": str(features_path),
                "features_sha256": feature_summaries[split]["sha256"],
                "epochs": 300,
                "learning_rate": 0.01,
                "seed": 20261005,
            },
        )
    checkpoint = out / "baseline.pt"
    print("training fixed CPU baseline on train split only", flush=True)
    training = train_model(out / "train-config.json", checkpoint)
    write_json(out / "training.json", training)
    predictions = predict_validation(
        out / "validation-config.json", checkpoint, out / "validation-predictions.json"
    )
    card = export_onnx(checkpoint, out / "baseline.onnx")
    by_method = {}
    for method in ("JMiPOD", "JUNIWARD", "UERD"):
        rows = [r for r in predictions["predictions"] if r["method"] in (None, method)]
        metrics = classification_metrics(
            [(r["label"] == "stego", r["score"] * 100) for r in rows],
            threshold=50,
            recommend_threshold=False,
        )
        metrics.pop("recommended_threshold")
        metrics.pop("average_precision")
        by_method[method] = metrics
    report = {
        "schema_version": "jpeg-development-run-v1",
        "scope": "same-source validation only",
        "source_sha256": source_sha256,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": predictions["checkpoint_sha256"],
        "onnx_sha256": card["onnx_sha256"],
        "prediction_sha256": hashlib.sha256(
            (out / "validation-predictions.json").read_bytes()
        ).hexdigest(),
        "quarantined_lineages": manifest["quarantined_lineages"],
        "features": feature_summaries,
        "training": training,
        "by_method": by_method,
        "calibrated": False,
        "deployed": False,
        "support_status": "experimental",
        "cross_source_gate": "unavailable",
        "old_test_images_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    acquire = sub.add_parser("import-development")
    acquire.add_argument("--source", type=Path, required=True)
    acquire.add_argument("--source-sha256", required=True)
    acquire.add_argument("--reserved-manifest", action="append", type=Path, required=True)
    acquire.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run-development")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--source-sha256", required=True)
    run.add_argument("--reserved-manifest", action="append", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    predict = sub.add_parser("predict-validation")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "run-development":
            result = run_development(
                args.source,
                args.out,
                source_sha256=args.source_sha256,
                reserved_manifests=args.reserved_manifest,
            )
            print(f"development complete; deployed={result['deployed']}")
        elif args.action == "import-development":
            result = import_development(
                args.source,
                args.out,
                source_sha256=args.source_sha256,
                reserved_manifests=args.reserved_manifest,
            )
            print(
                f"development rows: {len(result['samples'])}; quarantined lineages: "
                f"{len(result['quarantined_lineages'])}"
            )
        else:
            result = predict_validation(args.config, args.checkpoint, args.out)
            print(f"validation predictions: {len(result['predictions'])}; not a test result")
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"JPEG research failed ({type(exc).__name__}); details withheld\n")


if __name__ == "__main__":
    main()
