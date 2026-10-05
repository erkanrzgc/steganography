"""Checksum-bound, matched-family JPEG development from two declared origins."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from core.jpeg_context import FEATURE_NAMES, FEATURE_VERSION, summary_context_features
from core.jpeg_features import MAX_IMAGE_BYTES
from steganography.research import ResearchManifestError, verify_dataset_manifest
from steganography.research_features import MAX_DOCUMENT_BYTES, feature_inputs, read_document
from steganography.research_jpeg import write_json
from steganography.research_jpeg_corpus import METHODS


def prepare_dataset(
    alaska_experiment: Path,
    alaska_root: Path,
    boss: Path,
    out: Path,
    *,
    alaska_manifest_sha256: str,
    boss_manifest_sha256: str,
):
    """No downloads/label inference: preserve ancestry and original split roles."""
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("multi-source output exists or uses a symlink")
    original, original_hash = read_document(alaska_experiment / "manifest.json")
    generated, generated_hash = read_document(boss / "manifest.json")
    if original_hash != alaska_manifest_sha256 or generated_hash != boss_manifest_sha256:
        raise ResearchManifestError("multi-source input manifest checksum mismatch")
    verify_dataset_manifest(original, verify_files=False)
    verify_dataset_manifest(generated, verify_files=False)
    if any(s["split"] == "test" for m in (original, generated) for s in m["samples"]):
        raise ResearchManifestError("multi-source development must not include test rows")
    vectors = {}
    input_hashes = []
    for split in ("train", "validation"):
        config, _ = read_document(alaska_experiment / f"{split}-config.json")
        _, _, provenance = feature_inputs(config, split=split)
        if provenance["manifest_sha256"] != original_hash or (
            provenance["feature_version"] != "jpeg-dct-summary-v1"
        ):
            raise ResearchManifestError("ALASKA cached feature contract mismatch")
        artifact, digest = read_document(Path(config["features"]))
        input_hashes.append(digest)
        for row in artifact["rows"]:
            vectors[row["sha256"]] = summary_context_features(row["values"])
    artifact, digest = read_document(boss / "feature-rows.json")
    if digest != generated.get("feature_rows_sha256"):
        raise ResearchManifestError("BOSS cached feature checksum mismatch")
    input_hashes.append(digest)
    rows = artifact.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != len(generated["samples"])
        or any(
            not isinstance(r, dict) or any(r.get(k) != s[k] for k in ("sha256", "lineage", "label"))
            for r, s in zip(rows, generated["samples"], strict=True)
        )
    ):
        raise ResearchManifestError("BOSS cached feature identity mismatch")
    for row in rows:
        if row["sha256"] in vectors:
            raise ResearchManifestError("JPEG sources have duplicate image identities")
        values = row.get("values")
        if (
            not isinstance(values, list)
            or len(values) != len(FEATURE_NAMES)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in values
            )
        ):
            raise ResearchManifestError("invalid BOSS context feature vector")
        vectors[row["sha256"]] = row["values"]
    samples = []
    origins = {}
    sources = {}
    for index, (manifest, root) in enumerate(((original, alaska_root), (generated, boss))):
        for sample in manifest["samples"]:
            if sample["label"] != "cover" and sample.get("method") not in METHODS:
                continue
            source = sample.get("source_group")
            if not isinstance(source, str) or not source:
                raise ResearchManifestError("JPEG source groups must be declared")
            if index == 0:
                if source != "ALASKA2":
                    raise ResearchManifestError("ALASKA source declaration mismatch")
                origins[source] = {
                    "origin_manifest_sha256": original["source_sha256"],
                    "license": original["catalog"]["license"],
                    "source_url": original["catalog"]["source_url"],
                }
            else:
                origins[source] = generated["catalog"]["source_records"][source]
            relative = f"origin{index}/{sample['sha256']}.jpg"
            samples.append({**sample, "path": relative})
            sources[relative] = root / sample["path"]
    if len(origins) != 2 or len(samples) > 10_000 or sum(s["size"] for s in samples) > 2 * 1024**3:
        raise ResearchManifestError("multi-source corpus origins/count/byte limit mismatch")
    manifest = {
        "schema_version": "1.0",
        "source": ".",
        "samples": samples,
        "catalog": {"source_records": origins},
        "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
        "derivation": {
            "input_manifest_sha256": [original_hash, generated_hash],
            "input_features_sha256": input_hashes,
            "matched_methods": list(METHODS),
            "excluded_method": "JMiPOD: no matching simulator in conseal 2025.11",
            "scope": "declared multi-origin development, not independent blind qualification",
        },
    }
    verify_dataset_manifest(manifest, verify_files=False)
    out.mkdir(parents=True, exist_ok=False)
    for sample in samples:
        source_path = sources[sample["path"]]
        if (
            any(p.is_symlink() for p in (source_path, *source_path.parents))
            or not source_path.is_file()
        ):
            raise ResearchManifestError("JPEG source must be a regular nonsymlink file")
        with source_path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if (
            len(data) != sample["size"]
            or len(data) > MAX_IMAGE_BYTES
            or (hashlib.sha256(data).hexdigest() != sample["sha256"])
        ):
            raise ResearchManifestError("JPEG source size/hash mismatch")
        target = out / sample["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ResearchManifestError("multi-source artifact uses a symlink")
        with target.open("xb") as stream:
            stream.write(data)
    write_json(out / "manifest.json", manifest)
    artifacts = {}
    for split in ("train", "validation"):
        selected = [s for s in samples if s["split"] == split]
        path = out / f"{split}-features.json"
        artifact = {
            "schema_version": "research-features-1",
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
            "split": split,
            "rows": [
                {
                    **{k: s[k] for k in ("sha256", "lineage", "label")},
                    "values": vectors[s["sha256"]],
                }
                for s in selected
            ],
        }
        encoded = json.dumps(artifact, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > MAX_DOCUMENT_BYTES:
            raise ResearchManifestError("context feature artifact exceeds byte limit")
        with path.open("xb") as stream:
            stream.write(encoded)
        config = {
            "manifest": str(out / "manifest.json"),
            "features": str(path),
            "features_sha256": hashlib.sha256(encoded).hexdigest(),
            "sample_weighting": "jpeg-source-class-balanced-v1",
            "epochs": 300,
            "learning_rate": 0.01,
            "seed": 20261006,
            "standardize": True,
            "class_balanced": True,
            "inference_arithmetic": "float64",
        }
        _, _, provenance = feature_inputs(config, split=split)
        if split == "train":
            from steganography.research_weighting import training_weights

            training_weights(config, provenance)
        write_json(out / f"{split}-config.json", config)
        artifacts[split] = {
            "sha256": config["features_sha256"],
            "samples": len(selected),
            "bytes": len(encoded),
        }
    result = {
        "schema_version": "jpeg-multisource-preparation-v1",
        "sources": 2,
        "jpeg_files": len(samples),
        "feature_version": FEATURE_VERSION,
        "feature_artifacts": artifacts,
        "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
        "deployed": False,
        "support_status": "experimental",
    }
    write_json(out / "preparation.json", result)
    return result
