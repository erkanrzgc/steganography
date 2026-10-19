"""Audit-bound source/split feature preparation; no reserved media or fitting."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from core import nonjpeg_features as native
from core.nonjpeg_data import regular_read
from steganography.research_features import extract_features, read_document

MAX_SECONDS = 1800
MAX_OUTPUT_BYTES = 512 * 1024**2


def prepare_features(root: Path, audit_path: Path, out: Path) -> dict:
    started = time.monotonic()
    corpus, corpus_sha = read_document(root / "manifest.json")
    audit, audit_sha = read_document(audit_path)
    if (
        audit.get("status") != "completed"
        or audit.get("schema_version") != "nonjpeg-independent-audit-v1"
        or audit.get("manifest_sha256") != corpus_sha
        or audit.get("test_media_opened") is not False
        or audit.get("source_sha256") != corpus.get("source_sha256")
        or audit.get("artifact_count") != len(corpus.get("samples", []))
    ):
        raise ValueError("independent corpus audit binding failed")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("fresh non-symlink features directory required")
    if any(s.get("split") not in {"train", "validation"} for s in corpus["samples"]):
        raise ValueError("development-only corpus required")
    if any(
        s.get("source_group") not in {"BOWS2", "DIV2K", "ESC-50", "FSDD-v1.0.10"}
        for s in corpus["samples"]
    ):
        raise ValueError("unknown native source domain")
    out.mkdir(parents=True, mode=0o700)
    results = []
    total = 0
    for source in sorted({s["source_group"] for s in corpus["samples"]}):
        for split in ("train", "validation"):
            rows = [
                s for s in corpus["samples"] if s["source_group"] == source and s["split"] == split
            ]
            if not rows:
                continue
            versions = (
                [native.AUDIO_VERSION]
                if {s["format"] for s in rows} == {"WAV"}
                else [native.LOCAL_VERSION, *([native.RGB_VERSION] if source == "DIV2K" else [])]
            )
            manifest = {
                "schema_version": "1.0",
                "samples": rows,
                "partition": {
                    "policy": "identity-camera-device-development-v1",
                    "test_sources": [],
                },
                "preparation_manifest_sha256": corpus_sha,
                "independent_audit_sha256": audit_sha,
                "development_only": True,
            }
            name = f"{source.lower().replace('.', '-')}-{split}"
            manifest_path = out / f"{name}-manifest.json"
            with manifest_path.open("x") as stream:
                json.dump(manifest, stream, separators=(",", ":"), allow_nan=False)
            for version in versions:
                if time.monotonic() - started >= MAX_SECONDS:
                    raise TimeoutError("feature preparation deadline")
                artifact_path = out / f"{name}-{version}.json"
                result = extract_features(
                    manifest_path,
                    artifact_path,
                    split=split,
                    source=root,
                    feature_version=version,
                    workers=2,
                )
                total += artifact_path.stat().st_size
                if total > MAX_OUTPUT_BYTES:
                    raise ValueError("feature preparation output budget exceeded")
                results.append(
                    {
                        **result,
                        "source_group": source,
                        "manifest": manifest_path.name,
                        "manifest_sha256": hashlib.sha256(
                            regular_read(manifest_path, 8 * 1024**2)
                        ).hexdigest(),
                        "artifact": artifact_path.name,
                        "bytes": artifact_path.stat().st_size,
                        "features": len(native.CONTRACTS[version][0]),
                        "originals": len({s["lineage"] for s in rows}),
                    }
                )
    result = {
        "schema_version": "nonjpeg-feature-preparation-v1",
        "status": "ready",
        "corpus_sha256": corpus_sha,
        "audit_sha256": audit_sha,
        "artifacts": results,
        "test_media_opened": False,
        "optimizer_updates": 0,
    }
    with (out / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result
