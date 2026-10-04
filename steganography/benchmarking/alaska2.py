"""Frozen ALASKA2 native-score evaluation; no training or threshold selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import re
import shutil
import stat
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.service import AnalysisService
from steganography.benchmarking.metrics import classification_metrics

METHODS = ("JMiPOD", "JUNIWARD", "UERD")
REQUIRED = (
    "image_jpeg",
    "image_jpeg_dct",
    "filestruct_appended",
    "filestruct_exif",
    "file_structure",
    "signatures",
)
MAX_IMAGE = 2 * 1024 * 1024
MAX_MANIFEST = 8 * 1024 * 1024
SEED = 20261004
DETECTOR_REVISION = "f929429a1603e5819156c8462d124ba1fe84c36a"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def no_symlinks(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink paths are forbidden")


def bounded_read(path: Path, maximum: int) -> bytes:
    no_symlinks(path)
    # Nonblocking open prevents a FIFO from hanging before the regular-file check.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= maximum:
            raise ValueError("input is not a bounded regular file")
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("input exceeds byte limit")
    return data


def detector_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        [
            *root.joinpath("core").rglob("*.py"),
            *root.joinpath("modules").rglob("*.py"),
            root / "registry.py",
            root / "config.py",
        ]
    )
    return digest(
        b"".join(
            path.relative_to(root).as_posix().encode()
            + b"\0"
            + digest(bounded_read(path, MAX_MANIFEST)).encode()
            + b"\n"
            for path in paths
        )
    )


def load_source(root: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    raw = bounded_read(root / "source.json", MAX_MANIFEST)
    if digest(raw) != expected_sha256:
        raise ValueError("source manifest checksum mismatch")
    source = json.loads(raw)
    if source.get("schema_version") != "alaska2-acquisition-v1":
        raise ValueError("unsupported acquisition schema")
    selection = bounded_read(root / "selection.json", MAX_MANIFEST)
    if digest(selection) != source.get("selection_sha256"):
        raise ValueError("selection checksum mismatch")
    samples = source.get("samples", [])
    if not 4 <= len(samples) <= 4000 or source.get("source_group") != "ALASKA2":
        raise ValueError("invalid ALASKA2 sample count or source")
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in samples:
        path = row.get("path", "")
        if not isinstance(path, str) or not re.fullmatch(
            r"(Cover|JMiPOD|JUNIWARD|UERD)/[0-9]{5}\.jpg", path
        ):
            raise ValueError("unsafe sample path")
        folder, name = path.split("/")
        group = groups.setdefault(name, {})
        if folder in group:
            raise ValueError("duplicate sample path")
        group[folder] = row
        if (
            row.get("split") != "test"
            or row.get("source_group") != "ALASKA2"
            or row.get("label") != ("cover" if folder == "Cover" else "stego")
            or row.get("method") != (None if folder == "Cover" else folder)
            or row.get("format") != "JPEG"
            or (row.get("width"), row.get("height")) != (512, 512)
            or type(row.get("size")) is not int
            or not 0 < row["size"] <= MAX_IMAGE
            or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", "")))
        ):
            raise ValueError("invalid sample provenance")
    lineages = set()
    for group in groups.values():
        if set(group) != {"Cover", *METHODS}:
            raise ValueError("incomplete four-way lineage")
        lineage = group["Cover"]["sha256"]
        if lineage in lineages or any(r.get("lineage") != lineage for r in group.values()):
            raise ValueError("invalid or duplicate cover lineage")
        lineages.add(lineage)
    selected = json.loads(selection).get("members", [])
    if sorted(r["path"] for r in samples) != sorted(r["path"] for r in selected):
        raise ValueError("selection membership mismatch")
    return source, digest(raw)


def score_sample(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    root, sample = item
    # Copy only explicitly portable fields, never arbitrary source metadata.
    row = {key: sample[key] for key in ("path", "sha256", "label", "method", "lineage")}
    row.update(score=None, status="failed", coverage={}, signals=[])
    started = time.monotonic()
    try:
        if not re.fullmatch(r"(Cover|JMiPOD|JUNIWARD|UERD)/[0-9]{5}\.jpg", row["path"]):
            raise ValueError("unsafe sample path")
        path = Path(root) / row["path"]
        data = bounded_read(path, MAX_IMAGE)
        if digest(data) != sample["sha256"] or len(data) != sample["size"]:
            raise ValueError("sample checksum or size mismatch")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.size != (512, 512):
                raise ValueError("unexpected JPEG dimensions")
            image.load()
        report = AnalysisService(
            profile="balanced", ai_provider=None, max_file_size=MAX_IMAGE
        ).analyze(path)
        if (
            report.file.sha256 != sample["sha256"]
            or digest(bounded_read(path, MAX_IMAGE)) != row["sha256"]
        ):
            raise ValueError("sample changed during analysis")
        row.update(
            score=report.overall_score,
            status="completed",
            coverage={r.analyzer: r.status for r in report.results if r.analyzer != "ai_triage"},
            signals=[
                {"analyzer": r.analyzer, "code": s.name, "score": s.score}
                for r in report.results
                if r.analyzer != "ai_triage"
                for s in r.signals
                if s.score >= 30
            ],
        )
    except Exception as exc:
        # Exception messages and plugin details can contain host paths or secrets.
        row["failure"] = type(exc).__name__
    row["seconds"] = time.monotonic() - started
    return row


def fixed_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = classification_metrics(
        [(r["label"] == "stego", r["score"]) for r in rows],
        threshold=70,
        recommend_threshold=False,
    )
    for key in ("recommended_threshold", "average_precision"):
        metrics.pop(key)
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "failed_samples": sum(r["status"] != "completed" for r in rows),
        "analysis_error_samples": sum("error" in r["coverage"].values() for r in rows),
        "missing_required_samples": sum(
            any(r["coverage"].get(name) != "ok" for name in REQUIRED) for r in rows
        ),
        "coverage": {
            name: dict(Counter(r["coverage"].get(name, "missing") for r in rows))
            for name in sorted({*REQUIRED, *(n for r in rows for n in r["coverage"])})
        },
        "status": "unavailable",
        "metrics": None,
    }
    if not rows or any(
        result[k] for k in ("failed_samples", "analysis_error_samples", "missing_required_samples")
    ):
        return result
    result.update(status="completed", metrics=fixed_metrics(rows))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["lineage"], []).append(row)
    grouped = list(groups.values())
    draws: dict[str, list[float]] = {
        key: [] for key in ("roc_auc", "balanced_accuracy", "recall", "false_positive_rate")
    }
    rng = random.Random(SEED)  # noqa: S311 - fixed statistical bootstrap, not cryptography
    for _ in range(200):
        metrics = fixed_metrics([r for g in rng.choices(grouped, k=len(grouped)) for r in g])
        for key, values in draws.items():
            if metrics[key] is not None:
                values.append(metrics[key])
    result["lineage_bootstrap_95_ci"] = {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in draws.items()
        if values
    }
    return result


def evaluate(
    root: Path, out: Path, *, source_sha256: str, protocol: Path, workers: int = 4
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    source, source_hash = load_source(root, source_sha256)
    protocol_hash = digest(bounded_read(protocol, MAX_MANIFEST))
    no_symlinks(out)
    out.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": "alaska2-baseline-v1",
        "reference_detector_revision": DETECTOR_REVISION,
        "detector_code_sha256": detector_fingerprint(),
        "source_sha256": source_hash,
        "selection_sha256": source["selection_sha256"],
        "protocol_sha256": protocol_hash,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "threshold": 70,
        "profile": "balanced",
        "ai_enabled": False,
        "training_performed": False,
        "threshold_search_performed": False,
        "support_status": "experimental",
        "cross_source_gate": "unavailable",
        "ece": {"status": "unavailable", "reason": "uncalibrated heuristic scores"},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "workers": workers,
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "Pillow")},
            "tools_available": {
                n: bool(shutil.which(n)) for n in ("exiftool", "stegseek", "zsteg")
            },
        },
    }
    try:
        provenance["environment"]["packages"]["jpeglib"] = importlib.metadata.version("jpeglib")
    except importlib.metadata.PackageNotFoundError:
        provenance["environment"]["packages"]["jpeglib"] = "unavailable"
    write_json(out / "run.json", provenance)
    started = time.monotonic()
    rows = []
    with (
        (out / "scores.jsonl").open("x", encoding="utf-8") as stream,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        for row in executor.map(score_sample, ((str(root), s) for s in source["samples"])):
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 100 == 0:
                print(f"analyzed {len(rows)}/{len(source['samples'])}", flush=True)
    covers = {r["lineage"]: r for r in rows if r["label"] == "cover"}
    ambiguous = [
        r["path"]
        for r in rows
        if r["label"] == "stego" and r["sha256"] == covers[r["lineage"]]["sha256"]
    ]
    ambiguous_uerd = {
        r["lineage"] for r in rows if r["method"] == "UERD" and r["path"] in ambiguous
    }
    by_method = {m: summarize([r for r in rows if r["method"] in (None, m)]) for m in METHODS}
    result = {
        **provenance,
        "samples": len(rows),
        "lineages": len(covers),
        "status": "completed"
        if all(m["status"] == "completed" for m in by_method.values())
        else "unavailable",
        "by_method": by_method,
        "byte_identical_stego_paths": ambiguous,
        "uerd_sensitivity_excluding_identical_pairs": summarize(
            [
                r
                for r in rows
                if r["method"] in (None, "UERD") and r["lineage"] not in ambiguous_uerd
            ]
        ),
        "scores_sha256": digest(bounded_read(out / "scores.jsonl", 32 * 1024 * 1024)),
        "elapsed_seconds": time.monotonic() - started,
        "latency": {
            "median_seconds": float(np.median([r["seconds"] for r in rows])),
            "p95_seconds": float(np.percentile([r["seconds"] for r in rows], 95)),
        },
    }
    write_json(out / "report.json", result)
    return result


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        report = evaluate(
            args.source,
            args.out,
            source_sha256=args.source_sha256,
            protocol=args.protocol,
            workers=args.workers,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"evaluation failed ({type(exc).__name__}); details withheld\n")
    print(f"evaluation {report['status']}: {report['samples']} files", flush=True)
    if report["status"] != "completed":
        parser.exit(1)


if __name__ == "__main__":
    main()
