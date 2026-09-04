"""Reproducible user-supplied dataset manifests and split assignment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from steganography.benchmarking.metrics import classification_metrics

_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
_LABELS = {"cover", "stego", "unknown"}
_SPLITS = {"train", "validation", "test"}
_LABEL_COMPONENTS = {"cover", "clean", "negative", "stego", "steg", "positive", "embedded"}


class ResearchManifestError(ValueError):
    """Raised when a dataset or prediction manifest is unsafe or inconsistent."""


def import_dataset(source: Path, out: Path, *, seed: int = 20260813) -> dict[str, Any]:
    source = Path(source).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    samples: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _EXTENSIONS:
            continue
        digest = _sha256(path)
        label = _infer_label(path, source)
        samples.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": digest,
                "size": path.stat().st_size,
                "label": label,
                "source_group": _infer_source_group(path, source),
            }
        )
    if not samples:
        raise ValueError("dataset contains no supported image files")
    hashes = [sample["sha256"] for sample in samples]
    if len(hashes) != len(set(hashes)):
        raise ValueError("dataset contains duplicate file hashes")
    for sample in samples:
        sample["split"] = _stable_split(sample["sha256"], seed)
    manifest = {
        "schema_version": "1.0",
        "source": str(source),
        "seed": seed,
        "sample_count": len(samples),
        "splits": {
            name: sum(sample["split"] == name for sample in samples)
            for name in ("train", "validation", "test")
        },
        "samples": samples,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchManifestError(f"cannot read dataset manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchManifestError("dataset manifest root must be an object")
    return value


def verify_dataset_manifest(
    manifest: dict[str, Any],
    *,
    source: Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    samples = manifest.get("samples")
    if manifest.get("schema_version") != "1.0" or not isinstance(samples, list):
        raise ResearchManifestError("unsupported or malformed dataset manifest")
    source_dir = Path(source or manifest.get("source", "")).resolve()
    seen_hashes: set[str] = set()
    split_hashes: dict[str, set[str]] = {name: set() for name in _SPLITS}
    labels = {name: 0 for name in _LABELS}
    source_groups: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ResearchManifestError(f"sample {index} must be an object")
        relative = Path(str(sample.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ResearchManifestError(f"sample {index} has an unsafe path")
        digest = str(sample.get("sha256", "")).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ResearchManifestError(f"sample {index} has an invalid SHA-256")
        if digest in seen_hashes:
            raise ResearchManifestError(f"duplicate sample hash: {digest}")
        seen_hashes.add(digest)
        split = sample.get("split")
        label = sample.get("label")
        if split not in _SPLITS or label not in _LABELS:
            raise ResearchManifestError(f"sample {index} has an invalid split or label")
        split_hashes[split].add(digest)
        labels[label] += 1
        source_groups.add(str(sample.get("source_group") or "unspecified"))
        if verify_files:
            path = source_dir / relative
            if not path.is_file():
                raise ResearchManifestError(f"dataset file is missing: {relative.as_posix()}")
            if path.stat().st_size != sample.get("size") or _sha256(path) != digest:
                raise ResearchManifestError(f"dataset file integrity failed: {relative.as_posix()}")
    overlap = (split_hashes["train"] & split_hashes["test"]) | (
        split_hashes["validation"] & split_hashes["test"]
    )
    if overlap:
        raise ResearchManifestError("train/validation and test hashes overlap")
    return {
        "sample_count": len(samples),
        "labels": labels,
        "splits": {name: len(values) for name, values in split_hashes.items()},
        "source_groups": sorted(source_groups),
        "split_isolation": True,
    }


def benchmark_predictions(
    manifest_path: Path,
    predictions_path: Path,
    out: Path,
    *,
    source: Path | None = None,
    threshold: float = 0.9,
    min_samples: int = 5_000,
    min_source_groups: int = 2,
    min_roc_auc: float = 0.8,
    max_fpr: float = 0.05,
) -> dict[str, Any]:
    """Evaluate independent test predictions and enforce product-claim gates."""
    if not 0 <= threshold <= 1 or not 0 <= min_roc_auc <= 1 or not 0 <= max_fpr <= 1:
        raise ValueError("threshold, minimum ROC-AUC and maximum FPR must be between 0 and 1")
    if min_samples < 1 or min_source_groups < 1:
        raise ValueError("minimum sample and source-group counts must be positive")
    manifest_path = Path(manifest_path)
    predictions_path = Path(predictions_path)
    manifest = load_dataset_manifest(manifest_path)
    integrity = verify_dataset_manifest(manifest, source=source, verify_files=True)
    predictions = _load_predictions(predictions_path)
    eligible = [
        sample
        for sample in manifest["samples"]
        if sample["split"] == "test" and sample["label"] in {"cover", "stego"}
    ]
    observations: list[tuple[bool, int]] = []
    missing: list[str] = []
    used_keys: set[str] = set()
    groups: set[str] = set()
    for sample in eligible:
        key = sample["sha256"] if sample["sha256"] in predictions else sample["path"]
        if key not in predictions:
            missing.append(sample["sha256"])
            continue
        score = predictions[key]
        used_keys.add(key)
        groups.add(str(sample.get("source_group") or "unspecified"))
        observations.append((sample["label"] == "stego", round(score * 100)))
    if not observations:
        raise ResearchManifestError("no labeled test predictions are available")
    metrics = classification_metrics(observations, threshold=round(threshold * 100))
    reasons = []
    if len(eligible) < min_samples:
        reasons.append(f"held-out sample count {len(eligible)} is below {min_samples}")
    if len(groups) < min_source_groups:
        reasons.append(f"source-group count {len(groups)} is below {min_source_groups}")
    if missing:
        reasons.append(f"{len(missing)} held-out predictions are missing")
    if not metrics["positives"] or not metrics["negatives"]:
        reasons.append("held-out data must contain both cover and stego labels")
    roc_auc = metrics["roc_auc"]
    if roc_auc is None or roc_auc < min_roc_auc:
        reasons.append(f"ROC-AUC {roc_auc} is below {min_roc_auc}")
    if metrics["false_positive_rate"] > max_fpr:
        reasons.append(
            f"false-positive rate {metrics['false_positive_rate']} exceeds {max_fpr}"
        )
    report = {
        "schema_version": "1.0",
        "status": "passed" if not reasons else "failed",
        "manifest_sha256": _sha256(manifest_path),
        "predictions_sha256": _sha256(predictions_path),
        "integrity": integrity,
        "evaluation": {
            "split": "test",
            "eligible_samples": len(eligible),
            "evaluated_samples": len(observations),
            "missing_predictions": missing,
            "unused_prediction_count": len(set(predictions) - used_keys),
            "source_groups": sorted(groups),
            "metrics": metrics,
        },
        "gates": {
            "threshold": threshold,
            "min_samples": min_samples,
            "min_source_groups": min_source_groups,
            "min_roc_auc": min_roc_auc,
            "max_fpr": max_fpr,
            "reasons": reasons,
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verify_split_isolation(manifest: dict[str, Any]) -> bool:
    observed: dict[str, str] = {}
    for sample in manifest.get("samples", []):
        digest = sample["sha256"]
        split = sample["split"]
        if digest in observed and observed[digest] != split:
            return False
        observed[digest] = split
    return True


def _infer_label(path: Path, root: Path) -> str:
    components = {part.lower() for part in path.relative_to(root).parts[:-1]}
    if components & {"stego", "steg", "positive", "embedded"}:
        return "stego"
    if components & {"cover", "clean", "negative"}:
        return "cover"
    return "unknown"


def _infer_source_group(path: Path, root: Path) -> str:
    components = [
        part for part in path.relative_to(root).parts[:-1] if part.lower() not in _LABEL_COMPONENTS
    ]
    return "/".join(components) or "unspecified"


def _stable_split(digest: str, seed: int) -> str:
    value = hashlib.sha256(f"{seed}:{digest}".encode()).digest()
    fraction = int.from_bytes(value[:8], "big") / 2**64
    return "train" if fraction < 0.7 else "validation" if fraction < 0.85 else "test"


def _load_predictions(path: Path) -> dict[str, float]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchManifestError(f"cannot read predictions: {exc}") from exc
    raw = document.get("predictions") if isinstance(document, dict) else document
    if isinstance(raw, dict):
        items = [{"key": key, "score": score} for key, score in raw.items()]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ResearchManifestError("predictions must be a list or object")
    predictions: dict[str, float] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ResearchManifestError(f"prediction {index} must be an object")
        key = str(item.get("sha256") or item.get("path") or item.get("key") or "")
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchManifestError(f"prediction {index} has an invalid score") from exc
        if not key or key in predictions or not math.isfinite(score) or not 0 <= score <= 1:
            raise ResearchManifestError(f"prediction {index} is duplicate or outside [0, 1]")
        predictions[key] = score
    return predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
