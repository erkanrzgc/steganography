"""Reproducible user-supplied dataset manifests and split assignment."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any

from core.dataset import grouped_indices, identity_keys
from steganography.benchmarking.metrics import classification_metrics

_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
_LABELS = {"cover", "stego", "unknown"}
_SPLITS = {"train", "validation", "test"}
_LABEL_COMPONENTS = {"cover", "clean", "negative", "stego", "steg", "positive", "embedded"}


class ResearchManifestError(ValueError):
    """Raised when a dataset or prediction manifest is unsafe or inconsistent."""


def import_dataset(
    source: Path,
    out: Path,
    *,
    seed: int = 20260813,
    license_name: str = "user-supplied",
    source_url: str | None = None,
) -> dict[str, Any]:
    source = Path(source).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    samples: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _EXTENSIONS:
            continue
        digest = _sha256(path)
        label = _infer_label(path, source)
        source_group = _infer_source_group(path, source)
        lineage = _infer_lineage(path, source)
        samples.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": digest,
                "size": path.stat().st_size,
                "label": label,
                "source_group": source_group,
                "lineage": lineage,
                "camera": None,
                "device": None,
                "app": None,
                "method": None if label != "stego" else "unspecified",
                "payload_rate": None,
                "format": path.suffix.lower().lstrip("."),
                "quality_factor": None,
            }
        )
    if not samples:
        raise ValueError("dataset contains no supported image files")
    hashes = [sample["sha256"] for sample in samples]
    if len(hashes) != len(set(hashes)):
        raise ValueError("dataset contains duplicate file hashes")
    for sample in samples:
        grouping_key = f"{sample['source_group']}:{sample['lineage']}"
        sample["split"] = _stable_split(grouping_key, seed)
    manifest = {
        "schema_version": "1.0",
        "source": str(source),
        "catalog": {
            "license": license_name,
            "source_url": source_url,
            "downloaded": False,
            "import_mode": "local",
        },
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
    lineage_splits: dict[tuple[str, ...], str] = {}
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
        for key in identity_keys(sample):
            previous_split = lineage_splits.setdefault(key, split)
            if previous_split != split:
                raise ResearchManifestError("cover-lineage split leakage")
        if verify_files:
            path = source_dir / relative
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ResearchManifestError("dataset symlinks are not accepted")
            if not path.is_file():
                raise ResearchManifestError(f"dataset file is missing: {relative.as_posix()}")
            if path.stat().st_size != sample.get("size") or _sha256(path) != digest:
                raise ResearchManifestError(f"dataset file integrity failed: {relative.as_posix()}")
    overlap = (split_hashes["train"] & split_hashes["test"]) | (
        split_hashes["validation"] & split_hashes["test"]
    )
    if overlap:
        raise ResearchManifestError("train/validation and test hashes overlap")
    partition = manifest.get("partition")
    if partition is not None:
        if not isinstance(partition, dict) or partition.get("policy") not in {
            "identity-camera-device-components-v1",
            "identity-camera-device-development-v1",
        }:
            raise ResearchManifestError("unsupported partition policy")
        held_out = partition.get("test_sources")
        development = partition.get("policy") == "identity-camera-device-development-v1"
        if (
            not isinstance(held_out, list)
            or (not held_out and not development)
            or any(not isinstance(s, str) for s in held_out)
            or (development and (held_out or any(s["split"] == "test" for s in samples)))
        ):
            raise ResearchManifestError("partition requires test source names")
        for sample in samples:
            if (sample.get("source_group") in held_out) != (sample["split"] == "test"):
                raise ResearchManifestError("partition held-out source isolation violated")
        for indices in grouped_indices(samples):
            if len({samples[index]["split"] for index in indices}) != 1:
                raise ResearchManifestError("partition camera/device split leakage")
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
    min_samples: int = 2_000,
    min_source_groups: int = 2,
    min_roc_auc: float = 0.90,
    min_balanced_accuracy: float = 0.85,
    min_recall: float = 0.80,
    max_fpr: float = 0.03,
    max_ece: float = 0.05,
    bootstrap_samples: int = 200,
) -> dict[str, Any]:
    """Evaluate independent test predictions and enforce product-claim gates."""
    probability_values = (
        threshold,
        min_roc_auc,
        min_balanced_accuracy,
        min_recall,
        max_fpr,
        max_ece,
    )
    if any(not 0 <= value <= 1 for value in probability_values):
        raise ValueError("threshold and metric gates must be between 0 and 1")
    if min_samples < 1 or min_source_groups < 1 or bootstrap_samples < 1:
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
    scored_observations: list[tuple[bool, float]] = []
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
        scored_observations.append((sample["label"] == "stego", score))
    if not observations:
        raise ResearchManifestError("no labeled test predictions are available")
    metrics = classification_metrics(observations, threshold=round(threshold * 100))
    metrics["expected_calibration_error"] = _expected_calibration_error(scored_observations)
    intervals = _bootstrap_intervals(
        scored_observations,
        threshold=threshold,
        samples=bootstrap_samples,
        seed=20260813,
    )
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
    if metrics["balanced_accuracy"] < min_balanced_accuracy:
        reasons.append(
            f"balanced accuracy {metrics['balanced_accuracy']} is below {min_balanced_accuracy}"
        )
    if metrics["recall"] < min_recall:
        reasons.append(f"recall {metrics['recall']} is below {min_recall}")
    if metrics["false_positive_rate"] > max_fpr:
        reasons.append(f"false-positive rate {metrics['false_positive_rate']} exceeds {max_fpr}")
    if metrics["expected_calibration_error"] > max_ece:
        reasons.append(
            f"expected calibration error {metrics['expected_calibration_error']} exceeds {max_ece}"
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
            "bootstrap_95_percent_confidence_intervals": intervals,
        },
        "gates": {
            "threshold": threshold,
            "min_samples": min_samples,
            "min_source_groups": min_source_groups,
            "min_roc_auc": min_roc_auc,
            "min_balanced_accuracy": min_balanced_accuracy,
            "min_recall": min_recall,
            "max_fpr": max_fpr,
            "max_ece": max_ece,
            "bootstrap_samples": bootstrap_samples,
            "reasons": reasons,
        },
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def calibrate_predictions(
    manifest_path: Path,
    predictions_path: Path,
    out: Path,
    *,
    source: Path | None = None,
    split: str = "validation",
) -> dict[str, Any]:
    """Fit scalar temperature calibration on a declared non-test split."""
    if split not in {"train", "validation"}:
        raise ValueError("calibration split must be train or validation")
    manifest = load_dataset_manifest(manifest_path)
    verify_dataset_manifest(manifest, source=source, verify_files=True)
    predictions = _load_predictions(predictions_path)
    observations: list[tuple[bool, float]] = []
    for sample in manifest["samples"]:
        if sample["split"] != split or sample["label"] not in {"cover", "stego"}:
            continue
        key = sample["sha256"] if sample["sha256"] in predictions else sample["path"]
        if key not in predictions:
            raise ResearchManifestError(f"calibration prediction is missing: {sample['path']}")
        observations.append((sample["label"] == "stego", predictions[key]))
    if not observations or len({label for label, _score in observations}) != 2:
        raise ResearchManifestError("calibration requires cover and stego observations")
    candidates = [0.25 + index * 0.01 for index in range(376)]
    temperature = min(candidates, key=lambda value: _log_loss(observations, value))
    calibrated = [(label, _apply_temperature(score, temperature)) for label, score in observations]
    report = {
        "schema_version": "1.0",
        "method": "scalar_temperature",
        "split": split,
        "samples": len(observations),
        "temperature": round(temperature, 6),
        "before": {"ece": _expected_calibration_error(observations)},
        "after": {
            "ece": _expected_calibration_error(calibrated),
            "log_loss": round(_log_loss(observations, temperature), 6),
        },
        "manifest_sha256": _sha256(Path(manifest_path)),
        "predictions_sha256": _sha256(Path(predictions_path)),
    }
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_model(config_path: Path, out: Path) -> dict[str, Any]:
    """Train a declared binary feature model when the research extra is installed."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("research training requires the 'research' extra") from exc
    from steganography.research_features import read_document, training_inputs
    from steganography.research_weighting import training_weights

    config, _ = read_document(Path(config_path))
    features, labels, provenance = training_inputs(config)
    sample_weights = training_weights(config, provenance)
    destination = Path(out)
    if destination.exists() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise FileExistsError("checkpoint output already exists or uses a symlink")
    learning_rate = float(config.get("learning_rate", 1e-3))
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning rate must be finite and in (0, 1]")
    torch.manual_seed(int(config.get("seed", 20260813)))
    tensor = torch.from_numpy(features.reshape(features.shape[0], -1))
    targets = torch.from_numpy(labels.reshape(-1, 1))
    jpeg_model = provenance["feature_version"] in {
        "jpeg-dct-summary-v1",
        "jpeg-context-summary-v1",
        "jpeg-dct-residual-parity-v1",
    }
    residual_model = provenance["feature_version"] in {
        "spatial-cooccurrence-v1",
        "spatial-parity-residual-v1",
    }
    arithmetic = config.get("inference_arithmetic", "float32")
    if not isinstance(arithmetic, str) or arithmetic not in {"float32", "float64"}:
        raise ResearchManifestError("inference arithmetic must be float32 or float64")
    standardize = config.get("standardize", jpeg_model or residual_model)
    class_balanced = config.get("class_balanced", jpeg_model or residual_model)
    if type(standardize) is not bool or type(class_balanced) is not bool:
        raise ResearchManifestError("standardize/class_balanced must be booleans")
    normalization = None
    if standardize:
        mean = tensor.mean(dim=0)
        scale = tensor.std(dim=0, unbiased=False).clamp(min=1e-4)
        tensor = (tensor - mean) / scale
        normalization = {
            "method": "train-only-standardization",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }
    model = torch.nn.Linear(tensor.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = None if sample_weights is None else torch.from_numpy(sample_weights[:, None])
    negative_mass = float((labels == 0).sum())
    positive_mass = float((labels == 1).sum())
    if sample_weights is not None:
        negative_mass = float(sample_weights[labels == 0].sum())
        positive_mass = float(sample_weights[labels == 1].sum())
    loss_function = torch.nn.BCEWithLogitsLoss(
        weight=weights,
        pos_weight=torch.tensor(negative_mass / positive_mass) if class_balanced else None,
    )
    epochs = int(config.get("epochs", 10))
    if not 1 <= epochs <= 100_000:
        raise ValueError("training epochs must be between 1 and 100000")
    provenance["training"] = {
        "seed": int(config.get("seed", 20260813)),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "torch_version": str(torch.__version__),
        "class_balanced": class_balanced,
        "standardize": standardize,
        "inference_arithmetic": arithmetic,
    }
    if sample_weights is not None:
        recipe_metadata = {
            "recipe": config["sample_weighting"],
            "negative_mass": negative_mass,
            "positive_mass": positive_mass,
            "positive_class_weight": negative_mass / positive_mass if class_balanced else 1.0,
            "reduction": "mean-over-rows",
        }
        if config["sample_weighting"] == "jpeg-source-class-balanced-v1":
            recipe_metadata.update(
                {
                    "formula": "n / (2 * sources * source_label_count)",
                    "min_row_weight": float(sample_weights.min()),
                    "max_row_weight": float(sample_weights.max()),
                }
            )
        else:
            recipe_metadata.update(
                {
                    "cover_weight": 1.0,
                    "stego_5_percent_weight": 4.0,
                    "other_stego_weight": 1.0,
                }
            )
        provenance["training"]["sample_weighting"] = recipe_metadata
    loss = 0.0
    for _epoch in range(epochs):
        optimizer.zero_grad()
        output = model(tensor)
        loss_value = loss_function(output, targets)
        if not bool(torch.isfinite(loss_value)):
            raise ResearchManifestError("training produced nonfinite loss")
        loss_value.backward()
        optimizer.step()
        loss = float(loss_value.detach())
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_shape": list(features.shape[1:]),
        "features": int(tensor.shape[1]),
        "domain": "jpeg-dct-residual-parity-linear-v1"
        if provenance["feature_version"] == "jpeg-dct-residual-parity-v1"
        else "jpeg-context-summary-linear-v1"
        if provenance["feature_version"] == "jpeg-context-summary-v1"
        else "jpeg-dct-summary-linear-v1"
        if jpeg_model
        else "spatial-parity-residual-linear-v1"
        if provenance["feature_version"] == "spatial-parity-residual-v1"
        else "spatial-cooccurrence-linear-v1"
        if residual_model
        else "spatial-summary-linear-v1",
        "preprocessing": {
            "feature_version": provenance["feature_version"],
            "feature_names": provenance["feature_names"],
            "normalization": normalization,
            "inference_arithmetic": arithmetic,
        },
        "training_provenance": provenance,
    }
    with destination.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {
        "checkpoint": destination.name,
        "samples": len(labels),
        "loss": loss,
        "epochs": epochs,
        "training_provenance": provenance,
    }


def export_onnx(checkpoint_path: Path, out: Path) -> dict[str, Any]:
    """Export a trained checkpoint with its model/preprocessing contract."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ONNX export requires the 'research' extra") from exc
    from steganography.research_features import read_feature_checkpoint

    checkpoint = read_feature_checkpoint(Path(checkpoint_path))
    from core.feature_model import feature_model

    model = feature_model(checkpoint)
    model.eval()
    destination = Path(out)
    card = destination.with_suffix(destination.suffix + ".model-card.json")
    for path in (destination, card):
        if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
            raise FileExistsError("export output/card already exists or uses a symlink")
    dummy = torch.zeros((1, int(checkpoint["features"])), dtype=torch.float32)
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        (dummy,),
        buffer,  # type: ignore[arg-type]
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        # Preserve the small opset-17 contract across exporter default changes.
        dynamo=False,
    )
    data = buffer.getvalue()
    contract = {
        "schema_version": "1.0",
        "domain": checkpoint["domain"],
        "input_shape": checkpoint["input_shape"],
        "preprocessing": checkpoint["preprocessing"],
        "onnx_sha256": hashlib.sha256(data).hexdigest(),
        "calibrated": False,
        "training_provenance": checkpoint.get("training_provenance"),
    }
    if "inference_derivation" in checkpoint:
        contract["inference_derivation"] = checkpoint["inference_derivation"]
    card_data = json.dumps(contract, indent=2).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    with card.open("xb") as stream:
        stream.write(card_data)
    return contract


def verify_split_isolation(manifest: dict[str, Any]) -> bool:
    observed: dict[tuple[str, ...], str] = {}
    for sample in manifest.get("samples", []):
        split = sample["split"]
        for key in identity_keys(sample):
            if key in observed and observed[key] != split:
                return False
            observed[key] = split
    return True


def partition_dataset(
    manifest_path: Path,
    out: Path,
    *,
    test_sources: list[str],
    reserved_manifests: list[Path],
    source: Path | None = None,
    seed: int = 20260918,
) -> dict[str, Any]:
    """Prepare a new experiment without reusing published/frozen observations.

    Requires explicit ancestry and source labels; never guesses either from a
    filename. This validates declared provenance, not the truth of that metadata.
    """
    manifest = load_dataset_manifest(manifest_path)
    verify_dataset_manifest(manifest, source=source)
    samples = manifest["samples"]
    for sample in samples:
        if (
            not isinstance(sample.get("lineage"), str)
            or not sample["lineage"].strip()
            or not isinstance(sample.get("source_group"), str)
            or sample["source_group"] in {"", "unspecified"}
            or sample["label"] not in {"cover", "stego"}
        ):
            raise ResearchManifestError("partition requires explicit lineage, source and labels")
    sources = {sample["source_group"] for sample in samples}
    held_out = set(test_sources)
    if not held_out or not held_out < sources:
        raise ResearchManifestError("test sources must be a nonempty proper subset of sources")
    if not reserved_manifests:
        raise ResearchManifestError("at least one frozen/reserved manifest is required")
    reserved: set[tuple[str, ...]] = set()
    reserved_hashes = []
    for path in reserved_manifests:
        document = load_dataset_manifest(path)
        records = document.get("samples")
        if not isinstance(records, list) or not records:
            raise ResearchManifestError("reserved manifest must contain samples")
        for record in records:
            if not isinstance(record, dict) or not identity_keys(record):
                raise ResearchManifestError("reserved sample has no identity")
            reserved.update(identity_keys(record))
        reserved_hashes.append(_sha256(path))
    if any(identity_keys(sample) & reserved for sample in samples):
        raise ResearchManifestError("dataset overlaps frozen/reserved identities")
    groups = grouped_indices(samples)
    for indices in groups:
        group_sources = {samples[index]["source_group"] for index in indices}
        if group_sources & held_out and group_sources - held_out:
            raise ResearchManifestError("related samples cross held-out source boundary")
        keys = sorted({key for index in indices for key in identity_keys(samples[index])})
        fraction = (
            int.from_bytes(hashlib.sha256(json.dumps([seed, keys]).encode()).digest()[:8], "big")
            / 2**64
        )
        split = "test" if group_sources <= held_out else "train" if fraction < 0.8 else "validation"
        for index in indices:
            samples[index]["split"] = split
    counts = {
        split: {
            label: sum(s["split"] == split and s["label"] == label for s in samples)
            for label in ("cover", "stego")
        }
        for split in sorted(_SPLITS)
    }
    if any(not count for values in counts.values() for count in values.values()):
        raise ResearchManifestError(
            "each split needs cover and stego; supply more independent groups"
        )
    manifest["seed"] = seed
    manifest["sample_count"] = len(samples)
    manifest["splits"] = {split: sum(values.values()) for split, values in counts.items()}
    if source is not None:
        manifest["source"] = str(source.resolve())
    manifest["partition"] = {
        "policy": "identity-camera-device-components-v1",
        "input_manifest_sha256": _sha256(manifest_path),
        "reserved_manifest_sha256": sorted(set(reserved_hashes)),
        "test_sources": sorted(held_out),
        "group_count": len(groups),
        "counts": counts,
        "camera_device_metadata_complete": all(
            s.get("camera") and s.get("device") for s in samples
        ),
        "support_status": "experimental",
    }
    verify_dataset_manifest(manifest, source=source, verify_files=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


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


def _infer_lineage(path: Path, root: Path) -> str:
    parts = [
        part
        for part in path.relative_to(root).with_suffix("").parts
        if part.lower() not in _LABEL_COMPONENTS
    ]
    stem = parts[-1] if parts else path.stem
    for suffix in ("-stego", "_stego", "-embedded", "_embedded", "-cover", "_cover"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return "/".join((*parts[:-1], stem)) or stem


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


def _expected_calibration_error(observations: list[tuple[bool, float]], bins: int = 10) -> float:
    if not observations:
        return 0.0
    total = len(observations)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        values = [
            (label, score)
            for label, score in observations
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not values:
            continue
        confidence = sum(score for _, score in values) / len(values)
        frequency = sum(label for label, _ in values) / len(values)
        error += len(values) / total * abs(confidence - frequency)
    return round(error, 6)


def _apply_temperature(score: float, temperature: float) -> float:
    bounded = min(1 - 1e-9, max(1e-9, score))
    logit = math.log(bounded / (1 - bounded)) / temperature
    return 1 / (1 + math.exp(-logit))


def _log_loss(observations: list[tuple[bool, float]], temperature: float) -> float:
    total = 0.0
    for label, score in observations:
        probability = min(1 - 1e-9, max(1e-9, _apply_temperature(score, temperature)))
        total -= math.log(probability if label else 1 - probability)
    return total / len(observations)


def _bootstrap_intervals(
    observations: list[tuple[bool, float]],
    *,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    positives = [score for label, score in observations if label]
    negatives = [score for label, score in observations if not label]
    if not positives or not negatives:
        return {}
    generator = random.Random(seed)  # noqa: S311 - deterministic statistical resampling
    values: dict[str, list[float]] = {
        "roc_auc": [],
        "balanced_accuracy": [],
        "recall": [],
        "false_positive_rate": [],
        "expected_calibration_error": [],
    }
    for _index in range(samples):
        selected = [
            *((True, generator.choice(positives)) for _ in positives),
            *((False, generator.choice(negatives)) for _ in negatives),
        ]
        labels_and_ints = [(label, round(score * 100)) for label, score in selected]
        metrics = classification_metrics(labels_and_ints, threshold=round(threshold * 100))
        values["roc_auc"].append(float(metrics["roc_auc"]))
        values["balanced_accuracy"].append(float(metrics["balanced_accuracy"]))
        values["recall"].append(float(metrics["recall"]))
        values["false_positive_rate"].append(float(metrics["false_positive_rate"]))
        values["expected_calibration_error"].append(_expected_calibration_error(selected))
    intervals: dict[str, list[float]] = {}
    for name, distribution in values.items():
        distribution.sort()
        low = distribution[max(0, round(0.025 * (samples - 1)))]
        high = distribution[min(samples - 1, round(0.975 * (samples - 1)))]
        intervals[name] = [round(low, 6), round(high, 6)]
    return intervals
