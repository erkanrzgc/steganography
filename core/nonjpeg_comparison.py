"""Source-bound CPU feature comparison with closed test roles and bounded outputs."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from core import nonjpeg_compare_metrics as metrics
from core import nonjpeg_features as features
from core import nonjpeg_linear as linear
from core import srnet_cuda
from core.nonjpeg_data import basename, regular_read
from core.spatial_parity import FEATURE_NAMES, FEATURE_VERSION
from modules.audio_wav import AudioWav

SECONDS = 1800
OUTPUT_BYTES = 32 * 1024**2
INPUT_BYTES = 64 * 1024**2
SOURCES = ("BOWS2", "DIV2K", "ESC-50", "FSDD-v1.0.10")
VERSIONS = {
    "BOWS2": (FEATURE_VERSION, features.LOCAL_VERSION),
    "DIV2K": (FEATURE_VERSION, features.LOCAL_VERSION, features.RGB_VERSION),
    "ESC-50": (features.AUDIO_VERSION,), "FSDD-v1.0.10": (features.AUDIO_VERSION,),
}
NAMES = (
    "core/nonjpeg_linear.py", "core/nonjpeg_compare_metrics.py", "core/nonjpeg_comparison.py",
    "steganography/research_nonjpeg_compare.py", "core/nonjpeg_data.py",
    "core/nonjpeg_features.py", "core/spatial_parity.py", "core/spatial_cooccurrence.py",
    "core/features.py", "core/context.py", "core/result.py", "core/carrier.py",
    "core/payload.py", "core/crypto.py", "modules/audio_wav.py", "config.py",
    "core/srnet_cuda.py", "docs/NONJPEG_LINEAR_COMPARISON_PROTOCOL.md",
)
ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sources():
    return {name: sha(regular_read(ROOT / name, 1024**2)) for name in NAMES}


def cpu_bound(*, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    memory = srnet_cuda.host_bound(proc=proc, root=root)
    relative = [line[3:] for line in regular_read(proc, 4096).decode().splitlines()
                if line.startswith("0::")][0]
    folder = root / relative.lstrip("/")
    quotas: list[float] = []
    swap: list[float] = []
    while True:
        for name, values in (("cpu.max", quotas), ("memory.swap.max", swap)):
            path = folder / name
            if path.exists():
                text = regular_read(path, 64).decode().strip()
                if name == "cpu.max":
                    quota, period = text.split()
                    if quota != "max":
                        if not 0 < int(quota) <= 2**63 - 1 or not 0 < int(period) <= 10**6:
                            raise ValueError("invalid bounded CPU quota")
                        values.append(float(int(quota)) / int(period))
                elif text != "max":
                    values.append(float(int(text)))
        if folder == root:
            break
        folder = folder.parent
    if (
        memory > 2 * 1024**3 or not quotas or not 0 < min(quotas) <= 1
        or not swap or min(swap) != 0
        or any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    ):
        raise ValueError("bounded single-thread CPU service required")
    return {"memory_max_bytes": memory, "cpu_quota": min(quotas), "swap_max_bytes": 0,
            "numpy_version": np.__version__, "python_version": platform.python_version(),
            "gpu_used": False}


def read(path, expected=None):
    raw = regular_read(path, INPUT_BYTES)
    if expected is not None and sha(raw) != expected:
        raise ValueError("comparison checksum mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("comparison document must be an object")
    return value, sha(raw)


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if (
        len(raw) > OUTPUT_BYTES or path.exists()
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("bounded exclusive comparison output required")
    total = len(raw)
    for existing in path.parent.rglob("*"):
        if existing.is_symlink():
            raise ValueError("comparison output symlink")
        if existing.is_file():
            total += existing.stat().st_size
    if total > OUTPUT_BYTES:
        raise ValueError("comparison cumulative output limit")
    with path.open("xb") as stream:
        stream.write(raw)
    return sha(raw)


def expected_cells(source):
    formats = ("PNG", "BMP") if source in SOURCES[:2] else ("WAV",)
    channels = {"BOWS2": ("gray",), "DIV2K": ("r", "g", "b", "all")}.get(source, ("mono",))
    return set(itertools.product(formats, ("sequential", "scattered"), (5, 40), channels, (0, 1)))


def load_slice(record, root, split, source):
    manifest, manifest_sha = read(root["features"] / basename(record["manifest"]),
                                  record["manifest_sha256"])
    origin = root["baselines"] if record["feature_version"] == FEATURE_VERSION else root["features"]
    artifact, artifact_sha = read(origin / basename(record["artifact"]), record["sha256"])
    version = record["feature_version"]
    names = FEATURE_NAMES if version == FEATURE_VERSION else features.CONTRACTS[version][0]
    rows = manifest.get("samples")
    if (
        manifest.get("development_only") is not True or not isinstance(rows, list)
        or not 1 <= len(rows) <= 4096
        or artifact.get("schema_version") != "research-features-1"
        or artifact.get("feature_version") != version
        or artifact.get("feature_names") != list(names)
        or artifact.get("manifest_sha256") != manifest_sha or artifact.get("split") != split
        or any(r.get("split") != split or r.get("source_group") != source for r in rows)
        or linear.row_identity(rows) != record["ordered_rows_sha256"]
        or len({r["lineage"] for r in rows}) != (32 if split == "train" else 8)
        or len(artifact.get("rows", [])) != len(rows)
    ):
        raise ValueError("comparison source/split/domain binding failed")
    vectors = []
    for sample, row in zip(rows, artifact["rows"], strict=True):
        if any(row.get(k) != sample.get(k) for k in ("sha256", "lineage", "label")) or (
            not isinstance(row.get("values"), list) or len(row["values"]) != len(names)
            or any(type(v) not in (int, float) for v in row["values"])
        ):
            raise ValueError("comparison row/value contract failed")
        vectors.append(row["values"])
    x = np.asarray(vectors, dtype=np.float64)
    y = np.asarray([r["label"] == "stego" for r in rows], dtype=np.float64)
    linear.validate(x, y, rows)
    cells = metrics.pair_cells(rows, np.zeros(len(rows)))
    if {tuple(cell[k] for k in metrics.CELL_KEYS) for cell, _, _ in cells} != (
        expected_cells(source)
    ):
        raise ValueError("comparison requires all controlled cells")
    return {"x": x, "y": y, "rows": rows, "manifest_sha256": manifest_sha,
            "artifact_sha256": artifact_sha, "ordered_rows_sha256": linear.row_identity(rows),
            "corpus_sha256": manifest.get("preparation_manifest_sha256"),
            "audit_sha256": manifest.get("independent_audit_sha256")}


def load_inputs(config, protocol):
    readiness, _ = read(Path(config["readiness"]), protocol["readiness_sha256"])
    if (
        readiness.get("status") != "prepared_not_fitted"
        or readiness.get("schema_version") != "nonjpeg-comparison-input-readiness-v1"
        or readiness.get("originals") != 160
        or any(readiness.get(k) != protocol[k] for k in ("corpus_sha256", "audit_sha256"))
    ):
        raise ValueError("complete comparison input readiness required")
    root = {k: Path(config[k + "_root"]) for k in ("features", "baselines")}
    indexed = {}
    for record in readiness["artifacts"]:
        key = (record["source_group"], record["feature_version"], record["split"])
        if key in indexed:
            raise ValueError("duplicate comparison artifact")
        indexed[key] = record
    expected = {(source, version, split) for source in SOURCES for version in VERSIONS[source]
                for split in ("train", "validation")}
    if set(indexed) != expected:
        raise ValueError("comparison artifact matrix incomplete")
    result = {}
    identities: dict[tuple[str, str], str] = {}
    for key in sorted(expected):
        source, version, split = key
        value = load_slice(indexed[key], root, split, source)
        if any(value[k] != readiness[k] for k in ("corpus_sha256", "audit_sha256")):
            raise ValueError("comparison preparation/audit provenance mismatch")
        rows = value["rows"]
        if identities.setdefault((source, split), value["ordered_rows_sha256"]) != (
            value["ordered_rows_sha256"]
        ):
            raise ValueError("comparison exposure rows differ across arms")
        if source == "FSDD-v1.0.10":
            from core.nonjpeg_data import SPEAKERS

            if any(SPEAKERS.get(r.get("speaker")) != split for r in rows):
                raise ValueError("historical speaker roles changed")
        result[key] = value
    for source in SOURCES:
        version = VERSIONS[source][0]
        a, b = (result[(source, version, split)]["rows"] for split in ("train", "validation"))
        if any(set(function(r) for r in a) & set(function(r) for r in b)
               for function in (linear.group_key, lambda r: r["lineage"], lambda r: r["sha256"])):
            raise ValueError("comparison train/validation groups overlap")
        if source in SOURCES[:2]:
            for split in ("train", "validation"):
                reference = result[(source, FEATURE_VERSION, split)]["x"]
                if any(not np.array_equal(reference, result[(source, v, split)]["x"][:, :len(
                    FEATURE_NAMES)]) for v in VERSIONS[source][1:]):
                    raise ValueError("spatial baseline prefix differs across feature arms")
    return result


def wav_reference(rows, root, out, deadline):
    scores = []
    with tempfile.TemporaryDirectory(prefix="native-reference-", dir=out) as folder:
        for row in rows:
            deadline()
            raw = regular_read(root / basename(row["path"]), 1024**2)
            if len(raw) != row["size"] or sha(raw) != row["sha256"]:
                raise ValueError("native reference input integrity failed")
            path = Path(folder) / "bounded.wav"
            with path.open("xb") as stream:
                stream.write(raw)
            result = AudioWav().analyze(path)
            path.unlink()
            if result.status != "ok":
                raise ValueError("native WAV reference coverage unavailable")
            scores.append(result.suspicion / 100)
    return np.asarray(scores)


def execute(config, out, *, clock=time.monotonic):
    started = clock()
    if config.get("schema_version") != "nonjpeg-linear-comparison-config-v1":
        raise ValueError("comparison configuration schema mismatch")
    before = sources()
    protocol, protocol_sha = read(Path(config["protocol"]), config["protocol_sha256"])
    if (
        protocol.get("schema_version") != "nonjpeg-linear-comparison-protocol-v1"
        or protocol.get("status") != "frozen"
        or protocol.get("source_sha256") != before
        or protocol.get("seeds") != list(linear.SEEDS) or protocol.get("epochs") != linear.EPOCHS
        or protocol.get("budget_seconds") != SECONDS or protocol.get("budget_bytes") != OUTPUT_BYTES
        or out.exists() or any(p.is_symlink() for p in (out, *out.parents))
    ):
        raise ValueError("comparison protocol/fresh output mismatch")
    if any(out.resolve().is_relative_to(Path(config[key]).resolve()) for key in (
        "features_root", "baselines_root", "dataset_root", "readiness", "protocol"
    )):
        raise ValueError("comparison output cannot modify frozen inputs")
    out.mkdir(parents=True, mode=0o700)
    report = {"schema_version": "nonjpeg-linear-comparison-v1", "source_sha256": before,
              "protocol_sha256": protocol_sha, "status": "unavailable", "trials": [],
              "test_media_opened": False, "new_media_used": False, "gpu_used": False,
              "deep_models_trained": False, "production_qualification": "unavailable",
              "deployed": False, "real_data_optimizer_updates": 0,
              "limitations": ["same-source controlled and previously inspected development",
                              "eight validation originals; two FSDD validation speakers",
                              "image camera independence unavailable",
                              "natural-source, blind, codec and deep-model gates remain open"]}

    def deadline():
        marker = out / "CANCEL"
        if marker.is_symlink():
            raise ValueError("comparison cancellation symlink")
        if marker.exists():
            raise InterruptedError
        if clock() - started >= SECONDS:
            raise TimeoutError
        total = 0
        for path in out.rglob("*"):
            if path.is_symlink():
                raise ValueError("comparison artifact symlink")
            if path.is_file():
                total += path.stat().st_size
        if total + 1024**2 > OUTPUT_BYTES:
            raise ValueError("comparison output budget exceeded")

    def record_update():
        report["real_data_optimizer_updates"] += 1

    try:
        deadline()
        report["phase"] = "resource_preflight"
        report["execution"] = cpu_bound()
        if any(report["execution"].get(k) != protocol["runtime"].get(k)
               for k in ("numpy_version", "python_version")):
            raise ValueError("CPU comparison runtime binding mismatch")
        report["phase"] = "input_integrity"
        datasets = load_inputs(config, protocol)
        report["input_artifacts"] = [
            {"source_group": s, "version": v, "split": p,
             "manifest_sha256": d["manifest_sha256"], "artifact_sha256": d["artifact_sha256"],
             "ordered_rows_sha256": d["ordered_rows_sha256"],
             "originals": len({r["lineage"] for r in d["rows"]}),
             "independent_groups": len({linear.group_key(r) for r in d["rows"]})}
            for (s, v, p), d in sorted(datasets.items())]
        for source in SOURCES:
            report["phase"] = "fit_and_measure"
            reference = {}
            validation = datasets[(source, VERSIONS[source][0], "validation")]
            if source in SOURCES[2:]:
                scores = wav_reference(
                    validation["rows"], Path(config["dataset_root"]), out, deadline)
                reference = {seed: (scores, 0.7) for seed in linear.SEEDS}
            for version in VERSIONS[source]:
                train = datasets[(source, version, "train")]
                validation = datasets[(source, version, "validation")]
                for seed in linear.SEEDS:
                    deadline()
                    model, cv = linear.grouped_fit(
                        train["x"], train["y"], train["rows"], seed,
                        deadline=deadline, on_update=record_update)
                    scores = linear.predict(model, validation["x"])
                    reference_scores, reference_threshold = reference.get(
                        seed, (scores, cv["threshold"]))
                    cells = []
                    paired = metrics.pair_cells(validation["rows"], scores)
                    prior = metrics.pair_cells(validation["rows"], reference_scores)
                    for (cell, pairs, groups), (prior_cell, prior_pairs, prior_groups) in zip(
                        paired, prior, strict=True
                    ):
                        deadline()
                        if cell != prior_cell or groups != prior_groups:
                            raise ValueError("comparison reference pairing failed")
                        cells.append({**cell, **metrics.cell_statistics(
                            pairs, groups, cv["threshold"], baseline=prior_pairs,
                            baseline_threshold=reference_threshold)})
                    model_name = f"model-{len(report['trials']):02d}.npz"
                    with (out / model_name).open("xb") as stream:
                        np.savez(stream, mean=np.asarray(model["mean"], dtype="<f8"),
                                 scale=np.asarray(model["scale"], dtype="<f8"),
                                 weights=np.asarray(model["weights"], dtype="<f8"),
                                 bias=np.asarray(model["bias"], dtype="<f8"))
                    model_sha = sha(regular_read(out / model_name, 1024**2))
                    cv.pop("oof_scores")
                    cv["fold_assignments_sha256"] = sha(
                        cv.pop("fold_assignments").astype("<i4").tobytes())
                    trial = {"source_group": source, "feature_version": version, "seed": seed,
                             "model_domain": version + ":balanced-sgd20-f64-v1",
                             "model_sha256": model_sha, "model": model_name, "cv": cv,
                             "training": {k: v for k, v in model.items()
                                          if k not in ("mean", "scale", "weights", "bias")},
                             "cells": cells, "validation_scores": scores.tolist(),
                             "reference_scores": reference_scores.tolist(),
                             "reference_threshold": reference_threshold,
                             "validation_rows_sha256": validation["ordered_rows_sha256"],
                             "reference": "native-wav-fixed-70" if source in SOURCES[2:]
                             else FEATURE_VERSION, "score_calibration": "unavailable"}
                    write(out / f"trial-{len(report['trials']):02d}.json", trial)
                    report["trials"].append(trial)
                    if version == VERSIONS[source][0] and source in SOURCES[:2]:
                        reference[seed] = scores, cv["threshold"]
        deadline()
        if sources() != before:
            raise ValueError("comparison source mutation")
        report["status"] = "completed"
        report["phase"] = "completed"
    except (TimeoutError, InterruptedError) as exc:
        report["reason"] = "time_budget_exhausted" if isinstance(exc, TimeoutError) else "cancelled"
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ZeroDivisionError) as exc:
        report["reason"] = {
            "comparison checksum mismatch": "input_checksum_mismatch",
            "bounded single-thread CPU service required": "cpu_resource_bounds_unavailable",
            "CPU comparison runtime binding mismatch": "cpu_runtime_mismatch",
            "comparison source/split/domain binding failed": "source_split_domain_mismatch",
            "native reference input integrity failed": "reference_input_integrity_failed",
            "native WAV reference coverage unavailable": "native_wav_reference_unavailable",
            "linear nonfinite state": "linear_nonfinite_state",
            "comparison output budget exceeded": "output_budget_exhausted",
            "comparison cumulative output limit": "output_budget_exhausted",
            "comparison source mutation": "source_mutation",
        }.get(str(exc), "comparison_contract_failed")
    finally:
        report["seconds"] = clock() - started
        write(out / "report.json", report)
    return report
