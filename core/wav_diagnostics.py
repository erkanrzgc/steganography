"""Read-only train-feature diagnosis of the completed native WAV pilot."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from types import FunctionType

import numpy as np

from core import nonjpeg_comparison as comparison
from core import nonjpeg_linear as linear
from core import nonjpeg_process_preflight as parent
from core.nonjpeg_features import AUDIO_NAMES, AUDIO_VERSION

SECONDS = 300
OUTPUT_BYTES = 16 * 1024**2
MEMORY_BYTES = 512 * 1024**2
SOURCES = ("ESC-50", "FSDD-v1.0.10")
NAMES = (*parent.NAMES, "core/wav_diagnostics.py",
         "steganography/research_wav_diagnostics.py", "docs/WAV_READONLY_PROTOCOL.md")


def sources():
    return {name: parent.source_hash(parent.ROOT / name) for name in NAMES}


def write(path, value):
    writer = FunctionType(comparison.write.__code__, {
        **comparison.write.__globals__, "OUTPUT_BYTES": OUTPUT_BYTES})
    return writer(path, value)


def pair_indices(rows):
    # The shared pairing contract validates completeness, uniqueness and groups.
    checked = comparison.metrics.pair_cells(rows, np.zeros(len(rows)))
    covers = {(r["lineage"], r["format"]): i for i, r in enumerate(rows)
              if r["label"] == "cover"}
    result = []
    for cell, _, groups in checked:
        positive = np.array([i for i, r in enumerate(rows) if r["label"] == "stego"
                             and all(r[k] == v for k, v in cell.items())], dtype=int)
        negative = np.array([covers[(rows[i]["lineage"], rows[i]["format"])]
                             for i in positive], dtype=int)
        result.append((cell, negative, positive, groups))
    return result


def feature_contrasts(x, rows, *, deadline=lambda: None):
    linear.validate(x, np.array([r["label"] == "stego" for r in rows]), rows)
    if x.shape[1] != len(AUDIO_NAMES) or any(r["split"] != "train" for r in rows):
        raise ValueError("WAV diagnostics require native train-only features")
    covers = [i for i, r in enumerate(rows) if r["label"] == "cover"]
    spread = np.std(x[covers], axis=0)
    content_rms = float(np.sqrt(np.mean(spread**2)))
    cells = []
    for cell, negative, positive, groups in pair_indices(rows):
        deadline()
        delta = x[positive] - x[negative]
        shift = np.mean(delta, axis=0)
        rms = float(np.sqrt(np.mean(shift**2)))
        cells.append({**cell, "pairs": len(positive), "independent_groups": len(set(groups)),
                      "exactly_equal_feature_pairs": int(np.sum(np.all(delta == 0, axis=1))),
                      "paired_delta_rms": float(np.sqrt(np.mean(delta**2))),
                      "aligned_shift_rms": rms,
                      "aligned_shift_to_cover_variation": rms / content_rms
                      if content_rms else None,
                      "zero_delta_features": int(np.sum(np.all(delta == 0, axis=0))),
                      "same_strict_sign_features": int(np.sum(np.all(delta > 0, axis=0)
                                                               | np.all(delta < 0, axis=0))),
                      "mean_paired_feature_delta": shift.tolist(),
                      "rms_paired_feature_delta": np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                      "regions": {region: float(np.sqrt(np.mean(delta[:, i*152:(i+1)*152]**2)))
                                  for i, region in enumerate(("full", "q0", "q1", "q2", "q3"))}})
    return {"rows": len(rows), "originals": len(covers), "features": len(AUDIO_NAMES),
            "independent_groups": len({linear.group_key(r) for r in rows}),
            "constant_features": int(np.sum(np.ptp(x, axis=0) == 0)),
            "cover_variation_rms": content_rms, "cover_feature_std": spread.tolist(),
            "cells": cells}


def numeric_model(raw):
    if len(raw) > 1024**2:
        raise ValueError("diagnostic model byte bound")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 4 or sum(m.file_size for m in members) > 1024**2 or (
            {m.filename for m in members} != {"mean.npy", "scale.npy", "weights.npy", "bias.npy"}
        ):
            raise ValueError("diagnostic numeric model members")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        model = dict(archive)
    for name, value in model.items():
        if value.dtype != np.dtype("<f8") or not np.isfinite(value).all() or (
            value.shape != (() if name == "bias" else (len(AUDIO_NAMES),))
        ):
            raise ValueError("diagnostic numeric model contract")
    return model


def model_diagnosis(model, data, trial, *, deadline=lambda: None):
    x, y, rows = (data[k] for k in ("x", "y", "rows"))
    weights = linear.validate(x, y, rows)
    if x.shape[1] != len(AUDIO_NAMES) or any(r["split"] != "train" for r in rows):
        raise ValueError("WAV diagnostics require native train-only features")
    mean = weights @ x
    scale = np.maximum(np.sqrt(weights @ ((x - mean)**2)), linear.SCALE_FLOOR)
    if not np.allclose(mean, model["mean"], rtol=0, atol=1e-12) or not np.allclose(
        scale, model["scale"], rtol=0, atol=1e-12
    ):
        raise ValueError("diagnostic train normalization mismatch")
    logits = ((x - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]
    scores = linear.predict(model, x)
    loss = float(weights @ (np.logaddexp(0, logits) - y*logits))
    cached_loss = trial["training"]["losses"][-1]
    if not np.isclose(loss, cached_loss, rtol=0, atol=1e-12):
        raise ValueError("diagnostic train loss replay mismatch")
    threshold = trial["cv"]["threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("diagnostic cached threshold invalid")
    cached = {tuple(c[k] for k in comparison.metrics.CELL_KEYS): c for c in trial["cells"]}
    cells = []
    for cell, negative, positive, groups in pair_indices(rows):
        deadline()
        pairs = np.stack((scores[negative], scores[positive]), axis=1)
        point = comparison.metrics.measured(pairs, np.ones((1, len(pairs))), threshold)
        logit_delta = logits[positive] - logits[negative]
        cells.append({**cell, "pairs": len(positive), "independent_groups": len(set(groups)),
                      "train_metrics": {k: float(v[0]) for k, v in point.items()},
                      "same_pair_decisions": int(np.sum((pairs[:, 0] >= threshold)
                                                        == (pairs[:, 1] >= threshold))),
                      "stego_score_gt_cover_fraction": float(np.mean(pairs[:, 1] > pairs[:, 0])),
                      "mean_logit_delta": float(np.mean(logit_delta)),
                      "mean_score_delta": float(np.mean(pairs[:, 1] - pairs[:, 0])),
                      "cached_development_metrics": cached[tuple(
                          cell[k] for k in comparison.metrics.CELL_KEYS)]["metrics"]})
    return {"source_group": trial["source_group"], "seed": trial["seed"],
            "model_sha256": trial["model_sha256"], "frozen_threshold": threshold,
            "training_loss_replayed": loss, "training_loss_history": trial["training"]["losses"],
            "train_score_range": [float(scores.min()), float(scores.max())],
            "cached_development_score_range": [min(trial["validation_scores"]),
                                               max(trial["validation_scores"])],
            "oof_score_replay": "unavailable; fold models and OOF scores were not retained",
            "inference_rows": len(rows), "cells": cells}


def execute(config, out, *, clock=time.monotonic):
    started = clock()
    if config.get("schema_version") != "wav-readonly-diagnostics-config-v1":
        raise ValueError("diagnostic configuration schema mismatch")
    protocol, protocol_sha = comparison.read(Path(config["protocol"]), config["protocol_sha256"])
    original, _ = comparison.read(Path(config["comparison_config"]),
                                  protocol["comparison_config_sha256"])
    before = sources()
    if (
        protocol.get("schema_version") != "wav-readonly-diagnostics-protocol-v1"
        or protocol.get("status") != "frozen" or protocol.get("source_sha256") != before
        or protocol.get("budget_seconds") != SECONDS or protocol.get("budget_bytes") != OUTPUT_BYTES
        or out.exists() or any(p.is_symlink() for p in (out, *out.parents))
        or any(out.resolve().is_relative_to(Path(path).resolve()) for path in (
            config["model_root"], config["comparison_results"], config["comparison_config"],
            config["protocol"], original["protocol"], original["readiness"],
            original["features_root"], original["baselines_root"], original["dataset_root"]
        ))
    ):
        raise ValueError("diagnostic frozen protocol/fresh output mismatch")
    out.mkdir(parents=True, mode=0o700)
    report = {"schema_version": "wav-readonly-diagnostics-v1", "status": "unavailable",
              "source_sha256": before, "protocol_sha256": protocol_sha,
              "feature_version": AUDIO_VERSION, "feature_names": list(AUDIO_NAMES),
              "new_optimizer_updates": 0, "media_opened": False, "test_media_opened": False,
              "validation_features_opened": False, "opened_feature_splits": ["train"],
              "production_qualification": "unavailable", "deployed": False,
              "feature_diagnostics": {}, "models": []}

    def deadline():
        marker = out / "CANCEL"
        if marker.is_symlink():
            raise ValueError("diagnostic cancellation symlink")
        if marker.exists():
            raise InterruptedError
        if clock() - started >= SECONDS:
            raise TimeoutError

    try:
        deadline()
        report["phase"] = "resource_preflight"
        execution = parent.cpu_bound()
        if execution["memory_max_bytes"] > MEMORY_BYTES:
            raise ValueError("diagnostic bounded resident memory required")
        report["execution"] = execution
        report["phase"] = "input_integrity"
        result, _ = comparison.read(Path(config["comparison_results"]),
                                    protocol["comparison_results_sha256"])
        base_protocol, _ = comparison.read(Path(original["protocol"]), original["protocol_sha256"])
        if (
            result.get("status") != "completed" or result.get("read_only_replay", {}).get(
                "status") != "passed" or result["read_only_replay"]["models_replayed"] != 14
            or result["read_only_replay"]["cells_replayed"] != 480
            or result["fit"]["source_sha256"] != parent.sources()
            or result["protocol_sha256"] != original["protocol_sha256"]
            or base_protocol.get("schema_version") != "nonjpeg-linear-comparison-protocol-v1"
            or base_protocol.get("status") != "frozen"
            or base_protocol.get("source_sha256") != parent.sources()
            or original.get("schema_version") != "nonjpeg-linear-comparison-config-v1"
            or any(execution.get(k) != base_protocol["runtime"][k]
                   for k in ("numpy_version", "python_version"))
        ):
            raise ValueError("diagnostic completed comparison binding mismatch")
        trials = result["fit"]["trials"]
        expected = {t["model"]: t["model_sha256"] for t in trials}
        if len(trials) != 14 or len(expected) != 14 or expected != protocol["model_sha256"]:
            raise ValueError("diagnostic complete model inventory required")

        def snapshots():
            blobs = {}
            for name, sha in expected.items():
                deadline()
                raw = comparison.regular_read(Path(config["model_root"]) / comparison.basename(
                    name), 1024**2)
                if comparison.sha(raw) != sha:
                    raise ValueError("diagnostic model checksum mismatch")
                blobs[name] = raw
            return blobs

        blobs = snapshots()
        report["model_sha256_before"] = expected
        ready, _ = comparison.read(Path(original["readiness"]), base_protocol["readiness_sha256"])
        if (
            ready.get("schema_version") != "nonjpeg-comparison-input-readiness-v1"
            or ready.get("status") != "prepared_not_fitted" or ready.get("originals") != 160
            or any(ready.get(k) != base_protocol.get(k) for k in ("corpus_sha256", "audit_sha256"))
        ):
            raise ValueError("diagnostic original readiness binding mismatch")
        roots = {key: Path(original[key + "_root"]) for key in ("features", "baselines")}
        for source in SOURCES:
            deadline()
            report["phase"] = "train_only_diagnosis"
            selected = [r for r in ready["artifacts"] if r["source_group"] == source
                        and r["feature_version"] == AUDIO_VERSION and r["split"] == "train"]
            if len(selected) != 1:
                raise ValueError("diagnostic train artifact missing or duplicated")
            data = comparison.load_slice(selected[0], roots, "train", source)
            if any(data[k] != ready[k] for k in ("corpus_sha256", "audit_sha256")) or (
                source == "FSDD-v1.0.10" and any(r.get("speaker") not in ("lucas", "nicolas")
                                                for r in data["rows"])
            ):
                raise ValueError("diagnostic historical train roles changed")
            report["feature_diagnostics"][source] = feature_contrasts(
                data["x"], data["rows"], deadline=deadline)
            report["feature_diagnostics"][source].update({
                k: data[k] for k in ("manifest_sha256", "artifact_sha256", "ordered_rows_sha256")})
            for seed in linear.SEEDS:
                matched = [t for t in trials if t["source_group"] == source and t.get(
                    "feature_version") == AUDIO_VERSION and t["seed"] == seed]
                if len(matched) != 1:
                    raise ValueError("diagnostic audio source/seed model missing")
                trial = matched[0]
                model = numeric_model(blobs[trial["model"]])
                report["models"].append(model_diagnosis(model, data, trial, deadline=deadline))
            # Recheck the exact opened documents without opening a validation slice.
            record = selected[0]
            comparison.read(roots["features"] / comparison.basename(record["artifact"]),
                            record["sha256"])
            comparison.read(roots["features"] / comparison.basename(record["manifest"]),
                            record["manifest_sha256"])
        report["model_sha256_after"] = {name: comparison.sha(raw)
                                       for name, raw in snapshots().items()}
        deadline()
        if sources() != before:
            raise ValueError("diagnostic source mutation")
        report["weights_unchanged"] = True
        report["status"] = "completed"
        report["phase"] = "completed"
    except (TimeoutError, InterruptedError) as exc:
        report["reason"] = "time_budget_exhausted" if isinstance(exc, TimeoutError) else "cancelled"
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile):
        report["reason"] = "diagnostic_integrity_or_resource_unavailable"
    finally:
        report["seconds"] = clock() - started
        write(out / "report.json", report)
    return report
