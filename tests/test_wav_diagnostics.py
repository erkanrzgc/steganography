"""Independent scalar diagnosis fixtures, closed roles and hostile snapshots."""

import copy
import io
import json
import math
import os
import platform
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import wav_diagnostics as d
from steganography import research_wav_diagnostics as cli
from tests.test_nonjpeg_comparison import dump, generated

SHA = "a" * 64


def pack(model, compressed=False):
    stream = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(stream, **model)
    return stream.getvalue()


def scalar_model(x, rows, sign):
    weights = [1/64 if r["label"] == "cover" else 1/512 for r in rows]
    mean0 = math.fsum(w*float(v) for w, v in zip(weights, x[:, 0], strict=True))
    scale0 = math.sqrt(math.fsum(w*(float(v)-mean0)**2
                                 for w, v in zip(weights, x[:, 0], strict=True)))
    model = {"mean": np.zeros(760), "scale": np.full(760, 1e-6),
             "weights": np.zeros(760), "bias": np.asarray(0.025)}
    model["mean"][:2] = mean0, 0.25
    model["scale"][0] = scale0
    model["weights"][0] = sign
    logits = [sign*(float(v)-mean0)/scale0+0.025 for v in x[:, 0]]
    loss = math.fsum(w*(math.log1p(math.exp(z))-(r["label"] == "stego")*z)
                     for w, z, r in zip(weights, logits, rows, strict=True))
    return model, loss


def bind(f):
    f.base_protocol["readiness_sha256"] = dump(f.ready_path, f.ready)
    f.comparison["protocol_sha256"] = dump(f.parent_path, f.base_protocol)
    f.result["protocol_sha256"] = f.comparison["protocol_sha256"]
    f.protocol["comparison_config_sha256"] = dump(f.comparison_path, f.comparison)
    f.protocol["comparison_results_sha256"] = dump(f.result_path, f.result)
    f.config["protocol_sha256"] = dump(f.protocol_path, f.protocol)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    features, models = tmp_path / "features", tmp_path / "models"
    features.mkdir()
    models.mkdir()
    artifacts, datasets = [], {}
    for source in d.SOURCES:
        rows = generated(source, "train")
        x = np.zeros((len(rows), 760))
        x[:, 1] = 0.25
        for i, row in enumerate(rows):
            x[i, 0] = int(row["path"][:-4])/64 + (0.01 if row["rate_percent"] == 5 else 0.08
                                                  if row["rate_percent"] == 40 else 0)
        datasets[source] = {"x": x, "y": np.array([r["label"] == "stego" for r in rows]),
                            "rows": rows}
        manifest_name = source + "-train.json"
        manifest_sha = dump(features / manifest_name, {
            "development_only": True, "samples": rows,
            "preparation_manifest_sha256": SHA, "independent_audit_sha256": SHA})
        artifact_name = source + "-train-features.json"
        artifact_sha = dump(features / artifact_name, {
            "schema_version": "research-features-1", "feature_version": d.AUDIO_VERSION,
            "feature_names": list(d.AUDIO_NAMES), "manifest_sha256": manifest_sha,
            "split": "train", "rows": [{"sha256": r["sha256"], "lineage": r["lineage"],
                                          "label": r["label"], "values": v.tolist()}
                                         for r, v in zip(rows, x, strict=True)]})
        artifacts.append({"source_group": source, "feature_version": d.AUDIO_VERSION,
                          "split": "train", "manifest": manifest_name,
                          "manifest_sha256": manifest_sha, "artifact": artifact_name,
                          "sha256": artifact_sha,
                          "ordered_rows_sha256": d.linear.row_identity(rows)})
        # Opening either closed validation artifact would be an integrity failure.
        closed = source + "-validation.json"
        os.mkfifo(features / closed)
        artifacts.append({"source_group": source, "feature_version": d.AUDIO_VERSION,
                          "split": "validation", "manifest": closed, "artifact": closed,
                          "sha256": SHA, "manifest_sha256": SHA})
    trials = []
    for source in d.comparison.SOURCES:
        for version in d.comparison.VERSIONS[source]:
            for seed in d.linear.SEEDS:
                name = f"model-{len(trials):02d}.npz"
                trial = {"model": name, "source_group": source, "feature_version": version,
                         "seed": seed}
                if source in d.SOURCES:
                    data = datasets[source]
                    model, loss = scalar_model(data["x"], data["rows"],
                                               1 if seed == d.linear.SEEDS[0] else -1)
                    raw = pack(model)
                    trial.update({"training": {"losses": [loss]*20}, "cv": {"threshold": 0.5},
                                  "validation_scores": [0.5]*72,
                                  "cells": [{**cell, "metrics": {"roc_auc": 0.5,
                                             "balanced_accuracy": 0.5, "recall": 1.0,
                                             "false_positive_rate": 1.0, "ece": 0.0}}
                                            for cell, *_ in d.pair_indices(data["rows"])]})
                else:
                    raw = b"opaque generated spatial snapshot; hash-only access"
                (models / name).write_bytes(raw)
                trial["model_sha256"] = d.comparison.sha(raw)
                trials.append(trial)
    f = SimpleNamespace(root=tmp_path, features=features, models=models, datasets=datasets,
                        ready_path=tmp_path / "ready.json", parent_path=tmp_path / "parent.json",
                        comparison_path=tmp_path / "comparison.json",
                        result_path=tmp_path / "result.json",
                        protocol_path=tmp_path / "protocol.json")
    f.ready = {"schema_version": "nonjpeg-comparison-input-readiness-v1",
               "status": "prepared_not_fitted", "originals": 160,
               "corpus_sha256": SHA, "audit_sha256": SHA, "artifacts": artifacts}
    f.base_protocol = {"schema_version": "nonjpeg-linear-comparison-protocol-v1",
                       "status": "frozen", "source_sha256": d.parent.sources(),
                       "corpus_sha256": SHA, "audit_sha256": SHA,
                       "runtime": {"numpy_version": np.__version__,
                                   "python_version": platform.python_version()}}
    f.comparison = {"schema_version": "nonjpeg-linear-comparison-config-v1",
                    "protocol": str(f.parent_path), "readiness": str(f.ready_path),
                    "features_root": str(features), "baselines_root": str(tmp_path / "baselines"),
                    "dataset_root": str(tmp_path / "closed-media")}
    f.result = {"status": "completed", "read_only_replay": {
        "status": "passed", "models_replayed": 14, "cells_replayed": 480},
        "fit": {"source_sha256": d.parent.sources(), "trials": trials}}
    f.protocol = {"schema_version": "wav-readonly-diagnostics-protocol-v1", "status": "frozen",
                  "source_sha256": d.sources(), "budget_seconds": 300,
                  "budget_bytes": d.OUTPUT_BYTES,
                  "model_sha256": {t["model"]: t["model_sha256"] for t in trials}}
    f.config = {"schema_version": "wav-readonly-diagnostics-config-v1",
                "protocol": str(f.protocol_path), "comparison_config": str(f.comparison_path),
                "comparison_results": str(f.result_path), "model_root": str(models)}
    bind(f)
    monkeypatch.setattr(d.parent, "cpu_bound", lambda: {"memory_max_bytes": d.MEMORY_BYTES,
                        **f.base_protocol["runtime"], "gpu_used": False})
    return f


def test_scalar_feature_contrasts_and_model_metrics(fixture):
    data = fixture.datasets["ESC-50"]
    result = d.feature_contrasts(data["x"], data["rows"])
    assert result["constant_features"] == 759 and result["originals"] == 32
    spread = math.sqrt(sum((i/64-15.5/64)**2 for i in range(32))/32)/math.sqrt(760)
    assert result["cover_variation_rms"] == pytest.approx(spread)
    model, _ = scalar_model(data["x"], data["rows"], 1)
    trial = fixture.result["fit"]["trials"][10]
    diagnosis = d.model_diagnosis(model, data, trial)
    for cell, (_, negative, positive, _) in zip(result["cells"], d.pair_indices(data["rows"]),
                                               strict=True):
        change = 0.01 if cell["rate_percent"] == 5 else 0.08
        assert cell["paired_delta_rms"] == pytest.approx(change/math.sqrt(760))
        assert cell["aligned_shift_to_cover_variation"] == pytest.approx(
            change/math.sqrt(760)/spread)
        assert cell["zero_delta_features"] == 759 and cell["same_strict_sign_features"] == 1
        def score(i):
            return 1/(1+math.exp(-((float(data['x'][i, 0])-model['mean'][0])
                                     /model['scale'][0]+0.025)))
        a, b = [score(i) for i in negative], [score(i) for i in positive]
        auc = sum(float(p > n)+0.5*float(p == n) for p in b for n in a)/32**2
        found = next(c for c in diagnosis["cells"] if all(c[k] == cell[k]
                     for k in d.comparison.metrics.CELL_KEYS))
        assert found["train_metrics"]["roc_auc"] == pytest.approx(auc)
        assert found["train_metrics"]["balanced_accuracy"] == pytest.approx(
            (sum(p >= 0.5 for p in b)+sum(n < 0.5 for n in a))/64)
        assert found["stego_score_gt_cover_fraction"] == 1
    assert diagnosis["oof_score_replay"].startswith("unavailable")
    constant = {**data, "x": np.full_like(data["x"], 0.25)}
    assert all(c["aligned_shift_to_cover_variation"] is None
               for c in d.feature_contrasts(constant["x"], constant["rows"])["cells"])


def test_complete_readonly_diagnosis_forbids_optimizers_and_validation_reads(fixture, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("optimizer must never be called")

    monkeypatch.setattr(d.linear, "fit", forbidden)
    monkeypatch.setattr(d.linear, "grouped_fit", forbidden)
    original = d.comparison.regular_read

    def read(path, limit):
        assert "validation" not in path.name and "closed-media" not in path.parts
        return original(path, limit)

    monkeypatch.setattr(d.comparison, "regular_read", read)
    before = {p.name: p.read_bytes() for p in fixture.models.iterdir()}
    report = d.execute(fixture.config, fixture.root / "out")
    assert report["status"] == "completed" and report["weights_unchanged"]
    assert len(report["models"]) == 4 and report["new_optimizer_updates"] == 0
    assert report["opened_feature_splits"] == ["train"]
    assert not report["validation_features_opened"] and not report["media_opened"]
    assert {p.name: p.read_bytes() for p in fixture.models.iterdir()} == before


@pytest.mark.parametrize("fault", ["raw_size", "not_zip", "extra", "compressed_size", "object",
                                  "dtype", "shape", "nonfinite", "bias_shape"])
def test_numeric_snapshot_members_dtype_shapes_and_byte_limits(fault):
    model = {"mean": np.zeros(760), "scale": np.ones(760), "weights": np.zeros(760),
             "bias": np.asarray(0.0)}
    if fault == "raw_size":
        raw = b"x"*(1024**2+1)
    elif fault == "not_zip":
        raw = b"not a model"
    else:
        if fault == "extra":
            model["../outside"] = np.zeros(1)
        elif fault == "compressed_size":
            model["weights"] = np.zeros(200_000)
        elif fault == "object":
            model["weights"] = np.array([object()], dtype=object)
        elif fault == "dtype":
            model["weights"] = np.zeros(760, dtype=np.float32)
        elif fault == "shape":
            model["weights"] = np.zeros(759)
        elif fault == "nonfinite":
            model["weights"][0] = np.nan
        else:
            model["bias"] = np.zeros(1)
        raw = pack(model, compressed=fault == "compressed_size")
    with pytest.raises((ValueError, d.zipfile.BadZipFile)):
        d.numeric_model(raw)


@pytest.mark.parametrize("fault", ["result_hash", "source", "inventory", "readiness", "missing",
                                  "roles", "runtime", "parent_status", "parent_replay",
                                  "parent_source", "comparison_schema", "model_hash", "memory",
                                  "audio_domain"])
def test_invalid_inputs_remain_unavailable_without_new_work(fixture, monkeypatch, fault):
    if fault == "result_hash":
        fixture.protocol["comparison_results_sha256"] = "0"*64
        fixture.config["protocol_sha256"] = dump(fixture.protocol_path, fixture.protocol)
    else:
        if fault == "source":
            fixture.protocol["source_sha256"] = {}
        elif fault == "inventory":
            fixture.result["fit"]["trials"].pop()
        elif fault == "readiness":
            fixture.ready["originals"] = 159
        elif fault == "missing":
            fixture.ready["artifacts"] = [r for r in fixture.ready["artifacts"]
                                           if r["source_group"] != "ESC-50"]
        elif fault == "roles":
            original = d.comparison.load_slice

            def load(*a):
                data = original(*a)
                if a[-1] == "FSDD-v1.0.10":
                    data["rows"][0]["speaker"] = "george"
                return data

            monkeypatch.setattr(d.comparison, "load_slice", load)
        elif fault == "runtime":
            fixture.base_protocol["runtime"] = {"numpy_version": "other", "python_version": "other"}
            monkeypatch.setattr(d.parent, "cpu_bound", lambda: {"memory_max_bytes": d.MEMORY_BYTES})
        elif fault == "parent_status":
            fixture.result["status"] = "unavailable"
        elif fault == "parent_replay":
            fixture.result["read_only_replay"]["status"] = "unavailable"
        elif fault == "parent_source":
            fixture.base_protocol["source_sha256"] = {}
        elif fault == "comparison_schema":
            fixture.comparison["schema_version"] = "unknown"
        elif fault == "model_hash":
            (fixture.models / "model-00.npz").write_bytes(b"changed")
        elif fault == "audio_domain":
            fixture.result["fit"]["trials"][10]["feature_version"] = "unknown"
        else:
            monkeypatch.setattr(d.parent, "cpu_bound", lambda: {"memory_max_bytes": 1024**3})
        bind(fixture)
    if fault == "source":
        with pytest.raises(ValueError):
            d.execute(fixture.config, fixture.root / "out")
    else:
        result = d.execute(fixture.config, fixture.root / "out")
        assert result["status"] == "unavailable" and result["new_optimizer_updates"] == 0
        assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("fault", ["normalization", "loss", "threshold", "split", "dimension"])
def test_scalar_replay_and_train_only_guards(fixture, fault):
    data = copy.deepcopy(fixture.datasets["ESC-50"])
    model, _ = scalar_model(data["x"], data["rows"], 1)
    trial = copy.deepcopy(fixture.result["fit"]["trials"][10])
    if fault == "normalization":
        model["mean"][0] += 0.1
    elif fault == "loss":
        trial["training"]["losses"][-1] += 1
    elif fault == "threshold":
        trial["cv"]["threshold"] = 1.1
    elif fault == "split":
        data["rows"][0]["split"] = "validation"
    else:
        data["x"] = data["x"][:, :-1]
    with pytest.raises(ValueError):
        d.model_diagnosis(model, data, trial)
    if fault in ("split", "dimension"):
        with pytest.raises(ValueError):
            d.feature_contrasts(data["x"], data["rows"])


@pytest.mark.parametrize("fault", ["schema", "exists", "protected", "out_symlink", "cancel",
                                  "cancel_symlink", "timeout", "mutation", "source_mutation"])
def test_owned_outputs_cancellation_and_mutation_guards(fixture, monkeypatch, fault):
    out = fixture.root / "out"
    clock = d.time.monotonic
    if fault == "schema":
        fixture.config["schema_version"] = "unknown"
    elif fault == "exists":
        out.mkdir()
    elif fault == "protected":
        out = fixture.models / "new-output"
    elif fault == "out_symlink":
        out.symlink_to(fixture.models, target_is_directory=True)
    elif fault in ("cancel", "cancel_symlink"):
        original = Path.mkdir

        def mkdir(path, *a, **kw):
            result = original(path, *a, **kw)
            if path == out:
                if fault == "cancel":
                    (out / "CANCEL").touch()
                else:
                    (out / "CANCEL").symlink_to(fixture.root / "absent")
            return result

        monkeypatch.setattr(Path, "mkdir", mkdir)
    elif fault == "timeout":
        times = iter([0, 301, 302])
        def clock():
            return next(times)
    elif fault == "mutation":
        original = d.linear.predict

        def predict(*a):
            (fixture.models / "model-00.npz").write_bytes(b"external mutation")
            return original(*a)

        monkeypatch.setattr(d.linear, "predict", predict)
    else:
        before = d.sources()
        values = iter([before, {}])
        monkeypatch.setattr(d, "sources", lambda: next(values))
    if fault in ("schema", "exists", "protected", "out_symlink", "cancel_symlink"):
        with pytest.raises(ValueError):
            d.execute(fixture.config, out, clock=clock)
    else:
        result = d.execute(fixture.config, out, clock=clock)
        assert result["status"] == "unavailable" and result["new_optimizer_updates"] == 0
        assert result.get("weights_unchanged") is not True


def test_output_budget_and_cli_adapter(fixture, monkeypatch, capsys):
    monkeypatch.setattr(d, "OUTPUT_BYTES", 64)
    (fixture.root / "allocated").write_bytes(b"x"*64)
    with pytest.raises(ValueError):
        d.write(fixture.root / "bounded.json", {"ok": True})
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(cli.documents, "read", lambda *a: ({}, SHA))
    monkeypatch.setattr(cli.service, "execute", lambda *a: {"status": "completed"})
    args = ["--config", "private", "--out", "fresh"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.setattr(cli.service, "execute", lambda *a: {"status": "unavailable"})
    assert cli.main(args) == 2
    monkeypatch.setattr(sys, "argv", [cli.__file__, *args])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert caught.value.code == 2
