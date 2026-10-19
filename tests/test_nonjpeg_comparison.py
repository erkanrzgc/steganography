"""Complete generated matrix plus checksum/domain/split/resource failure gates."""

import copy
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import nonjpeg_comparison as c
from core import nonjpeg_linear as linear
from steganography import research_nonjpeg_compare as cli

SHA = "a" * 64


def dump(path, value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def generated(source, split):
    result = []
    for i in range(32 if split == "train" else 8):
        formats = ("PNG", "BMP") if source in c.SOURCES[:2] else ("WAV",)
        cells = [(fmt, None, 0, None, None) for fmt in formats] + sorted(c.expected_cells(source))
        for cell in cells:
            fmt, method, rate, channel, plane = cell
            identity = f"{source}:{split}:{i}:{cell}"
            result.append({"sha256": hashlib.sha256(identity.encode()).hexdigest(),
                           "lineage": hashlib.sha256(f"{source}:{split}:{i}".encode()).hexdigest(),
                           "source_group": source, "split": split,
                           "label": "cover" if method is None else "stego", "format": fmt,
                           "method": method, "rate_percent": rate, "channel": channel,
                           "plane": plane, "path": f"{i}.wav", "size": 1,
                           "speaker": ("lucas", "nicolas")[i % 2] if split == "train"
                           else ("theo", "yweweler")[i % 2],
                           "group_key": "ESC-50:" + split + ":" + str(
                               0 if split == "train" and i == 1 else i)})
    return result


@pytest.fixture
def matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "FEATURE_NAMES", ("a", "b"))
    for version in c.features.CONTRACTS:
        monkeypatch.setitem(c.features.CONTRACTS, version, (("a", "b"), None))
    feature_root, baseline_root = tmp_path / "features", tmp_path / "baselines"
    feature_root.mkdir()
    baseline_root.mkdir()
    artifacts = []
    for source in c.SOURCES:
        for split in ("train", "validation"):
            rows = generated(source, split)
            manifest = {"schema_version": "1.0", "development_only": True, "samples": rows,
                        "preparation_manifest_sha256": SHA, "independent_audit_sha256": SHA}
            manifest_name = source + "-" + split + ".json"
            manifest_sha = dump(feature_root / manifest_name, manifest)
            for version in c.VERSIONS[source]:
                name = source + "-" + split + "-" + version + ".json"
                values = [{"sha256": r["sha256"], "lineage": r["lineage"], "label": r["label"],
                           "values": [0.2, float(r["label"] == "stego")]}
                          for r in rows]
                feature = {"schema_version": "research-features-1", "feature_version": version,
                           "feature_names": ["a", "b"], "manifest_sha256": manifest_sha,
                           "split": split, "rows": values}
                root = baseline_root if version == c.FEATURE_VERSION else feature_root
                artifacts.append({"source_group": source, "feature_version": version,
                                  "split": split, "manifest": manifest_name,
                                  "manifest_sha256": manifest_sha, "artifact": name,
                                  "sha256": dump(root / name, feature),
                                  "ordered_rows_sha256": linear.row_identity(rows)})
    readiness = {"schema_version": "nonjpeg-comparison-input-readiness-v1",
                 "status": "prepared_not_fitted", "originals": 160,
                 "corpus_sha256": SHA, "audit_sha256": SHA, "artifacts": artifacts}
    ready_path = tmp_path / "readiness.json"
    ready_sha = dump(ready_path, readiness)
    runtime = {"numpy_version": "fixture", "python_version": "fixture"}
    protocol = {"schema_version": "nonjpeg-linear-comparison-protocol-v1", "status": "frozen",
                "source_sha256": c.sources(), "readiness_sha256": ready_sha,
                "corpus_sha256": SHA, "audit_sha256": SHA, "seeds": list(linear.SEEDS),
                "epochs": 20, "budget_seconds": 1800, "budget_bytes": c.OUTPUT_BYTES,
                "runtime": runtime}
    protocol_path = tmp_path / "protocol.json"
    config = {"schema_version": "nonjpeg-linear-comparison-config-v1",
              "protocol": str(protocol_path), "protocol_sha256": dump(protocol_path, protocol),
              "readiness": str(ready_path), "features_root": str(feature_root),
              "baselines_root": str(baseline_root), "dataset_root": str(tmp_path / "media")}
    monkeypatch.setattr(c, "cpu_bound", lambda: runtime)
    return SimpleNamespace(root=tmp_path, config=config, protocol=protocol, readiness=readiness,
                           artifacts=artifacts, roots={"features": feature_root,
                                                      "baselines": baseline_root})


def test_complete_generated_matrix_and_parameter_replay(matrix, monkeypatch):
    monkeypatch.setattr(c, "wav_reference", lambda rows, *a: np.zeros(len(rows)))
    report = c.execute(matrix.config, matrix.root / "out")
    assert report["status"] == "completed" and len(report["trials"]) == 14
    assert report["real_data_optimizer_updates"] > 0
    assert report["real_data_optimizer_updates"] == sum(
        t["cv"]["all_optimizer_updates"] for t in report["trials"])
    assert not report["gpu_used"] and not report["test_media_opened"]
    assert not report["deep_models_trained"] and not report["deployed"]
    first = report["trials"][0]
    with np.load(matrix.root / "out" / first["model"], allow_pickle=False) as archive:
        model = dict(archive)
    rows = generated("BOWS2", "validation")
    features = np.array([[0.2, float(r["label"] == "stego")] for r in rows])
    np.testing.assert_array_equal(linear.predict(model, features), first["validation_scores"])
    assert first["cells"][0]["paired_delta_ci95"]["roc_auc"] == [0, 0]
    for seed in linear.SEEDS:
        matched = [t for t in report["trials"]
                   if t["source_group"] == "DIV2K" and t["seed"] == seed]
        assert len({t["training"]["exposure_sha256"] for t in matched}) == 1


@pytest.mark.parametrize("fault", ["hash", "source", "split", "domain", "names", "row_order",
                                  "value_type", "nonfinite", "missing_cell", "original_count"])
def test_bound_slice_rejects_adversarial_variants(matrix, fault):
    record = copy.deepcopy(matrix.artifacts[0])
    manifest_path = matrix.roots["features"] / record["manifest"]
    feature_path = matrix.roots["baselines"] / record["artifact"]
    manifest, _ = c.read(manifest_path)
    artifact, _ = c.read(feature_path)
    if fault == "hash":
        record["sha256"] = "0" * 64
    elif fault == "source":
        manifest["samples"][0]["source_group"] = "unknown"
    elif fault == "split":
        manifest["samples"][0]["split"] = "test"
    elif fault == "domain":
        artifact["feature_version"] = "unknown"
    elif fault == "names":
        artifact["feature_names"] = ["wrong", "names"]
    elif fault == "row_order":
        artifact["rows"].reverse()
    elif fault == "value_type":
        artifact["rows"][0]["values"][0] = True
    elif fault == "nonfinite":
        artifact["rows"][0]["values"][0] = 2
    elif fault == "missing_cell":
        cell = manifest["samples"][2]
        keep = [i for i, row in enumerate(manifest["samples"])
                if tuple(row[k] for k in c.metrics.CELL_KEYS) !=
                tuple(cell[k] for k in c.metrics.CELL_KEYS)]
        manifest["samples"] = [manifest["samples"][i] for i in keep]
        artifact["rows"] = [artifact["rows"][i] for i in keep]
    else:
        lineage = manifest["samples"][0]["lineage"]
        keep = [i for i, row in enumerate(manifest["samples"]) if row["lineage"] != lineage]
        manifest["samples"] = [manifest["samples"][i] for i in keep]
        artifact["rows"] = [artifact["rows"][i] for i in keep]
    if fault != "hash":
        record["ordered_rows_sha256"] = linear.row_identity(manifest["samples"])
        record["manifest_sha256"] = dump(manifest_path, manifest)
        artifact["manifest_sha256"] = record["manifest_sha256"]
        record["sha256"] = dump(feature_path, artifact)
    with pytest.raises(ValueError):
        c.load_slice(record, matrix.roots, "train", "BOWS2")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "readiness", "audit", "exposure",
                                  "roles", "overlap", "prefix"])
def test_complete_matrix_integrity_and_role_isolation(matrix, monkeypatch, fault):
    readiness = matrix.readiness
    if fault == "missing":
        readiness["artifacts"] = readiness["artifacts"][:-1]
    elif fault == "duplicate":
        readiness["artifacts"].append(readiness["artifacts"][0])
    elif fault == "readiness":
        readiness["status"] = "incomplete"
    if fault in {"missing", "duplicate", "readiness"}:
        matrix.protocol["readiness_sha256"] = dump(Path(matrix.config["readiness"]), readiness)
    else:
        original = c.load_slice

        def load(record, root, split, source):
            value = original(record, root, split, source)
            if fault == "audit":
                value["audit_sha256"] = "0" * 64
            elif fault == "exposure" and record["feature_version"] != c.FEATURE_VERSION:
                value["ordered_rows_sha256"] = "0" * 64
            elif fault == "roles" and source == "FSDD-v1.0.10":
                value["rows"][0]["speaker"] = "george"
            elif fault == "overlap" and source == "BOWS2" and split == "validation":
                reference = generated("BOWS2", "train")[0]["lineage"]
                old = value["rows"][0]["lineage"]
                for row in value["rows"]:
                    if row["lineage"] == old:
                        row["lineage"] = reference
            elif fault == "prefix" and record["feature_version"] != c.FEATURE_VERSION:
                value["x"][0, 0] = 0.9
            return value

        monkeypatch.setattr(c, "load_slice", load)
    with pytest.raises(ValueError):
        c.load_inputs(matrix.config, matrix.protocol)


@pytest.mark.parametrize("fault", ["protocol", "output", "timeout", "cancel", "marker_symlink",
                                  "runtime", "contract", "mutation", "partial"])
def test_failures_never_qualify_or_erase_partial_work(matrix, monkeypatch, fault):
    out = matrix.root / "out"
    if fault == "protocol":
        matrix.config["protocol_sha256"] = "0" * 64
    if fault == "output":
        out.mkdir()
    if fault in {"protocol", "output"}:
        with pytest.raises(ValueError):
            c.execute(matrix.config, out)
        return
    def clock():
        return 0

    if fault == "timeout":
        times = iter([0, 1801, 1802])

        def clock():
            return next(times)
    if fault in {"cancel", "marker_symlink"}:
        original = Path.mkdir

        def mkdir(path, *a, **k):
            result = original(path, *a, **k)
            if path == out:
                if fault == "cancel":
                    (out / "CANCEL").touch()
                else:
                    (out / "CANCEL").symlink_to(matrix.root / "missing")
            return result

        monkeypatch.setattr(Path, "mkdir", mkdir)
    if fault == "runtime":
        monkeypatch.setattr(c, "cpu_bound", lambda: {})
    elif fault == "contract":
        monkeypatch.setattr(c, "load_inputs", lambda *a: (_ for _ in ()).throw(
            ValueError("/private/password=secret")))
    elif fault == "mutation":
        before = c.sources()
        calls = iter([before, {}])
        monkeypatch.setattr(c, "sources", lambda: next(calls))
        monkeypatch.setattr(c, "wav_reference", lambda rows, *a: np.zeros(len(rows)))
    elif fault == "partial":
        def fit(*a, **kwargs):
            kwargs["on_update"]()
            raise TimeoutError

        monkeypatch.setattr(linear, "grouped_fit", fit)
    if fault == "marker_symlink":
        with pytest.raises(ValueError):
            c.execute(matrix.config, out, clock=clock)
        return
    report = c.execute(matrix.config, out, clock=clock)
    assert report["status"] == "unavailable" and not report["deployed"]
    assert "/private/" not in json.dumps(report) and "password" not in json.dumps(report)
    if fault == "partial":
        assert report["real_data_optimizer_updates"] == 1


def test_documents_symlinks_fifo_overwrite_and_cumulative_limits(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text('{"ok":true}')
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        c.read(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        c.read(fifo)
    source.write_text("[]")
    with pytest.raises(ValueError):
        c.read(source)
    with pytest.raises(ValueError):
        c.write(source, {})
    link.unlink()
    fifo.unlink()
    monkeypatch.setattr(c, "OUTPUT_BYTES", 40)
    with pytest.raises(ValueError):
        c.write(tmp_path / "large.json", {"large": "x" * 100})
    (tmp_path / "existing").write_bytes(b"x" * 40)
    with pytest.raises(ValueError):
        c.write(tmp_path / "overflow.json", {})


@pytest.mark.parametrize("fault", [None, "memory", "quota", "period", "swap", "threads"])
def test_kernel_cpu_and_memory_bounds_are_mandatory(tmp_path, monkeypatch, fault):
    root = tmp_path / "cgroup"
    limited = root / "limited"
    limited.mkdir(parents=True)
    proc = tmp_path / "identity"
    proc.write_text("0::/limited\n")
    (limited / "memory.max").write_text(str(2 * 1024**3 if fault != "memory" else 3 * 1024**3))
    (limited / "memory.swap.max").write_text("0" if fault != "swap" else "4096")
    (limited / "cpu.max").write_text({"quota": "200000 100000", "period": "100000 0"}.get(
        fault, "100000 100000"))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1" if fault != "threads" else "2")
    if fault:
        with pytest.raises(ValueError):
            c.cpu_bound(proc=proc, root=root)
    else:
        assert c.cpu_bound(proc=proc, root=root)["cpu_quota"] == 1


def test_native_reference_checks_bytes_and_analyzer_coverage(tmp_path, monkeypatch):
    root, out = tmp_path / "media", tmp_path / "out"
    root.mkdir()
    out.mkdir()
    raw = b"bounded generated reference fixture"
    (root / "a.wav").write_bytes(raw)
    rows = [{"path": "a.wav", "size": len(raw), "sha256": c.sha(raw)}]
    monkeypatch.setattr(c.AudioWav, "analyze", lambda *a: SimpleNamespace(
        status="ok", suspicion=42))
    np.testing.assert_array_equal(c.wav_reference(rows, root, out, lambda: None), [0.42])
    monkeypatch.setattr(c.AudioWav, "analyze", lambda *a: SimpleNamespace(status="unavailable"))
    with pytest.raises(ValueError):
        c.wav_reference(rows, root, out, lambda: None)
    rows[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.wav_reference(rows, root, out, lambda: None)


def test_cli_is_a_thin_adapter(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(cli.service, "read", lambda *a: ({}, SHA))
    monkeypatch.setattr(cli.service, "execute", lambda *a: {"status": "completed"})
    assert cli.main(["--config", "private", "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.setattr(cli.service, "execute", lambda *a: {"status": "unavailable"})
    assert cli.main(["--config", "private", "--out", str(tmp_path)]) == 2
    monkeypatch.setattr(sys, "argv", [str(Path(cli.__file__)), "--config", "private",
                                      "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert caught.value.code == 2


@pytest.mark.parametrize("fault", ["artifact_symlink", "output_budget", "reference_pairing"])
def test_live_output_and_reference_guards_stop_before_qualification(matrix, monkeypatch, fault):
    out = matrix.root / "out"
    if fault == "reference_pairing":
        datasets = c.load_inputs(matrix.config, matrix.protocol)
        monkeypatch.setattr(c, "load_inputs", lambda *a: datasets)
        original = c.metrics.pair_cells
        calls = 0

        def pair_cells(*args):
            nonlocal calls
            calls += 1
            result = original(*args)
            if calls == 2:
                cell, pairs, groups = result[0]
                result[0] = cell, pairs, ["wrong-group"] * len(groups)
            return result

        monkeypatch.setattr(c.metrics, "pair_cells", pair_cells)
    else:
        original = Path.mkdir

        def mkdir(path, *a, **k):
            result = original(path, *a, **k)
            if path == out:
                if fault == "artifact_symlink":
                    (out / "unexpected").symlink_to(matrix.root / "absent")
                else:
                    (out / "allocation.bin").write_bytes(b"x" * 1024)
                    monkeypatch.setattr(c, "OUTPUT_BYTES", 65536)
            return result

        monkeypatch.setattr(Path, "mkdir", mkdir)
    if fault == "artifact_symlink":
        with pytest.raises(ValueError):
            c.execute(matrix.config, out)
    else:
        report = c.execute(matrix.config, out)
        assert report["status"] == "unavailable"
        if fault == "output_budget":
            assert report["real_data_optimizer_updates"] == 0


@pytest.mark.parametrize("fault", ["schema", "features_root", "dataset_root", "baselines_root"])
def test_output_cannot_contaminate_inputs_and_config_schema_is_explicit(matrix, fault):
    out = matrix.root / "out"
    if fault == "schema":
        matrix.config["schema_version"] = "unknown"
    else:
        out = Path(matrix.config[fault]) / "new-models"
    with pytest.raises(ValueError):
        c.execute(matrix.config, out)
    assert not out.exists()
