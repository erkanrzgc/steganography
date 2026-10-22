"""Independent receipt fixtures; no fitting, media download or CUDA required."""

import copy
import hashlib
import json
import os
import runpy
import sys

import pytest

from core import wav_control_gate as gate
from steganography import research_wav_control as cli


@pytest.fixture
def protocol():
    return {
        "schema_version": "wav-full-control-protocol-v1", "protocol_id": gate.PROTOCOL_ID,
        "spec": copy.deepcopy(gate.SPEC), "real_fit_authorized": False,
        "source_closure_sha256": dict.fromkeys(gate.SOURCES, hashlib.sha256(b"").hexdigest()),
    }


@pytest.fixture
def receipt(protocol):
    return {
        "schema_version": "wav-full-control-result-v1", "protocol_sha256": "a" * 64,
        "source_closure_sha256": protocol["source_closure_sha256"].copy(),
        "recipe": "audio-pcm16-tlu-parity-gn1d-v3", "generated_only": True,
        "real_media_opened": False, "reserved_media_opened": False,
        "checkpoint_written": False, "real_optimizer_updates": 0,
        "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
        "corpus": {
            "train_files": 400, "development_files": 200,
            "train_groups": 20, "development_groups": 10, "files_per_group": 20,
            "group_overlap": 0, "decoded_pcm_overlap": 0, "payload_oracles_passed": 4800,
            "length_schedule": "exact_metadata_order",
        },
        "wall_seconds": 3000, "host_peak_bytes": 2 * 1024**3,
        "cuda_peak_bytes": 200 * 1024**2, "swap_peak_bytes": 0,
        "seeds": [
            {
                "seed": seed, "epochs": 20, "optimizer_updates": 1140,
                "row_presentations": 72000, "train_files": 400, "development_files": 200,
                "virtual_batch": 64, "final_batch_rows": 16, "train_auc_all": 0.60,
                "development_cells": [
                    {"method": method, "rate_percent": rate, "plane": plane, "roc_auc": 0.70}
                    for method in ("sequential", "scattered")
                    for rate in (5, 40) for plane in (0, 1)
                ],
            } for seed in (20261016, 20261017)
        ],
    }


def test_complete_control_pass_never_authorizes_real_fit(protocol, receipt):
    result = gate.evaluate(protocol, "a" * 64, receipt)
    assert result["status"] == "passed_generated_only"
    assert not result["real_fit_authorized"] and not result["production_qualified"]
    assert [r["seed"] for r in result["seeds"]] == [20261016, 20261017]


@pytest.mark.parametrize("side", ("train", "sequential", "scattered"))
def test_one_failed_seed_or_required_cell_fails_combined_rule(protocol, receipt, side):
    run = receipt["seeds"][1]
    if side == "train":
        run["train_auc_all"] = 0.5999
    else:
        next(c for c in run["development_cells"] if c["method"] == side and
             c["rate_percent"] == 40 and c["plane"] == 0)["roc_auc"] = 0.6999
    result = gate.evaluate(protocol, "a" * 64, receipt)
    assert result["status"] == "failed_learning_gate" and not result["seeds"][1]["passed"]


@pytest.mark.parametrize(("key", "value"), [
    ("generated_only", False), ("generated_only", 1), ("real_media_opened", True),
    ("reserved_media_opened", True), ("checkpoint_written", True),
    ("real_optimizer_updates", 1), ("real_optimizer_updates", False),
    ("device", "cpu"), ("recipe", "v1"), ("schema_version", "old"),
    ("protocol_sha256", "b" * 64), ("source_closure_sha256", {}),
])
def test_provenance_mismatch_is_unavailable(protocol, receipt, key, value):
    receipt[key] = value
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "provenance_or_recipe_mismatch"


@pytest.mark.parametrize(("key", "value"), [
    ("wall_seconds", 3600.1), ("wall_seconds", 0), ("wall_seconds", True),
    ("host_peak_bytes", 8 * 1024**3 + 1), ("cuda_peak_bytes", 4 * 1024**3 + 1),
    ("swap_peak_bytes", 1), ("swap_peak_bytes", None), ("wall_seconds", float("nan")),
    ("cuda_peak_bytes", float("inf")), ("host_peak_bytes", 10**1000),
])
def test_resources_must_have_finite_in_bound_measurements(protocol, receipt, key, value):
    receipt[key] = value
    result = gate.evaluate(protocol, "a" * 64, receipt)
    assert result["reason"] == "resource_evidence_missing_or_exceeded"


@pytest.mark.parametrize(("key", "value"), [
    ("group_overlap", 1), ("group_overlap", False), ("decoded_pcm_overlap", 1),
    ("train_groups", 12),
    ("payload_oracles_passed", 4799), ("length_schedule", "random_sample"),
])
def test_complete_disjoint_corpus_required(protocol, receipt, key, value):
    receipt["corpus"][key] = value
    result = gate.evaluate(protocol, "a" * 64, receipt)
    assert result["reason"] == "complete_disjoint_corpus_required"


@pytest.mark.parametrize("value", (None, [], [{}], [1, 2]))
def test_two_complete_seeds_required(protocol, receipt, value):
    receipt["seeds"] = value
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "two_complete_seeds_required"


@pytest.mark.parametrize("value", (True, 20261016, 20261018))
def test_seed_identity_and_uniqueness(protocol, receipt, value):
    receipt["seeds"][1]["seed"] = value
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "two_complete_seeds_required"


@pytest.mark.parametrize(("key", "value"), [
    ("epochs", 10), ("optimizer_updates", 170), ("row_presentations", 10800),
    ("train_files", 120), ("development_files", 60), ("virtual_batch", True),
    ("final_batch_rows", 64), ("epochs", 20.0),
])
def test_reduced_exposure_never_counts_as_full_control(protocol, receipt, key, value):
    receipt["seeds"][0][key] = value
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "full_exposure_required"


@pytest.mark.parametrize("change", ("missing", "duplicate", "unknown", "nonlist"))
def test_eight_distinct_cells_required(protocol, receipt, change):
    cells = receipt["seeds"][0]["development_cells"]
    if change == "missing":
        cells.pop()
    elif change == "duplicate":
        cells[-1] = cells[0]
    elif change == "unknown":
        cells[-1]["rate_percent"] = 99
    else:
        receipt["seeds"][0]["development_cells"] = None
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "eight_complete_cells_required"


@pytest.mark.parametrize("value", (None, True, -0.1, 1.1, float("inf"), float("nan")))
def test_invalid_train_auc(protocol, receipt, value):
    receipt["seeds"][0]["train_auc_all"] = value
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "invalid_metric_evidence"


@pytest.mark.parametrize("cell", (None, {"method": [], "plane": 0, "rate_percent": 40},
                                  {"method": "sequential", "plane": True, "rate_percent": 40},
                                  {"method": "sequential", "plane": 0, "rate_percent": 40,
                                   "roc_auc": float("nan")}))
def test_invalid_development_evidence(protocol, receipt, cell):
    receipt["seeds"][0]["development_cells"][0] = cell
    assert gate.evaluate(protocol, "a" * 64, receipt)["reason"] == "invalid_metric_evidence"


def test_protocol_requires_exact_sources_and_detects_changed_code(tmp_path, protocol):
    for name in gate.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    doc = tmp_path / "protocol.json"
    doc.write_text(json.dumps(protocol))
    loaded, digest = gate.read_protocol(tmp_path, doc)
    assert loaded == protocol and len(digest) == 64
    (tmp_path / gate.SOURCES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source binding"):
        gate.read_protocol(tmp_path, doc)
    del protocol["source_closure_sha256"][gate.SOURCES[0]]
    doc.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="contract"):
        gate.read_protocol(tmp_path, doc)


def test_source_hash_rejects_symlink_fifo_and_size_limit(tmp_path):
    empty = tmp_path / "empty"
    empty.touch()
    assert gate.source_hash(empty) == hashlib.sha256(b"").hexdigest()
    link = tmp_path / "link"
    link.symlink_to(empty)
    with pytest.raises(ValueError, match="symlink"):
        gate.source_hash(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        gate.source_hash(fifo)
    empty.write_bytes(b"x" * (1024**2 + 1))
    with pytest.raises(ValueError, match="size"):
        gate.source_hash(empty)


def test_document_rejects_nonobject_and_oversize(tmp_path):
    doc = tmp_path / "doc.json"
    doc.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        gate.read_document(doc)
    doc.write_bytes(b"x" * (1024**2 + 1))
    with pytest.raises(ValueError, match="byte bound"):
        gate.read_document(doc)


def test_cli_reports_missing_document_without_host_path(tmp_path, capsys):
    assert cli.main(["--protocol", str(tmp_path / "missing"), "--receipt", "missing"]) == 2
    result = capsys.readouterr().out
    assert str(tmp_path) not in result and json.loads(result)["status"] == "unavailable"


def test_cli_checks_receipt_without_invoking_any_trainer(monkeypatch, protocol, receipt, capsys):
    monkeypatch.setattr(gate, "read_protocol", lambda *_: (protocol, "a" * 64))
    monkeypatch.setattr(gate, "read_document", lambda *_: (receipt, "b" * 64))
    assert cli.main(["--protocol", "bound.json", "--receipt", "run.json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed_generated_only"


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["research_wav_control", "--protocol", "missing",
                                     "--receipt", "missing"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path("steganography/research_wav_control.py", run_name="__main__")
    assert stopped.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
