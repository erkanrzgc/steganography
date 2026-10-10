"""Generated independent scalar replay, corruption and bounded-input cases."""

import copy
import hashlib
import math

import pytest

from core import jpeg_norm_audit as audit


def fixture():
    rows, outputs, cells = [], [], []
    for source, qualities in (("A", (None,)), ("B", (75, 95)), ("C", (75, 95))):
        for original in range(32):
            for quality in qualities:
                for method in (None, "JUNIWARD", "UERD"):
                    index = len(rows)
                    rows.append(
                        {
                            "source_group": source,
                            "lineage": f"{source}-{original:02}",
                            "quality_factor": quality,
                            "method": method,
                            "label": "cover" if method is None else "stego",
                            "split": "train",
                        }
                    )
                    if original >= 24:
                        outputs.append(
                            {"row": index, "logits": [0.0, 1.0 if original % 2 else -1.0]}
                        )
        for quality in qualities:
            for method in ("JUNIWARD", "UERD"):
                cells.append(
                    {
                        "source_group": source,
                        "quality_factor": quality,
                        "method": method,
                        "cover_rows": 8,
                        "stego_rows": 8,
                        "true_positive": 4,
                        "true_negative": 4,
                        "false_positive": 4,
                        "false_negative": 4,
                        "balanced_accuracy": 0.5,
                        "recall": 0.5,
                        "false_positive_rate": 0.5,
                        "cross_entropy": math.log1p(math.exp(-1)) + 0.5,
                    }
                )
    report = {
        "schema_version": "jpeg-bn-gn-learning-pilot-v1",
        "status": "completed",
        "manifest_sha256": audit.MANIFEST_SHA,
        "audit_sha256": audit.AUDIT_SHA,
        "epochs": 8,
        "optimizer_updates": 1152,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "validation_test_used": False,
        "probe_previously_inspected": True,
        "probe_logits": outputs,
        "probe_cells": cells,
        "arm": "bn",
        "source_sha256": {"mock": "0" * 64},
        "initial_parameter_sha256": "1" * 64,
        "schedule_sha256": "2" * 64,
        "schedules": [],
        "seed": 20261010,
        "optimizer": {"name": "mock"},
    }
    return rows, report


def test_scalar_counts_and_loss_match_independent_closed_form():
    rows, report = fixture()
    result = audit.replay(report, rows)
    assert len(result) == 10 and all(c["counts_exact"] for c in result)


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "metadata",
        "role",
        "originals",
        "output_count",
        "duplicate_row",
        "nan",
        "huge",
        "cell_count",
        "duplicate_cell",
        "unknown_context",
        "wrong_loss",
        "bool_metric",
    ],
)
def test_replay_rejects_incomplete_tampered_or_nonfinite_evidence(change):
    rows, report = fixture()
    if change == "status":
        report["status"] = "failed"
    elif change == "metadata":
        rows.pop()
    elif change == "role":
        rows[0]["split"] = "test"
    elif change == "originals":
        rows[0]["lineage"] = "extra"
    elif change == "output_count":
        report["probe_logits"].pop()
    elif change == "duplicate_row":
        report["probe_logits"][1] = report["probe_logits"][0]
    elif change == "nan":
        report["probe_logits"][0]["logits"][0] = float("nan")
    elif change == "huge":
        report["probe_logits"][0]["logits"][0] = 1e308
    elif change == "cell_count":
        report["probe_cells"].pop()
    elif change == "duplicate_cell":
        report["probe_cells"][1] = report["probe_cells"][0]
    elif change == "unknown_context":
        report["probe_cells"][0]["source_group"] = "unknown"
    elif change == "wrong_loss":
        report["probe_cells"][0]["cross_entropy"] += 0.01
    else:
        report["probe_cells"][0]["recall"] = True
    with pytest.raises(ValueError):
        audit.replay(report, rows)


def test_bound_checksum_and_symlink(tmp_path):
    path = tmp_path / "value.json"
    raw = b'{"test": true}'
    path.write_bytes(raw)
    assert audit.read(path, hashlib.sha256(raw).hexdigest()) == {"test": True}
    with pytest.raises(ValueError):
        audit.read(path, "0" * 64)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises((ValueError, OSError)):
        audit.read(link, hashlib.sha256(raw).hexdigest())
    path.write_bytes(b"x" * (2 * 1024**2 + 1))
    with pytest.raises(ValueError):
        audit.read(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("change", [None, "arm", "initialization"])
def test_pair_comparison_binds_both_arms(tmp_path, monkeypatch, change):
    rows, bn = fixture()
    gn = copy.deepcopy(bn)
    gn["arm"] = "gn"
    if change == "arm":
        gn["arm"] = "bn"
    elif change == "initialization":
        gn["initial_parameter_sha256"] = "3" * 64
    values = {"manifest": {"samples": rows}, "bn": bn, "gn": gn}
    monkeypatch.setattr(audit, "read", lambda path, checksum: values[path])
    if change:
        with pytest.raises(ValueError):
            audit.audit("bn", "b" * 64, "gn", "g" * 64, "manifest")
    else:
        report = audit.audit("bn", "b" * 64, "gn", "g" * 64, "manifest")
        assert report["matched_initialization_schedule_optimizer"]
        assert not report["independent_whole_model_oracle"]
        assert report["accuracy_qualification"] == "unavailable"
