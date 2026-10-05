"""Independent publication math and replay command, no detector accuracy claims."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit_script():
    path = Path(__file__).resolve().parents[1] / "scripts/audit-jrm-reference.py"
    spec = importlib.util.spec_from_file_location("jrm_publication_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows():
    return [
        {"lineage": "one", "label": "cover", "method": None, "score": 0.1},
        {"lineage": "one", "label": "stego", "method": "JUNIWARD", "score": 0.5},
        {"lineage": "two", "label": "cover", "method": None, "score": 0.5},
        {"lineage": "two", "label": "stego", "method": "JUNIWARD", "score": 0.9},
    ]


def test_scalar_metrics_ties_and_paired_changes(audit_script):
    values, confusion = audit_script.independent_metrics(rows())
    np.testing.assert_allclose(values, [0.875, 0.75, 1, 0.5, 0.05], rtol=0, atol=1e-12)
    assert confusion == {"tp": 2, "tn": 1, "fp": 1, "fn": 0}
    assert all(pair == [0, 0] for pair in audit_script.paired_deltas(rows(), rows()).values())
    samples = [{**r, "source_group": "one", "quality_factor": 75} for r in rows()]
    assert audit_script.context_ids(samples, "pooled", "all", "JUNIWARD") == [0, 1, 2, 3]
    assert audit_script.context_ids(samples, "quality_factor", "qf-95", "JUNIWARD") == []
    assert audit_script.context_ids(samples, "source", "unknown", "JUNIWARD") == []
    assert audit_script.context_ids(samples, "pooled", "all", "UERD") == [0, 2]


def test_file_identity_and_checks(audit_script, tmp_path):
    with pytest.raises(ValueError, match="regular"):
        audit_script.sha(tmp_path / "missing")
    path = tmp_path / "file"
    path.write_bytes(b"fixture")
    digest = audit_script.sha(path)
    assert len(digest) == 64
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        audit_script.sha(link)
    with pytest.raises(ValueError, match="reason"):
        audit_script.check(False, "reason")


def test_replay_entrypoint(audit_script, tmp_path, monkeypatch):
    out = tmp_path / "record.json"
    args = ["audit"]
    for flag in ("manifest", "corpus", "experiment", "reference-predictions", "reference-record"):
        args.extend(["--" + flag, str(tmp_path)])
    args.extend(["--out", str(out)])
    monkeypatch.setattr(audit_script.sys, "argv", args)
    monkeypatch.setattr(audit_script, "audit", lambda *a: {"passed": True, "deployed": False})
    audit_script.main()
    assert out.is_file()
    with pytest.raises(FileExistsError):
        audit_script.main()
