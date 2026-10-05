"""Independent scalar transfer audit and exclusive-output CLI contract."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_jrm_reference import numeric_model


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/audit-jrm-transfer.py"
    spec = importlib.util.spec_from_file_location("jrm_transfer_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalar_margin_oracle():
    module = script()
    model = numeric_model()
    features = np.zeros((3, 11255), dtype=np.float32)
    features[1] = 0.05
    features[2] = 0.5
    np.testing.assert_array_equal(module.scalar_votes(model, features), model.predict(features))


def test_publication_checksum_preflight(tmp_path):
    module = script()
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    with pytest.raises(ValueError, match="publication checksum"):
        module.audit(tmp_path / "absent", tmp_path, tmp_path, tmp_path, reference, "0" * 64)


def test_audit_cli_exclusive_output(tmp_path, monkeypatch):
    module = script()
    monkeypatch.setattr(module, "audit", lambda *a: {"passed": True})
    out = tmp_path / "audit.json"
    argv = ["audit"]
    for name in ("manifest", "corpus", "experiment", "reference-cache", "reference-record"):
        argv.extend(["--" + name, str(tmp_path)])
    argv.extend(["--reference-record-sha256", "1" * 64, "--out", str(out)])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert json.loads(out.read_bytes()) == {"passed": True}
    with pytest.raises(FileExistsError):
        module.main()
