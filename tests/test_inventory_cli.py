"""Exclusive bounded inventory report export and immutable baseline comparison."""

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fault", [None, "mismatch", "oversized", "symlink", "exists", "parent"])
def test_inventory_cli_report_and_baseline_fail_closed(tmp_path, monkeypatch, capsys, fault):
    code = runpy.run_path(str(ROOT / "scripts/full-image-inventory.py"))
    main = code["main"]
    report = {"schema_version": "full-image-inventory-v1", "status": "verified", "entries": []}
    monkeypatch.setitem(main.__globals__, "verify_inventory", lambda *a, **k: report.copy())
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({**report, "entries": [1]} if fault == "mismatch" else report))
    output = tmp_path / "output.json"
    if fault == "oversized":
        baseline.write_bytes(b"x" * 65537)
    if fault == "symlink":
        link = tmp_path / "baseline-link"
        link.symlink_to(baseline)
        baseline = link
    if fault == "exists":
        output.write_text("preserve")
    if fault == "parent":
        link = tmp_path / "output-dir"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "output.json"
    monkeypatch.setattr(sys, "argv", [
        "inventory", "--lock", str(ROOT / "tools/full-image.lock.json"),
        "--baseline", str(baseline), "--out", str(output),
    ])
    if fault in {"oversized", "symlink", "parent", "exists"}:
        with pytest.raises((ValueError, FileExistsError)):
            main()
        if fault == "exists":
            assert output.read_text() == "preserve"
        else:
            assert not output.exists()
    else:
        assert main() == (1 if fault == "mismatch" else 0)
        actual = json.loads(output.read_text())
        assert actual["status"] == ("unavailable" if fault == "mismatch" else "verified")
        assert str(tmp_path) not in json.dumps(actual)
        assert json.loads(capsys.readouterr().out) == actual
