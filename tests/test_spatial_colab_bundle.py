"""Portable source/notebook syntax and verified storage-package contracts."""

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from core import colab_artifacts as a
from core import spatial_colab as p
from core import spatial_colab_bundle as b
from tests.test_spatial_colab_data import source

ROOT = Path(__file__).parent.parent


def test_notebook_handoff_is_reproducible_and_source_bound(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    result = b.build(ROOT, first)
    assert b.build(ROOT, second) == result
    assert (first / "sources.json").read_bytes() == (second / "sources.json").read_bytes()
    package = a.document(first / "sources.json", 4 * a.MIB)
    notebook = a.document(first / "png_bmp_capacity.ipynb")
    assert notebook["nbformat"] == 4
    text = (first / "png_bmp_capacity.ipynb").read_text()
    assert result["source_package_sha256"] in text and p.BASE_COMMIT in text
    assert not any(cell.get("outputs") for cell in notebook["cells"])
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            body = "".join(cell["source"])
            compile(body, "colab-cell", "exec")
            code.append(body)
    assert "RUN_MEASUREMENT = False" in "\n".join(code)
    assert "requirements-research-ci.txt" not in "\n".join(code)
    for name, text in package["files"].items():
        assert a.digest(text.encode()) == package["protocol"]["source_sha256"][name]
    p.verify(ROOT, first / "protocol.json")
    ruff = Path(sys.executable).with_name("ruff")
    if not ruff.is_file():
        pytest.skip("notebook lint unavailable: optional development Ruff missing")
    checked = subprocess.run(  # noqa: S603 — provisioned checker and generated test artifact.
        [
            str(ruff),
            "check",
            "--config",
            str(ROOT / "ruff.toml"),
            str(first / "png_bmp_capacity.ipynb"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout


def test_receipt_export_and_tampering(tmp_path):
    folder = tmp_path / "receipts"
    folder.mkdir()
    a.put(folder / "measurement.json", a.encode({"status": "unavailable"}))
    a.put(folder / "worker.json", a.encode({"status": "unavailable"}))
    a.put(folder / "cell-00.json", b"{}")
    (folder / "worker.log").write_text("Private diagnostic log must not be exported")
    target = tmp_path / "package.json"
    assert b.export_receipts(folder, target) == a.digest(target.read_bytes())
    package = a.document(target)
    b.verify_receipts(package)
    assert "worker.log" not in package["files"]
    for fault in ("schema", "members", "hash"):
        bad = copy.deepcopy(package)
        if fault == "schema":
            bad["schema"] = "old"
        elif fault == "members":
            bad["files"]["../../escape"] = {}
        else:
            bad["files"]["measurement.json"]["text"] += "x"
        with pytest.raises(ValueError):
            b.verify_receipts(bad)
    with pytest.raises(ValueError, match="required"):
        b.export_receipts(tmp_path, tmp_path / "empty.json")


def test_adapter_all_phases_are_thin_and_explicit(tmp_path, monkeypatch, capsys):
    cli = runpy.run_path(str(ROOT / "scripts/spatial-colab.py"))
    main = cli["main"]
    monkeypatch.setattr(cli["resource"], "setrlimit", lambda *args: None)
    protocol = tmp_path / "protocol.json"
    assert main(["freeze", "--protocol", str(protocol)]) == 0
    assert main(["verify", "--protocol", str(protocol)]) == 0
    assert main(["verify"]) == 2
    assert main(["profile", "--protocol", str(protocol)]) == 2
    assert main(["validate-data"]) == 2
    manifest = tmp_path / "manifest.json"
    a.put(manifest, a.encode({"schema": "spatial-commons-v1", "sources": [source(b"test")]}))
    assert main(["validate-data", "--manifest", str(manifest)]) == 0
    assert main(["prepare", "--manifest", str(manifest), "--out", str(tmp_path / "out")]) == 2
    assert main(["acquire", "--manifest", str(manifest), "--out", str(tmp_path / "out")]) == 2
    monkeypatch.setattr(cli["data"], "prepare", lambda *a, **k: {"status": "ready"})
    monkeypatch.setattr(cli["data"], "acquire", lambda *a, **k: {"status": "ready"})
    monkeypatch.setattr(cli["service"], "supervise", lambda *a: {"status": "unavailable"})
    monkeypatch.setattr(cli["state"], "generated_probe", lambda *a: {"status": "ready"})
    ledger = tmp_path / "ledger.json"
    a.put(ledger, a.encode({"historical_bytes": None}))
    assert (
        main(
            [
                "acquire",
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "out"),
                "--allow-download",
                "--prior-ledger",
                str(ledger),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "prepare",
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "out"),
                "--originals",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert main(["profile", "--protocol", str(protocol), "--out", str(tmp_path / "out")]) == 2
    assert main(["state-probe", "--out", str(tmp_path / "state")]) == 0
    monkeypatch.setattr(cli["bundle"], "build", lambda *a: {"status": "ready"})
    assert main(["bundle", "--out", str(tmp_path / "bundle")]) == 0
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    monkeypatch.setattr(cli["service"], "worker", lambda *a: {"status": "ready"})
    assert main(["_worker", "--protocol", str(protocol), "--out", str(worker_dir)]) == 0
    (worker_dir / "worker.json").unlink()

    def fail(*args):
        raise RuntimeError("generated OOM; suppress internal paths")

    monkeypatch.setattr(cli["service"], "worker", fail)
    assert main(["_worker", "--protocol", str(protocol), "--out", str(worker_dir)]) == 2
    assert "generated OOM" not in capsys.readouterr().out


def test_metadata_candidates_do_not_authorize_acquisition():
    from core import spatial_colab_data as d

    value = json.loads((ROOT / "benchmarks/png-bmp-commons-candidates-20261023.json").read_text())
    assert len(d.manifest(value)) == 2
    assert all(
        row["role"] == "unassigned" and row["ancestry_review"] == "pending"
        for row in value["sources"]
    )
