"""Independent bounded pixel-audit oracle and failure/exclusive-output contracts."""

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.jpeg_pixels import decode
from tests.test_pixel_research import jpeg


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/audit-pixel-preparation.py"
    spec = importlib.util.spec_from_file_location("pixel_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_independent_oracle_and_input_bounds():
    module = script()
    data = jpeg()
    np.testing.assert_array_equal(
        module.independent_crop(data).ravel(), np.frombuffer(decode(data), dtype="u1")
    )
    with pytest.raises(ValueError, match="input"):
        module.independent_crop(b"x" * (module.MAX_IMAGE_BYTES + 1))


@pytest.mark.parametrize("fault", ["failed", "short", "excess", "timeout"])
def test_oracle_process_failures(monkeypatch, fault):
    module = script()

    def run(*a, stdout, **k):
        if fault == "timeout":
            raise subprocess.TimeoutExpired("fixed", 15)
        raw = bytes(16384)
        stdout.write(raw[:-1] if fault == "short" else raw + b"x" if fault == "excess" else raw)
        return SimpleNamespace(returncode=1 if fault == "failed" else 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        module.independent_crop(jpeg())


def test_native_oracle_entrypoint(monkeypatch):
    import resource

    module = script()
    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["audit", "--oracle"])
    for data, code in ((jpeg(), 0), (b"invalid", 1), (jpeg(width=127), 1)):
        output = io.BytesIO()
        monkeypatch.setattr(module.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
        monkeypatch.setattr(module.sys, "stdout", SimpleNamespace(buffer=output))
        assert module.main() == code
        assert len(output.getvalue()) == (16384 if code == 0 else 0)


def test_audit_binding_and_cli(tmp_path, monkeypatch):
    module = script()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        module.audit(manifest, tmp_path, tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(manifest)
    with pytest.raises(ValueError, match="symlink"):
        module.sha(link)
    monkeypatch.setattr(module, "audit", lambda *a: {"trained": False, "accuracy": "unavailable"})
    out = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--manifest",
            str(manifest),
            "--corpus",
            str(tmp_path),
            "--experiment",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert module.main() == 0 and not json.loads(out.read_bytes())["trained"]
    with pytest.raises(FileExistsError):
        module.main()
