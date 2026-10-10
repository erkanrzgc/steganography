import hashlib
import io
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from steganography import research_jpeg_corpus as corpus
from steganography import research_jpeg_timing as cli


def test_cli_success_and_redacted_failure(tmp_path, monkeypatch, capsys):
    options = []
    for name in ("root", "bows", "out", "protocol", "audit", "bows-audit"):
        options.extend(["--" + name, str(tmp_path / name)])
    for name in (
        "protocol-sha256",
        "index-sha256",
        "audit-sha256",
        "bows-sha256",
        "bows-audit-sha256",
    ):
        options.extend(["--" + name, "a" * 64])
    called = []
    monkeypatch.setattr(
        cli.jpeg_timing, "prepare", lambda **kw: called.append(kw) or {"status": "completed"}
    )
    assert cli.main(options) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}
    assert callable(called[0]["deadline"]) and called[0]["generator"] is cli.generate

    def failed(**kw):
        raise ValueError(str(tmp_path / "secret"))

    monkeypatch.setattr(cli.jpeg_timing, "prepare", failed)
    assert cli.main(options) == 1
    raw = capsys.readouterr().out
    assert str(tmp_path) not in raw and json.loads(raw)["reason"] == "ValueError"


@pytest.mark.parametrize("fault", ["success", "version", "exit", "bytes", "json", "timeout"])
def test_worker_parent_command_and_limits(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(
        cli.importlib.metadata, "version", lambda name: "wrong" if fault == "version" else "2025.11"
    )

    def run(command, **kw):
        assert command[1:4] == ["-m", "steganography.research_jpeg_timing", "worker"]
        assert kw["timeout"] == 90 and kw["stderr"] == subprocess.DEVNULL
        assert kw["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kw["input"] == b"pgm" and "shell" not in kw
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 90)
        raw = b"[]" if fault != "json" else b"not JSON"
        if fault == "bytes":
            raw = b" " * (256 * 1024 + 1)
        kw["stdout"].write(raw)
        return SimpleNamespace(returncode=1 if fault == "exit" else 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    if fault == "success":
        assert cli.generate(b"pgm", tmp_path / "out", "a" * 64) == []
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)):
            cli.generate(b"pgm", tmp_path / "out", "a" * 64)


def test_child_worker_bows_identity_and_failure(tmp_path, monkeypatch, capsys):
    import resource

    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda key, values: calls.append((key, values)))
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"pgm")))
    seen = []
    monkeypatch.setattr(corpus, "generate_lineage", lambda *a, **kw: seen.append((a, kw)) or [])
    assert cli.main(["worker", str(tmp_path / "out"), "a" * 64]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert seen[0][1] == {"source_group": "BOWS2"}
    assert seen[0][0][3] == "train"
    assert dict(calls)[resource.RLIMIT_AS] == (2 * 1024**3,) * 2
    assert cli.main(["worker"]) == 1

    def fail(*a, **kw):
        raise ValueError("no original")

    monkeypatch.setattr(corpus, "generate_lineage", fail)
    assert cli.main(["worker", str(tmp_path / "out"), "a" * 64]) == 1


def test_real_bows_opt_in_preserves_default_and_validates_source(tmp_path):
    pytest.importorskip("conseal")
    pytest.importorskip("jpeglib")
    data = io.BytesIO()
    pixels = np.random.default_rng(7).integers(0, 256, (512, 512), dtype=np.uint8)
    Image.fromarray(pixels).save(data, format="PPM")
    raw = data.getvalue()
    lineage = hashlib.sha256(raw).hexdigest()
    generated = corpus.generate_lineage(
        raw, tmp_path / "bows", lineage, "train", source_group="BOWS2"
    )
    assert len(generated) == 6 and {r["source_group"] for r in generated} == {"BOWS2"}
    assert all(r["coefficient_changes"] > 0 for r in generated if r["label"] == "stego")
    with pytest.raises(ValueError, match="unsupported"):
        corpus.generate_lineage(raw, tmp_path / "bad", lineage, "train", source_group="ALASKA2")
