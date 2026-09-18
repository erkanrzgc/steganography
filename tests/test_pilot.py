import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from steganography.benchmarking import pilot


def test_pilot_generation_integrity_and_exact_decoder_recovery(tmp_path, monkeypatch):
    covers = tmp_path / "covers"
    covers.mkdir()
    originals = []
    for i in range(4):
        path = covers / f"{i}.pgm"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (64, 64), dtype=np.uint8)).save(
            path
        )
        originals.append({"path": path.name, "sha256": pilot.digest(path.read_bytes())})
    (covers / "source.json").write_text(
        json.dumps({"source_group": "fixture", "samples": originals})
    )
    monkeypatch.setattr(pilot.shutil, "which", lambda tool: tool)

    def embedding(_tool, cover, payload, output):
        # A transparent independent fixture adapter; not a real tool recovery claim.
        output.write_bytes(cover.read_bytes() + payload.read_bytes())

    monkeypatch.setattr(pilot, "embed_external", embedding)
    root = tmp_path / "pilot"
    manifest = pilot.generate(covers, root, pairs=4, workers=1)
    assert len(manifest["samples"]) == 8
    assert len(manifest["challenges"]) == 30
    for index in range(0, 8, 2):
        clean, stego = manifest["samples"][index : index + 2]
        assert clean["lineage"] == stego["lineage"]
        assert clean["split"] == stego["split"] == "test"
        assert clean["sha256"] != stego["sha256"]
    # Keep actual extraction tests on standard encodings, independent of stego adapters.
    manifest["challenges"] = manifest["challenges"][2:6]
    (root / "manifest.json").write_text(json.dumps(manifest))
    report = pilot.evaluate(root, tmp_path / "ctf", ctf=True)
    assert report["exact_recovery_rate"] == 1.0
    assert report["blind"] is False and report["support_status"] == "experimental"
    assert report["cross_source_gate"] == "unavailable"
    monkeypatch.setattr(pilot, "ProcessPoolExecutor", ThreadPoolExecutor)
    # Use the real native analysis engine; no optional executable can run here.
    monkeypatch.setattr(pilot.shutil, "which", lambda tool: None)
    scored = pilot.evaluate(root, tmp_path / "detection", workers=1)
    assert scored["metrics"]["positives"] == 4
    assert scored["metrics"]["negatives"] == 4
    assert set(scored["by_method"]) == {"steghide", "openstego"}
    assert "lineage_bootstrap_95_ci" in scored["metrics"]
    original_sample = root / manifest["samples"][0]["path"]
    original_sample.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        pilot.score_sample((str(root), manifest["samples"][0]))
    with pytest.raises(FileExistsError):
        pilot.evaluate(root, tmp_path / "ctf", ctf=True)
    case = root / manifest["challenges"][0]["path"]
    case.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        pilot.evaluate(root, tmp_path / "tampered-ctf", ctf=True)
    with pytest.raises(ValueError, match="workers"):
        pilot.evaluate(root, tmp_path / "workers", workers=0)
    with pytest.raises(ValueError, match="pairs"):
        pilot.generate(covers, tmp_path / "bad", pairs=0)
    with pytest.raises(ValueError, match="insufficient"):
        pilot.generate(covers, tmp_path / "insufficient", pairs=10)
    with pytest.raises(RuntimeError, match="unavailable"):
        pilot.generate(covers, tmp_path / "absent", pairs=1)
    assert pilot.make_challenges(root, [])[0]["status"] == "unavailable"


def test_pilot_bounds_generators_and_missing_cases(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unsafe"):
        pilot.checked_path(tmp_path, "../escape")
    with pytest.raises(ValueError, match="missing"):
        pilot.checked_path(tmp_path, "missing")
    symlink = tmp_path / "link"
    symlink.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        pilot.checked_path(tmp_path, "link")
    out = tmp_path / "stego"
    monkeypatch.setattr(pilot.shutil, "which", lambda tool: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        pilot.embed_external("steghide", out, out, out)
    monkeypatch.setattr(pilot.shutil, "which", lambda tool: tool)
    with pytest.raises(ValueError, match="unsupported"):
        pilot.embed_external("other", out, out, out)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="invalid generator"):
        pilot.embed_external("steghide", out, out, out)

    def write_output(*args, **kwargs):
        assert kwargs["timeout"] == 30
        out.write_bytes(b"fixture")

    monkeypatch.setattr(subprocess, "run", write_output)
    pilot.embed_external("openstego", tmp_path / "cover", tmp_path / "payload", out)
    with pytest.raises(FileExistsError):
        pilot.embed_external("openstego", out, out, out)
    root = tmp_path / "missing-cases"
    root.mkdir()
    pilot.write_json(root / "manifest.json", {"challenges": [{"id": "x", "status": "unavailable"}]})
    report = pilot.evaluate(root, tmp_path / "missing-report", ctf=True)
    assert report["exact_recovery_rate"] == 0
    assert report["cases"][0]["status"] == "unavailable"


def test_pilot_cli_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pilot, "generate", lambda *args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(pilot, "evaluate", lambda *args, **kw: calls.append((args, kw)))
    for action in ("generate", "detect", "ctf"):
        monkeypatch.setattr(
            "sys.argv", ["pilot", action, "--source", str(tmp_path), "--out", str(tmp_path)]
        )
        pilot.main()
    assert len(calls) == 3
    assert calls[-1][1]["ctf"] is True
