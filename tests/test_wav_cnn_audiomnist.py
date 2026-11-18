"""AudioMNIST raw-waveform CNN: data guards, seeds, masking and a tiny CPU run."""

import hashlib
import io
import json
import wave
import zipfile

import numpy as np
import pytest

from research import wav_cnn_audiomnist as cnn

SPEAKERS = {"01": "fit", "02": "fit", "03": "calibration", "04": "test"}


def wav_bytes(samples, channels=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes(np.asarray(samples, "<i2").tobytes())
    return stream.getvalue()


def member(speaker, digit, repetition):
    return f"AudioMNIST-{cnn.COMMIT}/data/{speaker}/{digit}_{speaker}_{repetition}.wav"


def corpus(tmp_path, length=1024):
    """Synthetic archive with four opened speakers and one reserved speaker."""
    rng = np.random.default_rng(0)
    files, archive = {}, tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"AudioMNIST-{cnn.COMMIT}/README.md", "not data")
        for speaker in (*SPEAKERS, "05"):
            for digit in (0, 1):
                for repetition in (0, 1):
                    x = rng.integers(-300, 300, length + 37 * digit).astype(np.int16)
                    raw = wav_bytes(x)
                    bundle.writestr(member(speaker, digit, repetition), raw)
                    if speaker in SPEAKERS:
                        files[f"data/{speaker}/{digit}_{speaker}_{repetition}.wav"] = (
                            hashlib.sha256(raw).hexdigest())
    layout = {"archive_sha256": "0" * 64, "roles": dict(SPEAKERS), "files": files,
              "old_index": {"data/03/0_03_0.wav": 7}}
    return archive, layout


def with_digests(tmp_path, archive, layout):
    """Fill the layout digests from a scratch extraction (plumbing check)."""
    scratch = tmp_path / "scratch"
    cnn.extract(archive, scratch, layout)
    digests = {"covers": {}, "calibration": {}}
    for role in ("fit", "calibration"):
        clips = cnn.rows(layout, role)
        covers = [cnn.read_pcm(scratch / c["path"]) for c in clips]
        digests["covers"][role] = cnn.digest(covers)
    for rate in cnn.RATES:
        fixed = cnn.stegos(covers, clips, rate, layout["old_index"])
        digests["calibration"][str(rate)] = cnn.digest(fixed)
    return {**layout, "digests": digests}


def test_seed_rule_keeps_old_seeds_and_separates_new_cells():
    row = {"path": "data/05/1_05_2.wav", "speaker": "05", "digit": 1, "repetition": 2}
    assert cnn.seed(row, 0.1, {"data/05/1_05_2.wav": 7}) == [2026111201, 1, 5, 7]
    assert cnn.seed(row, 0.05, {}) == [2026111801, 0, 5, 1, 2]
    assert cnn.seed(row, 0.1, {}) == [2026111801, 1, 5, 1, 2]
    assert cnn.seed(row, 0.4, {}) == [2026111901, 0, 5, 1, 2]
    with pytest.raises(ValueError, match="rate"):
        cnn.seed(row, 0.3, {})
    with pytest.raises(ValueError, match="role"):
        cnn.rows({"files": {}, "roles": {}}, "holdout")


def test_extract_never_reads_reserved_members(tmp_path, monkeypatch):
    archive, layout = corpus(tmp_path)
    read = []
    original = zipfile.ZipFile.read

    def recorded(self, info, *args):
        read.append(info.filename)
        return original(self, info, *args)

    monkeypatch.setattr(zipfile.ZipFile, "read", recorded)
    assert cnn.extract(archive, tmp_path / "out", layout) == 16
    assert read and not any("/data/05/" in name for name in read)
    assert not (tmp_path / "out/data/05").exists()


def test_extract_rejects_changed_or_missing_files(tmp_path):
    archive, layout = corpus(tmp_path)
    changed = {**layout, "files": {**layout["files"], "data/01/0_01_0.wav": "f" * 64}}
    with pytest.raises(ValueError, match="hash mismatch"):
        cnn.extract(archive, tmp_path / "a", changed)
    missing = {**layout, "files": {**layout["files"], "data/01/9_01_0.wav": "f" * 64}}
    with pytest.raises(ValueError, match="missing"):
        cnn.extract(archive, tmp_path / "b", missing)


def test_read_pcm_and_bank_roundtrip(tmp_path):
    (tmp_path / "stereo.wav").write_bytes(wav_bytes([1, 2, 3, 4], channels=2))
    with pytest.raises(ValueError, match="format"):
        cnn.read_pcm(tmp_path / "stereo.wav")
    signals = [np.arange(5, dtype=np.int16), np.array([-3, 7], np.int16)]
    cnn.save_bank(tmp_path / "bank.npz", signals)
    restored = cnn.unpack(cnn.load_bank(tmp_path / "bank.npz"))
    assert cnn.digest(restored) == cnn.digest(signals)
    with pytest.raises(FileExistsError):
        cnn.save_bank(tmp_path / "bank.npz", signals)
    with pytest.raises(ValueError, match="digest"):
        cnn.checked(signals, "0" * 64, "fixture")


def test_prepare_banks_roles_once(tmp_path, capsys):
    archive, layout = corpus(tmp_path)
    layout = with_digests(tmp_path, archive, layout)
    summary = cnn.prepare(archive, tmp_path / "data", layout)
    assert summary["opened_files"] == 16 and "archive hash differs" in capsys.readouterr().out
    assert len(cnn.load_bank(tmp_path / "data/covers-fit.npz")["lengths"]) == 8
    assert cnn.prepare(archive, tmp_path / "data", layout) == summary
    bad = {**layout, "digests": {**layout["digests"], "calibration": {
        **layout["digests"]["calibration"], "0.05": "0" * 64}}}
    with pytest.raises(ValueError, match="stegos 0.05"):
        cnn.prepare(archive, tmp_path / "other", bad)


def test_metrics_selection_and_intervals():
    covers, stegos = np.array([0.0, 1.0, 2.0, 3.0]), np.array([2.5, 3.5, 4.0, 1.5])
    m = cnn.calibrate(covers, stegos)
    assert m["balanced_accuracy"] == 0.75 and 0 < m["auc"] <= 1
    record = {"calibration": m}
    assert cnn.better(record, None)
    assert not cnn.better(record, {"calibration": dict(m)})
    spans = cnn.intervals(covers, stegos, m["threshold"], ["a", "a", "b", "b"])
    for low, high in spans.values():
        assert 0 <= low <= high <= 1


def test_network_output_ignores_padding():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = cnn.network().eval()
    rng = np.random.default_rng(1)
    signals = [rng.integers(-500, 500, n).astype(np.float32) for n in (777, 1031)]
    alone = [model(torch.from_numpy(s)[None], torch.tensor([len(s)])) for s in signals]
    batch = np.zeros((2, 1031), np.float32)
    batch[0, :777], batch[1] = signals
    together = model(torch.from_numpy(batch), torch.tensor([777, 1031]))
    assert torch.allclose(together, torch.cat(alone), atol=1e-5)


def test_fresh_embedding_is_plus_minus_one_and_respects_bounds():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(3)
    x = torch.randint(-1000, 1000, (4, 20000), generator=generator).float()
    x[0, :50], x[1, :50] = 32767, -32768
    y = cnn.embed_batch(x, 0.4, generator)
    delta = y - x
    assert set(delta.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert (delta[0, :50] <= 0).all() and (delta[1, :50] >= 0).all()
    assert abs(float((delta != 0).float().mean()) - 0.2) < 0.01


def test_crops_stay_inside_their_clip(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cnn, "CROP", 8)
    signals = [np.arange(10, dtype=np.int16), np.arange(100, 130, dtype=np.int16)]
    bank = {k: torch.from_numpy(v) for k, v in cnn.pack(signals).items()}
    windows = cnn.crops(bank, torch.tensor([0, 1, 1]), torch.Generator().manual_seed(0))
    assert windows.shape == (3, 8)
    assert windows[0].min() >= 0 and windows[0].max() <= 9
    assert windows[1].min() >= 100 and windows[2].max() <= 129
    assert (windows[:, 1:] - windows[:, :-1] == 1).all()


def tiny(monkeypatch, gate):
    monkeypatch.setattr(cnn, "STAGES", tuple((r, 1, 1e-3) for r in cnn.RATES))
    monkeypatch.setattr(cnn, "CROP", 256)
    monkeypatch.setattr(cnn, "BATCH", 4)
    monkeypatch.setattr(cnn, "GATE", gate)


def test_tiny_cpu_curriculum_freezes_then_tests_once(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    tiny(monkeypatch, 0.0)
    archive, layout = corpus(tmp_path)
    layout = with_digests(tmp_path, archive, layout)
    cnn.prepare(archive, tmp_path / "data", layout)
    out = tmp_path / "out"
    frozen = cnn.train(tmp_path / "data", out, "cpu")
    assert set(frozen["choices"]) == {"0.1", "0.05"}
    assert len(json.loads((out / "history.json").read_text())) == 4
    assert cnn.train(tmp_path / "data", out, "cpu") == frozen  # resume is idempotent
    result = cnn.evaluate(tmp_path / "data", out, layout, "cpu")
    assert result["0.1"]["status"] == "evaluated_once" and result["0.1"]["test_pairs"] == 4
    with pytest.raises(FileExistsError):
        cnn.evaluate(tmp_path / "data", out, layout, "cpu")


def test_failed_positive_control_stops_before_low_payloads(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    tiny(monkeypatch, 1.01)
    archive, layout = corpus(tmp_path)
    layout = with_digests(tmp_path, archive, layout)
    cnn.prepare(archive, tmp_path / "data", layout)
    result = cnn.train(tmp_path / "data", tmp_path / "out", "cpu")
    assert result["status"] == "positive_control_failed"
    assert not (tmp_path / "out/best-0.1.pt").exists()
    assert not (tmp_path / "out/choice-freeze.json").exists()


def test_gate_failures_are_not_evaluated(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    choice = {"passes_gate": False}
    (out / "choice-freeze.json").write_text(json.dumps({"choices": {"0.1": choice,
                                                                    "0.05": choice}}))
    result = cnn.evaluate(tmp_path, out, {}, "cpu")
    assert result == {"0.1": {"status": "not_evaluated_calibration_gate"},
                      "0.05": {"status": "not_evaluated_calibration_gate"}}


def test_cli_dispatch(tmp_path, monkeypatch, capsys):
    layout = tmp_path / "layout.json"
    layout.write_text("{}")
    calls = []
    monkeypatch.setattr(cnn, "prepare", lambda *a: calls.append("prepare") or {"ok": 1})
    monkeypatch.setattr(cnn, "train", lambda *a: calls.append("train") or {"ok": 2})
    monkeypatch.setattr(cnn, "evaluate", lambda *a: calls.append("test") or {"ok": 3})
    for command in ("prepare", "train", "test"):
        cnn.main([command, "--layout", str(layout), "--data", str(tmp_path),
                  "--out", str(tmp_path), "--archive", str(layout)])
    assert calls == ["prepare", "train", "test"] and '"ok": 3' in capsys.readouterr().out


def test_interrupted_stage_resumes_with_its_optimizer(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    tiny(monkeypatch, 0.0)
    monkeypatch.setattr(cnn, "STAGES", tuple((r, 2, 1e-3) for r in cnn.RATES))
    archive, layout = corpus(tmp_path)
    layout = with_digests(tmp_path, archive, layout)
    cnn.prepare(archive, tmp_path / "data", layout)
    original, calls = cnn._epoch, []

    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(*args)

    monkeypatch.setattr(cnn, "_epoch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cnn.train(tmp_path / "data", tmp_path / "out", "cpu")
    monkeypatch.setattr(cnn, "_epoch", original)
    frozen = cnn.train(tmp_path / "data", tmp_path / "out", "cpu")
    history = json.loads((tmp_path / "out/history.json").read_text())
    assert [(h["rate"], h["epoch"]) for h in history][:2] == [(0.4, 0), (0.4, 1)]
    assert len(history) == 8 and set(frozen["choices"]) == {"0.1", "0.05"}
