"""AudioMNIST CNN v4: TTA maps, padded crops, ensemble freeze and a tiny CPU run."""

import json

import numpy as np
import pytest

from research import wav_cnn_audiomnist as cnn
from research import wav_cnn_audiomnist_ensemble as ens
from tests.test_wav_cnn_audiomnist import corpus, tiny, with_digests

TINY_MEMBERS = (
    {"name": "seed-b", "seed": 11, "crop": 256, "batch": 4},
    {"name": "crop-32k", "seed": 12, "crop": 2048, "batch": 2},  # longer than every clip
)


def test_transforms_flip_sign_and_time_without_overflow():
    x = np.array([-32768, 1, 2, 32767], np.int16)
    assert ens.transform(x, "identity").tolist() == [-32768, 1, 2, 32767]
    assert ens.transform(x, "negate").tolist() == [32768, -1, -2, -32767]
    assert ens.transform(x, "reverse").tolist() == [32767, 2, 1, -32768]
    assert ens.transform(x, "negate-reverse").tolist() == [-32767, -2, -1, 32768]
    assert ens.transform(x, "reverse").flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        ens.transform(x, "shift")


def test_crops_pad_short_clips_and_stay_inside():
    torch = pytest.importorskip("torch")
    samples = torch.arange(1, 31, dtype=torch.int16)
    bank = {"samples": samples, "offsets": torch.tensor([0, 10]),
            "lengths": torch.tensor([10, 20])}
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        x, valid = ens.crops(bank, torch.tensor([0, 1]), generator, 16)
        assert valid.tolist() == [10, 16]
        assert x[0, :10].tolist() == list(range(1, 11)) and not x[0, 10:].any()
        assert x[1].min() >= 11 and x[1].max() <= 30
        assert (x[1, 1:] - x[1, :-1] == 1).all()


def run_base(tmp_path, monkeypatch):
    tiny(monkeypatch, 0.0)
    monkeypatch.setattr(ens, "MEMBERS", TINY_MEMBERS)
    monkeypatch.setattr(ens, "EPOCHS", 1)
    archive, layout = corpus(tmp_path)
    layout = with_digests(tmp_path, archive, layout)
    cnn.prepare(archive, tmp_path / "data", layout)
    base = tmp_path / "v3"
    speakers = [r["speaker"] for r in cnn.rows(layout, "calibration")]
    cnn.train(tmp_path / "data", base, "cpu", speakers)
    return layout, base, speakers


def test_tiny_ensemble_trains_freezes_and_tests_once(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    layout, base, speakers = run_base(tmp_path, monkeypatch)
    data, out = tmp_path / "data", tmp_path / "v4"
    for spec in ens.MEMBERS:
        best = ens.train_member(data, base, out, spec, "cpu")
        assert best["member"] == spec["name"] and best["epoch"] == 0
        assert best["init_sha256"] == cnn.sha256((base / "best-0.2.pt").read_bytes())
        assert ens.train_member(data, base, out, spec, "cpu") == best  # finished: cached
    frozen = ens.freeze(data, base, out, speakers, "cpu")
    ten = frozen["choices"]["0.1"]
    assert [r["candidate"] for r in ten["candidates"]] == ["v3", "v3-tta", "ensemble",
                                                           "ensemble-tta"]
    assert ten["selected"] in {"v3", "v3-tta", "ensemble", "ensemble-tta"}
    assert set(ten["model_sha256"]) == set(ten["members"]) and ten["passes_gate"]
    assert [r["candidate"] for r in frozen["choices"]["0.05"]["candidates"]] == ["v3",
                                                                                "v3-tta"]
    assert len(list((out / "scores").glob("*.npy"))) == 2 * 4 * (3 + 1)
    assert ens.freeze(data, base, out, speakers, "cpu") == frozen  # frozen once
    result = ens.evaluate(data, base, out, layout, "cpu")
    assert result["0.1"]["status"] == "evaluated_once" and result["0.1"]["test_pairs"] == 4
    assert result["0.1"]["evidence"] == "second look at the v3 test speakers"
    with pytest.raises(FileExistsError):
        ens.evaluate(data, base, out, layout, "cpu")


def test_member_resumes_from_its_epoch_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _, base, _ = run_base(tmp_path, monkeypatch)
    data, out, spec = tmp_path / "data", tmp_path / "v4", TINY_MEMBERS[0]
    ens.train_member(data, base, out, spec, "cpu")
    (out / spec["name"] / "best.json").unlink()  # interrupted before the last epoch
    monkeypatch.setattr(ens, "EPOCHS", 2)
    best = ens.train_member(data, base, out, spec, "cpu")
    history = json.loads((out / spec["name"] / "history.json").read_text())
    assert [r["epoch"] for r in history] == [0, 1] and best["epoch"] in (0, 1)


def test_gate_failures_skip_and_changed_models_stop_the_test(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    (base / "best-0.1.pt").write_bytes(b"weights")
    choices = {"0.05": {"passes_gate": False},
               "0.1": {"passes_gate": True, "members": ["v3"], "transforms": ["identity"],
                       "model_sha256": {"v3": "0" * 64}, "threshold": 0.0}}
    (out / "choice-freeze.json").write_text(json.dumps({"choices": choices}))
    with pytest.raises(ValueError, match="frozen model changed"):
        ens.evaluate(tmp_path, base, out, {"files": {}, "roles": {}}, "cpu")
    choices["0.1"]["passes_gate"] = False
    (out / "choice-freeze.json").write_text(json.dumps({"choices": choices}))
    result = ens.evaluate(tmp_path, base, out, {"files": {}, "roles": {}}, "cpu")
    assert result == {"0.05": {"status": "not_evaluated_calibration_gate"},
                      "0.1": {"status": "not_evaluated_calibration_gate"}}


def test_cli_dispatch(tmp_path, monkeypatch, capsys):
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"files": {}, "roles": {}}))
    calls = []
    monkeypatch.setattr(ens, "train_member", lambda *a: calls.append(("train", a[3]["name"]))
                        or {})
    monkeypatch.setattr(ens, "freeze", lambda *a: calls.append(("freeze", a[3])) or {})
    monkeypatch.setattr(ens, "evaluate", lambda *a: calls.append(("test",)) or {})
    for command in ("train", "freeze", "test"):
        ens.main([command, "--layout", str(layout), "--data", str(tmp_path),
                  "--base", str(tmp_path), "--out", str(tmp_path), "--device", "cpu"])
    assert calls == [("train", "seed-b"), ("train", "crop-32k"), ("freeze", []), ("test",)]
    capsys.readouterr()
