import builtins
import hashlib
import io
import json
import runpy
import wave
from pathlib import Path

import numpy as np
import pytest

from core import wav_pilot, wav_pilot_training


def wav_bytes(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def test_generated_pcm_payload_oracle_and_update_schedule():
    assert wav_pilot.generated_row_smoke() == {
        "status": "ready",
        "recipes_checked": 8,
        "optimizer_updates": 0,
        "source_media_opened": False,
    }
    assert wav_pilot_training.generated_schedule_smoke() == {
        "status": "ready",
        "rows_per_epoch": 3600,
        "updates_per_epoch": 57,
        "rows_last_batch": 16,
        "updates_two_seeds": 2280,
        "optimizer_updates": 0,
    }
    assert wav_pilot_training.balanced_row_weight(0) == 4.5
    assert wav_pilot_training.balanced_row_weight(1) == 0.5625
    assert 400 * 4.5 == 3200 * 0.5625


@pytest.mark.parametrize("recipe", wav_pilot.RECIPES)
def test_materialize_is_byte_preserving_except_requested_sample_bits(tmp_path, recipe):
    raw = wav_bytes(np.resize(np.array([-32768, -1, 0, 1, 32767], dtype=np.int16), 160))
    source = tmp_path / "source.wav"
    source.write_bytes(raw)
    row = {"relative_source_path": "source.wav", "sha256": hashlib.sha256(raw).hexdigest()}
    changed = wav_pilot.materialize(tmp_path, row, recipe)
    offset = wav_pilot._wav_data_offset(raw)
    before = np.frombuffer(raw[offset:], dtype="<u2")
    after = np.frombuffer(changed[offset:], dtype="<u2")
    method, _rate, plane = recipe
    diff = np.bitwise_xor(before, after)
    assert len(changed) == len(raw) and changed[:offset] == raw[:offset]
    assert np.all(diff <= (1 << plane)) and np.any(diff)
    if method == "sequential":
        assert np.all(diff[after == before] == 0)


def test_materialize_rejects_changed_source_and_symlink(tmp_path):
    raw = wav_bytes(np.arange(160, dtype=np.int16))
    real = tmp_path / "real.wav"
    real.write_bytes(raw)
    row = {"relative_source_path": "real.wav", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum"):
        wav_pilot.materialize(tmp_path, row, None)
    link = tmp_path / "link.wav"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        wav_pilot.materialize(
            tmp_path,
            {
                "relative_source_path": "link.wav",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            None,
        )


def test_schedule_rejects_invalid_bounds():
    for rows, batch in ((0, 64), (-1, 2), (3, 0), (True, 2)):
        with pytest.raises(ValueError):
            wav_pilot_training.epoch_schedule(rows, batch)


def test_source_acquisition_uses_only_frozen_train_and_development_speakers(tmp_path):
    acquisition = runpy.run_path("steganography/research_wav_prepare.py")
    read_manifest = acquisition["_read_manifest"]
    draft_path = Path("benchmarks/wav-next-experiment-draft-20261016.json")
    draft = read_manifest(draft_path)
    assert len(draft["proposed_files"]) == 600
    assert {row["role"] for row in draft["proposed_files"]} == {"train", "development"}
    modified = json.loads(draft_path.read_text())
    modified["proposed_files"][0]["role"] = "reserved"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="draft hash"):
        read_manifest(tampered)


def test_torch_absence_is_explicitly_unavailable_before_any_fit(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("optional torch unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="optional Torch missing"):
        wav_pilot_training.fit_seed(
            tmp_path, [{}] * 3600, 20261016, device="cuda", deadline_seconds=1
        )
    with pytest.raises(RuntimeError, match="optional Torch missing"):
        wav_pilot_training.generated_cnn_preflight([160] * 400, [160] * 200)


def test_development_metrics_pair_by_file_and_cluster_by_speaker():
    rows = []
    scores = []
    for index in range(200):
        lineage = f"source-{index}"
        group = f"speaker-{index % 10}"
        source = {"role": "development"}
        rows.append(
            {"source": source, "lineage": lineage, "group": group, "label": "cover", "recipe": None}
        )
        scores.append(0.1)
        for recipe in wav_pilot.RECIPES:
            rows.append(
                {
                    "source": source,
                    "lineage": lineage,
                    "group": group,
                    "label": "stego",
                    "recipe": recipe,
                }
            )
            scores.append(0.9)
    result = wav_pilot_training.development_metrics(rows, np.asarray(scores))
    assert len(result) == 8
    assert all(cell["independent_groups"] == 10 for cell in result)
    assert all(cell["metrics"]["roc_auc"] == 1.0 for cell in result)
    assert all(cell["metrics"]["balanced_accuracy"] == 1.0 for cell in result)
