import copy
import io
import json
import os
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from steganography.benchmarking import wav


def recording(index=0):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        # Distinct speech-like fixture, both signed extremes and odd/even samples.
        values = np.arange(1000, dtype=np.int16) * (index + 1) - 1000
        stream.writeframes(values.astype("<i2").tobytes())
    return output.getvalue()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    samples = []
    for i, speaker in enumerate(wav.SPEAKERS):
        data = recording(i)
        name = f"0_{speaker}_0.wav"
        (root / name).write_bytes(data)
        samples.append(
            {
                "path": name,
                "speaker": speaker,
                "sha256": wav.digest(data),
                "lineage": wav.digest(data),
                "label": "cover",
                "size": len(data),
            }
        )
    document = {
        "schema_version": "fsdd-acquisition-v1",
        "samples": samples,
        "license": "CC-BY-SA-4.0",
        "source_url": "https://example.test/fsdd",
        "source_commit": "a" * 40,
    }
    (root / "source.json").write_text(json.dumps(document))
    return root, document


def source_hash(root):
    return wav.digest((root / "source.json").read_bytes())


@pytest.mark.parametrize("method", wav.METHODS)
@pytest.mark.parametrize("rate", wav.RATES)
def test_independent_pairs_preserve_structure_and_exact_bits(method, rate):
    data = recording()
    encoded, evidence = wav.make_pair(data, method=method, rate=rate)
    assert encoded == wav.make_pair(data, method=method, rate=rate)[0]
    offset, frames = wav.pcm_region(data)
    assert len(encoded) == len(data) and data[:offset] == encoded[:offset]
    original = np.frombuffer(data[offset:], dtype="<i2").astype(np.int32)
    actual = np.frombuffer(encoded[offset:], dtype="<i2").astype(np.int32)
    assert np.max(np.abs(actual - original)) <= 1
    assert evidence["extraction_verified"] and evidence["changed_samples"] > 0
    seed = f"{wav.SEED}:{wav.digest(data)}:{method}:{rate}"
    order = wav.positions(frames, evidence["payload_bytes"] * 8, method, seed)
    assert len(order) == len(set(order))
    bits = [(struct.unpack_from("<h", encoded, offset + 2 * i)[0] & 1) for i in order]
    recovered = bytes(sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, len(bits), 8))
    assert wav.digest(recovered) == evidence["payload_sha256"]
    assert np.count_nonzero(actual - original) == evidence["changed_samples"]
    assert evidence["actual_bits_per_sample"] <= rate / 100


def test_generator_rejects_invalid_recipe_and_zero_change(monkeypatch):
    with pytest.raises(ValueError, match="rate"):
        wav.make_pair(recording(), method="sequential", rate=99)
    for method, count in (("bad", 8), ("sequential", 0), ("scattered", 1001)):
        with pytest.raises(ValueError, match="method or length"):
            wav.positions(1000, count, method, "seed")
    with pytest.raises(ValueError):
        wav.make_pair(b"bad", method="sequential", rate=5)
    original = wav.struct.unpack_from
    monkeypatch.setattr(
        wav.struct,
        "unpack_from",
        lambda fmt, *args: (2,) if fmt == "<h" else original(fmt, *args),
    )
    with pytest.raises(ValueError, match="verification"):
        wav.make_pair(recording(), method="sequential", rate=5)


@pytest.mark.parametrize(
    "kind", ["size", "header", "extent", "short", "frame", "chunk", "many", "duplicate"]
)
def test_pcm_adversarial_bounds(kind, monkeypatch):
    data = bytearray(recording())
    if kind == "size":
        monkeypatch.setattr(wav, "MAX_FILE", 10)
    elif kind == "header":
        data[:4] = b"RIFX"
    elif kind == "extent":
        data += b"trailer"
    elif kind == "short":
        data = b"RIFF"
    elif kind == "frame":
        struct.pack_into("<I", data, 24, 44100)
    elif kind == "chunk":
        data += b"JUNK" + struct.pack("<I", 100)
        struct.pack_into("<I", data, 4, len(data) - 8)
    elif kind == "many":
        data += (b"JUNK" + struct.pack("<I", 0)) * 65
        struct.pack_into("<I", data, 4, len(data) - 8)
    else:
        data += b"data" + struct.pack("<I", 0)
        struct.pack_into("<I", data, 4, len(data) - 8)
    with pytest.raises((ValueError, wave.Error)):
        wav.pcm_region(data)


def test_partition_integrity_and_exclusive_outputs(source, tmp_path):
    root, document = source
    out = tmp_path / "generated"
    result = wav.prepare(root, out, source_sha256=source_hash(root), count=2)
    assert len(result["samples"]) == 14 and result["lineages"] == 2
    assert {r["speaker"] for r in result["samples"]} == {"george", "jackson"}
    assert {r["split"] for r in result["samples"]} == {"test"}
    assert len(result["reserved_originals"]) == 6
    assert len({r["sha256"] for r in result["samples"]}) == 14
    assert str(tmp_path) not in json.dumps(result)
    for row in result["samples"]:
        assert wav.digest((out / row["path"]).read_bytes()) == row["sha256"]
    with pytest.raises(FileExistsError):
        wav.prepare(root, out, source_sha256=source_hash(root), count=2)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        wav.prepare(link, tmp_path / "bad", source_sha256=source_hash(root), count=2)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        wav.bounded_read(fifo, 100)


@pytest.mark.parametrize(
    "kind",
    [
        "hash",
        "schema",
        "empty",
        "path",
        "speaker",
        "lineage",
        "file",
        "duplicate",
        "missing",
        "count",
        "insufficient",
        "budget",
        "changing",
    ],
)
def test_preparation_fails_without_success_manifest(source, tmp_path, monkeypatch, kind):
    root, document = source
    count, expected = 2, None
    if kind == "hash":
        expected = "0" * 64
    elif kind == "schema":
        document["schema_version"] = "bad"
    elif kind == "empty":
        document["samples"] = []
    elif kind == "path":
        document["samples"][0]["path"] = "../escape.wav"
    elif kind == "speaker":
        document["samples"][0]["speaker"] = "lucas"
    elif kind == "lineage":
        document["samples"][0]["lineage"] = "0" * 64
    elif kind == "file":
        (root / document["samples"][0]["path"]).write_bytes(b"changed")
    elif kind == "duplicate":
        document["samples"].append(copy.deepcopy(document["samples"][0]))
    elif kind == "missing":
        row = copy.deepcopy(document["samples"][-2])
        row["path"] = "1_theo_0.wav"
        data = recording(10)
        (root / row["path"]).write_bytes(data)
        row.update(sha256=wav.digest(data), lineage=wav.digest(data), size=len(data))
        document["samples"][-1] = row
    elif kind == "count":
        count = 0
    elif kind == "insufficient":
        count = 3
    elif kind == "budget":
        monkeypatch.setattr(wav, "MAX_OUTPUT", 1)
    else:
        original = wav.bounded_read
        reads = 0

        def changing(path, maximum):
            nonlocal reads
            data = original(path, maximum)
            if path.name == "0_george_0.wav":
                reads += 1
                if reads == 2:
                    return b"changed"
            return data

        monkeypatch.setattr(wav, "bounded_read", changing)
    (root / "source.json").write_text(json.dumps(document))
    with pytest.raises(ValueError):
        wav.prepare(
            root, tmp_path / "out", source_sha256=expected or source_hash(root), count=count
        )
    assert not (tmp_path / "out/manifest.json").exists()


def test_shared_service_real_path_and_error_redaction(source, tmp_path, monkeypatch):
    root, _ = source
    out = tmp_path / "generated"
    manifest = wav.prepare(root, out, source_sha256=source_hash(root), count=2)
    row = wav.score_sample((str(out), manifest["samples"][0]))
    assert row["status"] == "completed"
    assert all(row["coverage"].get(k) == "ok" for k in wav.REQUIRED)
    bad = manifest["samples"][0].copy()
    bad["path"] = "../../secret.wav"
    assert wav.score_sample((str(out), bad))["failure"] == "ValueError"
    bad = manifest["samples"][0].copy()
    bad["sha256"] = "0" * 64
    assert wav.score_sample((str(out), bad))["status"] == "failed"

    def service(**options):
        assert options["ai_provider"] is None

        def fail(path):
            raise RuntimeError(f"secret at {path}")

        return SimpleNamespace(analyze=fail)

    monkeypatch.setattr(wav, "AnalysisService", service)
    failure = wav.score_sample((str(out), manifest["samples"][0]))
    assert failure["failure"] == "RuntimeError" and str(tmp_path) not in json.dumps(failure)


def test_metrics_pairing_coverage_and_no_search(monkeypatch):
    from steganography.benchmarking import metrics

    def forbidden(*_):
        pytest.fail("no held-out threshold search")

    monkeypatch.setattr(metrics, "_recommended_threshold", forbidden)
    rows = [
        {
            "lineage": str(i),
            "label": label,
            "score": score,
            "coverage": dict.fromkeys(wav.REQUIRED, "ok"),
            "status": "completed",
        }
        for i in range(2)
        for label, score in [("cover", 0), ("stego", 90)]
    ]
    result = wav.summarize(rows)
    assert result["metrics"]["roc_auc"] == 1 and result["metrics"]["recall"] == 1
    assert "recommended_threshold" not in result["metrics"]
    assert all(
        v == [1.0, 1.0]
        for k, v in result["lineage_bootstrap_95_ci"].items()
        if k != "false_positive_rate"
    )
    assert wav.summarize([])["status"] == "unavailable"
    for status in ("error", "unavailable", "unsupported"):
        altered = copy.deepcopy(rows)
        altered[0]["coverage"]["audio_wav"] = status
        assert wav.summarize(altered)["metrics"] is None


def test_evaluation_and_cli(source, tmp_path, monkeypatch, capsys):
    root, _ = source
    protocol = tmp_path / "protocol.md"
    protocol.write_text("fixture protocol")
    monkeypatch.setattr(wav, "ProcessPoolExecutor", ThreadPoolExecutor)
    out = tmp_path / "run"
    wav.main(
        [
            "--source",
            str(root),
            "--source-sha256",
            source_hash(root),
            "--out",
            str(out),
            "--protocol",
            str(protocol),
            "--count",
            "2",
        ]
    )
    report = json.loads((out / "report.json").read_text())
    assert (
        report["status"] == "completed"
        and report["samples"] == 14
        and len(report["by_method_rate"]) == 6
    )
    assert report["scores_sha256"] == wav.digest((out / "scores.jsonl").read_bytes())
    assert report["cross_source_gate"] == "unavailable" and not report["training_performed"]
    assert str(tmp_path) not in (out / "report.json").read_text()
    with pytest.raises(ValueError, match="workers"):
        wav.evaluate(
            root, tmp_path / "bad", source_sha256=source_hash(root), protocol=protocol, workers=5
        )
    with pytest.raises(SystemExit) as error:
        wav.main(
            [
                "--source",
                str(root),
                "--source-sha256",
                "bad",
                "--out",
                str(tmp_path / "bad"),
                "--protocol",
                str(protocol),
            ]
        )
    assert error.value.code == 2 and "details" not in capsys.readouterr().err
    monkeypatch.setattr(wav, "evaluate", lambda *a, **k: {"status": "unavailable", "samples": 14})
    with pytest.raises(SystemExit) as error:
        wav.main(
            [
                "--source",
                str(root),
                "--source-sha256",
                "bad",
                "--out",
                str(tmp_path / "bad"),
                "--protocol",
                str(protocol),
            ]
        )
    assert error.value.code == 1
