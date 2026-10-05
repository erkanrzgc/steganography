"""Bounded reference math, cache isolation and optional upstream smoke, not accuracy."""

import hashlib
import io
import json
import subprocess
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core import jpeg_jrm as jrm
from core.fld_reference import LEARNERS, SUBSPACE, FLDReference
from steganography import research_jrm as rj
from tests.test_research_features import setup_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    _, manifest, path = setup_features(tmp_path)
    rows = []
    for split in ("train", "validation"):
        for source in ("one", "two"):
            lineage = f"{split}-{source}"
            for method in (None, "JUNIWARD", "UERD"):
                file = tmp_path / f"jrm-{len(rows)}.jpg"
                Image.new("RGB", (64, 64), (len(rows) * 15, 10, 90)).save(file)
                rows.append(
                    {
                        **manifest["samples"][0],
                        "path": file.name,
                        "sha256": sha(file),
                        "size": file.stat().st_size,
                        "split": split,
                        "source_group": source,
                        "lineage": lineage,
                        "format": "JPEG",
                        "label": "cover" if method is None else "stego",
                        "method": method,
                        "quality_factor": 75,
                    }
                )
    manifest["samples"] = rows
    manifest["catalog"] = {
        "source_records": {
            s: {
                "origin_manifest_sha256": str(i) * 64,
                "license": "fixture",
                "source_url": "https://example.org/fixture",
            }
            for i, s in enumerate(("one", "two"), 1)
        }
    }
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(jrm, "require_version", lambda: None)
    monkeypatch.setattr(
        jrm,
        "jpeg_jrm_batch",
        lambda files: np.array(
            [
                np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:8]))
                .random(jrm.DIMENSIONS)
                .astype("<f4")
                for data in files
            ]
        ),
    )
    return path, tmp_path


def test_cache_roundtrip_limits_and_exclusive_output(corpus, tmp_path):
    manifest, source = corpus
    out = tmp_path / "cache"
    result = rj.extract_cache(manifest, out, source=source, split="train", workers=2)
    x, rows, _ = rj.load_cache(
        manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train"
    )
    assert x.shape == (6, jrm.DIMENSIONS) and result["rows"] == [
        {k: s[k] for k in ("sha256", "lineage", "label")} for s in rows
    ]
    with pytest.raises(FileExistsError):
        rj.extract_cache(manifest, out, source=source, split="train")
    with pytest.raises(ValueError, match="contract"):
        rj.load_cache(manifest, out / "cache.json", checksum="0" * 64, split="train")
    with pytest.raises(ValueError, match="contract"):
        rj.load_cache(
            manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="validation"
        )
    data = out / "features.f32"
    raw = data.read_bytes()
    data.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="byte limits"):
        rj.load_cache(manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train")
    data.write_bytes(bytes(len(raw)))
    with pytest.raises(ValueError, match="checksum"):
        rj.load_cache(manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train")
    data.unlink()
    data.symlink_to(source / "jrm-0.jpg")
    with pytest.raises(ValueError, match="symlink"):
        rj.load_cache(manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train")


@pytest.mark.parametrize("fault", ["missing", "symlink", "integrity", "workers", "rows", "values"])
def test_failed_cache_never_becomes_complete(corpus, tmp_path, monkeypatch, fault):
    manifest, source = corpus
    path = source / "jrm-0.jpg"
    if fault == "missing":
        path.unlink()
    if fault == "symlink":
        path.unlink()
        path.symlink_to(source / "jrm-1.jpg")
    if fault == "integrity":
        path.write_bytes(b"changed")
    if fault == "rows":
        monkeypatch.setattr(rj, "selected_samples", lambda *a: [{}] * 4001)
    if fault == "values":
        monkeypatch.setattr(
            jrm, "jpeg_jrm_batch", lambda *a: (_ for _ in ()).throw(ValueError("invalid"))
        )
    with pytest.raises(ValueError):
        rj.extract_cache(
            manifest,
            tmp_path / "failed",
            source=source,
            split="train",
            workers=0 if fault == "workers" else 1,
        )
    assert not (tmp_path / "failed/cache.json").exists()


def numeric_model():
    spaces = np.tile(np.arange(SUBSPACE), (LEARNERS, 1))
    weights = np.ones((LEARNERS, SUBSPACE))
    biases = np.arange(LEARNERS, dtype=float)
    return FLDReference(spaces, weights, biases)


def test_independent_scalar_votes_and_immutability():
    model = numeric_model()
    x = np.zeros((3, jrm.DIMENSIONS), dtype=np.float32)
    x[1, :] = 0.05
    x[2, :] = 0.5
    oracle = np.array(
        [
            (
                sum(
                    np.sign(
                        sum(float(row[i]) * float(w) for i, w in zip(s, weights, strict=True))
                        - bias
                    )
                    for s, weights, bias in zip(
                        model.subspaces, model.weights, model.biases, strict=True
                    )
                )
                / LEARNERS
                + 1
            )
            / 2
            for row in x
        ]
    )
    np.testing.assert_array_equal(model.predict(x), oracle)
    with pytest.raises(ValueError):
        model.weights[0, 0] = 1


@pytest.mark.parametrize(
    "fault", ["shape", "index", "duplicate", "nan", "bias", "input", "nonfinite", "overflow"]
)
def test_numeric_bounds(fault):
    m = numeric_model()
    spaces = m.subspaces.copy()
    w = m.weights.copy()
    b = m.biases.copy()
    if fault == "shape":
        spaces = spaces[:1]
    if fault == "index":
        spaces[0, 0] = jrm.DIMENSIONS
    if fault == "duplicate":
        spaces[0, 0] = spaces[0, 1]
    if fault == "nan":
        w[0, 0] = float("nan")
    if fault == "bias":
        b = b[:1]
    if fault in {"input", "nonfinite", "overflow"}:
        if fault == "overflow":
            w[:] = 1e308
        model = FLDReference(spaces, w, b)
        x = np.ones((1, jrm.DIMENSIONS), dtype=np.float32)
        if fault == "input":
            x = x[:, :-1]
        if fault == "nonfinite":
            x[0, 0] = float("nan")
        with pytest.raises(ValueError):
            model.predict(x)
    else:
        with pytest.raises(ValueError):
            FLDReference(spaces, w, b)


def jpeg():
    stream = io.BytesIO()
    Image.fromarray(np.random.default_rng(6).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        stream, format="JPEG"
    )
    return stream.getvalue()


def test_upstream_real_worker_and_layout(monkeypatch):
    sw = pytest.importorskip("sealwatch")
    import jpeglib

    from core.jpeg_features import worker

    data = jpeg()
    expected = np.array(worker(data, jrm.coefficient_features), dtype="<f4")
    np.testing.assert_array_equal(jrm.jpeg_jrm_features(data), expected)
    y = np.random.default_rng(7).integers(-4, 5, (8, 8, 8, 8), dtype=np.int16)
    independent = np.concatenate(
        [v.flatten() for v in sw.jrm.extract(y.astype(np.int32), calibrated=False).values()]
    ).astype("<f4")
    np.testing.assert_array_equal(jrm.coefficient_features(y, np.ones((8, 8))), independent)
    with pytest.raises(ValueError, match="eight blocks"):
        jrm.coefficient_features(y[:2, :2], np.ones((8, 8)))
    monkeypatch.setattr(jrm, "LAYOUT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="layout"):
        jrm.coefficient_features(y, np.ones((8, 8)))
    assert jpeglib is not None


@pytest.mark.parametrize("fault", ["failed", "short", "excess", "nan", "timeout"])
def test_worker_failure_limits(monkeypatch, fault):
    monkeypatch.setattr(jrm, "require_version", lambda: None)

    def run(*args, stdout, **kwargs):
        if fault == "timeout":
            raise subprocess.TimeoutExpired("fixed", 15)
        values = np.zeros(jrm.DIMENSIONS, dtype="<f4")
        if fault == "nan":
            values[0] = np.nan
        raw = values.tobytes()
        stdout.write(raw[:-4] if fault == "short" else raw + b"x" if fault == "excess" else raw)
        return SimpleNamespace(returncode=1 if fault == "failed" else 0)

    monkeypatch.setattr(jrm.subprocess, "run", run)
    with pytest.raises((ValueError, RuntimeError)):
        jrm.jpeg_jrm_features(jpeg())


def test_dependency_version_absence(monkeypatch):
    monkeypatch.setattr(jrm.importlib.metadata, "version", lambda *a: "wrong")
    with pytest.raises(RuntimeError, match="pinned"):
        jrm.require_version()

    def missing(*a):
        raise jrm.importlib.metadata.PackageNotFoundError()

    monkeypatch.setattr(jrm.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="optional"):
        jrm.require_version()


def test_model_archive_and_checksum_guards(tmp_path, monkeypatch):
    model = numeric_model()
    path = tmp_path / "model.npz"
    np.savez(path, subspaces=model.subspaces, weights=model.weights, biases=model.biases)
    loaded = rj.load_model(path, checksum=sha(path))
    assert np.array_equal(loaded.weights, model.weights)
    with pytest.raises(ValueError, match="checksum"):
        rj.load_model(path, checksum="0" * 64)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("unexpected", b"x")
    with pytest.raises(ValueError, match="archive"):
        rj.load_model(path, checksum=sha(path))
    monkeypatch.setattr(rj, "MAX_MODEL_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        rj.load_model(path, checksum=sha(path))


def test_pairing_and_stage_validation(tmp_path):
    with pytest.raises(ValueError, match="pairs"):
        rj.paired_indices([{"lineage": "one", "label": "cover"}])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"stage": "unknown"}))
    with pytest.raises(ValueError, match="stage"):
        rj.run_reference(config, tmp_path / "out")


def test_real_fld_training_prediction_and_cli(corpus, tmp_path, monkeypatch):
    pytest.importorskip("sealwatch")
    import cli

    manifest, source = corpus
    configs = {}
    for split in ("train", "validation"):
        config = tmp_path / f"{split}.json"
        config.write_text(
            json.dumps(
                {
                    "stage": "features",
                    "manifest": str(manifest),
                    "source": str(source),
                    "split": split,
                }
            )
        )
        assert rj.run_reference(config, tmp_path / split)["rows"] == 6
        configs[split] = tmp_path / split / "cache.json"
    config = tmp_path / "train-config.json"
    config.write_text(
        json.dumps(
            {
                "stage": "train",
                "manifest": str(manifest),
                "cache": str(configs["train"]),
                "cache_sha256": sha(configs["train"]),
            }
        )
    )
    card = rj.run_reference(config, tmp_path / "model")
    assert card["paired_training_rows"] == 4 and card["upstream_training_vote_parity"]
    assert card["source_validation"]["declared_sources"] == 2
    assert not card["deployed"] and not card["calibrated"]
    with pytest.raises(FileExistsError):
        rj.run_reference(config, tmp_path / "model")
    config.write_text(
        json.dumps(
            {
                "stage": "predict",
                "manifest": str(manifest),
                "cache": str(configs["validation"]),
                "cache_sha256": sha(configs["validation"]),
                "model_dir": str(tmp_path / "model"),
                "card_sha256": sha(tmp_path / "model/model-card.json"),
            }
        )
    )
    report = rj.run_reference(config, tmp_path / "predictions.json")
    assert len(report["predictions"]) == 6 and not report["calibrated"]
    assert all(0 <= row["score"] <= 1 for row in report["predictions"])
    with pytest.raises(FileExistsError):
        rj.run_reference(config, tmp_path / "predictions.json")
    broken = json.loads(config.read_bytes())
    broken["card_sha256"] = "0" * 64
    config.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="contract"):
        rj.run_reference(config, tmp_path / "bad.json")
    monkeypatch.setattr(rj, "run_reference", lambda *a: {"deployed": False})
    assert (
        cli.main(
            [
                "research",
                "jrm-reference",
                "--config",
                str(config),
                "--out",
                str(tmp_path / "cli.json"),
            ]
        )
        == 0
    )


def test_training_refuses_numeric_vote_mismatch(corpus, tmp_path, monkeypatch):
    pytest.importorskip("sealwatch")
    manifest, source = corpus
    cache = tmp_path / "cache"
    rj.extract_cache(manifest, cache, source=source, split="train")
    monkeypatch.setattr(FLDReference, "predict", lambda *a: np.full(6, -1))
    with pytest.raises(ValueError, match="vote mismatch"):
        rj.train_reference(
            manifest,
            cache / "cache.json",
            tmp_path / "model",
            cache_sha256=sha(cache / "cache.json"),
        )
    assert not (tmp_path / "model").exists()


def test_cache_rejects_nonfinite_bound_values(corpus, tmp_path):
    manifest, source = corpus
    root = tmp_path / "cache"
    rj.extract_cache(manifest, root, source=source, split="train")
    data = root / "features.f32"
    x = np.frombuffer(data.read_bytes(), dtype="<f4").copy()
    x[0] = np.nan
    data.write_bytes(x.tobytes())
    cache = root / "cache.json"
    d = json.loads(cache.read_bytes())
    d["data_sha256"] = sha(data)
    cache.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="values"):
        rj.load_cache(manifest, cache, checksum=sha(cache), split="train")


def test_worker_main_failure_and_binary_success(monkeypatch):
    import resource

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    payload = (1).to_bytes(4, "little") + (4).to_bytes(4, "little") + b"jpeg"
    monkeypatch.setattr(jrm.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload)))
    output = io.BytesIO()
    monkeypatch.setattr(jrm.sys, "stdout", SimpleNamespace(buffer=output))
    monkeypatch.setattr(jrm.jpeg, "worker", lambda *a: [0] * jrm.DIMENSIONS)
    assert jrm.main() == 0 and len(output.getvalue()) == jrm.OUTPUT_BYTES
    monkeypatch.setattr(
        jrm.jpeg, "worker", lambda *a: (_ for _ in ()).throw(ValueError("secret host path"))
    )
    monkeypatch.setattr(jrm.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload)))
    assert jrm.main() == 1


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        (9).to_bytes(4, "little"),
        (1).to_bytes(4, "little"),
        (1).to_bytes(4, "little") + (jrm.jpeg.MAX_IMAGE_BYTES + 1).to_bytes(4, "little"),
        (1).to_bytes(4, "little") + (4).to_bytes(4, "little") + b"x",
        (1).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"xx",
    ],
)
def test_worker_rejects_malformed_frames(monkeypatch, payload):
    import resource

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(jrm.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload)))
    monkeypatch.setattr(jrm.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    monkeypatch.setattr(jrm.jpeg, "worker", lambda *a: [0] * jrm.DIMENSIONS)
    assert jrm.main() == 1


def test_batch_limits():
    with pytest.raises(ValueError, match="batch"):
        jrm.jpeg_jrm_batch([])
    with pytest.raises(ValueError, match="batch"):
        jrm.jpeg_jrm_batch([b""] * 9)


def test_extraction_deadlines(corpus, tmp_path, monkeypatch):
    manifest, source = corpus
    counter = iter([0, 1801])
    monkeypatch.setattr(rj.time, "monotonic", lambda: next(counter))
    with pytest.raises(ValueError, match="deadline"):
        rj.extract_cache(manifest, tmp_path / "timeout", source=source, split="train")
    assert not (tmp_path / "timeout/cache.json").exists()
    counter = iter([0, 0, 1801])
    with pytest.raises(ValueError, match="deadline"):
        rj.extract_cache(manifest, tmp_path / "late", source=source, split="train")
    assert not (tmp_path / "late/cache.json").exists()


def test_model_rejects_object_arrays(tmp_path):
    path = tmp_path / "object.npz"
    model = numeric_model()
    np.savez(
        path,
        subspaces=model.subspaces,
        weights=np.array([["untrusted object"]], dtype=object),
        biases=model.biases,
    )
    with pytest.raises(ValueError, match="Object arrays"):
        rj.load_model(path, checksum=sha(path))
