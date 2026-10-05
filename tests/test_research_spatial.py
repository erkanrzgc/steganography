import hashlib
import io
import json
import zlib

import numpy as np
import pytest
from PIL import Image

from core import spatial_cooccurrence as sc
from steganography import research_spatial as rs
from steganography.research import ResearchManifestError


def test_residual_scalar_oracle_and_contract(tmp_path, monkeypatch):
    pixels = np.arange(12 * 13 * 3, dtype=np.int16).reshape(12, 13, 3) % 11
    # Independent first-horizontal-difference histogram.
    counts = dict.fromkeys(sc.BINS, 0)
    for y in range(12):
        for x in range(10):
            for c in range(3):
                triple = tuple(
                    max(-2, min(2, int(pixels[y, x + i + 1, c] - pixels[y, x + i, c])))
                    for i in range(3)
                )
                counts[sc.canonical(triple)] += 1
    expected = np.array(list(counts.values())) / sum(counts.values())
    np.testing.assert_allclose(sc.residual_histogram(pixels, "h1"), expected)
    assert len(sc.BINS) == 39 and len(sc.FEATURE_NAMES) == 468
    assert sc.canonical((1, -2, 0)) == sc.canonical((0, 2, -1))
    with pytest.raises(ValueError, match="filter"):
        sc.residual_histogram(pixels, "unknown")
    outputs = []
    for fmt in ("PNG", "BMP"):
        buffer = io.BytesIO()
        Image.fromarray(pixels.astype(np.uint8)).save(buffer, format=fmt)
        outputs.append(sc.spatial_cooccurrence_features(buffer.getvalue()))
    np.testing.assert_allclose(*outputs)
    assert np.isfinite(outputs).all()
    np.testing.assert_allclose(np.array(outputs[0]).reshape(12, 39).sum(axis=1), 1)
    for size, fmt in (((7, 8), "PNG"), ((8, 8), "JPEG")):
        buffer = io.BytesIO()
        Image.new("L", size).save(buffer, format=fmt)
        with pytest.raises(ValueError):
            sc.spatial_cooccurrence_features(buffer.getvalue())
    monkeypatch.setattr(sc, "MAX_PIXELS", 1)
    with pytest.raises(ValueError):
        sc.spatial_cooccurrence_features(buffer.getvalue())
    monkeypatch.setattr(sc, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="byte"):
        sc.spatial_cooccurrence_features(b"xx")


@pytest.fixture
def development(tmp_path):
    root = tmp_path / "originals"
    root.mkdir()
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    rows = []
    members = []
    rng = np.random.default_rng(42)
    splits = set()
    while splits != {"train", "validation"}:
        pixels = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PPM")
        data = buffer.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        split = "train" if int(sha[:16], 16) / 2**64 < 0.8 else "validation"
        if split in splits:
            continue
        splits.add(split)
        name = f"{len(rows):04d}.pgm"
        (root / name).write_bytes(data)
        member = f"images/{name}"
        rows.append(
            {
                "path": name,
                "sha256": sha,
                "lineage": sha,
                "split": split,
                "size": len(data),
                "source_group": "BOSSbase-1.01",
                "label": "cover",
                "method": None,
                "upstream_member": member,
            }
        )
        members.append({"upstream_member": member, "size": len(data), "crc32": zlib.crc32(data)})
    (root / "selection.json").write_text(json.dumps({"members": members}))
    source = {
        "schema_version": "boss-development-acquisition-v1",
        "purpose": "development covers only",
        "samples": rows,
        "selection_sha256": hashlib.sha256((root / "selection.json").read_bytes()).hexdigest(),
        "reserved_manifest_sha256": [hashlib.sha256(reserved.read_bytes()).hexdigest()],
        "source_url": "https://example.org/fixture",
        "license": "test fixture",
    }
    (root / "source.json").write_text(json.dumps(source))
    return root, reserved, source


def options(root, reserved):
    return {
        "source_sha256": hashlib.sha256((root / "source.json").read_bytes()).hexdigest(),
        "reserved_manifests": [reserved],
    }


def test_corpus_and_fixed_training_pipeline(development, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    root, reserved, _ = development
    out = tmp_path / "run"
    report = rs.run_development(root, out, **options(root, reserved))
    assert report["counts"]["sample_count"] == 14
    assert not report["deployed"] and report["cross_source_gate"] == "unavailable"
    assert str(tmp_path) not in json.dumps(report)
    for experiment in report["experiments"].values():
        assert len(experiment["by_method_rate"]) == 6
        assert experiment["onnx_parity"]["samples"] == 7
    manifest = json.loads((out / "corpus/manifest.json").read_text())
    for row in manifest["samples"]:
        assert row["split"] != "test"
        assert row["payload_bits"] % 8 == 0
    with pytest.raises(FileExistsError):
        rs.run_development(root, out, **options(root, reserved))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda s: s.update(purpose="test"), "development covers"),
        (lambda s: s.update(selection_sha256="0" * 64), "checksum"),
        (lambda s: s.update(reserved_manifest_sha256=[]), "provenance"),
        (lambda s: s.update(samples=[]), "count"),
        (lambda s: s["samples"][0].update(upstream_member="other"), "membership"),
        (lambda s: s["samples"][0].update(lineage="bad"), "identity"),
        (lambda s: s["samples"][0].update(path="../escape.pgm"), "identity"),
        (lambda s: s["samples"][0].update(size=0), "integrity"),
    ],
)
def test_provenance_mutations(development, tmp_path, mutation, match):
    root, reserved, source = development
    mutation(source)
    (root / "source.json").write_text(json.dumps(source))
    with pytest.raises(ResearchManifestError, match=match):
        rs.generate_corpus(root, tmp_path / "out", **options(root, reserved))
    assert not (tmp_path / "out").exists()


def test_limits_symlinks_and_invalid_recipe(development, tmp_path, monkeypatch):
    root, reserved, source = development
    opts = options(root, reserved)
    with pytest.raises(ResearchManifestError, match="checksum"):
        rs.generate_corpus(root, tmp_path / "out", **{**opts, "source_sha256": "0" * 64})
    with pytest.raises(ResearchManifestError, match="provenance"):
        rs.generate_corpus(root, tmp_path / "out", **{**opts, "reserved_manifests": []})
    with pytest.raises(ResearchManifestError, match="recipe"):
        rs.replacement(np.zeros((8, 8), dtype=np.uint8), "x", "bad", 5)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ResearchManifestError, match="symlink"):
        rs.run_development(root, link / "out", **opts)
    with pytest.raises(ResearchManifestError, match="symlink"):
        rs.generate_corpus(root, link / "out", **opts)
    monkeypatch.setattr(rs, "MAX_CORPUS_BYTES", 1)
    with pytest.raises(ResearchManifestError, match="budget"):
        rs.generate_corpus(root, tmp_path / "small", **opts)
    path = root / source["samples"][0]["path"]
    original = tmp_path / "original.pgm"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ResearchManifestError, match="nonsymlink"):
        rs.generate_corpus(root, tmp_path / "unsafe", **opts)
