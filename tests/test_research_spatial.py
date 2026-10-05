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
    torch = pytest.importorskip("torch")
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
    from steganography.research import train_model
    from steganography.research_features import feature_inputs

    for version in ("spatial-summary-v1", "spatial-cooccurrence-v1"):
        directory = out / version
        config = json.loads((directory / "train-config.json").read_text())
        x, _, _ = feature_inputs(config, split="train")
        saved = torch.load(directory / "baseline.pt", weights_only=True)
        np.testing.assert_allclose(
            saved["preprocessing"]["normalization"]["mean"], x.mean(axis=0), rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            saved["preprocessing"]["normalization"]["scale"],
            np.maximum(x.std(axis=0), 1e-4),
            rtol=1e-5,
            atol=1e-7,
        )
        assert saved["training_provenance"]["training"]["standardize"]
        for key in ("standardize", "class_balanced"):
            bad = tmp_path / f"{version}-{key}.json"
            bad.write_text(json.dumps({**config, key: "yes"}))
            with pytest.raises(ResearchManifestError, match="booleans"):
                train_model(bad, tmp_path / "invalid.pt")
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


def test_geometry_selection_reserved_and_cli_failures(development, tmp_path, monkeypatch):
    root, reserved, source = development
    options_before = options(root, reserved)
    for content, match in (({"samples": []}, "empty"), ({"samples": [{}]}, "identity")):
        reserved.write_text(json.dumps(content))
        with pytest.raises(ResearchManifestError, match=match):
            rs.generate_corpus(root, tmp_path / "invalid", **options_before)
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    selection = json.loads((root / "selection.json").read_text())
    selection["members"][0]["crc32"] = 0
    (root / "selection.json").write_text(json.dumps(selection))
    source["selection_sha256"] = hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
    (root / "source.json").write_text(json.dumps(source))
    with pytest.raises(ResearchManifestError, match="member integrity"):
        rs.generate_corpus(root, tmp_path / "crc", **options(root, reserved))
    monkeypatch.setattr(rs, "run_development", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "sys.argv",
        [
            "spatial",
            str(root),
            str(tmp_path / "cli"),
            "--source-sha256",
            "a" * 64,
            "--reserved-manifest",
            str(reserved),
        ],
    )
    rs.main()


def test_all_residual_filters_independent_scalar():
    pixels = np.random.default_rng(2).integers(0, 256, (9, 10, 1), dtype=np.int16)
    for name in sc.FILTERS:
        residual = []
        h, w = pixels.shape[:2]
        for y in range(h - (2 if name == "v2" else 1 if name in ("v1", "d1", "a1") else 0)):
            line = []
            for x in range(w - (2 if name == "h2" else 1 if name in ("h1", "d1", "a1") else 0)):
                a = int(pixels[y, x, 0])
                if name == "h1":
                    value = int(pixels[y, x + 1, 0]) - a
                elif name == "v1":
                    value = int(pixels[y + 1, x, 0]) - a
                elif name == "h2":
                    value = int(pixels[y, x + 2, 0]) - 2 * int(pixels[y, x + 1, 0]) + a
                elif name == "v2":
                    value = int(pixels[y + 2, x, 0]) - 2 * int(pixels[y + 1, x, 0]) + a
                elif name == "d1":
                    value = int(pixels[y + 1, x + 1, 0]) - a
                else:
                    value = int(pixels[y + 1, x, 0]) - int(pixels[y, x + 1, 0])
                line.append(max(-2, min(2, value)))
            residual.append(line)
        if name in ("v1", "v2"):
            residual = list(zip(*residual, strict=True))
        counts = dict.fromkeys(sc.BINS, 0)
        for line in residual:
            for x in range(len(line) - 2):
                counts[sc.canonical(tuple(line[x : x + 3]))] += 1
        np.testing.assert_allclose(
            sc.residual_histogram(pixels, name),
            np.array(list(counts.values())) / sum(counts.values()),
        )


def test_oracle_fault_and_input_change_are_not_success(development, tmp_path, monkeypatch):
    root, reserved, source = development
    pixels = np.zeros((16, 16), dtype=np.uint8)
    with monkeypatch.context() as patch:
        patch.setattr(rs.np, "unpackbits", lambda values: np.zeros(values.size * 8, dtype=np.uint8))
        with pytest.raises(ResearchManifestError, match="oracle"):
            rs.replacement(pixels, "fixture", "sequential", 40)
    opts = options(root, reserved)
    mkdir = rs.Path.mkdir

    def changing_mkdir(path, *args, **kwargs):
        mkdir(path, *args, **kwargs)
        (root / source["samples"][0]["path"]).write_bytes(b"changed after input preflight")

    monkeypatch.setattr(rs.Path, "mkdir", changing_mkdir)
    with pytest.raises(ResearchManifestError, match="changed after preflight"):
        rs.generate_corpus(root, tmp_path / "changed", **opts)
