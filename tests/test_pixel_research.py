"""Pixel preprocessing, raw-cache security and independent residual smoke oracles."""

import importlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import cli
from core import jpeg_cnn as cnn
from core import jpeg_pixels as pixels
from steganography import research_pixels as rp
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha


def jpeg(width=257, height=259, mode="RGB"):
    image = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype="u1")
    )
    output = io.BytesIO()
    image.convert(mode).save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def pixel_corpus(corpus, monkeypatch):
    manifest, source = corpus
    doc = json.loads(manifest.read_bytes())
    for row in doc["samples"]:
        path = source / row["path"]
        path.write_bytes(jpeg())
        # Distinct compressed bytes/identities without changing the decoded raster.
        with path.open("ab") as stream:
            stream.write(row["lineage"].encode() + str(row["method"]).encode())
        row["sha256"], row["size"] = sha(path), path.stat().st_size
    manifest.write_text(json.dumps(doc))
    monkeypatch.setattr(
        pixels,
        "pixel_batch",
        lambda files: np.frombuffer(
            b"".join(pixels.decode(data) for data in files), dtype="u1"
        ).reshape(len(files), 1, 128, 128),
    )
    return manifest, source


def test_independent_crop_and_real_worker():
    for mode in ("RGB", "L"):
        data = jpeg(mode=mode)
        with Image.open(io.BytesIO(data)) as image:
            full = np.asarray(image.convert("L"))
        expected = full[65:193, 64:192]
        assert pixels.region(data) == {"image_size": [257, 259], "region": [64, 65, 128, 128]}
        np.testing.assert_array_equal(
            np.frombuffer(pixels.decode(data), dtype="u1").reshape(128, 128), expected
        )
        actual = pixels.pixel_batch([data, data])
        np.testing.assert_array_equal(actual[0, 0], expected)
        np.testing.assert_array_equal(actual[0], actual[1])
        assert not actual.flags.writeable
    for bad in (jpeg(width=127), jpeg(mode="CMYK"), b"not JPEG", b"x" * (2 * 1024**2 + 1)):
        with pytest.raises((ValueError, OSError)):
            pixels.region(bad)
    for bad in ([], [b"x"] * 9):
        with pytest.raises(ValueError, match="batch"):
            pixels.pixel_batch(bad)


@pytest.mark.parametrize("fault", ["failed", "short", "excess", "timeout"])
def test_native_output_failure(monkeypatch, fault):
    def run(*a, stdout, **k):
        if fault == "timeout":
            raise subprocess.TimeoutExpired("fixed", 15)
        raw = bytes(pixels.PIXELS)
        stdout.write(raw[:-1] if fault == "short" else raw + b"x" if fault == "excess" else raw)
        return SimpleNamespace(returncode=1 if fault == "failed" else 0)

    monkeypatch.setattr(pixels.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        pixels.pixel_batch([jpeg()])


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\x09\x00\x00\x00",
        b"\x01\x00\x00\x00",
        b"\x01\x00\x00\x00" + (2 * 1024**2 + 1).to_bytes(4, "little"),
        b"\x01\x00\x00\x00\x04\x00\x00\x00x",
    ],
)
def test_worker_rejects_malformed_frames(monkeypatch, raw):
    import resource

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(pixels.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(pixels.sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
    assert pixels.main() == 1


def test_worker_valid_and_trailing_input(monkeypatch):
    import resource

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    data = jpeg()
    frame = b"\x01\x00\x00\x00" + len(data).to_bytes(4, "little") + data
    for extra, code in ((b"", 0), (b"x", 1)):
        output = io.BytesIO()
        monkeypatch.setattr(pixels.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(frame + extra)))
        monkeypatch.setattr(pixels.sys, "stdout", SimpleNamespace(buffer=output))
        assert pixels.main() == code and len(output.getvalue()) == pixels.PIXELS


def test_cache_and_thin_cli(pixel_corpus, tmp_path, capsys):
    manifest, source = pixel_corpus
    out = tmp_path / "cache"
    assert (
        cli.main(
            [
                "research",
                "pixel-cache",
                "--manifest",
                str(manifest),
                "--source",
                str(source),
                "--split",
                "train",
                "--workers",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert not json.loads(capsys.readouterr().out)["deployed"]
    values, samples, descriptor = rp.load_pixels(
        manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train"
    )
    assert values.shape == (6, 1, 128, 128) and not values.flags.writeable
    assert descriptor["decoder"] == pixels.decoder_contract() and len(samples) == 6
    with pytest.raises(FileExistsError):
        rp.extract_pixels(manifest, out, source=source, split="train")
    with pytest.raises(ValueError, match="contract"):
        rp.load_pixels(manifest, out / "cache.json", checksum="0" * 64, split="train")
    with pytest.raises(ValueError, match="contract"):
        rp.load_pixels(
            manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="validation"
        )


@pytest.mark.parametrize(
    "fault",
    [
        "rows",
        "shape",
        "version",
        "decoder",
        "identity",
        "region",
        "giant-size",
        "bool-region",
        "missing-size",
    ],
)
def test_descriptor_tamper(pixel_corpus, tmp_path, fault):
    manifest, source = pixel_corpus
    out = tmp_path / "cache"
    rp.extract_pixels(manifest, out, source=source, split="train")
    path = out / "cache.json"
    doc = json.loads(path.read_bytes())
    if fault == "rows":
        doc["rows"].pop()
    if fault == "shape":
        doc["shape"] = [10**12, 1, 128, 128]
    if fault == "version":
        doc["feature_version"] = "unknown"
    if fault == "decoder":
        doc["decoder"]["pillow"] = "unknown"
    if fault == "identity":
        doc["rows"][0]["sha256"] = "0" * 64
    if fault == "region":
        doc["rows"][0]["region"][0] += 1
    if fault == "giant-size":
        doc["rows"][0]["image_size"] = [10**12, 128]
    if fault == "bool-region":
        doc["rows"][0]["region"][0] = True
    if fault == "missing-size":
        doc["rows"][0].pop("image_size")
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        rp.load_pixels(manifest, path, checksum=sha(path), split="train")


@pytest.mark.parametrize("fault", ["truncated", "changed", "symlink", "byte-limit"])
def test_cache_data_guards(pixel_corpus, tmp_path, monkeypatch, fault):
    manifest, source = pixel_corpus
    out = tmp_path / "cache"
    rp.extract_pixels(manifest, out, source=source, split="train")
    path = out / "pixels.u8"
    raw = path.read_bytes()
    if fault == "truncated":
        path.write_bytes(raw[:-1])
    if fault == "changed":
        path.write_bytes(bytes(len(raw)))
    if fault == "symlink":
        path.unlink()
        path.symlink_to(source / "jrm-0.jpg")
    if fault == "byte-limit":
        monkeypatch.setattr(rp, "MAX_CACHE_BYTES", 1)
    with pytest.raises(ValueError):
        rp.load_pixels(
            manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train"
        )


@pytest.mark.parametrize(
    "fault",
    [
        "source-symlink",
        "source-change",
        "workers",
        "rows",
        "format",
        "failed",
        "size",
        "before-deadline",
        "after-deadline",
        "out-symlink",
    ],
)
def test_incomplete_cache_is_never_complete(pixel_corpus, tmp_path, monkeypatch, fault):
    manifest, source = pixel_corpus
    out = tmp_path / "bad"
    if fault == "source-symlink":
        file = source / "jrm-0.jpg"
        file.unlink()
        file.symlink_to(source / "jrm-1.jpg")
    if fault == "source-change":
        (source / "jrm-0.jpg").write_bytes(b"changed")
    if fault == "rows":
        monkeypatch.setattr(rp, "MAX_ROWS", 1)
    if fault == "format":
        selected = rp.selected_samples
        monkeypatch.setattr(
            rp, "selected_samples", lambda *a: [{**s, "format": "PNG"} for s in selected(*a)]
        )
    if fault == "failed":
        monkeypatch.setattr(
            pixels, "pixel_batch", lambda *a: (_ for _ in ()).throw(RuntimeError("failed"))
        )
    if fault == "size":
        monkeypatch.setattr(pixels, "pixel_batch", lambda *a: np.zeros(1, dtype="u1"))
    if fault in {"before-deadline", "after-deadline"}:
        times = iter([0, 1801] if fault == "before-deadline" else [0, 0, 1801])
        monkeypatch.setattr(rp.time, "monotonic", lambda: next(times))
    if fault == "out-symlink":
        out.symlink_to(source, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        rp.extract_pixels(
            manifest, out, source=source, split="train", workers=0 if fault == "workers" else 1
        )
    assert not (out / "cache.json").exists()


def test_cnn_residual_oracle_gradient_and_inference():
    torch = pytest.importorskip("torch")
    torch.manual_seed(20261009)
    model = cnn.network()
    pixels_input = np.random.default_rng(42).integers(0, 256, (2, 1, 128, 128), dtype="u1")
    values = torch.from_numpy(pixels_input.astype(np.float32) / 255)
    windows = np.lib.stride_tricks.sliding_window_view(values.numpy(), (3, 3), axis=(2, 3))
    expected = np.stack(
        [
            np.clip((windows[:, 0] * np.array(f, dtype=np.float32)).sum(axis=(-2, -1)), -1, 1)
            for f in cnn.FILTERS
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        model.residuals(values).detach().numpy(), expected, atol=5e-7, rtol=0
    )
    initial = model.filters.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(values), torch.tensor([[0.0], [1.0]])
    )
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert torch.equal(initial, model.filters) and model.filters.grad is None
    with pytest.raises(ValueError, match="eval"):
        cnn.pixel_logits(model, pixels_input)
    model.eval()
    actual = cnn.pixel_logits(model, pixels_input)
    expected = model(values).detach().numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        cnn.pixel_logits(model, pixels_input[:1]), actual[:1], atol=1e-6, rtol=0
    )
    for bad in (
        [],
        pixels_input.astype(float),
        pixels_input[:0],
        pixels_input[:, :, :64],
        np.zeros((65, 1, 128, 128), dtype="u1"),
    ):
        with pytest.raises(ValueError, match="tensor"):
            cnn.pixel_logits(model, bad)
    with torch.no_grad():
        model.layers[-1].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="output"):
        cnn.pixel_logits(model, pixels_input)


def test_torch_is_optional_at_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    importlib.reload(cnn)
    with pytest.raises(RuntimeError, match="optional research"):
        cnn.network()
    model = SimpleNamespace(training=False)
    with pytest.raises(RuntimeError, match="optional research"):
        cnn.pixel_logits(model, np.zeros((1, 1, 128, 128), dtype="u1"))
