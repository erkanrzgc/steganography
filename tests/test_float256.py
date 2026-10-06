"""Independent unrounded IDCT, bounded framing and incompatible-cache tests."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import cli
from core import jpeg_float256 as fp
from core import srnet
from steganography import research_pixels as rp
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_pixel_research import jpeg


def test_independent_idct_and_no_rounding_clipping():
    scipy = pytest.importorskip("scipy.fft")
    rng = np.random.default_rng(61)
    coeff = rng.integers(-100, 101, (65, 66, 8, 8), dtype=np.int16)
    q = rng.integers(1, 256, (8, 8), dtype=np.uint16)
    full = scipy.idctn(coeff.astype(float) * q, axes=(-2, -1), norm="ortho") + 128
    full = full.transpose(0, 2, 1, 3).reshape(520, 528)
    expected = full[128:384, 128:384].astype("<f4")
    np.testing.assert_allclose(fp.reconstruct(coeff, q, 527, 513), expected, atol=0.002, rtol=0)
    coeff = np.zeros((32, 32, 8, 8), dtype=np.int16)
    coeff[:, :, 0, 0] = 1
    assert np.all(fp.reconstruct(coeff, np.ones((8, 8), dtype=np.uint16), 256, 256) == 128.125)
    coeff[:, :, 0, 0] = -2000
    assert np.all(fp.reconstruct(coeff, np.ones((8, 8), dtype=np.uint16), 256, 256) == -122)


@pytest.mark.parametrize(
    "fault",
    [
        "width",
        "size",
        "shape",
        "dtype",
        "coefficient",
        "qshape",
        "qdtype",
        "qzero",
        "qhigh",
        "qnone",
    ],
)
def test_idct_guards(fault):
    coeff = np.zeros((32, 32, 8, 8), dtype=np.int16)
    q = np.ones((8, 8), dtype=np.uint16)
    width = True if fault == "width" else 256
    height = 40000 if fault == "size" else 256
    if fault == "shape":
        coeff = coeff[:1]
    if fault == "dtype":
        coeff = coeff.astype(float)
    if fault == "coefficient":
        coeff = np.full(coeff.shape, 32768, dtype=np.int32)
    if fault == "qshape":
        q = q[:1]
    if fault == "qdtype":
        q = q.astype(float)
    if fault == "qzero":
        q[:] = 0
    if fault == "qhigh":
        q = np.full((8, 8), 65536, dtype=np.int32)
    if fault == "qnone":
        q = None
    with pytest.raises(ValueError, match="bounded"):
        fp.reconstruct(coeff, q, width, height)


def test_optional_dependency_and_invalid_raster(monkeypatch):
    def absent(*a):
        raise fp.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(fp.importlib.metadata, "version", absent)
    with pytest.raises(RuntimeError, match="optional"):
        fp.decoder_contract()
    monkeypatch.setattr(fp.np, "einsum", lambda *a, **k: np.full((32, 32, 8, 8), float("nan")))
    with pytest.raises(ValueError, match="raster"):
        fp.reconstruct(
            np.zeros((32, 32, 8, 8), dtype=np.int16), np.ones((8, 8), dtype=np.int16), 256, 256
        )


def test_real_decode_worker_and_nonzero_table_assignment(tmp_path):
    jp = pytest.importorskip("jpeglib")
    data = jpeg(width=527, height=513, mode="L")
    path = tmp_path / "original.jpg"
    path.write_bytes(data)
    image = jp.read_dct(str(path))
    expected = fp.reconstruct(image.Y, image.get_component_qt(0), 527, 513)
    np.testing.assert_array_equal(fp.pixel_batch([data])[0, 0], expected)
    assert not fp.pixel_batch([data]).flags.writeable
    image.qt = np.stack([np.full((8, 8), 99, dtype="u2"), image.get_component_qt(0)])
    image.quant_tbl_no = np.array([1], dtype="i4")
    shifted = tmp_path / "shifted.jpg"
    image.write_dct(str(shifted))
    np.testing.assert_array_equal(
        np.frombuffer(fp.decode(shifted.read_bytes()), dtype="<f4").reshape(256, 256), expected
    )
    assert fp.region(data) == {"image_size": [527, 513], "region": [128, 128, 256, 256]}
    assert fp.pixel_batch([jpeg()]).shape == (1, 1, 256, 256)
    for bad in (jpeg(width=255), jpeg(mode="CMYK"), b"invalid"):
        with pytest.raises((ValueError, OSError)):
            fp.region(bad)
    for bad in ([], [data] * 5):
        with pytest.raises(ValueError, match="batch"):
            fp.pixel_batch(bad)


@pytest.mark.parametrize("fault", ["timeout", "exit", "large", "short", "nan", "magnitude"])
def test_worker_output_guards(monkeypatch, fault):
    def run(command, **kwargs):
        assert command[-1] == "core.jpeg_float256" and kwargs["timeout"] == 15
        if fault == "timeout":
            raise fp.subprocess.TimeoutExpired(command, 15)
        raw = np.zeros((256, 256), dtype="<f4")
        if fault == "nan":
            raw[0, 0] = float("nan")
        if fault == "magnitude":
            raw[0, 0] = 2**37
        payload = raw.tobytes()
        if fault == "large":
            payload += b"x"
        if fault == "short":
            payload = payload[:-1]
        kwargs["stdout"].write(payload)
        return SimpleNamespace(returncode=1 if fault == "exit" else 0)

    monkeypatch.setattr(fp.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError)):
        fp.pixel_batch([jpeg()])


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"\x00" * 4,
        (5).to_bytes(4, "little"),
        (1).to_bytes(4, "little"),
        (1).to_bytes(4, "little") + b"\x00" * 4,
        (1).to_bytes(4, "little") + (3).to_bytes(4, "little") + b"x",
        (1).to_bytes(4, "little") + (fp.MAX_IMAGE_BYTES + 1).to_bytes(4, "little"),
    ],
)
def test_worker_frame_guards(monkeypatch, frame):
    import resource

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(fp.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(frame)))
    assert fp.main() == 1


@pytest.mark.parametrize("trailing", [False, True])
def test_worker_frame_success_and_trailing(monkeypatch, trailing):
    import resource

    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(fp, "decode", lambda d: bytes(256 * 256 * 4))
    frame = (
        (1).to_bytes(4, "little") + (1).to_bytes(4, "little") + b"x" + (b"x" if trailing else b"")
    )
    output = io.BytesIO()
    monkeypatch.setattr(fp.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(frame)))
    monkeypatch.setattr(fp.sys, "stdout", SimpleNamespace(buffer=output))
    assert fp.main() == (1 if trailing else 0)
    assert len(output.getvalue()) == 256 * 256 * 4


def test_non_y_jpeg_rejected(monkeypatch):
    jp = pytest.importorskip("jpeglib")
    monkeypatch.setattr(
        jp, "read_dct", lambda *a: SimpleNamespace(jpeg_color_space=jp.Colorspace.JCS_RGB)
    )
    with pytest.raises(ValueError, match="Y component"):
        fp.decode(jpeg())


def test_float_cache_cli_and_domains(corpus, tmp_path, monkeypatch, capsys):
    manifest, source = corpus
    doc = json.loads(manifest.read_bytes())
    for s in doc["samples"]:
        path = source / s["path"]
        path.write_bytes(jpeg())
        s["sha256"], s["size"] = sha(path), path.stat().st_size
    # Distinct hashes per lineage, while preserving encoded JPEG raster.
    for s in doc["samples"]:
        path = source / s["path"]
        with path.open("ab") as stream:
            stream.write(s["lineage"].encode() + str(s["method"]).encode())
        s["sha256"], s["size"] = sha(path), path.stat().st_size
    manifest.write_text(json.dumps(doc))
    monkeypatch.setattr(
        fp,
        "pixel_batch",
        lambda files: np.stack(
            [np.frombuffer(fp.decode(d), dtype="<f4").reshape(1, 256, 256) for d in files]
        ),
    )
    out = tmp_path / "cache"
    assert (
        cli.main(
            [
                "research",
                "srnet-cache",
                "--manifest",
                str(manifest),
                "--source",
                str(source),
                "--out",
                str(out),
                "--split",
                "train",
            ]
        )
        == 0
    )
    capsys.readouterr()
    values, _, desc = rp.load_pixels(
        manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train", _float=True
    )
    assert (
        values.shape == (6, 1, 256, 256)
        and values.dtype == np.dtype("<f4")
        and not values.flags.writeable
    )
    assert desc["decoder"]["rounding"] == "none" and desc["rows"][0]["region"] == [0, 0, 256, 256]
    monkeypatch.setattr(rp, "MAX_FLOAT_CACHE_BYTES", 1)
    with pytest.raises(ValueError, match="byte limits"):
        rp.load_pixels(
            manifest,
            out / "cache.json",
            checksum=sha(out / "cache.json"),
            split="train",
            _float=True,
        )
    monkeypatch.setattr(rp, "MAX_FLOAT_CACHE_BYTES", 1024**3)
    with pytest.raises(ValueError, match="contract"):
        rp.load_pixels(
            manifest, out / "cache.json", checksum=sha(out / "cache.json"), split="train"
        )
    raw = np.full(values.shape, float("nan"), dtype="<f4").tobytes()
    (out / "pixels.f32").write_bytes(raw)
    import hashlib

    desc["data_sha256"] = hashlib.sha256(raw).hexdigest()
    (out / "cache.json").write_text(json.dumps(desc))
    with pytest.raises(ValueError, match="values"):
        rp.load_pixels(
            manifest,
            out / "cache.json",
            checksum=sha(out / "cache.json"),
            split="train",
            _float=True,
        )


def test_srnet_float_inference_and_guards():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = srnet.network().eval()
        raw = np.full((1, 1, 256, 256), 128.125, dtype="<f4")
        np.testing.assert_array_equal(
            srnet.float_logits(model, raw), model(torch.from_numpy(raw)).detach().numpy()
        )
        for bad in (
            raw.astype("f8"),
            raw[:, :, :128, :128],
            np.full(raw.shape, float("nan"), dtype="f4"),
            np.full(raw.shape, 2**37, dtype="f4"),
        ):
            with pytest.raises(ValueError, match="bounded"):
                srnet.float_logits(model, bad)
    finally:
        torch.set_num_threads(previous)
