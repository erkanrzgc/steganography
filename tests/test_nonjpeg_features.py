"""Scalar contracts and adversarial inputs; no detection-success claims."""

import hashlib
import io
import json
import wave

import numpy as np
import pytest
from PIL import Image

from cli import main
from core import nonjpeg_features as nf
from steganography import research_features as rf
from steganography.research import ResearchManifestError


def image_bytes(pixels=None, fmt="PNG", mode="RGB", size=(256, 256), color=0):
    image = Image.new(mode, size, color=color) if pixels is None else Image.fromarray(pixels)
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


def wav_bytes(samples=None, *, channels=1, rate=8000, width=2):
    if samples is None:
        samples = np.arange(163 * channels, dtype="<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as audio:
        audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())
    return out.getvalue()


def scalar_parity(pixels, direction, bit):
    counts = [0] * 18
    dy, dx = {"h": (0, 1), "v": (1, 0), "d": (1, 1), "a": (1, -1)}[direction]
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < pixels.shape[0] and 0 <= xx < pixels.shape[1]):
                continue
            for c in range(pixels.shape[2]):
                center, neighbor = int(pixels[y, x, c]), int(pixels[yy, xx, c])
                counts[((center >> bit) & 1) * 9 + max(-4, min(4, neighbor - center)) + 4] += 1
    return np.array(counts) / sum(counts)


@pytest.mark.parametrize(
    "rgb,version,count",
    [
        (False, nf.LOCAL_VERSION, 1548),
        (True, nf.RGB_VERSION, 1332),
    ],
)
def test_spatial_scalar_regions_and_fixed_base(rgb, version, count):
    pixels = np.random.default_rng(31).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    data = image_bytes(pixels)
    names, extractor = rf.feature_contract(version)
    values = extractor(data)
    from core.spatial_parity import spatial_parity_features

    assert values[:684] == spatial_parity_features(data)
    assert len(values) == len(set(names)) == count
    regions = (
        [pixels[..., c : c + 1] for c in range(3)]
        if rgb
        else [pixels[y : y + 128, x : x + 128] for y in (0, 128) for x in (0, 128)]
    )
    offset = 684
    for region in regions:
        for direction in ("h", "v", "d", "a"):
            for bit in (0, 1, 2):
                expected = scalar_parity(region, direction, bit)
                np.testing.assert_allclose(
                    values[offset : offset + 18], expected, atol=5.1e-9, rtol=0
                )
                offset += 18
    assert offset == count
    assert np.isfinite(values).all() and min(values) >= 0 and max(values) <= 1
    assert nf.CONTRACTS[version][1](image_bytes(pixels, fmt="BMP")) == values


def scalar_audio(samples, lag, bit):
    diff, transitions = [0] * 34, [0] * 4
    for index in range(len(samples) - lag):
        center, neighbor = int(samples[index]), int(samples[index + lag])
        a, b = (center >> bit) & 1, (neighbor >> bit) & 1
        diff[a * 17 + max(-8, min(8, neighbor - center)) + 8] += 1
        transitions[a * 2 + b] += 1
    return [v / (len(samples) - lag) for v in diff + transitions]


def test_pcm_scalar_signed_extremes_quarters_and_lsb():
    samples = np.resize(np.array([-32768, 32767, 0, -1, 1, 2, -2, 8, -8], dtype="<i2"), 163)
    values = nf.audio_features(wav_bytes(samples))
    assert len(values) == len(set(nf.AUDIO_NAMES)) == 760
    offset = 0
    for region in (samples, samples[:40], samples[40:80], samples[80:120], samples[120:]):
        for lag in (1, 2):
            for bit in (0, 1):
                np.testing.assert_allclose(
                    values[offset : offset + 38],
                    scalar_audio(region, lag, bit),
                    atol=5.1e-9,
                    rtol=0,
                )
                offset += 38
    assert np.isfinite(values).all() and min(values) >= 0 and max(values) <= 1
    assert nf.pcm16_samples(wav_bytes(samples)).dtype == np.int32
    changed = samples.copy()
    changed[123] ^= 1
    assert nf.audio_features(wav_bytes(changed)) != values
    for lag, bit in ((0, 0), (3, 1), (1, 2)):
        with pytest.raises(ValueError, match="lag/bit"):
            nf.audio_histogram(samples, lag, bit)


@pytest.mark.parametrize(
    "data,message",
    [
        (image_bytes(size=(255, 256)), "256x256"),
        (image_bytes(fmt="JPEG"), "PNG/BMP"),
        (image_bytes(mode="RGBA"), "native"),
        (image_bytes(mode="P"), "native"),
        (b"invalid", "identify"),
    ],
)
def test_spatial_rejects_implicit_conversion(data, message):
    for extractor in (nf.spatial_features, nf.rgb_features):
        with pytest.raises((ValueError, OSError), match=message):
            extractor(data)


def test_gray_native_and_byte_limits(monkeypatch):
    values = nf.spatial_features(image_bytes(mode="L"))
    assert len(values) == 1548
    with pytest.raises(ValueError, match="RGB"):
        nf.rgb_features(image_bytes(mode="L"))
    monkeypatch.setattr(nf, "MAX_IMAGE_BYTES", 1)
    for extractor in (nf.spatial_features, nf.audio_features):
        with pytest.raises(ValueError, match="byte"):
            extractor(b"xx")


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"RIFF0000WAVE",
        wav_bytes(channels=2),
        wav_bytes(rate=7999),
        wav_bytes(rate=192001),
        wav_bytes(width=1),
        wav_bytes(np.zeros(159, dtype="<i2")),
        wav_bytes(np.zeros(220501, dtype="<i2")),
        wav_bytes()[:-1],
        wav_bytes() + b"junk",
    ],
)
def test_audio_contract_rejections(data):
    with pytest.raises((ValueError, wave.Error, EOFError)):
        nf.audio_features(data)


def riff(body):
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + b"WAVE" + body


def test_riff_hostile_chunks():
    data = wav_bytes()
    for bad in (
        riff(data[12:] + b"data\0\0\0\0"),
        riff(b"JUNK" + (9999).to_bytes(4, "little")),
        riff((b"JUNK\0\0\0\0") * 129),
        riff(data[12:-2]),
        riff(data[12:] + b"x"),
    ):
        with pytest.raises(ValueError):
            nf.audio_features(bad)
    # A declared odd PCM extent with pad reaches the decoder but fails frame extent.
    offset = data.index(b"data")
    body = data[12 : offset + 4] + (327).to_bytes(4, "little") + data[offset + 8 :] + b"\0\0"
    with pytest.raises(ValueError, match="PCM extent"):
        nf.audio_features(riff(body))


@pytest.mark.parametrize("version", list(nf.CONTRACTS))
def test_manifest_bound_cli_and_integrity(tmp_path, capsys, version, monkeypatch):
    root = tmp_path / "media"
    root.mkdir()
    rows = []
    for label in ("cover", "stego"):
        audio = version == nf.AUDIO_VERSION
        name = label + (".wav" if audio else ".png")
        data = (
            wav_bytes(np.arange(163, dtype="<i2") + (label == "stego"))
            if audio
            else image_bytes(np.full((256, 256, 3), int(label == "stego"), dtype=np.uint8))
        )
        (root / name).write_bytes(data)
        rows.append(
            {
                "path": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "lineage": "one-original",
                "source_group": "native",
                "label": label,
                "split": "train",
            }
        )
    path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "1.0",
        "samples": rows,
        "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
    }
    path.write_text(json.dumps(manifest))
    out = tmp_path / "features.json"
    assert (
        main(
            [
                "--quiet",
                "research",
                "features",
                "--manifest",
                str(path),
                "--source",
                str(root),
                "--out",
                str(out),
                "--feature-version",
                version,
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg = {
        "manifest": str(path),
        "features": str(out),
        "features_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    x, y, _ = rf.feature_inputs(cfg, split="train")
    assert x.shape == (2, len(nf.CONTRACTS[version][0])) and y.tolist() == [0, 1]
    assert str(tmp_path) not in out.read_text()
    assert rf.model_domain(version).endswith(("v1", "v2"))
    with pytest.raises(ResearchManifestError, match="contract"):
        rf.model_domain(version, "wrong-domain")
    with pytest.raises(FileExistsError):
        rf.extract_features(path, out, source=root, feature_version=version)
    (root / rows[0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ResearchManifestError, match="integrity"):
        rf.extract_features(path, tmp_path / "corrupt", source=root, feature_version=version)
    monkeypatch.setattr(rf, "MAX_DOCUMENT_BYTES", 1000)
    # Restore source then enforce the unchanged output cap.
    (root / rows[0]["path"]).write_bytes(
        data
        if label == "cover"
        else (wav_bytes(np.arange(163, dtype="<i2")) if audio else image_bytes())
    )
    # Hash of the restored constant PNG need not match the fixture encoding;
    # output limits are independently exercised with the in-memory extractor.
    monkeypatch.setattr(rf, "read_document", lambda _: (manifest, "0" * 64))
    monkeypatch.setattr(rf, "feature_contract", lambda _: (("x",), lambda _: [0.5] * 2000))
    rows[0]["size"] = (root / rows[0]["path"]).stat().st_size
    rows[0]["sha256"] = hashlib.sha256((root / rows[0]["path"]).read_bytes()).hexdigest()
    with pytest.raises(ResearchManifestError, match="artifact exceeds"):
        rf.extract_features(path, tmp_path / "oversize", source=root, feature_version=version)
