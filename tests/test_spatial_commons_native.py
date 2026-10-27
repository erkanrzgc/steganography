"""Generated admission QA; these tests never fetch native media or fit models."""

import copy
import hashlib
import importlib.util
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import spatial_colab_data as frozen
from core import spatial_commons_native as service


def fixture(mode="L", seed=1, license="CC-BY-2.0"):
    shape = (270, 280) if mode == "L" else (270, 280, 3)
    pixels = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    raw = stream.getvalue()
    row = {
        "id": f"source-{seed}",
        "page": f"https://commons.wikimedia.org/wiki/File:Source{seed}.png",
        "original_url": f"https://upload.wikimedia.org/wikipedia/commons/0/00/Source{seed}.png",
        "author": f"Author {seed}",
        "license": license,
        "license_url": service.LICENSES[license],
        "original_sha1": hashlib.sha1(raw, usedforsecurity=False).hexdigest(),
        "original_bytes": len(raw),
        "ancestry_group": f"author-{seed}",
        "ancestry_review": "conservative-author-group-reviewed",
        "role": "unassigned",
        "media_status": "not_downloaded",
        "native_mode": mode,
        "page_revision": seed,
        "evidence_sha256": "a" * 64,
    }
    return {"schema": "spatial-commons-reviewed-v2", "sources": [row]}, raw, pixels


def prior():
    return {"sha256": "b" * 64, "historical_bytes": None}


@pytest.mark.parametrize("license", list(service.LICENSES))
@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_licenses_native_and_scalar_payloads(tmp_path, monkeypatch, license, mode):
    value, raw, pixels = fixture(mode, license=license)
    monkeypatch.setattr(frozen, "fetch", lambda url, limit: raw)
    assert service.manifest(value) == value["sources"]
    acquired = service.acquire(
        value, tmp_path / "originals", allow_download=True, prior_ledger=prior()
    )
    assert acquired["historical_total_complete"] is False
    assert acquired["reserved_transfer_bytes"] == len(raw) + 1
    assert acquired["acquired"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    result = service.prepare(value, tmp_path / "originals", tmp_path / "prepared")
    assert result["status"] == "ready" and result["optimizer_updates"] == 0
    assert (
        result["representative_benchmark"] is False
        and result["cover_cleanliness_verified"] is False
    )
    assert len(result["artifacts"]) == 4
    total = 0
    controls = []
    for record in result["artifacts"]:
        output = (tmp_path / "prepared" / record["name"]).read_bytes()
        total += len(output)
        assert record["sha256"] == hashlib.sha256(output).hexdigest()
        assert record["role"] == "unassigned" and record["group"] == "author-1"
        assert record["license"] == license and record["author"] == "Author 1"
        decoded = np.asarray(Image.open(io.BytesIO(output)))
        if record["kind"] == "cover":
            assert np.array_equal(decoded, pixels[:256, :256])
        else:
            controls.append(decoded)
            delta = np.abs(decoded.astype(int) - pixels[:256, :256].astype(int))
            assert delta.max() == 1
            recovered = bytes(
                sum((int(v) & 1) << (7 - k) for k, v in enumerate(decoded.ravel()[i : i + 8]))
                for i in range(0, 256, 8)
            )
            assert recovered == hashlib.sha256(b"source-1:generated-control").digest()
    assert np.array_equal(*controls)
    assert total == result["output_bytes"]
    with pytest.raises(FileExistsError):
        service.acquire(value, tmp_path / "originals", allow_download=True, prior_ledger=prior())
    with pytest.raises(FileExistsError):
        service.prepare(value, tmp_path / "originals", tmp_path / "prepared")


@pytest.mark.parametrize(
    "key,bad",
    [
        ("id", "../oops"),
        ("id", 3),
        ("ancestry_group", "Mixed"),
        ("page", "http://commons.wikimedia.org/wiki/File:x.png"),
        ("page", "https://commons.wikimedia.org.evil/wiki/File:x.png"),
        ("page", "https://commons.wikimedia.org/wiki/File:x.png?x=1"),
        ("original_url", "https://upload.wikimedia.org/wikipedia/commons/thumb/x.png"),
        ("original_url", "https://upload.wikimedia.org/other/x.png"),
        ("license", "CC-BY-NC-4.0"),
        ("license_url", "https://example.com"),
        ("author", ""),
        ("author", " Name"),
        ("author", "a" * 1025),
        ("native_mode", "RGBA"),
        ("ancestry_review", "pending"),
        ("role", "test"),
        ("media_status", "downloaded"),
        ("original_bytes", True),
        ("original_bytes", 0),
        ("original_bytes", frozen.MAX_FILE + 1),
        ("page_revision", False),
        ("page_revision", 0),
        ("original_sha1", "bad"),
        ("evidence_sha256", "bad"),
    ],
)
def test_manifest_rejects_unsafe_rows(key, bad):
    value, _, _ = fixture()
    value["sources"][0][key] = bad
    with pytest.raises(ValueError):
        service.manifest(value)


def test_schema_groups_duplicate_and_limits():
    value, _, _ = fixture()
    for invalid in [
        {},
        {**value, "extra": 1},
        {**value, "schema": "old"},
        {**value, "sources": []},
        {**value, "sources": "wrong"},
        {**value, "sources": [None]},
        {**value, "sources": [{}]},
        {**value, "sources": value["sources"] * 17},
    ]:
        with pytest.raises(ValueError):
            service.manifest(invalid)
    second, _, _ = fixture(seed=2)
    for key in ["id", "original_sha1", "author", "page"]:
        combined = copy.deepcopy(value)
        row = copy.deepcopy(second["sources"][0])
        row[key] = combined["sources"][0][key]
        combined["sources"].append(row)
        with pytest.raises(ValueError):
            service.manifest(combined)
    combined = copy.deepcopy(value)
    row = copy.deepcopy(second["sources"][0])
    row.update(author="AUTHOR 1", ancestry_group="author-1")
    combined["sources"].append(row)
    assert len(service.manifest(combined)) == 2
    for item in combined["sources"]:
        item["original_bytes"] = frozen.MAX_FILE
    with pytest.raises(ValueError, match="total transfer"):
        service.manifest(combined)


def header(
    raw, *, depth=8, color=0, width=270, height=280, compression=0, filtering=0, interlace=0
):
    ihdr = struct.pack(">IIBBBBB", width, height, depth, color, compression, filtering, interlace)
    return raw[:16] + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)) + raw[33:]


@pytest.mark.parametrize(
    "changes",
    [
        {"depth": 16},
        {"color": 6},
        {"color": 3},
        {"color": 4},
        {"compression": 1},
        {"filtering": 1},
        {"interlace": 2},
        {"width": 255},
        {"height": 0},
        {"width": 10000, "height": 10000},
    ],
)
def test_ihdr_rejects_before_decoder(monkeypatch, changes):
    _, raw, _ = fixture()
    monkeypatch.setattr(frozen, "crop", lambda raw: pytest.fail("decoder opened invalid header"))
    with pytest.raises(ValueError):
        service.native_pixels(header(raw, **changes), "L")


def test_malformed_png_and_rgb16():
    _, raw, _ = fixture("RGB")
    for bad in [b"", b"x" * 100, raw[:20], raw[:8] + b"x" * 8 + raw[16:]]:
        with pytest.raises(ValueError):
            service.native_pixels(bad, "RGB")
    with pytest.raises(ValueError):
        service.native_pixels(header(raw, depth=16, color=2), "RGB")
    with pytest.raises(ValueError):
        service.native_pixels(raw, "RGBA")
    assert service.native_pixels(raw, "RGB").shape == (256, 256, 3)


@pytest.mark.parametrize(
    "ledger",
    [
        {},
        {"sha256": "bad", "historical_bytes": None},
        {"sha256": "b" * 64, "historical_bytes": -1},
        {"sha256": "b" * 64, "historical_bytes": True},
    ],
)
def test_ledger_before_io(tmp_path, monkeypatch, ledger):
    value, _, _ = fixture()
    monkeypatch.setattr(frozen, "fetch", lambda *a: pytest.fail("network before ledger gate"))
    with pytest.raises(ValueError):
        service.acquire(value, tmp_path / "out", allow_download=True, prior_ledger=ledger)
    assert not (tmp_path / "out").exists()


def test_opt_in_failure_debit_hash_and_symlink(tmp_path, monkeypatch):
    value, raw, _ = fixture()
    with pytest.raises(ValueError):
        service.acquire(value, tmp_path / "disabled", allow_download=False, prior_ledger=prior())

    def fail(url, limit):
        reservation = json.loads((tmp_path / "failed" / "reservation.json").read_text())
        assert reservation["reserved_transfer_bytes"] == len(raw) + 1
        raise TimeoutError("generated failure")

    monkeypatch.setattr(frozen, "fetch", fail)
    with pytest.raises(TimeoutError):
        service.acquire(value, tmp_path / "failed", allow_download=True, prior_ledger=prior())
    assert not (tmp_path / "failed" / "acquired.json").exists()
    monkeypatch.setattr(frozen, "fetch", lambda *a: b"wrong")
    with pytest.raises(ValueError, match="hash/extent"):
        service.acquire(value, tmp_path / "wrong", allow_download=True, prior_ledger=prior())
    monkeypatch.setattr(frozen, "fetch", lambda *a: bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="hash/extent"):
        service.acquire(
            value, tmp_path / "same-size-wrong", allow_download=True, prior_ledger=prior()
        )
    (tmp_path / "link").symlink_to(tmp_path / "failed", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        service.prepare(value, tmp_path / "link", tmp_path / "prepared")


def test_known_prior_and_preparation_failures(tmp_path, monkeypatch):
    value, raw, _ = fixture()
    monkeypatch.setattr(frozen, "fetch", lambda *a: raw)
    result = service.acquire(
        value,
        tmp_path / "originals",
        allow_download=True,
        prior_ledger={"sha256": "c" * 64, "historical_bytes": 123},
    )
    assert result["historical_total_complete"] is True
    source = tmp_path / "originals" / "source-1.original"
    source.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash/extent"):
        service.prepare(value, tmp_path / "originals", tmp_path / "wrong")
    source.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="hash/extent"):
        service.prepare(value, tmp_path / "originals", tmp_path / "wrong-size-match")
    source.write_bytes(raw)
    original_recover = frozen.recover
    monkeypatch.setattr(frozen, "recover", lambda *a: b"wrong")
    with pytest.raises(ValueError, match="pixel/payload"):
        service.prepare(value, tmp_path / "originals", tmp_path / "payload-bad")
    monkeypatch.setattr(frozen, "recover", original_recover)
    monkeypatch.setattr(frozen, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError, match="output bound"):
        service.prepare(value, tmp_path / "originals", tmp_path / "output-bound")


def test_duplicate_decoded_crop_and_corrupt_pixels(tmp_path, monkeypatch):
    first, raw1, pixels = fixture()
    # Independent second PNG has a different outside-crop pixel and hash.
    pixels[-1, -1] ^= 1
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    raw2 = stream.getvalue()
    second, _, _ = fixture(seed=2)
    second["sources"][0].update(
        original_bytes=len(raw2),
        original_sha1=hashlib.sha1(raw2, usedforsecurity=False).hexdigest(),
    )
    value = {"schema": first["schema"], "sources": first["sources"] + second["sources"]}
    monkeypatch.setattr(frozen, "fetch", lambda url, limit: raw1 if "Source1" in url else raw2)
    service.acquire(value, tmp_path / "originals", allow_download=True, prior_ledger=prior())
    with pytest.raises(ValueError, match="duplicate crop"):
        service.prepare(value, tmp_path / "originals", tmp_path / "duplicate")
    real_crop = frozen.crop
    calls = 0

    def corrupt(raw):
        nonlocal calls
        calls += 1
        result = real_crop(raw)
        if calls > 1:
            result[0, 0] ^= 2
        return result

    monkeypatch.setattr(frozen, "crop", corrupt)
    with pytest.raises(ValueError, match="pixel/payload"):
        service.prepare(first, tmp_path / "originals", tmp_path / "corrupt")


def test_cli_explicit_adapter(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "native_adapter", Path("scripts/spatial-native-admit.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.resource, "setrlimit", lambda *a: None)
    value, raw, _ = fixture()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"historical_bytes":null}')
    monkeypatch.setattr(frozen, "fetch", lambda *a: raw)
    args = ["--manifest", str(path)]
    assert module.main(["validate", *args]) == 0
    assert module.main(["acquire", *args]) == 2
    assert module.main(["acquire", *args, "--out", str(tmp_path / "out")]) == 2
    assert module.main(["prepare", *args, "--out", str(tmp_path / "prep")]) == 2
    assert (
        module.main(
            ["acquire", *args, "--out", str(tmp_path / "out"), "--prior-ledger", str(ledger)]
        )
        == 2
    )
    assert (
        module.main(
            [
                "acquire",
                *args,
                "--out",
                str(tmp_path / "out"),
                "--prior-ledger",
                str(ledger),
                "--allow-download",
            ]
        )
        == 0
    )
    assert (
        module.main(
            [
                "prepare",
                *args,
                "--out",
                str(tmp_path / "prep"),
                "--originals",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert '"status": "ready"' in capsys.readouterr().out
