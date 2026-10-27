"""Independent generated admission, provenance and filesystem adversaries."""

import copy
import hashlib
import io
import os

import numpy as np
import pytest
from PIL import Image

from core import colab_artifacts as a
from core import spatial_colab_data as d


def image(mode="RGB", extension="PNG", side=260, seed=8):
    shape = (side, side, 3) if mode == "RGB" else (side, side)
    values = np.random.default_rng(seed).integers(0, 256, shape, dtype="u1")
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format=extension)
    return buffer.getvalue(), values


def source(raw, name="first"):
    return {
        "id": name,
        "page": f"https://commons.wikimedia.org/wiki/File:{name}.png",
        "original_url": f"https://upload.wikimedia.org/wikipedia/commons/a/ab/{name}.png",
        "author": "Generated test author",
        "license": "CC0-1.0",
        "license_url": d.LICENSES["CC0-1.0"],
        "original_sha1": hashlib.sha1(raw, usedforsecurity=False).hexdigest(),
        "original_bytes": len(raw),
        "ancestry_group": "whole-origin",
        "ancestry_review": "whole-source-group-reviewed",
        "role": "unassigned",
        "media_status": "not_downloaded",
    }


@pytest.fixture
def corpus(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    rows = []
    for index, mode in enumerate(("L", "RGB")):
        raw, _ = image(mode=mode)
        row = source(raw, f"source-{index}")
        (originals / (row["id"] + ".original")).write_bytes(raw)
        rows.append(row)
    return {"schema": "spatial-commons-v1", "sources": rows}, originals


def test_prepare_independent_pixels_payload_and_groups(corpus, tmp_path):
    manifest, originals = corpus
    out = tmp_path / "prepared"
    report = d.prepare(manifest, originals, out)
    assert report["status"] == "ready" and len(report["artifacts"]) == 8
    assert report["optimizer_updates"] == 0 and not report["representative_benchmark"]
    assert {r["group"] for r in report["artifacts"]} == {"whole-origin"}
    for record in report["artifacts"]:
        data = (out / record["name"]).read_bytes()
        assert a.digest(data) == record["sha256"] and record["role"] == "unassigned"
        raw = (originals / (record["source_id"] + ".original")).read_bytes()
        with Image.open(io.BytesIO(raw)) as img:
            original = np.asarray(img)[:256, :256].copy()
        with Image.open(io.BytesIO(data)) as img:
            prepared = np.asarray(img).copy()
        assert np.array_equal(original & 254, prepared & 254)
        if record["kind"] == "cover":
            assert np.array_equal(original, prepared)
        else:
            expected = bytes.fromhex(
                a.digest((record["source_id"] + ":generated-control").encode())
            )
            bits = [int(x) & 1 for x in prepared.reshape(-1)[:256]]
            recovered = bytes(
                sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, 256, 8)
            )
            assert recovered == expected
    with pytest.raises(FileExistsError):
        d.prepare(manifest, originals, out)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "../escape"),
        ("ancestry_group", "/secret"),
        ("page", "http://example.org"),
        ("original_url", "https://upload.wikimedia.org.evil/a.png"),
        ("original_url", "https://upload.wikimedia.org/wikipedia/commons/thumb/a.png"),
        ("license", "CC-BY-NC-4.0"),
        ("license_url", "https://invalid.test"),
        ("author", ""),
        ("role", "test"),
        ("original_sha1", "x" * 40),
        ("original_bytes", True),
        ("original_bytes", d.MAX_FILE + 1),
        ("ancestry_review", "guess"),
        ("media_status", "verified"),
    ],
)
def test_source_contract_rejects(corpus, field, value):
    manifest, _ = corpus
    manifest["sources"][0][field] = value
    with pytest.raises(ValueError):
        d.manifest(manifest)


@pytest.mark.parametrize("fault", ["schema", "count", "fields", "duplicate", "leak"])
def test_manifest_adversaries(corpus, fault):
    manifest, _ = corpus
    if fault == "schema":
        manifest["schema"] = "old"
    elif fault == "count":
        manifest["sources"] = []
    elif fault == "fields":
        manifest["sources"][0]["unexpected"] = "must reject"
    elif fault == "duplicate":
        manifest["sources"].append(copy.deepcopy(manifest["sources"][0]))
    else:
        manifest["sources"][1]["page"] = manifest["sources"][0]["page"]
        manifest["sources"][1]["ancestry_group"] = "another-role"
    with pytest.raises(ValueError):
        d.manifest(manifest)


def test_acquisition_is_explicit_charged_and_hash_bound(corpus, tmp_path, monkeypatch):
    manifest, originals = corpus
    ledger = {"sha256": "f" * 64, "historical_bytes": None}
    calls = []

    def fetch(url, limit):
        calls.append(url)
        row = next(r for r in manifest["sources"] if r["original_url"] == url)
        assert (tmp_path / "download" / "reservation.json").exists()
        return (originals / (row["id"] + ".original")).read_bytes()

    monkeypatch.setattr(d, "fetch", fetch)
    with pytest.raises(ValueError, match="explicit"):
        d.acquire(manifest, tmp_path / "no", allow_download=False, prior_ledger=ledger)
    assert not calls
    result = d.acquire(manifest, tmp_path / "download", allow_download=True, prior_ledger=ledger)
    assert result["status"] == "ready" and len(calls) == 2
    assert result["prior_ledger"] == ledger and not result["historical_total_complete"]
    with pytest.raises(FileExistsError):
        d.acquire(manifest, tmp_path / "download", allow_download=True, prior_ledger=ledger)
    monkeypatch.setattr(d, "fetch", lambda *args: b"wrong")
    with pytest.raises(ValueError, match="hash/size"):
        d.acquire(manifest, tmp_path / "bad", allow_download=True, prior_ledger=ledger)
    assert a.document(tmp_path / "bad" / "reservation.json")["status"] == "unavailable"


@pytest.mark.parametrize(
    "fault", ["pending", "ledger", "total", "hash", "duplicate", "output", "payload"]
)
def test_admission_failures(corpus, tmp_path, monkeypatch, fault):
    manifest, originals = corpus
    ledger = {"sha256": "f" * 64, "historical_bytes": 45}
    if fault == "pending":
        manifest["sources"][0]["ancestry_review"] = "pending"
        with pytest.raises(ValueError, match="ancestry"):
            d.acquire(manifest, tmp_path / "acquire", allow_download=True, prior_ledger=ledger)
    elif fault == "ledger":
        ledger["historical_bytes"] = -1
    elif fault in {"total", "output"}:
        monkeypatch.setattr(d, "MAX_OUTPUT", 1)
    elif fault == "hash":
        manifest["sources"][0]["original_sha1"] = "0" * 40
    elif fault == "duplicate":
        raw, _ = image(mode="RGB")
        second = source(raw + b"padding", "third")
        manifest["sources"].append(second)
        (originals / "third.original").write_bytes(raw + b"padding")
    elif fault == "payload":
        monkeypatch.setattr(d, "recover", lambda *args: b"bad")
    with pytest.raises(ValueError):
        if fault in {"ledger", "total"}:
            d.acquire(manifest, tmp_path / "acquire", allow_download=True, prior_ledger=ledger)
        else:
            d.prepare(manifest, originals, tmp_path / "out")


@pytest.mark.parametrize(
    "fault", ["format", "mode", "small", "bytes", "pixels", "corrupt", "alpha", "orientation"]
)
def test_carrier_limits(fault, monkeypatch):
    raw, _ = image()
    if fault == "format":
        raw, _ = image(extension="TIFF")
    elif fault in {"mode", "alpha"}:
        buffer = io.BytesIO()
        Image.new("RGBA" if fault == "alpha" else "P", (260, 260)).save(buffer, format="PNG")
        raw = buffer.getvalue()
    elif fault == "small":
        raw, _ = image(side=255)
    elif fault == "bytes":
        monkeypatch.setattr(d, "MAX_FILE", 5)
    elif fault == "pixels":
        monkeypatch.setattr(d, "MAX_PIXELS", 10)
    elif fault == "orientation":
        buffer = io.BytesIO()
        img = Image.new("RGB", (260, 260))
        exif = Image.Exif()
        exif[274] = 6
        img.save(buffer, format="PNG", exif=exif)
        raw = buffer.getvalue()
    else:
        raw = raw[:40]
    with pytest.raises((ValueError, OSError)):
        d.crop(raw)


def test_io_and_scalar_bounds(tmp_path):
    path = tmp_path / "file"
    a.put(path, b"data")
    assert a.read(path, 4) == b"data"
    with pytest.raises(ValueError):
        a.read(path, 3)
    with pytest.raises(ValueError):
        a.put(tmp_path / "too-large", b"ab", 1)
    link = tmp_path / "link"
    link.symlink_to(path)
    for operation in (lambda: a.read(link, 4), lambda: a.put(link, b"a")):
        with pytest.raises(ValueError, match="symlink"):
            operation()
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="nonregular"):
        a.read(fifo, 4)
    a.put(tmp_path / "list", b"[]")
    with pytest.raises(ValueError, match="object"):
        a.document(tmp_path / "list")
    with pytest.raises(ValueError, match="capacity"):
        d.embed(np.zeros(1, dtype="u1"), b"x")
    with pytest.raises(ValueError, match="length"):
        d.recover(np.zeros(8, dtype="u1"), 2)


def test_network_bounds_without_network(monkeypatch):
    class Response(io.BytesIO):
        pass

    class Opener:
        def open(self, request, timeout):
            assert timeout == 30
            return Response(b"abc")

    monkeypatch.setattr(d.urllib.request, "build_opener", lambda *args: Opener())
    url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/file.png"
    assert d.fetch(url, 3) == b"abc"
    with pytest.raises(ValueError, match="transfer"):
        d.fetch(url, 2)
    with pytest.raises(ValueError, match="allowlist"):
        d.fetch("file:///etc/passwd", 2)
    with pytest.raises(ValueError, match="redirect"):
        d._NoRedirect().redirect_request(None, None, 302, None, None, "https://example.com")
