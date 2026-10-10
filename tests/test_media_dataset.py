"""Generated acquisition/security fixtures; no live data or accuracy claims."""

import csv
import hashlib
import io
import json
import stat
import struct
import wave
import zipfile

import pytest
from PIL import Image

from core import media_dataset as service


def png(color=(20, 30, 40), mode="RGB", size=(256, 256)):
    output = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else 20).save(output, format="PNG")
    return output.getvalue()


def wav(rate=44100, frames=220500):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(frames * 2))
    return output.getvalue()


def archive(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in entries:
            output.writestr(name, data)
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"samples": [{"sha256": "a" * 64, "lineage": "a" * 64}]}))
    config = {**service.SOURCES["div2k-train"], "count": 2}
    monkeypatch.setitem(service.SOURCES, "div2k-train", config)

    def fetch(url, out, limit, deadline):
        deadline()
        raw = b"academic research purpose only"
        out.write_bytes(raw)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    monkeypatch.setattr(service, "fetch", fetch)
    path = archive(
        tmp_path / "input.zip",
        [("DIV2K_train_HR/0001.png", png()), ("DIV2K_train_HR/0002.png", png((80, 90, 100)))],
    )
    return prior, path


def test_acquire_provenance_exact_pixels_roles_no_overwrite(prepared, tmp_path):
    prior, path = prepared
    out = tmp_path / "out"
    result = service.acquire("div2k-train", out, reserved_paths=[prior], archive_path=path)
    assert result["originals"] == 2 and result["splits"] == {"train": 2}
    assert result["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["prior_manifest_sha256"] == [hashlib.sha256(prior.read_bytes()).hexdigest()]
    assert not result["model_trained"] and not result["cover_cleanliness_verified"]
    assert result["accuracy_qualification"] == "unavailable"
    assert str(tmp_path) not in (out / "source.json").read_text()
    assert (out / "0001.png").read_bytes() == png()
    assert (
        result["samples"][0]["decoded_sha256"]
        == hashlib.sha256(bytes((20, 30, 40)) * 256 * 256).hexdigest()
    )
    with pytest.raises(FileExistsError):
        service.acquire("div2k-train", out, reserved_paths=[prior], archive_path=path)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "no-prior",
        "prior-invalid",
        "prior-size",
        "prior-identity",
        "input-link",
        "output-link",
        "prior-link",
        "prior-duplicate",
        "license",
    ],
)
def test_acquire_input_and_publication_guards(prepared, tmp_path, monkeypatch, fault):
    prior, path = prepared
    key, out, paths = "div2k-train", tmp_path / "out", [prior]
    if fault == "unknown":
        key = "arbitrary-url"
    elif fault == "no-prior":
        paths = []
    elif fault == "prior-invalid":
        prior.write_text("{}")
    elif fault == "prior-size":
        prior.write_bytes(b"x" * (8 * service.MAX_DOCUMENT + 1))
    elif fault == "prior-identity":
        prior.write_text(json.dumps({"samples": [{"sha256": "bad", "lineage": "a" * 64}]}))
    elif fault == "prior-duplicate":
        digest = hashlib.sha256(png()).hexdigest()
        prior.write_text(json.dumps({"samples": [{"sha256": digest, "lineage": digest}]}))
    elif fault in ("input-link", "prior-link", "output-link"):
        link = tmp_path / "link"
        link.symlink_to(
            path if fault == "input-link" else prior if fault == "prior-link" else tmp_path
        )
        if fault == "input-link":
            path = link
        elif fault == "prior-link":
            paths = [link]
        else:
            out = link / "nested"
    elif fault == "license":

        def wrong(url, out, limit, deadline):
            out.write_bytes(b"unknown terms")
            return {"sha256": "a" * 64, "bytes": 13}

        monkeypatch.setattr(service, "fetch", wrong)
    with pytest.raises((ValueError, FileExistsError)):
        service.acquire(key, out, reserved_paths=paths, archive_path=path)
    assert not (out / "source.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "path",
        "link",
        "duplicate",
        "count",
        "name",
        "size",
        "expanded",
        "method",
        "directory",
        "decoded-duplicate",
    ],
)
def test_zip_members_fail_closed(prepared, tmp_path, monkeypatch, fault):
    prior, _ = prepared
    entries = [("DIV2K_train_HR/0001.png", png()), ("DIV2K_train_HR/0002.png", png((80, 90, 100)))]
    if fault == "path":
        entries.append(("../outside.py", b"do not execute"))
    elif fault in ("link", "method", "directory"):
        info = zipfile.ZipInfo("unused/" if fault == "directory" else "unused.py")
        info.external_attr = ((stat.S_IFLNK if fault == "link" else stat.S_IFREG) | 0o777) << 16
        info.compress_type = zipfile.ZIP_BZIP2 if fault == "method" else zipfile.ZIP_STORED
        entries.append((info, b"never execute"))
    elif fault == "duplicate":
        entries.append(entries[0])
    elif fault == "count":
        entries.pop()
    elif fault == "name":
        entries[1] = ("DIV2K_train_HR/0003.png", png((80, 90, 100)))
    elif fault == "size":
        monkeypatch.setattr(service, "MAX_FILE", 5)
    elif fault == "expanded":
        monkeypatch.setattr(service, "MAX_EXPANDED", 5)
    else:
        entries[1] = ("DIV2K_train_HR/0002.png", png())
    if fault == "duplicate":
        with pytest.warns(UserWarning):
            path = archive(tmp_path / "bad.zip", entries)
    else:
        path = archive(tmp_path / "bad.zip", entries)
    with pytest.raises(ValueError):
        service.acquire("div2k-train", tmp_path / "out", reserved_paths=[prior], archive_path=path)
    assert not (tmp_path / "out/source.json").exists()
    assert not (tmp_path / "outside.py").exists()


@pytest.mark.parametrize("fault", ["size", "garbage", "trailing", "disk", "count", "directory"])
def test_zip_directory_allocation(prepared, tmp_path, fault):
    _, path = prepared
    raw = bytearray(path.read_bytes())
    offset = raw.rfind(b"PK\x05\x06")
    limit = 1024**2
    if fault == "size":
        limit = 5
    elif fault == "garbage":
        raw = bytearray(b"x" * 100)
    elif fault == "trailing":
        raw.extend(b"suffix")
    elif fault == "disk":
        struct.pack_into("<H", raw, offset + 4, 1)
    elif fault == "count":
        struct.pack_into("<HH", raw, offset + 8, 6000, 6000)
    else:
        struct.pack_into("<I", raw, offset + 12, 3 * 1024**2)
    bad = tmp_path / "bad.zip"
    bad.write_bytes(raw)
    with pytest.raises(ValueError):
        service.zip_preflight(bad, limit)


def csv_data():
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["filename", "fold", "target", "category", "esc10", "src_file", "take"]
    )
    writer.writeheader()
    for fold in range(1, 6):
        for target in range(50):
            for i in range(8):
                source = fold * 10000 + target * 8 + i
                writer.writerow(
                    {
                        "filename": f"{fold}-{source}-A-{target}.wav",
                        "fold": str(fold),
                        "target": str(target),
                        "category": f"class-{target}",
                        "esc10": "False",
                        "src_file": str(source),
                        "take": "A",
                    }
                )
    return output.getvalue().encode()


def test_esc_metadata_original_grouping_and_balance():
    result = service.esc_metadata(csv_data())
    assert len(result) == 2000
    assert {r["fold"] for r in result.values()} == set("12345")
    with pytest.raises(ValueError):
        service.esc_metadata(b"filename,fold\n")
    with pytest.raises(ValueError):
        service.esc_metadata(csv_data().replace(b"1-10000-A-0.wav", b"../escape.wav"))
    with pytest.raises(ValueError):
        service.esc_metadata(csv_data().replace(b"class-0,False,10000,A", b"class-0,False,20000,A"))
    with pytest.raises(ValueError):
        service.esc_metadata(csv_data().replace(b"-A-0.wav,1,0,", b"-A-0.wav,1,1,"))


def test_media_decode_limits_and_pcm_integrity():
    assert service.media_details(wav(), "ESC-50")["frames"] == 220500
    for data in (wav(rate=8000), wav(frames=10), wav()[:-2]):
        with pytest.raises(ValueError):
            service.media_details(data, "ESC-50")
    for data in (png(mode="L"), png(size=(32, 32))):
        with pytest.raises(ValueError):
            service.media_details(data, "DIV2K")


class Response(io.BytesIO):
    status = 200
    headers = {}


@pytest.mark.parametrize(
    "fault", ["ok", "status", "declared", "stream", "truncated", "empty", "deadline"]
)
def test_fetch_boundaries(tmp_path, monkeypatch, fault):
    response = Response(b"abcd")
    if fault == "status":
        response.status = 206
    elif fault == "declared":
        response.headers = {"Content-Length": "5"}
    elif fault == "stream":
        response = Response(b"abcde")
    elif fault == "truncated":
        response = Response(b"ab")
        response.headers = {"Content-Length": "4"}
    elif fault == "empty":
        response = Response()

    class Opener:
        def open(self, url, timeout):
            assert timeout == 15
            return response

    monkeypatch.setattr(service.urllib.request, "build_opener", lambda *a: Opener())

    def deadline():
        if fault == "deadline":
            raise ValueError("deadline")

    if fault == "ok":
        result = service.fetch("https://fixed.test", tmp_path / "out", 4, deadline)
        assert result["bytes"] == 4 and (tmp_path / "out").read_bytes() == b"abcd"
    else:
        with pytest.raises(ValueError):
            service.fetch("https://fixed.test", tmp_path / "out", 4, deadline)
    assert service.NoRedirect().redirect_request(None, None, None, None, None, None) is None
