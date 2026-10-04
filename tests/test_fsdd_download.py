import hashlib
import importlib.util
import io
import json
import stat
import struct
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def downloader():
    spec = importlib.util.spec_from_file_location(
        "fsdd_download", Path(__file__).parents[1] / "scripts/fetch-fsdd-pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav(channels=1, width=2, rate=8000, frames=80):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00" * channels * width * frames)
    return output.getvalue()


def blob(module, *, entries=None, readme=b"https://creativecommons.org/licenses/by-sa/4.0/"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if readme is not None:
            archive.writestr(module.PREFIX + "README.md", readme)
        for name, data in entries if entries is not None else [("recordings/0_alice_0.wav", wav())]:
            if isinstance(name, str):
                name = module.PREFIX + name
            archive.writestr(name, data)
        archive.writestr(module.PREFIX + "unused.py", b"raise SystemExit('must never execute')")
    return output.getvalue()


def test_acquire_hashes_provenance_and_no_overwrite(tmp_path, monkeypatch, capsys):
    module = downloader()
    data = blob(module)
    monkeypatch.setattr(module, "download", lambda: data)
    out = tmp_path / "corpus"
    module.main(["--out", str(out)])
    assert "no benchmark performed" in capsys.readouterr().out
    manifest = json.loads((out / "source.json").read_text())
    row = manifest["samples"][0]
    assert row["sha256"] == row["lineage"] == hashlib.sha256(wav()).hexdigest()
    assert row["speaker"] == "alice" and row["split"] is None
    assert manifest["archive_sha256"] == hashlib.sha256(data).hexdigest()
    assert not manifest["upstream_sha256_verified"]
    assert manifest["license"] == "CC-BY-SA-4.0"
    assert set(p.name for p in out.iterdir()) == {"0_alice_0.wav", "source.json"}
    assert str(tmp_path) not in (out / "source.json").read_text()
    with pytest.raises(FileExistsError):
        module.acquire(out, archive=data)
    link = tmp_path / "link"
    link.symlink_to(out, target_is_directory=True)
    for path in (link, link / "nested"):
        with pytest.raises(ValueError, match="symlink"):
            module.acquire(path, archive=data)


@pytest.mark.parametrize("readme", [None, b"", b"changed", b"x" * 32769])
def test_license_required_before_writing(tmp_path, readme):
    module = downloader()
    with pytest.raises(ValueError, match="license"):
        module.acquire(tmp_path / "out", archive=blob(module, readme=readme))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "kind", ["empty", "path", "symlink", "duplicate", "oversize", "method", "expanded"]
)
def test_archive_members_are_bounded(tmp_path, monkeypatch, kind):
    module = downloader()
    entries = [("recordings/0_alice_0.wav", wav())]
    if kind == "empty":
        entries = []
    elif kind == "path":
        entries = [("recordings/../../escape.wav", wav())]
    elif kind in ("symlink", "method"):
        info = zipfile.ZipInfo(module.PREFIX + "recordings/0_alice_0.wav")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16 if kind == "symlink" else 0
        info.compress_type = zipfile.ZIP_STORED if kind == "symlink" else zipfile.ZIP_BZIP2
        entries = [(info, wav())]
    elif kind == "duplicate":
        entries *= 2
    elif kind == "oversize":
        monkeypatch.setattr(module, "MAX_FILE", 100)
    elif kind == "expanded":
        entries = [("recordings/0_alice_0.wav", wav(frames=10000))]
        monkeypatch.setattr(module, "MAX_ARCHIVE", 10000)
    if kind == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate"):
            data = blob(module, entries=entries)
    else:
        data = blob(module, entries=entries)
    with pytest.raises(ValueError):
        module.acquire(tmp_path / "out", archive=data)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "escape.wav").exists()


@pytest.mark.parametrize("kind", ["garbage", "trailing", "disk", "count", "directory", "budget"])
def test_directory_preflight(tmp_path, monkeypatch, kind):
    module = downloader()
    data = bytearray(blob(module))
    offset = data.rfind(b"PK\x05\x06")
    if kind == "garbage":
        data = b"not zip"
    elif kind == "trailing":
        data += b"suffix"
    elif kind == "disk":
        struct.pack_into("<H", data, offset + 4, 1)
    elif kind == "count":
        struct.pack_into("<HH", data, offset + 8, 5000, 5000)
    elif kind == "directory":
        struct.pack_into("<I", data, offset + 12, 3 * 1024 * 1024)
    else:
        monkeypatch.setattr(module, "MAX_ARCHIVE", 10)
    with pytest.raises(ValueError):
        module.acquire(tmp_path / "out", archive=data)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "args", [{"channels": 2}, {"width": 1}, {"rate": 44100}, {"frames": 0}, {"frames": 80001}]
)
def test_pcm_contract(tmp_path, args):
    module = downloader()
    with pytest.raises(ValueError, match="PCM"):
        module.acquire(
            tmp_path / "out",
            archive=blob(module, entries=[("recordings/0_alice_0.wav", wav(**args))]),
        )
    assert not (tmp_path / "out/source.json").exists()


def test_truncated_pcm_and_corrupted_zip(tmp_path):
    module = downloader()
    data = wav()[:-2]
    with pytest.raises(ValueError, match="truncated"):
        module.acquire(
            tmp_path / "truncated",
            archive=blob(module, entries=[("recordings/0_alice_0.wav", data)]),
        )
    archive = bytearray(blob(module))
    offset = archive.find(b"PK\x03\x04", 4)
    name_len, extra_len = struct.unpack_from("<HH", archive, offset + 26)
    archive[offset + 30 + name_len + extra_len + 3] ^= 0x01
    with pytest.raises((zipfile.BadZipFile, ValueError)):
        module.acquire(tmp_path / "corrupt", archive=archive)


class Response(io.BytesIO):
    def __init__(self, data=b"", status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


@pytest.mark.parametrize("kind", ["ok", "status", "length", "stream", "time"])
def test_download_network_limits(monkeypatch, kind):
    module = downloader()
    monkeypatch.setattr(module, "MAX_ARCHIVE", 4)
    response = Response(b"abcd")
    if kind == "status":
        response.status = 206
    elif kind == "length":
        response.headers["Content-Length"] = "5"
    elif kind == "stream":
        response = Response(b"abcde")
    elif kind == "time":
        ticks = iter([0, 181])
        monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    def open_request(url, timeout):
        assert url == module.URL and timeout == 15 and "https://" in url
        return response

    monkeypatch.setattr(
        module.urllib.request, "build_opener", lambda cls: SimpleNamespace(open=open_request)
    )
    assert (
        module.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example")
        is None
    )
    if kind == "ok":
        assert module.download() == b"abcd"
    else:
        with pytest.raises(ValueError):
            module.download()


def test_cli_redacts_exception(tmp_path, monkeypatch, capsys):
    module = downloader()

    def fail(*_):
        raise ValueError(f"secret /host/path/{tmp_path}")

    monkeypatch.setattr(module, "acquire", fail)
    with pytest.raises(SystemExit) as error:
        module.main(["--out", str(tmp_path / "out")])
    assert error.value.code == 2
    assert (
        capsys.readouterr().err == "FSDD acquisition failed (ValueError); partial output retained\n"
    )
