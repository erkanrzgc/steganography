import importlib.util
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pytest


def load_downloader():
    path = Path(__file__).parents[1] / "scripts" / "fetch-boss-pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_range_download_resume_and_tamper_rejection(tmp_path, monkeypatch):
    module = load_downloader()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("../../one.pgm", b"P5\n1 1\n255\n\x01")
        archive.writestr("two.pgm", b"P5\n1 1\n255\n\x02")
    blob = stream.getvalue()

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Length": str(len(blob)), "ETag": '"fixture"'}

    def request(req, **kwargs):
        assert req.full_url == module.URL
        assert kwargs["timeout"] <= 60
        if req.get_method() == "HEAD":
            return Response(b"")
        left, right = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        return Response(blob[left : right + 1])

    monkeypatch.setattr(urllib.request, "urlopen", request)
    out = tmp_path / "samples"
    args = ["fetch", "--out", str(out), "--count", "2"]
    monkeypatch.setattr("sys.argv", args)
    module.main()
    manifest = json.loads((out / "source.json").read_text())
    assert len(manifest["samples"]) == 2
    assert (out / "0000.pgm").read_bytes().endswith(b"\x01")
    assert not (tmp_path / "one.pgm").exists()
    assert manifest["upstream_sha256_verified"] is False
    monkeypatch.setattr("sys.argv", [*args, "--resume"])
    module.main()
    (out / "0000.pgm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="CRC"):
        module.main()
    remote = module.RemoteZip()
    assert remote.seekable()
    assert remote.seek(0) == remote.tell() == 0
    assert remote.read(0) == b""
    with pytest.raises(ValueError, match="seek"):
        remote.seek(-1)
    remote.size = 400 * 1024 * 1024
    remote.seek(0)
    with pytest.raises(ValueError, match="budget"):
        remote.read()


def test_range_download_rejects_unbounded_response(tmp_path, monkeypatch):
    module = load_downloader()

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "100", "ETag": '"fixture"'}

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response(b"x" * 100))
    remote = module.RemoteZip()
    with pytest.raises(ValueError, match="range"):
        remote.read(1)
    Response.status = 206
    with pytest.raises(ValueError, match="incomplete"):
        remote.read(1)
    out = tmp_path / "link"
    out.symlink_to(tmp_path)
    monkeypatch.setattr("sys.argv", ["fetch", "--out", str(out)])
    with pytest.raises(ValueError, match="symlink"):
        module.main()
