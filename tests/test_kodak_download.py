import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image


def downloader():
    spec = importlib.util.spec_from_file_location(
        "kodak_download", Path(__file__).parents[1] / "scripts" / "fetch-kodak-pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_download(module, monkeypatch, *, size=(768, 512), status=200, redirect=False):
    class Response(io.BytesIO):
        headers = {"ETag": "fixture"}

        def geturl(self):
            return "https://invalid.test/" if redirect else self.url

    def get(url, timeout):
        assert url.startswith(module.SOURCE_URL + "kodak/kodim")
        assert timeout == 30
        index = int(url[-6:-4])
        buffer = io.BytesIO()
        Image.new("RGB", size, (index, 0, 0)).save(buffer, format="PNG")
        response = Response(buffer.getvalue())
        response.url = url
        response.status = status
        return response

    monkeypatch.setattr(module.urllib.request, "urlopen", get)
    return get


def reserved_file(tmp_path):
    path = tmp_path / "reserved.json"
    path.write_text(json.dumps({"samples": [{"sha256": "a" * 64, "lineage": "b" * 64}]}))
    return path


def test_kodak_download_all_hashes_and_exclusive_output(tmp_path, monkeypatch):
    module = downloader()
    fake_download(module, monkeypatch)
    reserved = reserved_file(tmp_path)
    out = tmp_path / "source"
    monkeypatch.setattr(
        "sys.argv", ["fetch", "--out", str(out), "--reserved-manifest", str(reserved)]
    )
    module.main()
    manifest = json.loads((out / "source.json").read_text())
    assert len(manifest["samples"]) == 24
    assert manifest["reserved_sha256_overlap"] == 0
    assert not manifest["upstream_sha256_verified"]
    for row in manifest["samples"]:
        assert hashlib.sha256((out / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    with pytest.raises(FileExistsError):
        module.fetch(out, reserved)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(ValueError, match="symlink"):
        module.fetch(link, reserved)


@pytest.mark.parametrize(
    "failure", ["size", "status", "redirect", "bytes", "reserved", "duplicate"]
)
def test_kodak_rejects_bad_or_overlapping_sources(tmp_path, monkeypatch, failure):
    module = downloader()
    get = fake_download(
        module,
        monkeypatch,
        size=(1, 1) if failure == "size" else (768, 512),
        status=500 if failure == "status" else 200,
        redirect=failure == "redirect",
    )
    reserved = reserved_file(tmp_path)
    if failure == "bytes":
        monkeypatch.setattr(module, "MAX_BYTES", 1)
    if failure == "reserved":
        data = get(module.SOURCE_URL + "kodak/kodim01.png", 30).getvalue()
        reserved.write_text(
            json.dumps({"samples": [{"lineage": hashlib.sha256(data).hexdigest()}]})
        )
    if failure == "duplicate":

        def same(url, timeout):
            response = get(module.SOURCE_URL + "kodak/kodim01.png", timeout)
            response.url = url
            return response

        monkeypatch.setattr(module.urllib.request, "urlopen", same)
    out = tmp_path / "bad-source"
    with pytest.raises(ValueError):
        module.fetch(out, reserved)
    assert not (out / "source.json").exists()
