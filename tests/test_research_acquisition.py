import io
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import research_acquisition as acq
from steganography import research_media_pilot as frontend
from tests.test_nonjpeg_features import image_bytes, wav_bytes


def gif_bytes():
    out = io.BytesIO()
    Image.new("P", (12, 12)).save(out, format="GIF", comment=b"ordinary note", loop=0, duration=100)
    return out.getvalue()


def mp3_bytes():
    return (b"\xff\xfb\x90\0" + bytes(413)) * 2


def page(family, data, index=1):
    return {
        "pageid": index,
        "index": index,
        "title": f"File:native{index}.{family}",
        "imageinfo": [
            {
                "size": len(data),
                "url": f"https://upload.wikimedia.org/native{index}.{family}",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/File:native{index}.{family}",
                "extmetadata": {
                    "LicenseShortName": {"value": "CC BY-SA 4.0"},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                    "Artist": {"value": "<b>Example &amp; author</b>"},
                },
            }
        ],
    }


@pytest.fixture
def offline(monkeypatch):
    data = {
        "png": image_bytes(),
        "gif": gif_bytes(),
        "wav": wav_bytes(np.arange(320, dtype="<i2"), channels=2),
        "mp3": mp3_bytes(),
    }
    monkeypatch.setattr(acq, "TARGETS", dict.fromkeys(data, 1))
    monkeypatch.setattr(
        acq, "UNICODE_FILES", {"WordBreakTest.txt": "https://www.unicode.org/17.0.txt"}
    )
    monkeypatch.setattr(acq.time, "sleep", lambda _: None)
    calls = []

    def fetch(self, url, limit=acq.MAX_FILE):
        calls.append(url)
        if "api.php" in url:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            family = next(
                k for k, v in acq.MIMES.items() if query["gsrsearch"] == [f"filemime:{v}"]
            )
            result = json.dumps({"query": {"pages": {"1": page(family, data[family])}}}).encode()
        elif "unicode.org" in url:
            result = b"# WordBreakTest-17.0.0.txt\n\xc3\xb7 0041 \xc3\x97 200D \xc3\xb7\n"
        else:
            result = data[Path(urllib.parse.urlsplit(url).path).suffix[1:]]
        self.bytes += len(result)
        return result

    monkeypatch.setattr(acq.Budget, "fetch", fetch)
    return data, calls


def test_offline_acquisition_entrypoint_and_safe_outputs(tmp_path, monkeypatch, capsys, offline):
    data, calls = offline
    out = tmp_path / "pilot"
    result = acq.acquire(out)
    assert all(v["status"] == "ready" for v in result["families"].values())
    assert len(calls) == 9 and result["downloaded_bytes"] > sum(map(len, data.values()))
    assert str(tmp_path) not in (out / "report.json").read_text()
    for family, raw in data.items():
        row = result["families"][family]["samples"][0]
        assert row["original_sha256"] == acq.digest(raw)
        assert row["author"] == "Example & author" and row["split"] == "development_unassigned"
        assert json.loads((out / f"{family}-000.{family}.source.json").read_text()) == row
    with pytest.raises(FileExistsError):
        acq.acquire(out)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        acq.acquire(link / "pilot2")
    with pytest.raises(ValueError, match="symlink"):
        acq.write_json(link / "anything", {})
    monkeypatch.setattr(frontend, "acquire", lambda *a, **kw: result)
    monkeypatch.setattr(sys, "argv", ["pilot", "--out", str(tmp_path / "front"), "--unicode-only"])
    frontend.main()
    assert "png" in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["metadata", "original"])
def test_rate_limit_stops_without_retry_and_preserves_unicode(
    tmp_path, monkeypatch, offline, stage
):
    original = acq.Budget.fetch

    def limited(self, url, *args):
        if "unicode.org" not in url and (("api.php" in url) == (stage == "metadata")):
            raise urllib.error.HTTPError(url, 429, "rate limited", {}, None)
        return original(self, url, *args)

    monkeypatch.setattr(acq.Budget, "fetch", limited)
    result = acq.acquire(tmp_path / "out")
    assert result["families"]["png"]["failed_checks"] == {"http_429": 1}
    assert all(v["status"] == "unavailable" for v in result["families"].values())
    assert result["unicode"]["WordBreakTest.txt"]["status"] == "ready"


@pytest.mark.parametrize("kind", ["empty", "network", "badunicode", "unicode_network"])
def test_missing_sources_never_pass(tmp_path, monkeypatch, offline, kind):
    original = acq.Budget.fetch

    def failed(self, url, *args):
        if "api.php" in url:
            if kind == "network":
                raise OSError("offline")
            return b"{}"
        if "unicode.org" in url:
            if kind == "unicode_network":
                raise OSError("offline")
            if kind == "badunicode":
                return b"# 16.0.0\n"
        return original(self, url, *args)

    monkeypatch.setattr(acq.Budget, "fetch", failed)
    result = acq.acquire(tmp_path / "out")
    assert all(v["count"] == 0 and v["status"] != "ready" for v in result["families"].values())
    assert result["unicode"]["WordBreakTest.txt"]["status"] == (
        "unavailable" if "unicode" in kind else "ready"
    )


def test_duplicate_author_limits_and_rejections(tmp_path, monkeypatch, offline):
    data, _ = offline
    monkeypatch.setattr(acq, "TARGETS", {"png": 4})
    original = acq.Budget.fetch

    def candidates(self, url, *args):
        if "api.php" in url:
            pages = {str(i): page("png", data["png"], i) for i in range(1, 5)}
            pages["4"]["imageinfo"][0]["size"] += 1
            return json.dumps({"query": {"pages": pages}}).encode()
        return original(self, url, *args)

    monkeypatch.setattr(acq.Budget, "fetch", candidates)
    cell = acq.acquire(tmp_path / "dup")["families"]["png"]
    assert cell["count"] == 1 and cell["failed_checks"]["duplicate_carrier"] > 0
    assert cell["failed_checks"]["source_rejected_or_unavailable"] > 0

    def distinct(self, url, *args):
        if "api.php" in url:
            pages = {
                str(i): page("png", image_bytes(np.full((256, 256, 3), i, dtype=np.uint8)), i)
                for i in range(1, 5)
            }
            return json.dumps({"query": {"pages": pages}}).encode()
        if "upload.wikimedia" in url:
            i = int(Path(url).stem.replace("native", ""))
            return image_bytes(np.full((256, 256, 3), i, dtype=np.uint8))
        return original(self, url, *args)

    monkeypatch.setattr(acq.Budget, "fetch", distinct)
    cell = acq.acquire(tmp_path / "author")["families"]["png"]
    assert cell["count"] == 2 and cell["failed_checks"]["author_diversity_limit"] > 0


def test_metadata_contract():
    row = page("png", image_bytes())
    assert acq.metadata(row, "png")["author"] == "Example & author"
    row["imageinfo"][0]["extmetadata"] = {"LicenseShortName": {"value": "Public domain"}}
    assert acq.metadata(row, "png")["author"] is None
    with pytest.raises(ValueError, match="extension"):
        acq.metadata(row, "gif")
    row["imageinfo"][0]["size"] = acq.MAX_FILE + 1
    with pytest.raises(ValueError, match="byte"):
        acq.metadata(row, "png")
    row["imageinfo"][0]["extmetadata"] = {}
    with pytest.raises(ValueError, match="license"):
        acq.metadata(row, "png")


@pytest.mark.parametrize(
    "url",
    [
        "http://www.unicode.org/x",
        "file:///etc/passwd",
        "https://evil.test/x",
        "https://www.unicode.org:444/x",
        "https://user@www.unicode.org/x",
    ],
)
def test_fixed_https_hosts(url):
    with pytest.raises(ValueError, match="HTTPS"):
        acq.check_url(url)
    with pytest.raises(ValueError, match="HTTPS"):
        acq.SafeRedirect().redirect_request(None, None, 302, "", {}, url)


class Response(io.BytesIO):
    def __init__(self, data, *, size=None, status=200):
        super().__init__(data)
        self.headers = {} if size is None else {"Content-Length": str(size)}
        self.status, self.url = status, "https://www.unicode.org/x"


@pytest.mark.parametrize(
    "data,size,status,limit,error",
    [
        (b"abcd", 4, 200, 4, None),
        (b"abcd", None, 200, 4, None),
        (b"", None, 200, 4, "empty"),
        (b"ab", 3, 200, 4, "truncated"),
        (b"abcde", 5, 200, 4, "size"),
        (b"abcde", None, 200, 4, "byte bound"),
        (b"ab", 2, 503, 4, "response"),
    ],
)
def test_streaming_limits(monkeypatch, data, size, status, limit, error):
    class Opener:
        def open(self, *args, **kwargs):
            return Response(data, size=size, status=status)

    monkeypatch.setattr(acq.urllib.request, "build_opener", lambda *a: Opener())
    budget = acq.Budget()
    if error:
        with pytest.raises(ValueError, match=error):
            budget.fetch("https://www.unicode.org/x", limit)
    else:
        assert budget.fetch("https://www.unicode.org/x", limit) == data
        assert budget.bytes == len(data)
    monkeypatch.setattr(acq, "MAX_DOWNLOAD", 1)
    with pytest.raises(ValueError, match="budget"):
        acq.Budget().fetch("https://www.unicode.org/x")
    budget.started -= acq.MAX_SECONDS
    with pytest.raises(TimeoutError, match="deadline"):
        budget.deadline()


def test_format_adversaries(monkeypatch):
    for data, family in (
        (image_bytes(fmt="BMP"), "png"),
        (image_bytes(mode="RGBA"), "png"),
        (wav_bytes(), "wav"),
        (wav_bytes(np.zeros(320, dtype="<i2"), channels=2), "wav"),
        (wav_bytes(channels=2)[:-3], "wav"),
    ):
        with pytest.raises(ValueError):
            acq.inspect(data, family)
    with pytest.raises(ValueError, match="unknown"):
        acq.inspect(b"", "aac")
    for raw in (
        b"ID3",
        b"ID3\x04\0\0\xff\0\0\0",
        b"bogus",
        b"\xff\xfb\x90\0",
        (b"\xff\xff\x90\0" + bytes(413)) * 2,
        (b"\xff\xfb\x00\0" + bytes(413)) * 2,
    ):
        with pytest.raises(ValueError):
            acq.inspect(raw, "mp3")
    assert acq.inspect(b"ID3\x04\0\0\0\0\0\0" + mp3_bytes(), "mp3")["native_layer3_headers"] == 2
    original = Image.open

    def enormous(*args, **kwargs):
        image = original(*args, **kwargs)
        image._n_frames = 257
        return image

    monkeypatch.setattr(Image, "open", enormous)
    with pytest.raises(ValueError, match="frame/pixel"):
        acq.inspect(gif_bytes(), "gif")
