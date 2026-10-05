import importlib.util
import io
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


def downloader():
    spec = importlib.util.spec_from_file_location(
        "boss_dev", Path(__file__).parents[1] / "scripts/fetch-boss-development.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, monkeypatch, *, kind="valid"):
    module = downloader()
    helper = module.helpers()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i in range(4):
            image = io.BytesIO()
            shape = (1, 1) if kind == "geometry" else (512, 512)
            Image.fromarray(np.full(shape, i, dtype=np.uint8)).save(image, format="PPM")
            info = zipfile.ZipInfo(f"images/{i}.pgm")
            if kind == "unsafe":
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, image.getvalue())
    data = buffer.getvalue()

    class Response(io.BytesIO):
        def __init__(self, body=b"", *, status=200, headers=None):
            super().__init__(body)
            self.status = status
            self.headers = headers or {}

    def request(req, timeout=30):
        assert req.full_url == module.URL and not req.has_header("Authorization")
        headers = {"Content-Length": str(len(data)), "ETag": '"fixture"'}
        if req.get_method() == "HEAD":
            return Response(headers=headers)
        left, right = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
        headers.update(
            {
                "Content-Range": f"bytes {left}-{right}/{len(data)}",
                "Content-Length": str(right - left + 1),
            }
        )
        return Response(data[left : right + 1], status=206, headers=headers)

    monkeypatch.setattr(helper, "open_request", request)
    monkeypatch.setattr(module, "helpers", lambda: helper)
    reserved = tmp_path / "reserved.json"
    reserved.write_text(
        json.dumps({"samples": [{"sha256": "a" * 64, "upstream_member": "images/0.pgm"}]})
    )
    return module, helper, reserved


def test_acquire_reserved_safe_resume_and_integrity(tmp_path, monkeypatch, capsys):
    module, _, reserved = fixture(tmp_path, monkeypatch)
    out = tmp_path / "data"
    result = module.acquire(out, [reserved], count=2)
    assert all(r["upstream_member"] != "images/0.pgm" for r in result["samples"])
    assert result["purpose"] == "development covers only"
    assert {r["split"] for r in result["samples"]} <= {"train", "validation"}
    assert str(tmp_path) not in json.dumps(result)
    with pytest.raises(FileExistsError):
        module.acquire(out, [reserved], count=2, resume=True)
    (out / "source.json").unlink()
    assert module.acquire(out, [reserved], count=2, resume=True)["samples"] == result["samples"]
    (out / "source.json").unlink()
    (out / "0000.pgm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="CRC"):
        module.acquire(out, [reserved], count=2, resume=True)
    with pytest.raises(ValueError, match="origin"):
        module.fixed_origin("https://other.example/archive")
    with pytest.raises(ValueError):
        module.acquire(tmp_path / "bad", [reserved], count=0)
    with pytest.raises(ValueError):
        module.acquire(tmp_path / "bad", [], count=2)
    monkeypatch.setattr(module, "acquire", lambda *a, **k: result)
    module.main(["--out", str(out), "--reserved-manifest", str(reserved), "--count", "2"])
    assert "no evaluation performed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kind",
    [
        "unsafe",
        "empty",
        "identity",
        "overlap",
        "selection",
        "insufficient",
        "duplicate",
        "size",
        "image",
        "geometry",
    ],
)
def test_archive_identity_and_resume_failures(tmp_path, monkeypatch, kind):
    module, helper, reserved = fixture(tmp_path, monkeypatch, kind=kind)
    if kind == "empty":
        reserved.write_text('{"samples":[]}')
    elif kind == "identity":
        reserved.write_text('{"samples":[{}]}')
    elif kind == "overlap":
        image = io.BytesIO()
        Image.fromarray(np.ones((512, 512), dtype=np.uint8)).save(image, format="PPM")
        import hashlib

        reserved.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sha256": hashlib.sha256(image.getvalue()).hexdigest(),
                            "upstream_member": "images/0.pgm",
                        }
                    ]
                }
            )
        )
        # Select all eligible members so the overlap must occur.
    elif kind == "selection":
        out = tmp_path / "data"
        module.acquire(out, [reserved], count=2)
        (out / "source.json").unlink()
        (out / "selection.json").write_text("{}")
    elif kind in ("duplicate", "size"):
        original = helper.catalog_members

        def members(remote):
            rows = original(remote)
            if kind == "duplicate":
                rows.append(rows[-1])
            else:
                for r in rows:
                    r.file_size = helper.MAX_FILE + 1
            return rows

        monkeypatch.setattr(helper, "catalog_members", members)
    elif kind == "image":
        monkeypatch.setattr(
            module, "Image", SimpleNamespace(open=lambda *_: Image.open(io.BytesIO(b"bad")))
        )
    with pytest.raises((ValueError, OSError)):
        module.acquire(
            tmp_path / "data",
            [reserved],
            count=4 if kind == "insufficient" else 3 if kind == "overlap" else 2,
            resume=kind == "selection",
        )
    assert not (tmp_path / "data/source.json").exists()


def test_cli_redacts_errors_and_symlinks(tmp_path, monkeypatch, capsys):
    module, _, reserved = fixture(tmp_path, monkeypatch)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.acquire(link / "out", [reserved], count=2)

    def fail(*a, **k):
        raise ValueError(f"secret {tmp_path}")

    monkeypatch.setattr(module, "acquire", fail)
    with pytest.raises(SystemExit) as error:
        module.main(["--out", str(tmp_path / "out"), "--reserved-manifest", str(reserved)])
    assert error.value.code == 2
    assert (
        capsys.readouterr().err == "BOSS development failed (ValueError); partial output retained\n"
    )
