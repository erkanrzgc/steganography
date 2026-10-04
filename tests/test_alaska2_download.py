import hashlib
import importlib.util
import io
import json
import stat
import struct
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

import pytest
from PIL import Image


def downloader():
    spec = importlib.util.spec_from_file_location(
        "alaska_download", Path(__file__).parents[1] / "scripts/fetch-alaska2-pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jpeg(color, size=(16, 16)):
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="JPEG", quality=95)
    return data.getvalue()


def archive_bytes(compression=zipfile.ZIP_DEFLATED, *, dimension_mismatch=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for index in range(1, 5):
            for family_index, family in enumerate(("Cover", "JMiPOD", "JUNIWARD", "UERD")):
                size = (8, 8) if dimension_mismatch and family == "UERD" else (16, 16)
                archive.writestr(
                    f"{family}/{index:05d}.jpg", jpeg((index * 30, family_index * 40, 0), size)
                )
        archive.writestr("Test/00001.jpg", b"unlabeled")
        archive.writestr("../../escape.jpg", b"unsafe path")
        archive.writestr("Cover/99999.jpg", b"incomplete lineage")
    return buffer.getvalue()


class Response(io.BytesIO):
    def __init__(self, data=b"", *, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def setup(tmp_path, monkeypatch, **kwargs):
    module = downloader()
    blob = archive_bytes(**kwargs)
    credential = tmp_path / "credential.json"
    credential.write_text(json.dumps({"username": "fixture", "key": "fixture-secret"}))
    credential.chmod(0o600)
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64, "lineage": "b" * 64}]}))
    requests = []

    def request(req, timeout=30):
        requests.append(req)
        assert 0 < timeout <= 30
        if req.full_url == module.API_URL:
            assert req.get_header("Authorization").startswith("Basic ")
            raise urllib.error.HTTPError(
                req.full_url,
                302,
                "redirect",
                {"Location": "https://storage.googleapis.com/fixture.zip?signature=SECRET"},
                io.BytesIO(),
            )
        assert req.full_url.startswith("https://storage.googleapis.com/")
        assert not req.has_header("Authorization")
        headers = {"Content-Length": str(len(blob)), "ETag": '"immutable-fixture"'}
        if req.get_method() == "HEAD":
            return Response(headers=headers)
        assert req.get_header("If-match") == '"immutable-fixture"'
        left, right = map(int, req.get_header("Range").removeprefix("bytes=").split("-"))
        assert 0 <= left <= right < len(blob)
        data = blob[left : right + 1]
        headers.update(
            {"Content-Length": str(len(data)), "Content-Range": f"bytes {left}-{right}/{len(blob)}"}
        )
        return Response(data, status=206, headers=headers)

    monkeypatch.setattr(module, "open_request", request)
    return module, credential, reserved, blob, requests, request


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_acquisition_integrity_lineage_and_exclusive_outputs(tmp_path, monkeypatch, compression):
    module, credential, reserved, _, requests, _ = setup(
        tmp_path, monkeypatch, compression=compression
    )
    out = tmp_path / "holdout"
    manifest = module.acquire(out, credential, [reserved], count=3)
    assert len(manifest["samples"]) == 12
    assert manifest["unique_cover_hashes"] == 3
    assert manifest["upstream_crc32_verified"] and not manifest["upstream_sha256_verified"]
    plan = json.loads((out / "selection.json").read_text())
    assert plan["eligible_lineages"] == 4
    assert plan["selected_bytes"] == manifest["total_bytes"]
    assert len(list(out.rglob("*.jpg"))) == 12
    assert not (tmp_path / "escape.jpg").exists()
    assert not (out / "Test").exists()
    for index in range(0, 12, 4):
        cover = manifest["samples"][index]
        for record in manifest["samples"][index : index + 4]:
            data = (out / record["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
            assert record["lineage"] == cover["sha256"]
            assert record["split"] == "test"
            assert record["camera"] is None and record["payload_rate"] is None
    report = (out / "source.json").read_text() + (out / "selection.json").read_text()
    assert all(secret not in report for secret in ("SECRET", "fixture-secret", str(tmp_path)))
    for resume in (True, False):
        with pytest.raises(module.AcquisitionError, match="completed"):
            module.acquire(out, credential, [reserved], count=3, resume=resume)
    assert requests[0].full_url == module.API_URL


def test_failed_job_can_resume_but_not_overwrite_tampered_output(tmp_path, monkeypatch):
    module, credential, reserved, _, _, _ = setup(tmp_path, monkeypatch)
    original = module.validate_image
    failed = False

    def fail_once(data, member, hashes):
        nonlocal failed
        if member.filename.startswith("JMiPOD/") and not failed:
            failed = True
            raise module.AcquisitionError("injected validation failure")
        return original(data, member, hashes)

    monkeypatch.setattr(module, "validate_image", fail_once)
    out = tmp_path / "resume"
    with pytest.raises(module.AcquisitionError):
        module.acquire(out, credential, [reserved], count=4)
    assert (out / "selection.json").is_file() and not (out / "source.json").exists()
    monkeypatch.setattr(module, "validate_image", original)
    existing = next(out.rglob("*.jpg"))
    valid = existing.read_bytes()
    existing.write_bytes(b"tampered")
    with pytest.raises(module.AcquisitionError):
        module.acquire(out, credential, [reserved], count=4, resume=True)
    assert existing.read_bytes() == b"tampered"
    existing.write_bytes(valid)
    with pytest.raises(module.AcquisitionError, match="provenance"):
        module.acquire(out, credential, [reserved], count=3, resume=True)
    result = module.acquire(out, credential, [reserved], count=4, resume=True)
    assert len(result["samples"]) == 16


@pytest.mark.parametrize(
    "mutation", ["status", "content_range", "etag", "length", "short", "overlong"]
)
def test_range_integrity_guards(tmp_path, monkeypatch, mutation):
    module, _, _, blob, _, request = setup(tmp_path, monkeypatch)
    remote = module.RemoteZip("https://storage.googleapis.com/fixture")

    def bad(req, timeout=30):
        response = request(req, timeout)
        if mutation == "status":
            response.status = 200
            response.read = lambda *a: pytest.fail("must not consume full archive response")
        elif mutation == "content_range":
            response.headers["Content-Range"] = "bytes 1-2/3"
        elif mutation == "etag":
            response.headers["ETag"] = "changed"
        elif mutation == "length":
            response.headers["Content-Length"] = "99"
        else:
            response = Response(
                blob[:1] if mutation == "short" else blob[:3], status=206, headers=response.headers
            )
        return response

    monkeypatch.setattr(module, "open_request", bad)
    with pytest.raises(module.AcquisitionError):
        remote.range(0, 2)


def test_remote_seek_bounds_deadline_and_aggregate_budget(tmp_path, monkeypatch):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch)
    remote = module.RemoteZip("https://storage.googleapis.com/fixture")
    assert remote.seekable()
    assert remote.seek(2) == remote.tell() == 2
    assert remote.read(0) == b""
    assert remote.seek(2, 1) == 4
    assert remote.seek(-2, 2) == len(blob) - 2
    assert remote.read() == blob[-2:]
    for args in [(-1,), (len(blob) + 1,), (0, 3)]:
        with pytest.raises(module.AcquisitionError):
            remote.seek(*args)
    for args in [(-1, 1), (0, module.MAX_METADATA + 1), (len(blob), 1)]:
        with pytest.raises(module.AcquisitionError):
            remote.range(*args)
    remote.requested_bytes = module.MAX_TOTAL
    with pytest.raises(module.AcquisitionError, match="budget"):
        remote.range(0, 1)
    remote.deadline = 0
    with pytest.raises(module.AcquisitionError, match="deadline"):
        remote.range(0, 1)


@pytest.mark.parametrize("code", [403, 429, 500])
def test_safe_retry_limits(tmp_path, monkeypatch, code):
    module, _, _, _, _, _ = setup(tmp_path, monkeypatch)
    remote = module.RemoteZip("https://storage.googleapis.com/fixture?SECRET")
    attempts = []

    def failed(req, timeout=30):
        attempts.append(req)
        raise urllib.error.HTTPError(req.full_url, code, "SECRET", {}, io.BytesIO())

    monkeypatch.setattr(module, "open_request", failed)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    with pytest.raises(module.AcquisitionError) as error:
        remote.range(0, 1)
    assert "SECRET" not in str(error.value)
    assert len(attempts) == (1 if code == 403 else 3)


@pytest.mark.parametrize(
    "origin",
    [
        "http://storage.googleapis.com/x",
        "https://evil.test/x",
        "https://storage.googleapis.com.evil.test/x",
        "https://user@storage.googleapis.com/x",
        "https://storage.googleapis.com/x#fragment",
    ],
)
def test_credentials_never_follow_untrusted_redirect(tmp_path, monkeypatch, origin):
    module, credential, _, _, _, _ = setup(tmp_path, monkeypatch)

    def redirect(req, timeout=30):
        assert req.full_url == module.API_URL
        raise urllib.error.HTTPError(
            req.full_url, 302, "redirect", {"Location": origin}, io.BytesIO()
        )

    monkeypatch.setattr(module, "open_request", redirect)
    with pytest.raises(module.AcquisitionError, match="origin"):
        module.archive_url(credential)
    assert module.NoRedirect().redirect_request(None, None, None, None, None, None) is None


def test_credential_and_local_file_guards(tmp_path, monkeypatch):
    module, credential, reserved, _, _, _ = setup(tmp_path, monkeypatch)
    credential.chmod(0o644)
    with pytest.raises(module.AcquisitionError, match="private"):
        module.archive_url(credential)
    credential.chmod(0o600)
    credential.write_text('{"username":"x","key":""}')
    with pytest.raises(module.AcquisitionError, match="credential"):
        module.archive_url(credential)
    with pytest.raises(module.AcquisitionError, match="bounded"):
        module.bounded_read(credential, 1)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(module.AcquisitionError, match="symlink"):
        module.acquire(link / "child", credential, [reserved], count=1)
    with pytest.raises(module.AcquisitionError, match="existing acquisition"):
        module.acquire(tmp_path / "absent", credential, [reserved], count=1, resume=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "symlink",
        "encrypted",
        "method",
        "size",
        "compressed_size",
        "duplicate",
        "count",
        "member_limit",
        "missing",
        "budget",
    ],
)
def test_selection_rejects_unsafe_or_incomplete_metadata(tmp_path, monkeypatch, mutation):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = archive.infolist()
    item = members[0]
    count = 4
    if mutation == "symlink":
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
    elif mutation == "encrypted":
        item.flag_bits |= 1
    elif mutation == "method":
        item.compress_type = zipfile.ZIP_LZMA
    elif mutation == "size":
        item.file_size = module.MAX_FILE + 1
    elif mutation == "compressed_size":
        item.compress_size = module.MAX_FILE + 1
    elif mutation == "duplicate":
        members.append(item)
    elif mutation == "count":
        count = 0
    elif mutation == "member_limit":
        monkeypatch.setattr(module, "MAX_MEMBERS", 1)
    elif mutation == "missing":
        members.remove(item)
    else:
        monkeypatch.setattr(module, "MAX_TOTAL", 1)
    with pytest.raises(module.AcquisitionError):
        module.select_members(members, count)


@pytest.mark.parametrize(
    "mutation",
    ["signature", "name", "method", "size", "crc", "body", "short", "extra", "deflate_bomb"],
)
def test_bounded_local_member_parsing(tmp_path, monkeypatch, mutation):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        member = archive.infolist()[0]
    data = bytearray(blob)
    if mutation == "signature":
        data[:4] = b"EVIL"
    elif mutation == "name":
        data[30] = ord("X")
    elif mutation == "method":
        struct.pack_into("<H", data, 8, 99)
    elif mutation == "size":
        struct.pack_into("<I", data, 22, 99)
    elif mutation == "crc":
        struct.pack_into("<I", data, 14, 0)
    elif mutation == "extra":
        struct.pack_into("<H", data, 28, 5000)
    elif mutation == "short":
        data = data[:4]
    elif mutation == "deflate_bomb":
        member.file_size = 1
        struct.pack_into("<I", data, 22, 1)
    else:
        offset = 30 + len(member.filename)
        data[offset : offset + member.compress_size] = b"x" * member.compress_size

    class Remote:
        size = len(data)

        def range(self, start, length):
            return bytes(data[start : start + length])

    with pytest.raises((module.AcquisitionError, zlib.error)):
        module.member_bytes(Remote(), member)


def test_reserved_overlap_and_family_dimensions_fail_closed(tmp_path, monkeypatch):
    module, credential, reserved, _, _, _ = setup(tmp_path, monkeypatch, dimension_mismatch=True)
    out = tmp_path / "mismatch"
    with pytest.raises(module.AcquisitionError, match="dimensions"):
        module.acquire(out, credential, [reserved], count=4)
    assert not (out / "source.json").exists()
    reserved.write_text(
        json.dumps({"samples": [{"lineage": hashlib.sha256(jpeg((30, 0, 0))).hexdigest()}]})
    )
    with pytest.raises(module.AcquisitionError):
        module.acquire(tmp_path / "overlap", credential, [reserved], count=4)


def test_cli_redacts_unknown_errors_and_success(tmp_path, monkeypatch, capsys):
    module, credential, reserved, _, _, _ = setup(tmp_path, monkeypatch)
    arguments = [
        "--out",
        str(tmp_path / "cli"),
        "--credential",
        str(credential),
        "--reserved-manifest",
        str(reserved),
        "--count",
        "1",
    ]
    assert module.main(arguments) == 0

    def failure(*a, **kw):
        raise OSError("https://storage.googleapis.com/?signature=SECRET")

    monkeypatch.setattr(module, "acquire", failure)
    assert module.main(arguments) == 1
    assert "SECRET" not in capsys.readouterr().out


class LocalArchive(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.size = len(data)
        self.blob = data

    def range(self, start, length):
        assert 0 <= start <= start + length <= self.size
        return self.blob[start : start + length]


def zip64_archive(blob):
    _, _, _, count, _, size, offset, _ = struct.unpack("<4s4H2IH", blob[-22:])
    record_offset = len(blob) - 22
    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count, size, offset)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, record_offset, 1)
    end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return blob[:-22] + record + locator + end


def test_zip64_catalog_and_explicit_https_port(tmp_path, monkeypatch):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch)
    remote = LocalArchive(zip64_archive(blob))
    members = module.catalog_members(remote)
    assert len(members) == 19
    assert module.member_bytes(remote, members[0]) == jpeg((30, 0, 0))
    module.validate_archive_url("https://storage.googleapis.com:443/data?signature=SECRET")
    with pytest.raises(module.AcquisitionError):
        module.RemoteZip("file:///etc/passwd")


@pytest.mark.parametrize(
    "mutation",
    [
        "end",
        "comment",
        "count_limit",
        "size_limit",
        "count_mismatch",
        "locator",
        "record",
        "offset",
    ],
)
def test_catalog_preflight_prevents_unbounded_directory_parsing(tmp_path, monkeypatch, mutation):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch)
    data = bytearray(zip64_archive(blob))
    if mutation == "end":
        data[-22:-18] = b"FAIL"
    elif mutation == "comment":
        struct.pack_into("<H", data, len(data) - 2, 1)
    elif mutation == "count_limit":
        monkeypatch.setattr(module, "MAX_MEMBERS", 1)
    elif mutation == "size_limit":
        monkeypatch.setattr(module, "MAX_METADATA", 1)
    elif mutation == "count_mismatch":
        record_offset = len(blob) - 22
        struct.pack_into("<Q", data, record_offset + 24, 20)
        struct.pack_into("<Q", data, record_offset + 32, 20)
    elif mutation == "locator":
        data[-42:-38] = b"FAIL"
    elif mutation == "record":
        data[len(blob) - 22 : len(blob) - 18] = b"FAIL"
    else:
        struct.pack_into("<Q", data, len(data) - 34, len(data))
    with pytest.raises(module.AcquisitionError):
        module.catalog_members(LocalArchive(bytes(data)))


@pytest.mark.parametrize("failure", ["direct", "denied", "head_status", "head_etag"])
def test_handshake_fails_without_consuming_unbounded_bodies(tmp_path, monkeypatch, failure):
    module, credential, _, _, _, _ = setup(tmp_path, monkeypatch)

    def failed(req, timeout=30):
        if failure == "denied":
            raise urllib.error.HTTPError(req.full_url, 403, "SECRET", {}, io.BytesIO())
        response = Response(
            status=403 if failure == "head_status" else 200, headers={"Content-Length": "100"}
        )
        response.read = lambda *a: pytest.fail("unexpected response body read")
        return response

    monkeypatch.setattr(module, "open_request", failed)
    with pytest.raises(module.AcquisitionError) as error:
        if failure in ("direct", "denied"):
            module.archive_url(credential)
        else:
            module.RemoteZip("https://storage.googleapis.com/data")
    assert "SECRET" not in str(error.value)


def test_transport_retry_and_post_read_deadline(tmp_path, monkeypatch):
    module, _, _, _, _, request = setup(tmp_path, monkeypatch)
    remote = module.RemoteZip("https://storage.googleapis.com/data")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    attempts = []

    def failed(req, timeout=30):
        attempts.append(req)
        raise urllib.error.URLError("SECRET")

    monkeypatch.setattr(module, "open_request", failed)
    with pytest.raises(module.AcquisitionError, match="network"):
        remote.range(0, 1)
    assert len(attempts) == 3

    def late(req, timeout=30):
        response = request(req, timeout)
        remote.deadline = 0
        return response

    monkeypatch.setattr(module, "open_request", late)
    with pytest.raises(module.AcquisitionError, match="deadline"):
        remote.range(0, 1)


def test_truncated_crc_malformed_images_and_local_growth(tmp_path, monkeypatch):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch, compression=zipfile.ZIP_STORED)
    members = module.catalog_members(LocalArchive(blob))
    member = members[0]
    bad = bytearray(blob)
    bad[30 + len(member.filename) + 100] ^= 1
    with pytest.raises(module.AcquisitionError, match="CRC"):
        module.member_bytes(LocalArchive(bad), member)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    data = buffer.getvalue()
    member.file_size, member.CRC = len(data), zlib.crc32(data)
    with pytest.raises(module.AcquisitionError, match="format"):
        module.validate_image(data, member, set())
    local = tmp_path / "growing"
    local.write_bytes(b"x")
    original = Path.open

    def open_growing(path, *args, **kwargs):
        return io.BytesIO(b"xx") if path == local else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_growing)
    with pytest.raises(module.AcquisitionError, match="byte limit"):
        module.bounded_read(local, 1)


def test_opener_keeps_redirects_disabled(monkeypatch):
    module = downloader()
    marker = object()

    class Opener:
        def open(self, request, timeout):
            assert request is marker and timeout == 7
            return "response"

    def build(handler):
        assert handler is module.NoRedirect
        return Opener()

    monkeypatch.setattr(module.urllib.request, "build_opener", build)
    assert module.open_request(marker, timeout=7) == "response"


def test_reproducible_selection_does_not_depend_on_catalog_order(tmp_path, monkeypatch):
    module, _, _, blob, _, _ = setup(tmp_path, monkeypatch)
    members = module.catalog_members(LocalArchive(blob))
    chosen, count = module.select_members(members, 2)
    reordered, _ = module.select_members(list(reversed(members)), 2)
    assert count == 4
    assert [m.filename for m in chosen] == [m.filename for m in reordered]


def test_resume_rejects_artifact_symlink_and_disk_tampering(tmp_path, monkeypatch):
    module, credential, reserved, _, _, _ = setup(tmp_path, monkeypatch)
    original = module.validate_image
    out = tmp_path / "disk"

    def corrupt_after_validation(data, member, hashes):
        record = original(data, member, hashes)
        record["sha256"] = "0" * 64
        return record

    monkeypatch.setattr(module, "validate_image", corrupt_after_validation)
    with pytest.raises(module.AcquisitionError, match="written file"):
        module.acquire(out, credential, [reserved], count=1)
    monkeypatch.setattr(module, "validate_image", original)
    existing = next(out.rglob("*.jpg"))
    existing.unlink()
    existing.symlink_to(reserved)
    with pytest.raises(module.AcquisitionError):
        module.acquire(out, credential, [reserved], count=1, resume=True)
    assert not (out / "source.json").exists()
