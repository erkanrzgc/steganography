"""Independent PGM/TAR fixtures and adversarial acquisition boundaries."""

import gzip
import hashlib
import io
import json
import tarfile

import pytest
from PIL import Image

from core import bows_dataset as b


def pgm(value=0, size=(512, 512), mode="L"):
    output = io.BytesIO()
    Image.new(mode, size, value).save(output, format="PPM")
    return output.getvalue()


def archive(entries=None):
    entries = entries or [("images/a.pgm", pgm(0), b"0"), ("images/b.pgm", pgm(1), b"0")]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.size, member.type = len(data), kind
            tar.addfile(member, io.BytesIO(data))
    return gzip.compress(output.getvalue())


@pytest.fixture
def reserved(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "COUNT", 2)
    path = tmp_path / "reserved.json"
    path.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    return [path]


def test_acquisition_complete_and_exact_lineage_split(tmp_path, reserved):
    out = tmp_path / "new"
    raw, page = archive(), b"source bows2-1g.tar.gz"
    result = b.acquire(out, reserved, archive=raw, page=page)
    assert result["status"] == "completed" and len(result["samples"]) == 2
    assert result["archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["source_page_sha256"] == hashlib.sha256(page).hexdigest()
    assert not result["model_trained"] and not result["upstream_sha256_verified"]
    assert result["accuracy_qualification"] == "unavailable"
    assert json.loads((out / "source.json").read_bytes()) == result
    for row in result["samples"]:
        data = (out / row["path"]).read_bytes()
        assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["lineage"]
        fraction = (
            int(hashlib.sha256(("bows2:20261010:" + row["sha256"]).encode()).hexdigest()[:16], 16)
            / 2**64
        )
        assert row["split"] == ("train" if fraction < 0.8 else "validation")
    with pytest.raises(FileExistsError):
        b.acquire(out, reserved, archive=raw, page=page)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape.pgm", b"0"),
        ("/absolute.pgm", b"0"),
        ("x\\bad.pgm", b"0"),
        ("link.pgm", b"2"),
        ("hard.pgm", b"1"),
        ("device", b"3"),
        ("pax", b"x"),
        ("long", b"L"),
        ("sparse", b"S"),
        ("run.sh", b"0"),
    ],
)
def test_unsafe_archive_fails_before_output(tmp_path, reserved, name, kind):
    out = tmp_path / "no-output"
    with pytest.raises(ValueError):
        b.acquire(out, reserved, archive=archive([(name, pgm(), kind)]), page=b"bows2-1g.tar.gz")
    assert not out.exists() and not (tmp_path / "escape.pgm").exists()


def test_duplicates_overlap_incomplete_and_geometry(reserved):
    for entries, previous in [
        ([("x.pgm", pgm(), b"0"), ("x.pgm", pgm(1), b"0")], set()),
        ([("x.pgm", pgm(), b"0"), ("y.pgm", pgm(), b"0")], set()),
        ([("x.pgm", pgm(), b"0")], {hashlib.sha256(pgm()).hexdigest()}),
        ([("x.pgm", pgm(), b"0")], set()),
        ([("x.pgm", pgm(size=(32, 32)), b"0")], set()),
        ([("x.pgm", pgm(mode="RGB"), b"0")], set()),
    ]:
        with pytest.raises(ValueError):
            b.inspect(archive(entries), previous, lambda: None)


def test_compressed_expanded_member_and_ending_limits(monkeypatch, reserved):
    raw = archive()
    monkeypatch.setattr(b, "MAX_ARCHIVE", len(raw) - 1)
    with pytest.raises(ValueError, match="compressed"):
        b.inspect(raw, set(), lambda: None)
    monkeypatch.setattr(b, "MAX_ARCHIVE", 200 * 1024**2)
    monkeypatch.setattr(b, "MAX_EXPANDED", 512)
    with pytest.raises(ValueError, match="expanded"):
        b.inspect(raw, set(), lambda: None)
    monkeypatch.setattr(b, "MAX_EXPANDED", 320 * 1024**2)
    monkeypatch.setattr(b, "MAX_FILE", 5)
    with pytest.raises(ValueError, match="count/size"):
        b.inspect(raw, set(), lambda: None)
    monkeypatch.setattr(b, "MAX_FILE", 300 * 1024)
    for changed in (gzip.decompress(raw)[:-10000], gzip.decompress(raw) + b"hidden"):
        with pytest.raises(ValueError):
            b.inspect(gzip.compress(changed), set(), lambda: None)


def test_source_and_prior_manifest_guards(tmp_path, reserved):
    with pytest.raises(ValueError):
        b.identities([])
    for document in ({}, {"samples": [{"sha256": "bad"}]}, {"samples": [{"lineage": "b" * 64}]}):
        reserved[0].write_text(json.dumps(document))
        with pytest.raises(ValueError):
            b.identities(reserved)
    with pytest.raises(ValueError):
        b.acquire(tmp_path / "out", reserved, archive=archive(), page=b"wrong")
    reserved[0].write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    with pytest.raises(ValueError, match="context"):
        b.acquire(tmp_path / "out", reserved, archive=archive(), page=b"wrong")
    with pytest.raises(ValueError, match="origin"):
        b.download("http://example.test", 10, lambda: None)


def test_no_symlink_or_oversize_document(tmp_path, reserved):
    symlink = tmp_path / "link"
    symlink.symlink_to(reserved[0])
    with pytest.raises(ValueError):
        b.read(symlink, 1000)
    with pytest.raises(ValueError):
        b.read(tmp_path, 1000)
    with pytest.raises(ValueError):
        b.read(reserved[0], 1)
    symlink.unlink()
    symlink.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        b.fresh(symlink / "child")


def test_fixed_download_transport_limits(monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}

    class Opener:
        response = Response(b"abc")

        def open(self, url, timeout):
            assert url == b.PAGE and timeout == 15
            return self.response

    opener = Opener()
    monkeypatch.setattr(b.urllib.request, "build_opener", lambda *args: opener)
    assert b.download(b.PAGE, 3, lambda: None) == b"abc"
    opener.response = Response(b"abcd")
    with pytest.raises(ValueError):
        b.download(b.PAGE, 3, lambda: None)
    opener.response = Response(b"")
    opener.response.status = 404
    with pytest.raises(ValueError):
        b.download(b.PAGE, 3, lambda: None)
    assert b.NoRedirect().redirect_request(None, None, None, None, None, None) is None


def test_directory_and_pixel_duplicate_with_distinct_headers(reserved):
    rows = b.inspect(
        archive(
            [("images", b"", b"5"), ("images/a.pgm", pgm(), b"0"), ("images/b.pgm", pgm(1), b"0")]
        ),
        set(),
        lambda: None,
    )
    assert len(rows) == 2
    commented = pgm().replace(b"P5\n", b"P5\n# different header\n", 1)
    with pytest.raises(ValueError, match="decoded"):
        b.inspect(
            archive([("a.pgm", pgm(), b"0"), ("b.pgm", commented, b"0")]), set(), lambda: None
        )


def test_missing_truncated_stream_and_deadline(monkeypatch, tmp_path, reserved):
    original = b.tarfile.TarFile.extractfile
    monkeypatch.setattr(b.tarfile.TarFile, "extractfile", lambda *a: None)
    with pytest.raises(ValueError, match="missing"):
        b.inspect(archive(), set(), lambda: None)
    monkeypatch.setattr(b.tarfile.TarFile, "extractfile", lambda *a: io.BytesIO(b"short"))
    with pytest.raises(ValueError, match="truncated"):
        b.inspect(archive(), set(), lambda: None)
    monkeypatch.setattr(b.tarfile.TarFile, "extractfile", original)
    clock = iter([0, 601])
    monkeypatch.setattr(b.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="deadline"):
        b.acquire(tmp_path / "late", reserved, archive=archive(), page=b"bows2-1g.tar.gz")
    assert not (tmp_path / "late").exists()


def test_native_mode_cannot_relax_checksum_or_download(tmp_path, reserved):
    with pytest.raises(ValueError, match="checksum"):
        b.inspect(archive(), set(), lambda: None, native=True)
    with pytest.raises(ValueError, match="cached archive"):
        b.acquire(tmp_path / "no-cache", reserved, page=b"bows2-1g.tar.gz", native=True)


def test_native_report_branch_with_explicit_stub(monkeypatch, tmp_path, reserved):
    # Report construction only; this stub is not live archive acceptance evidence.
    raw = archive()
    rows = b.inspect(raw, set(), lambda: None)
    monkeypatch.setattr(b, "inspect", lambda *a, **kw: rows)
    report = b.acquire(
        tmp_path / "native", reserved, archive=raw, page=b"bows2-1g.tar.gz", native=True
    )
    assert report["archive_layout"] == "verified-native-1001-v1"
