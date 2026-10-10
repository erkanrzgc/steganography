"""Generated acquisition/security fixtures; no live data or accuracy claims."""

import csv
import hashlib
import importlib.util
import io
import json
import stat
import struct
import wave
import zipfile
from pathlib import Path

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
    raw = csv_data().replace(b"5-50399-A-49.wav,5,49,", b"5-50399-A-48.wav,5,48,")
    with pytest.raises(ValueError, match="balance"):
        service.esc_metadata(raw)


def test_grouped_metadata_requires_pin_and_never_promotes_test_to_train(monkeypatch):
    raw = csv_data().replace(
        b"5-50000-A-0.wav,5,0,class-0,False,50000,A", b"5-40000-A-0.wav,5,0,class-0,False,40000,A"
    )
    with pytest.raises(ValueError, match="checksum"):
        service.esc_metadata(raw, group_folds=True)
    monkeypatch.setattr(service, "ESC_GROUP_CSV_SHA", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="cross folds"):
        service.esc_metadata(raw)
    result = service.esc_metadata(raw, group_folds=True)
    assert result["4-40000-A-0.wav"]["fold"] == "4"
    assert result["4-40000-A-0.wav"]["assigned_group_fold"] == "5"
    assert result["5-40000-A-0.wav"]["assigned_group_fold"] == "5"
    assert result["1-10000-A-0.wav"]["assigned_group_fold"] == "1"


def test_group_retry_is_only_explicit_pinned_cached_esc(prepared, tmp_path, monkeypatch):
    prior, path = prepared
    for key, cached, enabled in [
        ("div2k-train", path, True),
        ("esc50", None, True),
        ("esc50", path, 1),
    ]:
        with pytest.raises(ValueError, match="explicit cached ESC"):
            service.acquire(
                key,
                tmp_path / "out",
                reserved_paths=[prior],
                archive_path=cached,
                esc_group_retry=enabled,
            )

    def fetch(url, out, limit, deadline):
        raw = b"creativecommons.org/licenses/by-nc/3.0/"
        out.write_bytes(raw)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    monkeypatch.setattr(service, "fetch", fetch)
    with pytest.raises(ValueError, match="archive checksum"):
        service.acquire(
            "esc50",
            tmp_path / "bad",
            reserved_paths=[prior],
            archive_path=path,
            esc_group_retry=True,
        )


@pytest.mark.parametrize(
    "fault", ["ok", "grouped", "metadata-size", "membership", "truncated-member"]
)
def test_esc_acquisition_control_flow_with_stub_metadata(prepared, tmp_path, monkeypatch, fault):
    # Deliberately tiny control-flow stub, not a real ESC corpus/parity audit.
    prior, _ = prepared
    monkeypatch.setitem(service.SOURCES, "esc50", {**service.SOURCES["esc50"], "count": 3})
    metadata, entries = {}, []
    for fold in (1, 4, 5):
        name = f"{fold}-{fold}-A-0.wav"
        metadata[name] = {
            "fold": str(fold),
            "src_file": str(fold),
            "category": "dog",
            "target": "0",
            "esc10": "True",
        }
        if fault == "grouped":
            metadata[name]["assigned_group_fold"] = str(fold)
        data = wav()[:-2] + struct.pack("<h", fold)
        entries.append((service.ESC_PREFIX + "audio/" + name, data))
    csv_raw = b"stub metadata"
    if fault == "metadata-size":
        csv_raw = b"x" * (service.MAX_DOCUMENT + 1)
    if fault == "membership":
        metadata.pop(next(iter(metadata)))
    entries.append((service.ESC_PREFIX + "meta/esc50.csv", csv_raw))
    path = archive(tmp_path / "esc.zip", entries)
    monkeypatch.setattr(service, "esc_metadata", lambda raw, **kw: metadata)
    if fault == "grouped":
        monkeypatch.setattr(
            service, "ESC_GROUP_ARCHIVE_SHA", hashlib.sha256(path.read_bytes()).hexdigest()
        )

    def fetch(url, out, limit, deadline):
        raw = b"http://creativecommons.org/licenses/by-nc/3.0/"
        out.write_bytes(raw)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    monkeypatch.setattr(service, "fetch", fetch)
    if fault == "truncated-member":
        original = zipfile.ZipFile.open

        def truncated(self, member, *args, **kwargs):
            if member.filename.endswith(".wav"):
                return io.BytesIO(b"truncated")
            return original(self, member, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "open", truncated)
    out = tmp_path / "esc-out"
    if fault not in ("ok", "grouped"):
        with pytest.raises(ValueError):
            service.acquire("esc50", out, reserved_paths=[prior], archive_path=path)
        assert not (out / "source.json").exists()
    else:
        result = service.acquire(
            "esc50",
            out,
            reserved_paths=[prior],
            archive_path=path,
            esc_group_retry=fault == "grouped",
        )
        assert result["splits"] == {"train": 1, "validation": 1, "test": 1}
        assert result["source_commit"] == service.ESC_COMMIT
        assert {r["group_key"] for r in result["samples"]} == {"ESC-50:1", "ESC-50:4", "ESC-50:5"}
        assert (out / "upstream-metadata.csv").read_bytes() == csv_raw
        if fault == "grouped":
            assert result["split_policy"] == "esc-original-group-max-fold-v1"
            assert {r["assigned_group_fold"] for r in result["samples"]} == {1, 4, 5}


def test_download_branch_and_deadline(prepared, tmp_path, monkeypatch):
    prior, path = prepared
    data = path.read_bytes()

    def fetch(url, out, limit, deadline):
        deadline()
        raw = data if url.endswith(".zip") else b"academic research purpose only"
        out.write_bytes(raw)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    monkeypatch.setattr(service, "fetch", fetch)
    result = service.acquire("div2k-train", tmp_path / "downloaded", reserved_paths=[prior])
    assert result["archive_bytes"] == len(data)
    clock = iter((0, service.MAX_SECONDS + 1))
    monkeypatch.setattr(service.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="deadline"):
        service.acquire("div2k-train", tmp_path / "late", reserved_paths=[prior])


def audit_module():
    path = Path(__file__).parents[1] / "scripts/audit-media-diversity.py"
    spec = importlib.util.spec_from_file_location("independent_media_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "fault",
    [
        "ok",
        "bytes",
        "decoded",
        "split",
        "prior-bytes",
        "prior-pixel-overlap",
        "archive",
        "evidence",
        "member",
        "path",
        "link",
        "duplicate",
        "count",
        "no-prior",
    ],
)
def test_independent_audit_rereads_bytes_pixels_roles_and_archives(
    prepared, tmp_path, monkeypatch, fault
):
    prior, archive_path = prepared
    old = png((190, 170, 140)) if fault != "prior-pixel-overlap" else png()
    (tmp_path / "old.png").write_bytes(old)
    digest = hashlib.sha256(old).hexdigest()
    # For the overlap case encode identical pixels with a different compression
    # level: acquisition's exact-byte guard passes; independent pixels must fail.
    if fault == "prior-pixel-overlap":
        output = io.BytesIO()
        Image.new("RGB", (256, 256), (20, 30, 40)).save(output, format="PNG", compress_level=0)
        old = output.getvalue()
        (tmp_path / "old.png").write_bytes(old)
        digest = hashlib.sha256(old).hexdigest()
    prior.write_text(
        json.dumps({"samples": [{"path": "old.png", "sha256": digest, "lineage": digest}]})
    )
    out = tmp_path / "new"
    service.acquire("div2k-train", out, reserved_paths=[prior], archive_path=archive_path)
    (out / "archive.zip").write_bytes(archive_path.read_bytes())
    manifest_path = out / "source.json"
    manifest = json.loads(manifest_path.read_bytes())
    if fault == "bytes":
        (out / "0001.png").write_bytes(b"changed")
    elif fault == "decoded":
        manifest["samples"][0]["decoded_sha256"] = "b" * 64
    elif fault == "split":
        manifest["samples"][0]["split"] = "test"
    elif fault == "prior-bytes":
        (tmp_path / "old.png").write_bytes(b"changed")
    elif fault == "archive":
        (out / "archive.zip").write_bytes(b"changed")
    elif fault == "evidence":
        (out / "license-evidence.txt").write_bytes(b"changed")
    elif fault == "member":
        manifest["samples"][0]["path"] = "9999.png"
    elif fault == "path":
        manifest["samples"][0]["path"] = "../old.png"
    elif fault == "link":
        (out / "0001.png").unlink()
        (out / "0001.png").symlink_to(tmp_path / "old.png")
    elif fault == "duplicate":
        manifest["samples"][1] = manifest["samples"][0]
    elif fault == "count":
        manifest["samples"].pop()
    manifest_path.write_text(json.dumps(manifest))
    module = audit_module()
    monkeypatch.setitem(module.EXPECTED, "div2k-train", (2, {"train": 2}))
    prior_paths = [] if fault == "no-prior" else [prior]
    target = tmp_path / "audit.json"
    if fault == "ok":
        result = module.audit([manifest_path], prior_paths, target)
        assert result["originals_audited"] == 2 and result["prior_covers_reread"] == 1
        assert result["prior_decoded_overlap"] == 0 and not result["model_trained"]
        assert str(tmp_path) not in target.read_text()
    else:
        with pytest.raises((ValueError, OSError)):
            module.audit([manifest_path], prior_paths, target)
        assert not target.exists()


def test_media_decode_limits_and_pcm_integrity():
    assert service.media_details(wav(), "ESC-50")["frames"] == 220500
    for data in (wav(rate=8000), wav(frames=10), wav()[:-2]):
        with pytest.raises(ValueError):
            service.media_details(data, "ESC-50")
    for data in (png(mode="L"), png(size=(32, 32))):
        with pytest.raises(ValueError):
            service.media_details(data, "DIV2K")


def test_only_exact_pinned_legacy_manifests_can_omit_original_lineage(
    prepared, tmp_path, monkeypatch
):
    prior, path = prepared
    prior.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    with pytest.raises(ValueError, match="prior original identity"):
        service.acquire(
            "div2k-train", tmp_path / "rejected", reserved_paths=[prior], archive_path=path
        )
    monkeypatch.setattr(
        service, "LEGACY_ORIGINAL_MANIFESTS", {hashlib.sha256(prior.read_bytes()).hexdigest()}
    )
    result = service.acquire(
        "div2k-train", tmp_path / "accepted", reserved_paths=[prior], archive_path=path
    )
    assert result["originals"] == 2


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
