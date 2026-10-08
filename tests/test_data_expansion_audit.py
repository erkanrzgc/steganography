"""Independent byte/ancestry audit fixtures, not acquisition implementations."""

import hashlib
import importlib.util
import io
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def module():
    spec = importlib.util.spec_from_file_location(
        "expansion_audit", Path(__file__).parents[1] / "scripts/audit-data-expansion.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = module()
    reserved = tmp_path / "reserved.json"
    reserved.write_text(json.dumps({"samples": [{"sha256": "a" * 64}]}))
    sources = []
    for kind, group in m.KINDS.items():
        folder = tmp_path / group
        folder.mkdir()
        selection = b'{"independent_fixture":true}\n'
        (folder / "selection.json").write_bytes(selection)
        doc = {
            "schema_version": kind,
            "purpose": "development",
            "selection_seed": 20261008,
            "license": "MIT" if group == "WIFD" else "local research only",
            "selection_sha256": hashlib.sha256(selection).hexdigest(),
            "samples": [],
        }
        families = ["Cover", "JMiPOD", "JUNIWARD", "UERD"] if group == "ALASKA2" else [None]
        lineage = None
        for i, family in enumerate(families):
            buf = io.BytesIO()
            mode = "L" if group == "BOSSbase-1.01" else "RGB"
            size = (512, 512) if group != "WIFD" else (256, 256)
            Image.new(mode, size, i * 30 + (100 if group == "WIFD" else 0)).save(
                buf, format="PPM" if mode == "L" else "JPEG"
            )
            raw = buf.getvalue()
            sha = hashlib.sha256(raw).hexdigest()
            lineage = lineage or sha
            name = (
                f"{family}/12345.jpg"
                if group == "ALASKA2"
                else "cover.pgm"
                if mode == "L"
                else "cover.jpg"
            )
            target = folder / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(raw)
            fraction = int(sha[:16], 16) / 2**64
            if group == "ALASKA2":
                fraction = (
                    int.from_bytes(hashlib.sha256(b"20261008:12345.jpg").digest()[:8], "big")
                    / 2**64
                )
            row = {
                "path": name,
                "sha256": sha,
                "lineage": lineage,
                "source_group": group,
                "size": len(raw),
                "width": size[0],
                "height": size[1],
                "method": family if family not in (None, "Cover") else None,
                "label": "stego" if family not in (None, "Cover") else "cover",
                "split": "test" if group == "WIFD" else "train" if fraction < 0.8 else "validation",
                "upstream_crc32": f"{zlib.crc32(raw):08x}",
            }
            if group == "WIFD":
                row.update(
                    camera="device",
                    device="device",
                    upstream_member="device/sdr_image/cover.jpg",
                    git_blob_sha1=blob(raw),
                    format="JPEG",
                    declared_frames=1,
                    decoded_frames=1,
                )
            doc["samples"].append(row)
        if group == "WIFD":
            license_raw = b"MIT License independent fixture"
            (folder / "LICENSE.upstream").write_bytes(license_raw)
            monkeypatch.setattr(m, "WIFD_LICENSE_BLOB", blob(license_raw))
            doc["license_sha256"] = hashlib.sha256(license_raw).hexdigest()
        doc["total_bytes"] = sum(r["size"] for r in doc["samples"])
        source = folder / "source.json"
        source.write_text(json.dumps(doc))
        sources.append(source)
    return m, sources, reserved


def mutate(source, change):
    doc = json.loads(source.read_bytes())
    change(doc)
    source.write_text(json.dumps(doc))


def test_complete_independent_audit_all_groups_reserved_and_portable(setup, tmp_path):
    m, sources, reserved = setup
    out = tmp_path / "audit.json"
    report = m.audit(sources, [reserved], out)
    assert (
        report["files"] == 6 and not report["detection_measured"] and not report["models_trained"]
    )
    assert str(tmp_path) not in out.read_text()
    assert report["sources"][2]["training_eligible_lineages_after_preparation"] == 0
    assert report["sources"][2]["native_formats"] == {"JPEG": 1}
    with pytest.raises(FileExistsError):
        m.audit(sources, [reserved], out)


def test_unchanged_stego_quarantines_whole_lineage_without_deleting_files(setup, tmp_path):
    m, sources, reserved = setup
    doc = json.loads(sources[0].read_bytes())
    cover, stego = doc["samples"][:2]
    old_size = stego["size"]
    raw = (sources[0].parent / cover["path"]).read_bytes()
    (sources[0].parent / stego["path"]).write_bytes(raw)
    stego.update(sha256=cover["sha256"], size=cover["size"], upstream_crc32=cover["upstream_crc32"])
    doc["total_bytes"] += len(raw) - old_size
    sources[0].write_text(json.dumps(doc))
    r = m.audit(sources, [reserved], tmp_path / "quarantine.json")["sources"][0]
    assert r["training_eligible_lineages_after_preparation"] == 0
    assert r["quarantined_lineages"][0]["unchanged_stego_methods"] == ["JMiPOD"]
    assert (sources[0].parent / stego["path"]).exists()


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "rows",
        "identity",
        "lineage",
        "hash",
        "size",
        "source",
        "crc",
        "dimensions",
        "format",
        "path",
        "duplicate_path",
        "labels",
        "split",
        "policy",
        "ancestry",
        "missing_family",
        "total",
        "selection",
        "overlap",
        "duplicate_manifest",
        "cross_acquisition",
    ],
)
def test_acquisition_forgery_and_leakage_have_no_success_record(setup, tmp_path, fault):
    m, sources, reserved = setup
    doc = json.loads(sources[0].read_bytes())
    row = doc["samples"][0]
    if fault == "schema":
        doc["schema_version"] = "unknown"
    if fault == "rows":
        doc["samples"] = []
    if fault == "identity":
        row["sha256"] = "invalid"
    if fault == "lineage":
        row["lineage"] = "invalid"
    if fault == "hash":
        row["sha256"] = "0" * 64
    if fault == "size":
        row["size"] += 1
    if fault == "source":
        row["source_group"] = "fake"
    if fault == "crc":
        row["upstream_crc32"] = "00000000"
    if fault == "dimensions":
        row["width"] += 1
    if fault == "format":
        buf = io.BytesIO()
        Image.new("RGB", (512, 512)).save(buf, format="PNG")
        raw = buf.getvalue()
        (sources[0].parent / row["path"]).write_bytes(raw)
        row.update(
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            upstream_crc32=f"{zlib.crc32(raw):08x}",
        )
    if fault == "path":
        row["path"] = "../escape.jpg"
    if fault == "duplicate_path":
        doc["samples"][1]["path"] = row["path"]
    if fault == "labels":
        row["label"] = "stego"
    if fault == "split":
        row["split"] = "test"
    if fault == "policy":
        doc["purpose"] = "ignored"
    if fault == "ancestry":
        row["lineage"] = "f" * 64
    if fault == "missing_family":
        doc["samples"].pop()
    if fault == "total":
        doc["total_bytes"] += 1
    if fault == "selection":
        doc["selection_sha256"] = "0" * 64
    if fault == "overlap":
        reserved.write_text(json.dumps({"samples": [{"sha256": row["sha256"]}]}))
    if fault == "duplicate_manifest":
        sources.append(sources[0])
    if fault == "cross_acquisition":
        alternate = sources[0].parent / "copy.json"
        alternate.write_text(json.dumps({**doc, "extra": True}))
        sources.append(alternate)
    sources[0].write_text(json.dumps(doc))
    out = tmp_path / "failure.json"
    with pytest.raises(ValueError):
        m.audit(sources, [reserved], out)
    assert not out.exists()


@pytest.mark.parametrize(
    "fault", ["ancestry", "format", "duplicate", "blob", "license", "device", "frames"]
)
def test_boss_and_wifd_native_license_and_ancestry_fail_closed(setup, tmp_path, fault):
    m, sources, reserved = setup
    index = 1 if fault in ("ancestry", "format", "duplicate") else 2

    def change(d):
        r = d["samples"][0]
        if fault == "ancestry":
            r["label"] = "stego"
        if fault == "format":
            r["width"] = 16
        if fault == "duplicate":
            target = sources[1].parent / "second.pgm"
            target.write_bytes((sources[1].parent / r["path"]).read_bytes())
            d["samples"].append({**r, "path": target.name})
        if fault == "blob":
            r["git_blob_sha1"] = "0" * 40
        if fault == "license":
            d["license_sha256"] = "0" * 64
        if fault == "device":
            r["device"] = "different"
        if fault == "frames":
            r["declared_frames"] = 2

    mutate(sources[index], change)
    with pytest.raises(ValueError):
        m.audit(sources, [reserved], tmp_path / "bad.json")


def test_bounded_read_symlinks_output_and_cli_redaction(setup, tmp_path, monkeypatch, capsys):
    import resource

    m, sources, reserved = setup
    with pytest.raises(ValueError):
        m.audit([], [reserved], tmp_path / "bad.json")
    link = tmp_path / "linked"
    link.symlink_to(reserved)
    with pytest.raises(ValueError):
        m.read(link, 1000)
    with pytest.raises(ValueError):
        m.read(reserved, 1)
    original = Path.stat
    monkeypatch.setattr(
        Path,
        "stat",
        lambda p, *a, **kw: SimpleNamespace(st_size=0, st_mode=original(p, *a, **kw).st_mode),
    )
    with pytest.raises(ValueError, match="grew"):
        m.read(reserved, 1)
    monkeypatch.undo()
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    args = [
        "--source",
        str(sources[0]),
        "--reserved",
        str(reserved),
        "--out",
        str(tmp_path / "cli.json"),
    ]
    assert m.main(args) == 0 and len(calls) == 4
    assert m.main(args) == 2
    assert str(tmp_path) not in capsys.readouterr().out


def test_mpo_primary_audit_preserves_native_format_and_never_claims_jpeg(setup, tmp_path):
    m, sources, reserved = setup
    doc = json.loads(sources[2].read_bytes())
    row = doc["samples"][0]
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "purple").save(
        buffer, format="MPO", save_all=True, append_images=[Image.new("RGB", (256, 256))]
    )
    raw = buffer.getvalue()
    (sources[2].parent / row["path"]).write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    row.update(
        sha256=sha,
        lineage=sha,
        size=len(raw),
        git_blob_sha1=blob(raw),
        upstream_crc32=f"{zlib.crc32(raw):08x}",
        format="MPO",
        declared_frames=2,
    )
    doc.update(total_bytes=len(raw), allow_bounded_mpo=True)
    sources[2].write_text(json.dumps(doc))
    r = m.audit(sources, [reserved], tmp_path / "native.json")
    assert r["sources"][2]["native_formats"] == {"MPO": 1}
    doc["allow_bounded_mpo"] = False
    sources[2].write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        m.audit(sources, [reserved], tmp_path / "strict.json")
