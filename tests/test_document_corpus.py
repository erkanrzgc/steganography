import io
import json
import runpy
import stat
import sys
import zipfile
from pathlib import Path

import pytest

from core import document_corpus as corpus


def zipped(name="document.txt", contents=b"ordinary documentation", *, mode=0):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        member = zipfile.ZipInfo(name)
        member.external_attr = mode << 16
        archive.writestr(member, contents)
    return stream.getvalue()


@pytest.fixture
def config(tmp_path):
    license_path = tmp_path / "license.txt"
    license_path.write_text("Fixture authors. MIT license; independent generated test data.")
    records = []
    for i, (extension, raw) in enumerate([
        (".pdf", b"%PDF-1.4\n% generated parser fixture\n%%EOF\n"),
        (".txt", "Persian joiner: \u200c; ordinary emoji: 🙂\n".encode()),
        (".md", b"# A natural-looking generated fixture\n\nDocumentation.\n"),
        (".zip", zipped()),
    ]):
        path = tmp_path / f"source-{i}{extension}"
        path.write_bytes(raw)
        records.append({"input": str(path), "sha256": corpus.digest(raw),
                        "license_input": str(license_path),
                        "license_sha256": corpus.digest(license_path.read_bytes()),
                        "license_review": "declared_file_scope_reviewed", "license": "MIT",
                        "source_id": f"fixture-v1:{i}", "source_group": "generated-fixture"})
    return {"schema_version": "local-document-pilot-config-v1", "documents": records}


def test_prepare_scalar_oracle_and_public_manifest(config, tmp_path):
    out = tmp_path / "out"
    report = corpus.prepare(config, out)
    assert report == json.loads((out / "manifest.json").read_text())
    assert report["original_files"] == 4 and report["source_groups"] == 1
    assert not report["training_ready"] and report["optimizer_updates"] == 0
    assert report["statistical_detection"] == "unavailable"
    assert not report["test_media_opened"] and report["downloaded_bytes"] == 0
    assert str(tmp_path) not in json.dumps(report)
    for sample, record in zip(report["samples"], config["documents"], strict=True):
        variants = sample["variants"]
        original = (out / variants["original"]["artifact"]).read_bytes()
        assert original == Path(record["input"]).read_bytes()
        positive = (out / variants["controlled_positive"]["artifact"]).read_bytes()
        negative = (out / variants["whitespace_control"]["artifact"]).read_bytes()
        assert len(positive) == len(negative)
        assert positive[:len(original)] == negative[:len(original)] == original
        tail = positive[len(original):]
        if sample["extension"] in (".md", ".txt"):
            assert len(tail) == 768
            recovered = bytes(int("".join(str(ord(c) - 0x200B) for c in tail.decode()[i:i+8]), 2)
                              for i in range(0, 256, 8))
        else:
            recovered = tail
            assert len(recovered) == 32
        assert corpus.digest(recovered) == sample["payload_sha256"]
        assert negative[len(original):] == b" " * len(tail)
    assert report["artifact_bytes_excluding_manifest"] == sum(
        p.stat().st_size for p in out.iterdir() if p.name != "manifest.json")


@pytest.mark.parametrize("extension,raw", [
    (".txt", b"\0bad"), (".md", b" \n"), (".txt", b"\xff"),
    (".pdf", b"wrong %%EOF"), (".pdf", b"%PDF-1.4 no end"),
    (".zip", b"broken archive"), (".mp3", b"not decoded"),
    (".zip", zipped("../escape")), (".zip", zipped("/absolute")),
    (".zip", zipped("a\\..\\escape")),
    (".zip", zipped(mode=stat.S_IFLNK | 0o777)),
])
def test_inventory_rejects_malformed_unsafe(extension, raw):
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        corpus.inspect(raw, extension)


def test_archive_member_size_and_encryption_bounds(monkeypatch):
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(ValueError):
        corpus.inspect(empty.getvalue(), ".zip")
    monkeypatch.setattr(corpus, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError):
        corpus.inspect(zipped(), ".zip")
    monkeypatch.setattr(corpus, "MAX_OUTPUT", 1024)
    many = io.BytesIO()
    with zipfile.ZipFile(many, "w") as archive:
        for i in range(513):
            archive.writestr(str(i), b"")
    with pytest.raises(ValueError):
        corpus.inspect(many.getvalue(), ".zip")
    # ZIP encryption bit in both local and central headers; never extract it.
    raw = bytearray(zipped())
    raw[6] |= 1
    raw[raw.index(b"PK\x01\x02") + 8] |= 1
    with pytest.raises(ValueError):
        corpus.inspect(bytes(raw), ".zip")


@pytest.mark.parametrize("field,value", [
    ("schema_version", "wrong"), ("documents", []), ("documents", None),
    ("sha256", "0" * 64), ("license_sha256", "0" * 64),
    ("license_review", "unreviewed"), ("source_id", "/home/secret"),
    ("source_group", "../train"), ("license", "MIT\nsecret"), ("source_id", 12),
])
def test_configuration_and_identity_rejection(config, tmp_path, field, value):
    if field in ("schema_version", "documents"):
        config[field] = value
    else:
        config["documents"][0][field] = value
    with pytest.raises(ValueError):
        corpus.prepare(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicates_count_and_aggregate_bounds(config, tmp_path, monkeypatch):
    config["documents"].append(config["documents"][0])
    with pytest.raises(ValueError, match="duplicate"):
        corpus.prepare(config, tmp_path / "duplicate")
    config["documents"].pop()
    monkeypatch.setattr(corpus, "MAX_DOCUMENTS", 1)
    with pytest.raises(ValueError, match="configuration"):
        corpus.prepare(config, tmp_path / "count")
    monkeypatch.setattr(corpus, "MAX_DOCUMENTS", 64)
    monkeypatch.setattr(corpus, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError, match="aggregate"):
        corpus.prepare(config, tmp_path / "bytes")


def test_manifest_budget(config, tmp_path, monkeypatch):
    report = corpus.prepare(config, tmp_path / "first")
    monkeypatch.setattr(corpus, "MAX_OUTPUT", report["artifact_bytes_excluding_manifest"] + 1)
    with pytest.raises(ValueError, match="manifest/output"):
        corpus.prepare(config, tmp_path / "second")


@pytest.mark.parametrize("kind", ["empty", "large", "directory", "fifo", "symlink",
                                 "parent_symlink"])
def test_bounded_regular_reader(tmp_path, kind):
    target = tmp_path / "input"
    if kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        corpus.os.mkfifo(target)
    elif kind == "parent_symlink":
        source = tmp_path / "actual"
        source.mkdir()
        (source / "input").write_bytes(b"x")
        target = tmp_path / "link"
        target.symlink_to(source, target_is_directory=True)
        target = target / "input"
    elif kind == "symlink":
        target.symlink_to(tmp_path / "missing")
    else:
        target.write_bytes(b"" if kind == "empty" else b"12345")
    with pytest.raises((OSError, ValueError)):
        corpus.regular_read(target, 4)


def test_changed_reader_size(tmp_path, monkeypatch):
    target = tmp_path / "input"
    target.write_bytes(b"original")
    original = corpus.os.fstat
    def truncated(fd):
        info = original(fd)
        return type("Stat", (), {"st_mode": info.st_mode, "st_size": info.st_size - 1})()
    with monkeypatch.context() as scoped:
        scoped.setattr(corpus.os, "fstat", truncated)
        with pytest.raises(ValueError, match="changed"):
            corpus.regular_read(target, 100)


@pytest.mark.parametrize("kind", ["existing", "dangling_symlink", "parent_symlink", "inside"])
def test_output_protection(config, tmp_path, kind):
    out = tmp_path / "out"
    if kind == "existing":
        out.mkdir()
    elif kind == "dangling_symlink":
        out.symlink_to(tmp_path / "missing")
    elif kind == "parent_symlink":
        out.symlink_to(tmp_path, target_is_directory=True)
        out = out / "subdir"
    else:
        # A source under the proposed output is rejected before I/O.
        config["documents"][0]["input"] = str(out / "source.pdf")
    with pytest.raises(ValueError):
        corpus.prepare(config, out)


@pytest.mark.parametrize("raw,method", [(b"short", "opaque-append32-v1"),
                                        (b"bad", "unknown"),
                                        (b"bad", "unicode-suffix-bitpair-v1"),
                                        (b"x" * 256, "unicode-suffix-bitpair-v1")])
def test_oracle_rejects_extent_and_alphabet(raw, method):
    with pytest.raises(ValueError):
        corpus.recover(raw, 0, method)


def test_deadline(config, tmp_path):
    times = iter([0, 120])
    with pytest.raises(TimeoutError):
        corpus.prepare(config, tmp_path / "out", clock=lambda: next(times))


@pytest.mark.parametrize("stage", ["encoder", "source", "output", "disk_oracle", "late_source"])
def test_mutation_fails_without_success_manifest(config, tmp_path, monkeypatch, stage):
    out = tmp_path / "out"
    original = corpus.regular_read
    counts = {}
    if stage == "encoder":
        monkeypatch.setattr(corpus, "recover", lambda *a: b"not expected")
    else:
        def changed(path, limit):
            raw = original(path, limit)
            counts[path] = counts.get(path, 0) + 1
            is_source = path == Path(config["documents"][0]["input"])
            if stage == "source" and is_source and counts[path] == 2:
                return raw + b"changed"
            if stage == "late_source" and is_source and counts[path] == 3:
                return raw + b"changed"
            if stage == "output" and path.parent == out:
                return raw + b"changed"
            if (stage == "disk_oracle" and path.parent == out
                    and "controlled_positive" in path.name and counts[path] == 2):
                return raw + b"changed"
            return raw
        monkeypatch.setattr(corpus, "regular_read", changed)
    with pytest.raises(ValueError):
        corpus.prepare(config, out)
    assert not (out / "manifest.json").exists()


def test_exclusive_writer_and_symlink(tmp_path):
    path = tmp_path / "artifact"
    corpus._put(path, b"first")
    with pytest.raises(FileExistsError):
        corpus._put(path, b"second")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        corpus._put(link, b"second")
    assert path.read_bytes() == b"first"


def test_cli(config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["research_documents", "--config", str(path),
                                    "--out", str(tmp_path / "out")])
    runpy.run_module("steganography.research_documents", run_name="__main__")
    assert capsys.readouterr().out.strip() == "prepared_unassigned"
