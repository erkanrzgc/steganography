"""Tests for ArchiveSafeAnalyzer bounded inspection and archive anomaly detection."""

from __future__ import annotations

import io
import tarfile
import zipfile
from email.message import EmailMessage
from pathlib import Path

from modules.archive_safe import (
    ArchiveSafeAnalyzer,
    _inspect_members,
    _unsafe_path,
)


def test_unsafe_path_cases() -> None:
    assert _unsafe_path("../evil.txt")
    assert _unsafe_path("/etc/passwd")
    assert _unsafe_path("C:/Windows/System32/cmd.exe")
    assert _unsafe_path("d:\\autoexec.bat")
    assert not _unsafe_path("safe/path/file.txt")
    assert not _unsafe_path("safe_file.png")


def test_clean_zip_archive(tmp_path: Path) -> None:
    zip_path = tmp_path / "clean.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file1.txt", "hello")
        zf.writestr("dir/file2.txt", "world")

    analyzer = ArchiveSafeAnalyzer()
    res = analyzer.analyze(zip_path)
    assert res.status == "ok"
    assert res.suspicion == 0
    assert len(res.signals) == 0


def test_zip_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../etc/shadow", "root:x:0:0")

    analyzer = ArchiveSafeAnalyzer()
    res = analyzer.analyze(zip_path)
    signals = {s.name: s for s in res.signals}
    assert "archive_path_traversal" in signals
    assert signals["archive_path_traversal"].score == 95
    assert res.suspicion == 95


def test_inspect_members_resource_limits_and_compression_bomb() -> None:
    # 1. Resource limits: excessive depth
    deep_path = "/".join(f"d{i}" for i in range(25)) + "/leaf.txt"
    signals_depth = _inspect_members([(deep_path, 100, 100)])
    assert any(s.name == "archive_resource_limits" for s in signals_depth)

    # 2. Resource limits: excessive total size
    signals_size = _inspect_members([("huge.bin", 3 * 1024 * 1024 * 1024, 1000)])
    assert any(s.name == "archive_resource_limits" for s in signals_size)
    assert any(s.name == "archive_compression_bomb" for s in signals_size)


def test_tar_safe_and_unsupported(tmp_path: Path) -> None:
    # 1. Clean TAR archive
    tar_path = tmp_path / "clean.tar"
    with tarfile.open(tar_path, "w") as tf:
        data = b"hello from tar"
        info = tarfile.TarInfo(name="safe.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    analyzer = ArchiveSafeAnalyzer()
    res = analyzer.analyze(tar_path)
    assert res.status == "ok"
    assert res.suspicion == 0

    # 2. Traversal in TAR
    tar_traversal = tmp_path / "traversal.tar.gz"
    with tarfile.open(tar_traversal, "w:gz") as tf:
        data = b"secret"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    res_trav = analyzer.analyze(tar_traversal)
    assert any(s.name == "archive_path_traversal" for s in res_trav.signals)

    # 3. Corrupt TAR archive
    corrupt_tar = tmp_path / "corrupt.tar"
    corrupt_tar.write_bytes(b"not a tar file at all")
    res_corrupt = analyzer.analyze(corrupt_tar)
    assert res_corrupt.status == "unsupported"


def test_eml_safe_and_excessive(tmp_path: Path) -> None:
    analyzer = ArchiveSafeAnalyzer()

    # 1. Normal EML
    normal_eml = tmp_path / "normal.eml"
    msg = EmailMessage()
    msg["Subject"] = "Test"
    msg["From"] = "alice@example.com"
    msg["To"] = "bob@example.com"
    msg.set_content("Normal email body")
    normal_eml.write_bytes(msg.as_bytes())

    res = analyzer.analyze(normal_eml)
    assert res.status == "ok"
    assert res.suspicion == 0

    # 2. Excessive MIME parts
    excessive_eml = tmp_path / "excessive.eml"
    boundary = "bndry123"
    parts = [f"--{boundary}\nContent-Type: text/plain\n\npart {i}" for i in range(1005)]
    body = (
        f"Subject: Many parts\nContent-Type: multipart/mixed; boundary=\"{boundary}\"\n\n"
        + "\n".join(parts)
        + f"\n--{boundary}--\n"
    )
    excessive_eml.write_bytes(body.encode("ascii"))
    res_excessive = analyzer.analyze(excessive_eml)
    assert any(s.name == "excessive_mime_parts" for s in res_excessive.signals)
    assert res_excessive.suspicion == 80

    # 3. Unsupported extension
    unsupported = tmp_path / "test.unknown"
    unsupported.write_bytes(b"unknown format")
    assert analyzer.analyze(unsupported).status == "unsupported"
