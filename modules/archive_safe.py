"""Bounded ZIP/TAR/EML structure inspection without extraction."""

from __future__ import annotations

import email
import itertools
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

_MAX_FILES = 10_000
_MAX_DEPTH = 20
_MAX_UNCOMPRESSED = 2 * 1024 * 1024 * 1024
_MAX_RATIO = 1_000


class ArchiveSafeAnalyzer(Analyzer):
    name = "archive_safe"

    def analyze(self, src: Path) -> AnalysisResult:
        suffix = src.suffix.lower()
        signals: list[Signal] = []
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as archive:
                members = [
                    (item.filename, item.file_size, item.compress_size)
                    for item in archive.infolist()
                ]
            signals.extend(_inspect_members(members))
        elif suffix in {".tar", ".tgz", ".gz", ".bz2", ".xz"}:
            try:
                with tarfile.open(src, mode="r:*") as archive:
                    members = [
                        (item.name, item.size, item.size)
                        for item in itertools.islice(archive, _MAX_FILES + 1)
                    ]
            except tarfile.TarError:
                return AnalysisResult(self.name, 0, (), None, status="unsupported")
            signals.extend(_inspect_members(members))
        elif suffix == ".eml":
            message = email.message_from_bytes(src.read_bytes())
            count = sum(1 for _ in message.walk())
            if count > 1000:
                signals.append(
                    Signal(
                        "excessive_mime_parts",
                        80,
                        f"EML contains {count} MIME parts",
                        category="archive_limits",
                        evidence="strong",
                    )
                )
        else:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        suspicion = max((signal.score for signal in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)


def _inspect_members(members: list[tuple[str, int, int]]) -> list[Signal]:
    signals: list[Signal] = []
    total = sum(max(0, size) for _, size, _ in members)
    compressed = sum(max(1, size) for _, _, size in members)
    unsafe = [name for name, _, _ in members if _unsafe_path(name)]
    depth = max(
        (len(PurePosixPath(name.replace("\\", "/")).parts) for name, _, _ in members), default=0
    )
    ratio = total / max(1, compressed)
    if unsafe:
        signals.append(
            Signal(
                "archive_path_traversal",
                95,
                f"{len(unsafe)} unsafe member paths",
                category="archive_path",
                evidence="strong",
            )
        )
    if len(members) > _MAX_FILES or depth > _MAX_DEPTH or total > _MAX_UNCOMPRESSED:
        signals.append(
            Signal(
                "archive_resource_limits",
                85,
                f"files={len(members)}, depth={depth}, uncompressed={total}",
                category="archive_limits",
                evidence="strong",
            )
        )
    if ratio > _MAX_RATIO:
        signals.append(
            Signal(
                "archive_compression_bomb",
                90,
                f"aggregate compression ratio={ratio:.1f}:1",
                category="archive_ratio",
                evidence="strong",
            )
        )
    return signals


def _unsafe_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    return (
        path.is_absolute() or ".." in path.parts or (len(normalized) >= 2 and normalized[1] == ":")
    )
