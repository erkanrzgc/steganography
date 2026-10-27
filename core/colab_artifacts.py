"""Small bounded, exclusive artifacts for the explicit Colab preparation path."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

MIB = 1024**2


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(value: str, size: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", value):
        raise ValueError("invalid digest")
    return value


def safe(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink artifact")


def read(path: Path, limit: int) -> bytes:
    safe(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("nonregular artifact")
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError("artifact byte limit")
    return raw


def encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def document(path: Path, limit: int = MIB) -> dict:
    value = json.loads(read(path, limit))
    if not isinstance(value, dict):
        raise ValueError("artifact object required")
    return value


def put(path: Path, raw: bytes, limit: int = MIB) -> str:
    safe(path)
    if not raw or len(raw) > limit:
        raise ValueError("output byte limit")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def directory(path: Path) -> None:
    safe(path)
    path.mkdir()  # Existing attempts are consumed, including incomplete ones.
