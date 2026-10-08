"""Checksum-bound train-only float blocks, never a whole-corpus tensor allocation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from core import jpeg_float256 as backend
from core.jpeg_scale import identity, safe_name
from core.srnet_sampling import _buckets
from core.srnet_scale_sampling import MAX_ROWS

MAX_DOCUMENT = 16 * 1024**2
MAX_BLOCKS = 32
ROW_BYTES = 256 * 256 * 4
CHUNK_BYTES = 1024**2
MAX_BYTES = 4 * 1024**3


def regular_open(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("stream inputs cannot use symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("stream input must be a regular file")
    return os.fdopen(fd, "rb")


def document(path, checksum):
    identity(checksum)
    with regular_open(path) as stream:
        raw = stream.read(MAX_DOCUMENT + 1)
    if len(raw) > MAX_DOCUMENT or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("stream metadata checksum/size mismatch")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("stream metadata must be an object")
    return result


def signature(stream):
    info = os.fstat(stream.fileno())
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class TrainBlocks:
    """Own open train cache handles. Use as a context manager; reads <=4 rows."""

    def __init__(self, root: Path, *, index_sha256, audit: Path, audit_sha256, deadline):
        if not callable(deadline):
            raise ValueError("stream requires a propagated deadline")
        self._stack = ExitStack()
        self._deadline = deadline
        self._closed = False
        self.samples: list[dict] = []
        self._locations: list[tuple] = []
        self.train_bytes = 0
        self.max_batch_bytes = 0
        try:
            self._load(root, index_sha256, audit, audit_sha256)
        except BaseException:
            self.close()
            raise

    def _load(self, root, index_sha256, audit_path, audit_sha256):
        self._deadline()
        index = document(root / "index.json", index_sha256)
        audit = document(audit_path, audit_sha256)
        blocks = index.get("blocks")
        if (
            index.get("schema_version") != "jpeg-scale-blocks-v1"
            or index.get("status") != "completed"
            or audit.get("schema_version") != "jpeg-scale-independent-audit-v1"
            or audit.get("status") != "completed"
            or audit.get("index_sha256") != index_sha256
            or audit.get("protocol_sha256") != index.get("protocol_sha256")
            or not isinstance(blocks, list)
            or not 1 <= len(blocks) <= MAX_BLOCKS
            or audit.get("blocks_audited") != len(blocks)
            or audit.get("reserved_identity_overlap") != 0
        ):
            raise ValueError("stream requires a complete bound independent audit")
        hashes, lineages, all_rows = set(), set(), []
        for block in blocks:
            self._deadline()
            safe_name(block["path"])
            manifest_path = root / block["path"]
            manifest = document(manifest_path, block["manifest_sha256"])
            rows = manifest.get("samples")
            if (
                manifest.get("schema_version") != "1.0"
                or not isinstance(rows, list)
                or not 1 <= len(rows) <= 768
            ):
                raise ValueError("stream block row limit/schema")
            local = set()
            for sample in rows:
                if not isinstance(sample, dict):
                    raise ValueError("stream sample must be an object")
                digest, lineage = identity(sample.get("sha256")), identity(sample.get("lineage"))
                if (
                    digest in hashes
                    or lineage in lineages
                    or sample.get("split") not in ("train", "validation")
                    or sample.get("source_group") != block["source_group"]
                ):
                    raise ValueError("stream duplicate, split or source mismatch")
                hashes.add(digest)
                local.add(lineage)
            if local != set(block["lineages"]) or len(local) != len(block["lineages"]):
                raise ValueError("stream incomplete lineage membership")
            lineages.update(local)
            all_rows.extend(rows)
            if len(all_rows) > MAX_ROWS:
                raise ValueError("stream corpus row limit")
            # Validate both roles' family metadata, never open validation pixels.
            roles: dict[str, str] = {}
            for row in rows:
                if roles.setdefault(row["lineage"], row["split"]) != row["split"]:
                    raise ValueError("stream cover lineage crosses roles")
            for split in ("train", "validation"):
                chosen = [r for r in rows if r["split"] == split]
                if chosen:
                    _buckets([{**r, "split": "train"} for r in chosen], row_limit=768)
            chosen = [r for r in rows if r["split"] == "train"]
            if chosen:
                self._cache(root, manifest_path, block, chosen)
        counts = dict(Counter(r["split"] for r in all_rows))
        if (
            not self.samples
            or counts != index.get("splits")
            or counts != audit.get("splits")
            or len(all_rows) != index.get("jpeg_rows")
            or len(all_rows) != audit.get("jpeg_rows")
            or len(lineages) != index.get("original_lineages")
            or len(lineages) != audit.get("original_lineages")
        ):
            raise ValueError("stream incomplete corpus accounting")
        self.index_sha256, self.audit_sha256 = index_sha256, audit_sha256
        self._deadline()

    def _cache(self, root, manifest_path, block, samples):
        binding = block["caches"]["train"]
        safe_name(binding["path"])
        cache_path = root / binding["path"]
        if cache_path != manifest_path.parent / "train/cache.json":
            raise ValueError("stream cache must belong to its block")
        cache = document(cache_path, binding["sha256"])
        if (
            cache.get("schema_version") != "research-float256-cache-v1"
            or cache.get("feature_version") != backend.FEATURE_VERSION
            or cache.get("manifest_sha256") != block["manifest_sha256"]
            or cache.get("split") != "train"
            or cache.get("dtype") != "<f4"
            or cache.get("shape") != [len(samples), 1, 256, 256]
            or cache.get("decoder") != backend.decoder_contract()
            or cache.get("data_sha256") != binding["data_sha256"]
            or binding["rows"] != len(samples)
            or not isinstance(cache.get("rows"), list)
            or len(cache["rows"]) != len(samples)
        ):
            raise ValueError("stream cache contract mismatch")
        for row, sample in zip(cache["rows"], samples, strict=True):
            size = row.get("image_size") if isinstance(row, dict) else None
            if (
                not isinstance(size, list)
                or len(size) != 2
                or any(type(n) is not int or n < 256 for n in size)
                or size[0] * size[1] > 4_000_000
                or row.get("region") != backend.crop_box(*size)
                or any(row.get(k) != sample[k] for k in ("sha256", "lineage", "label"))
            ):
                raise ValueError("stream pixel identity/region mismatch")
        stream = self._stack.enter_context(regular_open(cache_path.parent / "pixels.f32"))
        initial = signature(stream)
        size = len(samples) * ROW_BYTES
        self.train_bytes += size
        if initial[2] != size or self.train_bytes > MAX_BYTES:
            raise ValueError("stream tensor byte limit/length mismatch")
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            self._deadline()
            raw = stream.read(min(CHUNK_BYTES, remaining))
            if not raw:
                raise ValueError("stream tensor truncated")
            self._finite(raw)
            digest.update(raw)
            remaining -= len(raw)
        if digest.hexdigest() != cache["data_sha256"] or signature(stream) != initial:
            raise ValueError("stream tensor checksum/mutation mismatch")
        self.samples.extend(samples)
        self._locations.extend((stream, i * ROW_BYTES, initial) for i in range(len(samples)))

    @staticmethod
    def _finite(raw):
        values = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(values).all() or np.any(np.abs(values) > backend.MAX_MAGNITUDE):
            raise ValueError("stream tensor contains invalid values")

    def batch(self, indices):
        self._deadline()
        if (
            self._closed
            or not isinstance(indices, np.ndarray)
            or indices.dtype.kind not in "iu"
            or indices.ndim != 1
            or not 1 <= len(indices) <= 4
            or np.any(indices < 0)
            or np.any(indices >= len(self.samples))
        ):
            raise ValueError("stream batch indices outside limits")
        result = np.empty((len(indices), 1, 256, 256), dtype="<f4")
        for target, index in zip(result, indices, strict=True):
            self._deadline()
            stream, offset, initial = self._locations[int(index)]
            if signature(stream) != initial:
                raise ValueError("stream tensor mutated after verification")
            stream.seek(offset)
            raw = stream.read(ROW_BYTES)
            if len(raw) != ROW_BYTES or signature(stream) != initial:
                raise ValueError("stream row truncated or mutated")
            self._finite(raw)
            target[:] = np.frombuffer(raw, dtype="<f4").reshape(1, 256, 256)
        self.max_batch_bytes = max(self.max_batch_bytes, result.nbytes)
        return result

    def close(self):
        self._closed = True
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def deadline_after(seconds):
    if type(seconds) is not int or not 1 <= seconds <= 1800:
        raise ValueError("stream deadline outside limits")
    end = time.monotonic() + seconds

    def check():
        if time.monotonic() >= end:
            raise ValueError("stream job deadline exceeded")

    return check
