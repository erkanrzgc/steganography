"""Bounded, original-disjoint Q75/JUNIWARD study preparation, no downloads."""

from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from core import jpeg_float256
from core.jpeg_scale import identity, safe_name
from core.jpeg_timing import read_bytes, write_document
from core.srnet_stream import document, regular_open, signature

SCHEMA = "jpeg-learning-study-data-v1"
ORIGINS = ("BOSSbase-1.01", "BOWS2")
SEED = 20261012
MAX_BYTES = 5 * 1024**3
ROW_BYTES = 256 * 256 * 4
SIMULATION = "conseal-2025.11-juniward-bounded-entropy-refinement-v1"


def fresh(path):
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("study output exists or uses a symlink")


def originals(root, checksum, origin, reserved):
    source = document(root / "source.json", checksum)
    expected = "boss-development-acquisition-v1" if origin == ORIGINS[0] else (
        "bows2-original-acquisition-v1"
    )
    rows = source.get("samples")
    if (
        origin not in ORIGINS or source.get("schema_version") != expected
        or not isinstance(rows, list) or not 1 <= len(rows) <= 2000
        or not source.get("license") or not source.get("source_url")
        or (origin == ORIGINS[0] and source.get("purpose") != "development covers only")
        or (origin == ORIGINS[1] and source.get("status") != "completed")
    ):
        raise ValueError("study requires licensed development original manifests")
    required = source.get("reserved_manifest_sha256")
    if not isinstance(required, list) or not required or not set(required) <= set(reserved):
        raise ValueError("study reserved provenance is incomplete")
    forbidden = set().union(*(reserved[h] for h in required))
    seen = set()
    eligible = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("study original must be an object")
        digest = identity(row.get("sha256"))
        safe_name(row.get("path"))
        if (
            row.get("lineage") != digest or digest in seen or digest in forbidden
            or row.get("source_group") != origin
            or row.get("split") not in {"train", "validation"}
            or row.get("label", "cover") != "cover"
        ):
            raise ValueError("study original identity/role/reservation mismatch")
        seen.add(digest)
        if row["split"] == "train":
            eligible.append(row)
    return sorted(eligible, key=lambda r: hashlib.sha256(
        f"jpeg-study:{SEED}:{origin}:{r['sha256']}".encode()
    ).digest())


def reservations(bindings):
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 32:
        raise ValueError("study requires bounded reserved manifests")
    result = {}
    for entry in bindings:
        value = document(Path(entry["path"]), entry["sha256"])
        rows = value.get("samples")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 15000:
            raise ValueError("study reserved manifest is empty/oversized")
        identities = set()
        for row in rows:
            identities.add(identity(row.get("sha256")))
            if row.get("lineage") is not None:
                identities.add(identity(row["lineage"]))
        result[entry["sha256"]] = identities
    return result


def prepare(inputs, reserved, out, *, generator, deadline, fit=256, development=64):
    if (
        type(fit) is not int or not 2 <= fit <= 256
        or type(development) is not int or not 2 <= development <= 64
        or not isinstance(inputs, list) or len(inputs) != 2
        or {e["source_group"] for e in inputs} != set(ORIGINS)
    ):
        raise ValueError("study selection counts/sources outside limits")
    fresh(out)
    forbidden = reservations(reserved)
    selection: list[tuple[Path, dict, str]] = []
    for entry in inputs:
        deadline()
        root = Path(entry["root"])
        rows = originals(root, entry["sha256"], entry["source_group"], forbidden)
        if len(rows) < fit + development:
            raise ValueError("insufficient study train originals")
        selection.extend((root, row, "fit" if i < fit else "development")
                         for i, row in enumerate(rows[:fit + development]))
    # Verify all originals and decoded identities before creating any output.
    pixel_hashes, byte_hashes = set(), set()
    for root, row, _ in selection:
        deadline()
        raw = read_bytes(root / row["path"], row["sha256"], row.get("size", row.get("bytes")),
                         maximum=1024**2)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
                raise ValueError("study original must be 512x512 gray PGM")
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_hash in pixel_hashes or row["sha256"] in byte_hashes:
            raise ValueError("study original/decoded duplicate crosses selection")
        pixel_hashes.add(pixel_hash)
        byte_hashes.add(row["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "jpeg").mkdir()
    rows_by_role: dict[str, list] = {"fit": [], "development": []}
    tensor_hashes = {role: hashlib.sha256() for role in rows_by_role}
    used = len(selection) * 2 * ROW_BYTES
    streams = {}
    try:
        for role in rows_by_role:
            streams[role] = (out / f"{role}.f32").open("xb")
        for root, row, role in selection:
            deadline()
            raw = read_bytes(root / row["path"], row["sha256"], row.get("size", row.get("bytes")),
                             maximum=1024**2)
            pair = generator(raw, row["sha256"])
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("study simulation must return a cover/stego pair")
            pixels = jpeg_float256.pixel_batch(pair)
            if np.array_equal(pixels[0], pixels[1]):
                # Reject the whole preparation; never cherry-pick labels/regions.
                raise ValueError("study simulated pair has identical selected inputs")
            for label, data, tensor in zip(("cover", "stego"), pair, pixels, strict=True):
                used += len(data)
                if used > MAX_BYTES or not 0 < len(data) <= 256 * 1024:
                    raise ValueError("study output byte limit")
                digest = hashlib.sha256(data).hexdigest()
                name = f"jpeg/{digest}.jpg"
                with (out / name).open("xb") as stream:
                    stream.write(data)
                encoded = tensor.astype("<f4", copy=False).tobytes()
                streams[role].write(encoded)
                tensor_hashes[role].update(encoded)
                rows_by_role[role].append({
                    "path": name, "sha256": digest, "size": len(data),
                    "lineage": row["sha256"], "source_group": row["source_group"],
                    "role": role, "historical_role": "train", "label": label,
                    "method": None if label == "cover" else "JUNIWARD",
                    "quality_factor": 75, "payload_rate": None if label == "cover" else 0.4,
                    "payload_unit": "bpnzAC", "ground_truth": "embedding simulation",
                })
    finally:
        for stream in streams.values():
            stream.close()
    result = {
        "schema_version": SCHEMA, "status": "completed", "seed": SEED,
        "simulation_contract": SIMULATION,
        "source_manifests_sha256": sorted(e["sha256"] for e in inputs),
        "reserved_manifests_sha256": sorted(forbidden), "originals_per_source": {
            "fit": fit, "development": development,
        }, "samples": rows_by_role, "tensors": {
            role: {"path": f"{role}.f32", "rows": len(rows),
                   "sha256": tensor_hashes[role].hexdigest(), "bytes": len(rows) * ROW_BYTES}
            for role, rows in rows_by_role.items()
        }, "decoder": jpeg_float256.decoder_contract(), "output_bytes": used,
        "camera_independence_verified": False, "blind_test": False,
        "accuracy_qualification": "unavailable", "deployed": False,
    }
    write_document(out / "manifest.json", result)
    return result


def validate_manifest(manifest):
    if (
        manifest.get("schema_version") != SCHEMA or manifest.get("status") != "completed"
        or manifest.get("simulation_contract") != SIMULATION
        or set(manifest.get("samples", {})) != {"fit", "development"}
        or manifest.get("decoder") != jpeg_float256.decoder_contract()
    ):
        raise ValueError("study data contract/decoder mismatch")
    seen, roles, source_groups = set(), {}, defaultdict(set)
    for role, rows in manifest["samples"].items():
        if not isinstance(rows, list) or not 8 <= len(rows) <= 1024 or len(rows) % 4:
            raise ValueError("study role row bounds")
        for i in range(0, len(rows), 2):
            cover, stego = rows[i:i + 2]
            lineage = identity(cover.get("lineage"))
            if (
                cover.get("label") != "cover" or stego.get("label") != "stego"
                or cover.get("method") is not None or stego.get("method") != "JUNIWARD"
                or cover.get("payload_rate") is not None or stego.get("payload_rate") != 0.4
                or cover.get("source_group") not in ORIGINS
                or stego.get("lineage") != lineage
                or stego.get("source_group") != cover["source_group"]
                or lineage in roles
            ):
                raise ValueError("study pair ancestry/labels mismatch")
            roles[lineage] = role
            source_groups[role].add(cover["source_group"])
            for row in (cover, stego):
                digest = identity(row.get("sha256"))
                if (digest in seen or row.get("role") != role
                    or row.get("historical_role") != "train" or row.get("quality_factor") != 75
                    or row.get("path") != f"jpeg/{digest}.jpg"
                    or type(row.get("size")) is not int or not 0 < row["size"] <= 256 * 1024):
                    raise ValueError("study sample identity/role mismatch")
                seen.add(digest)
        if source_groups[role] != set(ORIGINS):
            raise ValueError("study role requires both declared origins")
        counts = [sum(r["source_group"] == origin for r in rows) // 2 for origin in ORIGINS]
        if counts != [manifest["originals_per_source"][role]] * 2:
            raise ValueError("study original source accounting mismatch")


class Reader:
    """Only opens one assigned-role tensor, never loads a whole corpus."""

    def __init__(self, root, checksum, role, *, deadline):
        deadline()
        if role not in {"fit", "development"}:
            raise ValueError("study reader role invalid")
        self.manifest = document(root / "manifest.json", checksum)
        validate_manifest(self.manifest)
        self.samples = self.manifest["samples"][role]
        self.deadline = deadline
        tensor = self.manifest["tensors"][role]
        identity(tensor["sha256"])
        if (tensor["path"] != f"{role}.f32" or type(tensor["rows"]) is not int
            or type(tensor["bytes"]) is not int or tensor["rows"] != len(self.samples)
            or tensor["bytes"] != len(self.samples) * ROW_BYTES):
            raise ValueError("study tensor identity/accounting mismatch")
        self.sha256 = tensor["sha256"]
        self.stream = regular_open(root / tensor["path"])
        self.initial = signature(self.stream)
        try:
            if self.initial[2] != tensor["bytes"]:
                raise ValueError("study tensor length mismatch")
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        self.deadline()
        if signature(self.stream) != self.initial:
            raise ValueError("study tensor changed")
        self.stream.seek(0)
        digest = hashlib.sha256()
        while raw := self.stream.read(1024**2):
            self.deadline()
            values = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(values).all() or np.any(np.abs(values) > 2**36):
                raise ValueError("study tensor nonfinite/out of bounds")
            digest.update(raw)
        if digest.hexdigest() != self.sha256 or signature(self.stream) != self.initial:
            raise ValueError("study tensor checksum/mutation mismatch")

    def batch(self, indices):
        self.deadline()
        if (not isinstance(indices, np.ndarray) or indices.ndim != 1
            or indices.dtype.kind not in "iu" or not 1 <= len(indices) <= 32
            or np.any(indices < 0) or np.any(indices >= len(self.samples))
            or signature(self.stream) != self.initial):
            raise ValueError("study batch bounds/state mismatch")
        result = np.empty((len(indices), 1, 256, 256), dtype="<f4")
        for target, i in zip(result, indices, strict=True):
            self.deadline()
            self.stream.seek(int(i) * ROW_BYTES)
            raw = self.stream.read(ROW_BYTES)
            if len(raw) != ROW_BYTES or signature(self.stream) != self.initial:
                raise ValueError("study row truncated/changed")
            target[:] = np.frombuffer(raw, dtype="<f4").reshape(1, 256, 256)
        return result

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
