"""Local, train-only three-origin timing kit; not a detector training corpus."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core import jpeg_float256, srnet_diversity_sampling
from core.jpeg_features import MAX_IMAGE_BYTES
from core.jpeg_scale import identity, safe_name
from core.srnet_sampling import _buckets
from core.srnet_stream import ROW_BYTES, TrainBlocks, document, regular_open

ORIGINS = ("ALASKA2", "BOSSbase-1.01", "BOWS2")
SEED = 20261010
MAX_OUTPUT = 512 * 1024**2


def select(samples, source, count):
    """Select complete original families by identity, never pixels or scores."""
    if source not in ORIGINS or type(count) is not int or not 2 <= count <= 32:
        raise ValueError("timing selection source/count outside bounds")
    if not isinstance(samples, list) or len(samples) > 15000:
        raise ValueError("timing selection row limit")
    groups: dict[str, list] = defaultdict(list)
    roles: dict[str, str] = {}
    digests = set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("timing selection requires object rows")
        lineage, digest = identity(row.get("lineage")), identity(row.get("sha256"))
        if row.get("source_group") != source or row.get("split") not in {"train", "validation"}:
            raise ValueError("timing selection source/role mismatch")
        if digest in digests or roles.setdefault(lineage, row["split"]) != row["split"]:
            raise ValueError("timing selection duplicate or cross-role lineage")
        digests.add(digest)
        if row["split"] == "train":
            groups[lineage].append(row)
    if source == "BOWS2":
        if any(len(rows) != 1 or rows[0]["sha256"] != key for key, rows in groups.items()):
            raise ValueError("timing BOWS originals must be unique original identities")
    else:
        for rows in groups.values():
            _buckets(rows, row_limit=6)
            expected = {None} if source == "ALASKA2" else {75, 95}
            if (
                len(rows) != 3 * len(expected)
                or {r.get("quality_factor") for r in rows} != expected
            ):
                raise ValueError("timing quality families mismatch")
    if len(groups) < count:
        raise ValueError("insufficient timing train originals")
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"timing-kit:{SEED}:{source}:{key}".encode()).digest(),
    )[:count]
    return [row for key in ordered for row in groups[key]]


def read_bytes(path, digest, size, *, maximum=MAX_IMAGE_BYTES):
    identity(digest)
    if type(size) is not int or not 0 < size <= maximum:
        raise ValueError("timing input byte limit")
    with regular_open(path) as stream:
        data = stream.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("timing input byte identity mismatch")
    return data


def write_document(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 16 * 1024**2:
        raise ValueError("timing metadata exceeds limits")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(
    root: Path,
    bows: Path,
    out: Path,
    *,
    protocol: Path,
    protocol_sha256,
    index_sha256,
    audit: Path,
    audit_sha256,
    bows_sha256,
    bows_audit: Path,
    bows_audit_sha256,
    deadline,
    generator,
):
    """Fresh private output, checksum-bound audits, bounded worker callback.

    The callback must execute the fixed simulation recipe in an isolated,
    timeout/resource-limited worker. No validation image/cache is read here.
    """
    deadline()
    frozen = document(protocol, protocol_sha256)
    binding = {
        "index_sha256": index_sha256,
        "audit_sha256": audit_sha256,
        "bows_sha256": bows_sha256,
        "bows_audit_sha256": bows_audit_sha256,
    }
    if (
        frozen.get("schema_version") != "jpeg-real-timing-protocol-v1"
        or frozen.get("inputs") != binding
        or frozen.get("originals_per_source") != 32
        or frozen.get("seed") != SEED
    ):
        raise ValueError("timing protocol/input binding mismatch")
    source = document(bows / "source.json", bows_sha256)
    proof = document(bows_audit, bows_audit_sha256)
    if (
        source.get("schema_version") != "bows2-original-acquisition-v1"
        or source.get("status") != "completed"
        or proof.get("schema_version") != "bows2-independent-original-audit-v1"
        or proof.get("status") != "completed"
        or proof.get("source_manifest_sha256") != bows_sha256
        or proof.get("exact_identity_overlap") != 0
        or proof.get("decoded_boss_pixel_overlap") != 0
        or proof.get("reserved_manifest_sha256") != source.get("reserved_manifest_sha256")
    ):
        raise ValueError("timing requires bound BOWS original audit")
    originals = select(source.get("samples"), "BOWS2", 32)
    selected = []
    with TrainBlocks(
        root,
        index_sha256=index_sha256,
        audit=audit,
        audit_sha256=audit_sha256,
        deadline=deadline,
    ) as reader:
        for origin in ORIGINS[:2]:
            selected.extend(
                select([r for r in reader.samples if r["source_group"] == origin], origin, 32)
            )
    if {r["lineage"] for r in originals} & {r["lineage"] for r in selected}:
        raise ValueError("timing cross-source original overlap")
    index = document(root / "index.json", index_sha256)
    paths = {}
    for block in index["blocks"]:
        safe_name(block["path"])
        manifest = document(root / block["path"], block["manifest_sha256"])
        for row in manifest["samples"]:
            safe_name(row["path"])
            paths[row["sha256"]] = (root / block["path"]).parent / row["path"]
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("timing output exists or uses symlink")
    out.mkdir(parents=True, exist_ok=False)
    # Reserve the full tensor; generated JPEGs also remain in simulation/.
    used = 480 * ROW_BYTES
    rows = []

    def publish(row, path, *, simulated=False):
        nonlocal used
        deadline()
        data = read_bytes(path, row["sha256"], row["size"])
        name = f"jpeg/{row['sha256']}.jpg"
        target = out / name
        target.parent.mkdir(exist_ok=True)
        used += len(data) * (2 if simulated else 1)
        if used > MAX_OUTPUT:
            raise ValueError("timing output budget exceeded")
        with target.open("xb") as stream:
            stream.write(data)
        rows.append({**row, "path": name})

    for row in selected:
        publish(row, paths[row["sha256"]])
    for original in originals:
        deadline()
        safe_name(original["path"])
        data = read_bytes(
            bows / original["path"], original["sha256"], original["bytes"], maximum=1024**2
        )
        generated = generator(data, out / "simulation" / original["lineage"], original["lineage"])
        if not isinstance(generated, list) or len(generated) != 6:
            raise ValueError("timing worker must return six rows")
        for row in generated:
            if (
                row.get("source_group") != "BOWS2"
                or row.get("lineage") != original["lineage"]
                or row.get("split") != "train"
                or row.get("payload_unit") != "bpnzAC"
                or row.get("payload_rate") != (None if row.get("label") == "cover" else 0.2)
                or type(row.get("coefficient_changes")) is not int
                or row["coefficient_changes"] < (0 if row.get("label") == "cover" else 1)
            ):
                raise ValueError("timing worker recipe mismatch")
            safe_name(row.get("path"))
            if Path(row["path"]).parts[0] != original["lineage"]:
                raise ValueError("timing worker lineage path mismatch")
            publish(
                {k: v for k, v in row.items() if k != "values"},
                out / "simulation" / row["path"],
                simulated=True,
            )
        _buckets(generated, row_limit=6)
        if {r.get("quality_factor") for r in generated} != {75, 95}:
            raise ValueError("timing worker quality mismatch")
    if len(rows) != 480 or len({r["sha256"] for r in rows}) != 480:
        raise ValueError("timing complete unique row accounting failed")
    batches, schedule = srnet_diversity_sampling.epoch_batches(rows, seed=SEED, epoch=0)
    if len(batches) != 192:
        raise ValueError("timing schedule accounting failed")
    tensor_sha = hashlib.sha256()
    geometry: list[dict] = []
    with (out / "pixels.f32").open("xb") as stream:
        for start in range(0, len(rows), 4):
            deadline()
            group = rows[start : start + 4]
            files = [read_bytes(out / r["path"], r["sha256"], r["size"]) for r in group]
            tensor = jpeg_float256.pixel_batch(files)
            raw = tensor.astype("<f4", copy=False).tobytes()
            if len(raw) != len(group) * ROW_BYTES or not np.isfinite(tensor).all():
                raise ValueError("timing decoder output mismatch")
            stream.write(raw)
            tensor_sha.update(raw)
            geometry.extend(jpeg_float256.region(data) for data in files)
    manifest = {
        "schema_version": "jpeg-real-timing-kit-v1",
        "status": "completed",
        "protocol_sha256": protocol_sha256,
        "inputs": binding,
        "samples": rows,
        "bows_originals": originals,
        "shape": [480, 1, 256, 256],
        "dtype": "<f4",
        "tensor_bytes": 480 * ROW_BYTES,
        "tensor_sha256": tensor_sha.hexdigest(),
        "decoder": jpeg_float256.decoder_contract(),
        "geometry": geometry,
        "schedule": schedule,
        "purpose": "train-only GPU timing, not accuracy evaluation",
        "full_corpus_optimizer_updates_per_epoch": 4932,
        "model_trained": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "redistribution": "private local use only; source licenses unchanged",
    }
    deadline()
    digest = write_document(out / "manifest.json", manifest)
    return {
        "status": "completed",
        "manifest_sha256": digest,
        "jpeg_rows": len(rows),
        "originals": 96,
        "timing_updates": len(batches),
        "tensor_bytes": 480 * ROW_BYTES,
        "real_model_trained": False,
        "accuracy_qualification": "unavailable",
    }
