"""Independent full JPEG/block/cache membership and scalar-IDCT replay audit."""

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

MAX_META = 64 * 1024**2
MAX_JPEG = 2 * 1024**2
ROW_BYTES = 256 * 256 * 4
PROTOCOL_SHA = "e494621f8d3eee763fe14461fafa81458b8404827fa554b2bf060ea7c5cb3036"


def regular(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("regular nonsymlink audit file required")


def read(path, limit):
    regular(path)
    if path.stat().st_size > limit:
        raise ValueError("audit byte limit exceeded")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("audit input grew beyond byte limit")
    return raw


def document(path, expected=None):
    raw = read(path, MAX_META)
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("audit document checksum mismatch")
    return json.loads(raw), digest


def relative(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("unsafe audit relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe audit relative path")
    return root / name


def scalar_replay(jpeg_path, tensor_path, position):
    """Four scalar sums, not the shared vector decoder or model inference."""
    import jpeglib

    jpeglib.version.set("6b")
    jpeg = jpeglib.read_dct(str(jpeg_path))
    jpeg.load()
    table = jpeg.get_component_qt(0)
    with tensor_path.open("rb") as stream:
        stream.seek(position * ROW_BYTES)
        raw = stream.read(ROW_BYTES)
    if len(raw) != ROW_BYTES:
        raise ValueError("truncated scalar replay tensor")
    raster = np.frombuffer(raw, dtype="<f4").reshape(256, 256)
    differences = []
    for y, x in ((0, 0), (15, 31), (127, 127), (255, 255)):
        # All upstream/native/prepared carriers in this protocol are 512 square.
        by, bx = (128 + y) // 8, (128 + x) // 8
        py, px = (128 + y) % 8, (128 + x) % 8
        value = 128.0
        for u in range(8):
            for v in range(8):
                value += (
                    (1 / math.sqrt(2) if u == 0 else 1)
                    * (1 / math.sqrt(2) if v == 0 else 1)
                    * int(jpeg.Y[by, bx, u, v])
                    * int(table[u, v])
                    / 4
                    * math.cos(math.pi * u * (2 * py + 1) / 16)
                    * math.cos(math.pi * v * (2 * px + 1) / 16)
                )
        expected = float(np.float32(value))
        difference = abs(float(raster[y, x]) - expected)
        if not math.isfinite(difference) or difference > 2e-4:
            raise ValueError("independent scalar IDCT replay failed")
        differences.append(difference)
    return max(differences)


def audit(root, alaska_source, boss_source, acquisition_audit, reserved_paths, out):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("fresh nonsymlink audit output required")
    index, index_sha = document(root / "index.json")
    if (
        index.get("schema_version") != "jpeg-scale-blocks-v1"
        or index.get("status") != "completed"
        or index.get("protocol_sha256") != PROTOCOL_SHA
    ):
        raise ValueError("complete frozen scale index required")
    acquisition, _ = document(acquisition_audit, index["acquisition_audit_sha256"])
    if acquisition.get("status") != "completed" or acquisition.get("models_trained") is not False:
        raise ValueError("completed acquisition-only audit required")
    alaska, _ = document(alaska_source, index["source_manifest_sha256"][0])
    boss, _ = document(boss_source, index["source_manifest_sha256"][1])
    acquisition_sources = {r["source_group"]: r for r in acquisition["sources"]}
    q = {r["lineage"] for r in acquisition_sources["ALASKA2"]["quarantined_lineages"]}
    if q != set(index["quarantined_lineages"]):
        raise ValueError("scale quarantine changed")
    reserved, reserved_shas = set(), []
    if not 1 <= len(reserved_paths) <= 65:
        raise ValueError("bounded reserved audit manifests required")
    for path in reserved_paths:
        doc, sha = document(path)
        reserved_shas.append(sha)
        if not isinstance(doc.get("samples"), list) or not 1 <= len(doc["samples"]) <= 10000:
            raise ValueError("invalid reserved audit identities")
        for r in doc["samples"]:
            reserved.add(r["sha256"])
            if r.get("lineage"):
                reserved.add(r["lineage"])
    if sorted(reserved_shas) != index["reserved_manifest_sha256"]:
        raise ValueError("reserved audit bindings changed")
    originals = {r["sha256"]: r for r in boss["samples"]}
    if (
        not 1 <= len(alaska["samples"]) <= 4000
        or not 2 <= len(boss["samples"]) <= 1000
        or len(originals) != len(boss["samples"])
        or any(r["source_group"] != "ALASKA2" for r in alaska["samples"])
        or any(r["source_group"] != "BOSSbase-1.01" for r in boss["samples"])
    ):
        raise ValueError("unexpected audit acquisition source rows")
    expected = {
        (r["source_group"], r["lineage"], None, r["method"]): r
        for r in alaska["samples"]
        if r["lineage"] not in q and r["method"] in (None, "JUNIWARD", "UERD")
    }
    for r in originals.values():
        for quality in (75, 95):
            for method in (None, "JUNIWARD", "UERD"):
                expected[("BOSSbase-1.01", r["lineage"], quality, method)] = r
    seen, hashes, lineage_blocks, counts, source_counts = set(), set(), {}, Counter(), Counter()
    oracle_cells, oracles, cache_bytes, jpeg_bytes, decoder = set(), [], 0, 0, None
    blocks = index.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 16:
        raise ValueError("scale block limit")
    for block_no, block in enumerate(blocks):
        manifest_path = relative(root, block["path"])
        manifest, _ = document(manifest_path, block["manifest_sha256"])
        rows = manifest["samples"]
        if (
            not isinstance(rows, list)
            or not 1 <= len(rows) <= 768
            or len(set(block["lineages"])) > 128
        ):
            raise ValueError("scale block row/lineage limit")
        if {r["lineage"] for r in rows} != set(block["lineages"]):
            raise ValueError("scale block lineage declaration changed")
        for row in rows:
            key = (row["source_group"], row["lineage"], row.get("quality_factor"), row["method"])
            if (
                key not in expected
                or key in seen
                or row["sha256"] in hashes | reserved
                or row["lineage"] in reserved
            ):
                raise ValueError("incomplete/duplicate/reserved scale family")
            seen.add(key)
            hashes.add(row["sha256"])
            original = expected[key]
            if (
                row["split"] != original["split"]
                or row["label"] != ("cover" if row["method"] is None else "stego")
                or row["format"] != "JPEG"
                or row["source_group"] != block["source_group"]
            ):
                raise ValueError("scale role/source/split changed")
            if lineage_blocks.setdefault(row["lineage"], block_no) != block_no:
                raise ValueError("original split across preparation blocks")
            body = read(relative(manifest_path.parent, row["path"]), MAX_JPEG)
            if len(body) != row["size"] or hashlib.sha256(body).hexdigest() != row["sha256"]:
                raise ValueError("scale JPEG bytes changed")
            with Image.open(io.BytesIO(body)) as image:
                if (
                    image.format != "JPEG"
                    or image.size != (512, 512)
                    or image.mode not in ("L", "RGB")
                ):
                    raise ValueError("scale JPEG geometry changed")
                image.load()
            if row["source_group"] == "ALASKA2" and row["sha256"] != original["sha256"]:
                raise ValueError("native ALASKA bytes changed")
            if row["source_group"] == "BOSSbase-1.01" and row["method"] is None:
                raw = read(relative(boss_source.parent, original["path"]), 1024**2)
                if hashlib.sha256(raw).hexdigest() != row["lineage"]:
                    raise ValueError("BOSS original ancestry changed")
                with Image.open(io.BytesIO(raw)) as image:
                    encoded = io.BytesIO()
                    image.save(
                        encoded, format="JPEG", quality=row["quality_factor"], optimize=False
                    )
                if encoded.getvalue() != body:
                    raise ValueError("independent BOSS cover re-encoding failed")
            jpeg_bytes += len(body)
            counts[row["split"]] += 1
            source_counts[row["source_group"]] += 1
        if set(block["caches"]) != {r["split"] for r in rows}:
            raise ValueError("missing or unexpected scale cache")
        for split, record in block["caches"].items():
            path = relative(root, record["path"])
            cache, _ = document(path, record["sha256"])
            selected = [r for r in rows if r["split"] == split]
            if (
                cache.get("schema_version") != "research-float256-cache-v1"
                or cache.get("manifest_sha256") != block["manifest_sha256"]
                or cache.get("split") != split
                or cache.get("dtype") != "<f4"
                or cache.get("feature_version") != "jpeg-y-idct-center256-phase0-f32-v1"
                or cache.get("shape") != [len(selected), 1, 256, 256]
                or len(cache["rows"]) != len(selected)
                or record["rows"] != len(selected)
                or record["data_sha256"] != cache["data_sha256"]
            ):
                raise ValueError("scale cache contract mismatch")
            if decoder is None:
                decoder = cache["decoder"]
            if cache["decoder"] != decoder:
                raise ValueError("mixed cache decoder versions")
            for row, sample in zip(cache["rows"], selected, strict=True):
                if (
                    any(row.get(k) != sample[k] for k in ("sha256", "lineage", "label"))
                    or row.get("image_size") != [512, 512]
                    or row.get("region") != [128, 128, 256, 256]
                ):
                    raise ValueError("scale cache row/crop identity changed")
            tensor = path.parent / "pixels.f32"
            regular(tensor)
            size = len(selected) * ROW_BYTES
            if tensor.stat().st_size != size:
                raise ValueError("scale tensor size changed")
            hashed = hashlib.sha256()
            with tensor.open("rb") as stream:
                remaining = size
                while remaining:
                    raw = stream.read(min(1024**2, remaining))
                    if not raw or len(raw) % 4:
                        raise ValueError("truncated scale tensor")
                    values = np.frombuffer(raw, dtype="<f4")
                    if not np.isfinite(values).all() or np.any(np.abs(values) > 2**36):
                        raise ValueError("invalid scale tensor values")
                    hashed.update(raw)
                    remaining -= len(raw)
                if stream.read(1):
                    raise ValueError("scale tensor grew")
            if hashed.hexdigest() != cache["data_sha256"]:
                raise ValueError("scale tensor checksum changed")
            cache_bytes += size
            for i, r in enumerate(selected):
                cell = (r["source_group"], r.get("quality_factor"), r["label"], r["method"])
                if cell not in oracle_cells:
                    difference = scalar_replay(relative(manifest_path.parent, r["path"]), tensor, i)
                    oracle_cells.add(cell)
                    oracles.append(
                        {
                            "source_group": cell[0],
                            "quality_factor": cell[1],
                            "label": cell[2],
                            "method": cell[3],
                            "jpeg_sha256": r["sha256"],
                            "max_scalar_difference": difference,
                            "passed": True,
                        }
                    )
    if (
        seen != set(expected)
        or len(seen) != index["jpeg_rows"]
        or dict(counts) != index["splits"]
        or len(lineage_blocks) != index["original_lineages"]
    ):
        raise ValueError("incomplete whole-source scale membership")
    report = {
        "schema_version": "jpeg-scale-independent-audit-v1",
        "status": "completed",
        "index_sha256": index_sha,
        "protocol_sha256": PROTOCOL_SHA,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "blocks_audited": len(blocks),
        "jpeg_rows": len(seen),
        "original_lineages": len(lineage_blocks),
        "source_rows": dict(source_counts),
        "splits": dict(counts),
        "jpeg_bytes": jpeg_bytes,
        "tensor_bytes": cache_bytes,
        "decoder": decoder,
        "scalar_oracles": oracles,
        "reserved_identity_overlap": 0,
        "models_trained": False,
        "detection_measured": False,
        "accuracy_qualification": "unavailable",
        "raw_data_published": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alaska-source", type=Path, required=True)
    parser.add_argument("--boss-source", type=Path, required=True)
    parser.add_argument("--acquisition-audit", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_META, MAX_META))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        report = audit(
            args.root,
            args.alaska_source,
            args.boss_source,
            args.acquisition_audit,
            args.reserved,
            args.out,
        )
    except Exception as exc:
        print(f"scale audit failed ({type(exc).__name__}); no complete audit")
        return 2
    print(
        f"independent scale audit completed: {report['jpeg_rows']} JPEGs; "
        "no model or accuracy score"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
