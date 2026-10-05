"""Explicit reserved-safe BOSSbase development acquisition; local research only."""

import argparse
import hashlib
import importlib.util
import io
import json
import random
import re
import stat
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

URL = "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
SEED = 20261005


def helpers():
    # Reuse the independently tested bounded ZIP/range implementation, without
    # accessing Kaggle credentials or executing any downloaded source code.
    spec = importlib.util.spec_from_file_location(
        "boss_range_helpers", Path(__file__).with_name("fetch-alaska2-pilot.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("range implementation unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_origin(url):
    if url != URL:
        raise ValueError("unexpected BOSSbase origin")


def acquire(out, reserved_paths, *, count=1000, resume=False):
    if not 1 <= count <= 1000 or not reserved_paths:
        raise ValueError("count 1..1000 and reserved manifests required")
    module = helpers()
    reserved: set[str] = set()
    names, hashes = set(), []
    for path in reserved_paths:
        raw = module.bounded_read(path, 8 * 1024 * 1024)
        document = json.loads(raw)
        hashes.append(hashlib.sha256(raw).hexdigest())
        if not document.get("samples"):
            raise ValueError("empty reserved manifest")
        for row in document["samples"]:
            if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", ""))):
                raise ValueError("invalid reserved identity")
            reserved.update(row[k] for k in ("sha256", "lineage") if row.get(k))
            if row.get("upstream_member"):
                names.add(row["upstream_member"])
    module.no_symlinks(out)
    if (out / "source.json").exists():
        raise FileExistsError("completed output already exists")
    out.mkdir(parents=True, exist_ok=resume)
    remote = module.RemoteZip(URL, validate_origin=fixed_origin)
    members = module.catalog_members(remote)
    eligible = sorted(
        (m for m in members if m.filename.lower().endswith(".pgm") and m.filename not in names),
        key=lambda m: m.filename,
    )
    if len(eligible) < count or len({m.filename for m in eligible}) != len(eligible):
        raise ValueError("insufficient or duplicate source members")
    selected = sorted(random.Random(SEED).sample(eligible, count), key=lambda m: m.filename)  # noqa: S311
    for m in selected:
        if (
            PurePosixPath(m.filename).is_absolute()
            or ".." in PurePosixPath(m.filename).parts
            or "\\" in m.filename
            or stat.S_IFMT(m.external_attr >> 16) not in (0, stat.S_IFREG)
            or m.flag_bits & 1
            or m.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or not 0 < m.file_size <= module.MAX_FILE
            or not 0 < m.compress_size <= module.MAX_FILE
        ):
            raise ValueError("unsafe source member")
    if sum(m.file_size for m in selected) > 512 * 1024 * 1024:
        raise ValueError("expanded selection exceeds byte budget")
    selection = {
        "schema_version": "boss-development-selection-v1",
        "seed": SEED,
        "archive_bytes": remote.size,
        "archive_etag": remote.etag,
        "reserved_manifest_sha256": sorted(hashes),
        "members": [
            {"upstream_member": m.filename, "size": m.file_size, "crc32": m.CRC} for m in selected
        ],
    }
    encoded = json.dumps(selection, sort_keys=True, indent=2).encode()
    plan = out / "selection.json"
    if plan.exists():
        if module.bounded_read(plan, 8 * 1024 * 1024) != encoded:
            raise ValueError("resume selection/provenance changed")
    else:
        with plan.open("xb") as stream:
            stream.write(encoded)

    def fetch(item):
        index, member = item
        target = out / f"{index:04d}.pgm"
        module.no_symlinks(target)
        data = (
            module.bounded_read(target, module.MAX_FILE)
            if target.exists()
            else module.member_bytes(remote, member)
        )
        if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
            raise ValueError("member CRC or size changed")
        digest = hashlib.sha256(data).hexdigest()
        if digest in reserved:
            raise ValueError("development overlaps reserved identities")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
                raise ValueError("unexpected BOSSbase PGM")
            image.load()
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
        fraction = int(digest[:16], 16) / 2**64
        return {
            "path": target.name,
            "upstream_member": member.filename,
            "sha256": digest,
            "lineage": digest,
            "size": len(data),
            "source_group": "BOSSbase-1.01",
            "label": "cover",
            "method": None,
            "split": "train" if fraction < 0.8 else "validation",
            "camera": None,
            "device": None,
            "format": "PGM",
            "width": 512,
            "height": 512,
            "upstream_crc32": f"{member.CRC:08x}",
        }

    samples = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for row in executor.map(fetch, enumerate(selected)):
            samples.append(row)
            if len(samples) % 100 == 0:
                print(f"acquired {len(samples)}/{count}", flush=True)
    if len({r["sha256"] for r in samples}) != count:
        raise ValueError("duplicate acquired content")
    result = {
        "schema_version": "boss-development-acquisition-v1",
        "purpose": "development covers only",
        "source_url": URL,
        "source_group": "BOSSbase-1.01",
        "license": "unspecified on download page; local research only; do not redistribute",
        "selection_sha256": hashlib.sha256(encoded).hexdigest(),
        "reserved_manifest_sha256": sorted(hashes),
        "seed": SEED,
        "total_bytes": sum(r["size"] for r in samples),
        "requested_range_bytes": remote.requested_bytes,
        "range_requests": remote.requests,
        "upstream_sha256_verified": False,
        "samples": samples,
    }
    with (out / "source.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = acquire(args.out, args.reserved_manifest, count=args.count, resume=args.resume)
    except Exception as exc:
        parser.exit(2, f"BOSS development failed ({type(exc).__name__}); partial output retained\n")
    print(f"BOSS development acquired: {len(result['samples'])} covers; no evaluation performed")


if __name__ == "__main__":
    main()
