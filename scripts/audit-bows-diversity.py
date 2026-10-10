"""Independent byte/pixel/split audit; no acquisition implementation imported."""

import argparse
import hashlib
import io
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path

from PIL import Image

ARCHIVE_SHA = "3e26b0faf740f7cd6d3a67df5ac86dba1631b8bcc69f17302498c785e240bd49"


def bounded(path, maximum):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit paths must not use symlinks")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("audit inputs must be regular files")
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("audit byte limit exceeded")
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pixel_sha(raw):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
            raise ValueError("invalid original image format/geometry")
        return sha(image.tobytes())


def audit(root, source_sha, manifests):
    raw_source = bounded(root / "source.json", 8 * 1024**2)
    if sha(raw_source) != source_sha:
        raise ValueError("original source manifest checksum mismatch")
    source = json.loads(raw_source)
    if source["status"] != "completed" or source["archive_layout"] != "verified-native-1001-v1":
        raise ValueError("source acquisition did not complete")
    raw = bounded(root / "archive.tar.gz", 200 * 1024**2)
    if len(raw) != 183938734 or sha(raw) != ARCHIVE_SHA or source["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("original archive identity mismatch")
    page = bounded(root / "source-page.html", 65536)
    if sha(page) != source["source_page_sha256"] or b"bows2-1g.tar.gz" not in page:
        raise ValueError("source-page evidence mismatch")
    reserved, bindings, boss_pixels, boss_count = set(), [], set(), 0
    for path in manifests:
        document = bounded(path, 8 * 1024**2)
        bindings.append(sha(document))
        samples = json.loads(document)["samples"]
        for row in samples:
            reserved.add(row["sha256"])
            if row.get("lineage"):
                reserved.add(row["lineage"])
            if row["path"].endswith(".pgm"):
                if not re.fullmatch(r"[0-9]{4}\.pgm", row["path"]):
                    raise ValueError("unsafe previous image path")
                original = bounded(path.parent / row["path"], 300 * 1024)
                if sha(original) != row["sha256"]:
                    raise ValueError("previous original bytes changed")
                boss_pixels.add(pixel_sha(original))
                boss_count += 1
    if sorted(bindings) != source["reserved_manifest_sha256"] or len(bindings) != 8:
        raise ValueError("original exclusions changed")
    if boss_count != 3000 or len(boss_pixels) != 3000:
        raise ValueError("independent BOSS pixel audit scope mismatch")
    rows = source["samples"]
    if len(rows) != 1001 or {r["upstream_member"] for r in rows} != (
        {f"{i}.pgm" for i in range(1, 1001)} | {"3661.pgm"}
    ):
        raise ValueError("original archive selection changed")
    seen, pixels, paths, splits, total = set(), set(), set(), Counter(), 0
    for row in rows:
        name = row["path"]
        if not re.fullmatch(r"[0-9]{4}\.pgm", name) or name in paths:
            raise ValueError("unsafe/duplicate acquired path")
        data = bounded(root / name, 300 * 1024)
        digest, identity = sha(data), pixel_sha(data)
        if (
            digest != row["sha256"]
            or digest != row["lineage"]
            or len(data) != row["bytes"]
            or identity != row["pixel_sha256"]
            or digest in seen
            or digest in reserved
            or identity in pixels
            or identity in boss_pixels
        ):
            raise ValueError("image provenance/duplicate/prior overlap audit failed")
        split_value = int.from_bytes(
            hashlib.sha256(("bows2:20261010:" + digest).encode()).digest()[:8], "big"
        )
        expected = "train" if split_value < 0.8 * 2**64 else "validation"
        if row["split"] != expected or row["source_group"] != "BOWS2":
            raise ValueError("lineage split/source mismatch")
        seen.add(digest)
        pixels.add(identity)
        paths.add(name)
        splits[expected] += 1
        total += len(data)
    return {
        "schema_version": "bows2-independent-original-audit-v1",
        "status": "completed",
        "source_manifest_sha256": source_sha,
        "archive_sha256": ARCHIVE_SHA,
        "archive_bytes": 183938734,
        "originals": len(rows),
        "original_bytes": total,
        "split": dict(splits),
        "reserved_manifest_sha256": sorted(bindings),
        "prior_boss_originals_pixel_checked": boss_count,
        "exact_identity_overlap": 0,
        "decoded_boss_pixel_overlap": 0,
        "camera_scene_independence": "unverified",
        "WIFD_training_used": False,
        "license": source["license"],
        "model_trained": False,
        "accuracy_qualification": "unavailable",
        "prepared_training_rows": 0,
        "audit_script_sha256": sha(bounded(Path(__file__), 65536)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or any(p.is_symlink() for p in (args.out, *args.out.parents)):
        raise FileExistsError("fresh nonsymlink audit output required")
    result = audit(args.root, args.source_sha256, args.reserved_manifest)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"status": result["status"], "originals": result["originals"], "split": result["split"]}
        )
    )


if __name__ == "__main__":
    main()
