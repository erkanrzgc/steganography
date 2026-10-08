"""Read-only, independent file/split/ancestry audit of explicitly acquired data."""

import argparse
import hashlib
import io
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

MAX_META = 8 * 1024**2
MAX_FILE = 32 * 1024**2
WIFD_LICENSE_BLOB = "99108f413927f4937261f1e672454fcea4ddceea"
KINDS = {
    "alaska2-acquisition-v1": "ALASKA2",
    "boss-development-acquisition-v1": "BOSSbase-1.01",
    "wifd-acquisition-v1": "WIFD",
}


def read(path, limit):
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > limit
    ):
        raise ValueError("unbounded or symlink input")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input grew beyond byte limit")
    return raw


def identities(doc):
    rows = doc.get("samples")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("invalid manifest rows")
    hashes = set()
    members = set()
    for r in rows:
        if not isinstance(r, dict) or not re.fullmatch("[a-f0-9]{64}", str(r.get("sha256", ""))):
            raise ValueError("invalid manifest identity")
        if r.get("lineage") and not re.fullmatch("[a-f0-9]{64}", str(r["lineage"])):
            raise ValueError("invalid lineage identity")
        hashes.update(r[k] for k in ("sha256", "lineage") if r.get(k))
        member = r.get("upstream_member")
        if r.get("source_group") == "ALASKA2":
            name = PurePosixPath(r.get("path", "")).name
            if re.fullmatch(r"[0-9]{5}\.jpg", name):
                member = name
        if member:
            members.add((r.get("source_group"), member))
    return rows, hashes, members


def audit(sources, reserved_paths, out):
    if not sources or len(sources) > 16 or not reserved_paths or len(reserved_paths) > 64:
        raise ValueError("bounded source/reserved lists required")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("fresh output required")
    reserved, reserved_members = set(), set()
    reserved_shas = []
    for path in reserved_paths:
        raw = read(path, MAX_META)
        _, hashes, members = identities(json.loads(raw))
        reserved.update(hashes)
        reserved_members.update(members)
        reserved_shas.append(hashlib.sha256(raw).hexdigest())
    records = []
    acquired = set()
    acquired_members = set()
    manifests = set()
    for source in sources:
        raw = read(source, MAX_META)
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum in manifests:
            raise ValueError("duplicate source manifest")
        manifests.add(checksum)
        doc = json.loads(raw)
        kind = doc.get("schema_version")
        if kind not in KINDS:
            raise ValueError("unrecognized acquisition schema")
        rows, hashes, members = identities(doc)
        if (
            hashes & reserved
            or members & reserved_members
            or hashes & acquired
            or members & acquired_members
        ):
            raise ValueError("acquisition overlaps reserved or other acquired data")
        acquired.update(hashes)
        acquired_members.update(members)
        group = KINDS[kind]
        if group == "ALASKA2" and (
            doc.get("purpose") not in ("evaluation", "development")
            or type(doc.get("selection_seed")) is not int
        ):
            raise ValueError("invalid acquisition selection policy")
        labels = Counter()
        splits = Counter()
        cameras = Counter()
        formats = Counter()
        lineages = defaultdict(list)
        paths = set()
        total = 0
        for row in rows:
            name = row.get("path", "")
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or name in paths
            ):
                raise ValueError("unsafe or duplicate acquired path")
            paths.add(name)
            body = read(source.parent / name, MAX_FILE)
            digest = hashlib.sha256(body).hexdigest()
            if (
                digest != row["sha256"]
                or len(body) != row["size"]
                or row.get("source_group") != group
            ):
                raise ValueError("file hash/size/source mismatch")
            if "upstream_crc32" in row and f"{zlib.crc32(body):08x}" != row["upstream_crc32"]:
                raise ValueError("upstream CRC mismatch")
            if (
                group == "WIFD"
                and hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()  # noqa: S324
                != row["git_blob_sha1"]
            ):
                raise ValueError("Git blob mismatch")
            with Image.open(io.BytesIO(body)) as image:
                frames = getattr(image, "n_frames", 1)
                if (
                    image.size != (row["width"], row["height"])
                    or image.width * image.height > 32_000_000
                ):
                    raise ValueError("image size outside contract")
                if group == "BOSSbase-1.01":
                    if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
                        raise ValueError("invalid PGM")
                elif (
                    not (
                        (image.format == "JPEG" and frames == 1)
                        or (
                            group == "WIFD"
                            and doc.get("allow_bounded_mpo") is True
                            and image.format == "MPO"
                            and 2 <= frames <= 4
                        )
                    )
                    or image.mode not in ("RGB", "L")
                    or min(image.size) < 256
                    or (group == "ALASKA2" and image.size != (512, 512))
                ):
                    raise ValueError("invalid JPEG")
                if group == "WIFD" and (
                    row.get("format") != image.format
                    or row.get("declared_frames") != frames
                    or row.get("decoded_frames") != 1
                ):
                    raise ValueError("native format/frame declaration changed")
                formats[image.format] += 1
                image.load()
            if group == "BOSSbase-1.01":
                split = "train" if int(digest[:16], 16) / 2**64 < 0.8 else "validation"
                if row["label"] != "cover" or row["method"] is not None or row["lineage"] != digest:
                    raise ValueError("invalid BOSS ancestry")
            elif group == "ALASKA2":
                matched = re.fullmatch(r"(Cover|JMiPOD|JUNIWARD|UERD)/([0-9]{5}\.jpg)", name)
                if not matched:
                    raise ValueError("invalid ALASKA path")
                family, original = matched.groups()
                split = (
                    "test"
                    if doc["purpose"] == "evaluation"
                    else (
                        "train"
                        if int.from_bytes(
                            hashlib.sha256(f"{doc['selection_seed']}:{original}".encode()).digest()[
                                :8
                            ],
                            "big",
                        )
                        / 2**64
                        < 0.8
                        else "validation"
                    )
                )
                if row["label"] != ("cover" if family == "Cover" else "stego") or row["method"] != (
                    None if family == "Cover" else family
                ):
                    raise ValueError("invalid ALASKA labels")
            else:
                split = "test"
                if (
                    row["label"] != "cover"
                    or row["method"] is not None
                    or row["lineage"] != digest
                    or row["camera"] != row["device"]
                    or not row["upstream_member"].startswith(row["camera"] + "/sdr_image/")
                ):
                    raise ValueError("invalid reserved WIFD ancestry/device")
            if row["split"] != split:
                raise ValueError("lineage split changed")
            labels[str(row["method"]) if row["method"] is not None else "cover"] += 1
            splits[split] += 1
            if row.get("camera"):
                cameras[row["camera"]] += 1
            lineages[row["lineage"]].append(row)
            total += len(body)
            if total > 16 * 1024**3:
                raise ValueError("aggregate byte limit exceeded")
        quarantine = []
        if group != "ALASKA2" and len(lineages) != len(rows):
            raise ValueError("duplicate acquired cover identity")
        if group == "ALASKA2":
            for lineage, members in lineages.items():
                covers = [r for r in members if r["label"] == "cover"]
                if (
                    len(covers) != 1
                    or covers[0]["sha256"] != lineage
                    or len(members) != 4
                    or {r["method"] for r in members} != {None, "JMiPOD", "JUNIWARD", "UERD"}
                ):
                    raise ValueError("incomplete or forged four-way lineage")
                unchanged = [
                    r["method"] for r in members if r["label"] == "stego" and r["sha256"] == lineage
                ]
                if unchanged:
                    quarantine.append(
                        {"lineage": lineage, "unchanged_stego_methods": sorted(unchanged)}
                    )
        if total != doc["total_bytes"]:
            raise ValueError("aggregate bytes mismatch")
        if group == "WIFD":
            license_raw = read(source.parent / "LICENSE.upstream", 32768)
            if (
                doc["license"] != "MIT"
                or hashlib.sha256(license_raw).hexdigest() != doc["license_sha256"]
                or hashlib.sha1(f"blob {len(license_raw)}\0".encode() + license_raw).hexdigest()  # noqa: S324
                != WIFD_LICENSE_BLOB
            ):
                raise ValueError("WIFD license evidence changed")
        records.append(
            {
                "source_group": group,
                "manifest_sha256": checksum,
                "selection_sha256": hashlib.sha256(
                    read(source.parent / "selection.json", MAX_META)
                ).hexdigest(),
                "files": len(rows),
                "original_lineages": len(lineages),
                "bytes": total,
                "labels": dict(labels),
                "splits": dict(splits),
                "declared_camera_counts": dict(cameras),
                "native_formats": dict(formats),
                "license": doc["license"],
                "quarantined_lineages": quarantine,
                "training_eligible_lineages_after_preparation": 0
                if group == "WIFD"
                else len(lineages) - len(quarantine),
                "scene_independence_verified": False,
            }
        )
        if records[-1]["selection_sha256"] != doc["selection_sha256"]:
            raise ValueError("selection record changed")
    report = {
        "schema_version": "data-expansion-audit-v1",
        "status": "completed",
        "reserved_manifest_sha256": sorted(reserved_shas),
        "sources": records,
        "files": sum(r["files"] for r in records),
        "bytes": sum(r["bytes"] for r in records),
        "detection_measured": False,
        "models_trained": False,
        "accuracy_qualification": "unavailable",
        "raw_data_published": False,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--reserved", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_META, MAX_META))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        report = audit(args.source, args.reserved, args.out)
    except Exception as exc:
        print(f"expansion audit failed ({type(exc).__name__}); no success record")
        return 2
    print(
        f"independent expansion audit complete: {report['files']} files; no model/detection score"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
