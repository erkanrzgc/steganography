"""Explicit full expansion in bounded blocks; no fitting or automatic downloads."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

from core.jpeg_features import MAX_IMAGE_BYTES
from core.jpeg_scale import METHODS, identity, layout
from steganography.research import verify_dataset_manifest
from steganography.research_features import MAX_DOCUMENT_BYTES, read_document
from steganography.research_jpeg import write_json
from steganography.research_jpeg_corpus import generate_boss
from steganography.research_pixel_cnn import fresh
from steganography.research_pixels import extract_pixels

PROTOCOL_SHA = "e494621f8d3eee763fe14461fafa81458b8404827fa554b2bf060ea7c5cb3036"
AUDIT_SHA = "62b0c8218797c365b697af1068a6c9c1c896e94395430f91c2e818cc0f3eb28a"
WIFD_SHA = "303995e90fbeb90a52e5a6ee99063741092690b5b47d50c2703c66bdf0959f9c"
MAX_SECONDS = 3600
MAX_OUTPUT = 8 * 1024**3


def native_block(root, out, source, keys):
    """Original ALASKA bytes; shared matched-family schema, no resampling."""
    fresh(out)
    selected = [
        r
        for r in source["samples"]
        if r["lineage"] in set(keys) and r.get("method") in (None, *METHODS)
    ]
    rows = []
    for r in selected:
        rows.append(
            {**r, "path": r["sha256"] + ".jpg", "quality_factor": None, "payload_rate": None}
        )
    manifest = {
        "schema_version": "1.0",
        "source": ".",
        "samples": rows,
        "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
        "catalog": {
            "source_records": {
                "ALASKA2": {"license": source["license"], "source_url": source["source_url"]}
            }
        },
    }
    verify_dataset_manifest(manifest, verify_files=False)
    out.mkdir(parents=True, exist_ok=False)
    for original, row in zip(selected, rows, strict=True):
        path = root / original["path"]
        fresh_path_input(path)
        with path.open("rb") as stream:
            raw = stream.read(MAX_IMAGE_BYTES + 1)
        if (
            not 0 < len(raw) <= MAX_IMAGE_BYTES
            or len(raw) != row["size"]
            or hashlib.sha256(raw).hexdigest() != row["sha256"]
        ):
            raise ValueError("native scale JPEG integrity changed")
        with (out / row["path"]).open("xb") as stream:
            stream.write(raw)
    write_json(out / "manifest.json", manifest)


def fresh_path_input(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("scale input must be regular and non-symlink")


def prepare(
    alaska_root, boss_root, audit_path, wifd_manifest, reserved_paths, out, *, progress=None
):
    fresh(out)
    started = time.monotonic()
    protocol = Path(__file__).resolve().parents[1] / "docs/JPEG_SCALE_PREPARATION_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("frozen scale protocol changed")
    audit, audit_sha = read_document(audit_path)
    if (
        audit_sha != AUDIT_SHA
        or audit.get("status") != "completed"
        or audit.get("models_trained") is not False
    ):
        raise ValueError("scale acquisition audit mismatch")
    alaska, alaska_sha = read_document(alaska_root / "source.json")
    boss, boss_sha = read_document(boss_root / "source.json")
    wifd, wifd_sha = read_document(wifd_manifest)
    if wifd_sha != WIFD_SHA or not 1 <= len(reserved_paths) <= 64:
        raise ValueError("reserved scale origins required")
    records = {r["source_group"]: r for r in audit["sources"]}
    if (
        records["ALASKA2"]["manifest_sha256"] != alaska_sha
        or records["BOSSbase-1.01"]["manifest_sha256"] != boss_sha
        or records["WIFD"]["manifest_sha256"] != wifd_sha
    ):
        raise ValueError("scale source checksums changed")
    if alaska.get("purpose") != "development" or boss.get("purpose") != "development covers only":
        raise ValueError("scale requires development acquisitions")
    reserved = set()
    reserved_shas = []
    for document, digest in [(wifd, wifd_sha)] + [read_document(p) for p in reserved_paths]:
        samples = document.get("samples")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 10000:
            raise ValueError("bounded scale reserved identities required")
        reserved_shas.append(digest)
        for r in samples:
            reserved.add(identity(r.get("sha256")))
            if r.get("lineage"):
                reserved.add(identity(r["lineage"]))
    if sorted(reserved_shas) != sorted([*audit["reserved_manifest_sha256"], wifd_sha]):
        raise ValueError("scale reserved acquisition bindings changed")
    quarantine = {r["lineage"] for r in records["ALASKA2"]["quarantined_lineages"]}
    blocks = layout(alaska["samples"], boss["samples"], quarantine, reserved)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "plan.json",
        {"protocol_sha256": PROTOCOL_SHA, "blocks": blocks, "audit_sha256": audit_sha},
    )
    entries, all_rows, total_bytes = [], [], 0
    hashes: set[str] = set()
    for i, block in enumerate(blocks):
        if time.monotonic() - started > MAX_SECONDS:
            raise ValueError("scale preparation deadline exceeded")
        folder = out / f"block-{i:03d}"
        row_limit = len(block["lineages"]) * (3 if block["source_group"] == "ALASKA2" else 6)
        conservative = row_limit * (MAX_IMAGE_BYTES + 256 * 256 * 4) + 4 * MAX_DOCUMENT_BYTES
        if total_bytes + conservative > MAX_OUTPUT:
            raise ValueError("scale remaining output budget insufficient")
        if block["source_group"] == "ALASKA2":
            native_block(alaska_root, folder, alaska, block["lineages"])
        else:
            generate_boss(
                boss_root,
                folder,
                source_sha256=boss_sha,
                reserved_manifests=reserved_paths,
                count=len(block["lineages"]),
                selection_offset=block["offset"],
            )
        manifest, manifest_sha = read_document(folder / "manifest.json")
        verify_dataset_manifest(manifest, source=folder)
        rows = manifest["samples"]
        if {r["lineage"] for r in rows} != set(block["lineages"]) or len(rows) != len(
            block["lineages"]
        ) * (3 if block["source_group"] == "ALASKA2" else 6):
            raise ValueError("incomplete scale block membership")
        for row in rows:
            if row["sha256"] in hashes | reserved or row["lineage"] in reserved:
                raise ValueError("duplicate or reserved scale artifact")
            hashes.add(row["sha256"])
            all_rows.append({**row, "path": f"{folder.name}/{row['path']}"})
            total_bytes += row["size"]
        caches = {}
        for split in ("train", "validation"):
            if any(r["split"] == split for r in rows):
                descriptor = extract_pixels(
                    folder / "manifest.json",
                    folder / split,
                    source=folder,
                    split=split,
                    workers=2,
                    _float=True,
                )
                total_bytes += len(descriptor["rows"]) * 256 * 256 * 4
                caches[split] = {
                    "path": f"{folder.name}/{split}/cache.json",
                    "sha256": hashlib.sha256(
                        (folder / split / "cache.json").read_bytes()
                    ).hexdigest(),
                    "rows": len(descriptor["rows"]),
                    "data_sha256": descriptor["data_sha256"],
                }
        if total_bytes > MAX_OUTPUT or time.monotonic() - started > MAX_SECONDS:
            raise ValueError("scale preparation byte/time limit exceeded")
        entries.append(
            {
                "path": f"{folder.name}/manifest.json",
                "manifest_sha256": manifest_sha,
                "source_group": block["source_group"],
                "lineages": block["lineages"],
                "caches": caches,
            }
        )
        total_bytes = output_bytes(out)
        if progress is not None:
            progress(
                {
                    "completed_blocks": len(entries),
                    "total_blocks": len(blocks),
                    "jpeg_rows": len(all_rows),
                }
            )
    verify_dataset_manifest(
        {
            "schema_version": "1.0",
            "samples": all_rows,
            "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
        },
        verify_files=False,
    )
    index = {
        "schema_version": "jpeg-scale-blocks-v1",
        "status": "completed",
        "protocol_sha256": PROTOCOL_SHA,
        "acquisition_audit_sha256": audit_sha,
        "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layout_script_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "core/jpeg_scale.py").read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": [alaska_sha, boss_sha],
        "reserved_manifest_sha256": sorted(reserved_shas),
        "quarantined_lineages": sorted(quarantine),
        "excluded_method": "JMiPOD: no matched BOSS simulator in pinned recipe",
        "blocks": entries,
        "jpeg_rows": len(all_rows),
        "splits": dict(Counter(r["split"] for r in all_rows)),
        "original_lineages": len({r["lineage"] for r in all_rows}),
        "output_bytes_before_index": total_bytes,
        "seconds": time.monotonic() - started,
        "models_trained": False,
        "detection_measured": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
    write_json(out / "index.json", index)
    return index


def output_bytes(out):
    total, count = 0, 0
    for path in out.rglob("*"):
        if path.is_symlink():
            raise ValueError("scale output symlink forbidden")
        if path.is_dir():
            continue
        fresh_path_input(path)
        count += 1
        total += path.stat().st_size
        if count > 15000 or total > MAX_OUTPUT:
            raise ValueError("scale output byte/member limit exceeded")
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alaska", type=Path, required=True)
    parser.add_argument("--boss", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--wifd-reserved", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (7200, 7201))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        index = prepare(
            args.alaska,
            args.boss,
            args.audit,
            args.wifd_reserved,
            args.reserved,
            args.out,
            progress=lambda r: print(
                f"prepared {r['completed_blocks']}/{r['total_blocks']} blocks; "
                f"{r['jpeg_rows']} JPEG rows",
                flush=True,
            ),
        )
    except Exception as exc:
        print(
            f"scale preparation failed ({type(exc).__name__}); "
            "no complete index; partial blocks retained",
            flush=True,
        )
        return 2
    print(
        f"JPEG scale preparation completed: {index['jpeg_rows']} rows; "
        "no fitting or accuracy measurement"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
