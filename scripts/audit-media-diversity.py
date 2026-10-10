"""Independent complete media/ancestry audit; imports no acquisition service."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import time
import wave
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

EXPECTED = {
    "div2k-train": (800, {"train": 800}),
    "div2k-validation": (100, {"validation": 100}),
    "esc50": (2000, {"train": 1200, "validation": 400, "test": 400}),
}
COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
GROUP_ARCHIVE_SHA = "661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f"
GROUP_CSV_SHA = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
MAX_FILE = 64 * 1024**2


def open_regular(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit symlink rejected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("audit requires regular input")
    return os.fdopen(fd, "rb")


def read(path, limit):
    with open_regular(path) as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("audit file size limit")
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def document(path):
    raw = read(path, 8 * 1024**2)
    result = json.loads(raw)
    if not isinstance(result, dict) or not isinstance(result.get("samples"), list):
        raise ValueError("audit manifest schema")
    if not 1 <= len(result["samples"]) <= 15000:
        raise ValueError("audit row limit")
    return result, sha(raw)


def media_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("audit media path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("audit traversal")
    return root / name


def decoded(raw):
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw)) as audio:
            parameters = audio.getparams()
            if (
                parameters.comptype != "NONE"
                or parameters.nframes * parameters.nchannels * parameters.sampwidth > MAX_FILE
            ):
                raise ValueError("audit PCM bounds")
            data = audio.readframes(parameters.nframes)
            if len(data) != parameters.nframes * parameters.nchannels * parameters.sampwidth:
                raise ValueError("audit truncated PCM")
            return (
                "WAV",
                parameters.nchannels,
                parameters.sampwidth,
                parameters.framerate,
                parameters.nframes,
                sha(data),
            ), data
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 32_000_000:
            raise ValueError("audit image decode bound")
        image.seek(0)
        data = image.convert("RGB").tobytes()
        return ("image", image.width, image.height, sha(data)), data


def audit(sources, prior_paths, out):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("audit output exists or symlink")
    if not 1 <= len(sources) <= 3 or not 1 <= len(prior_paths) <= 128:
        raise ValueError("audit manifest count limit")
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > 2400:
            raise ValueError("independent audit deadline")

    prior_hashes, prior_decoded, prior_shas, prior_covers = set(), set(), [], 0
    for path in prior_paths:
        prior, digest = document(path)
        prior_shas.append(digest)
        for row in prior["samples"]:
            deadline()
            if not isinstance(row, dict) or not re.fullmatch(
                r"[a-f0-9]{64}", row.get("sha256", "")
            ):
                raise ValueError("audit prior identity")
            prior_hashes.add(row["sha256"])
            if row.get("lineage") is not None:
                if not re.fullmatch(r"[a-f0-9]{64}", row["lineage"]):
                    raise ValueError("audit prior lineage")
                prior_hashes.add(row["lineage"])
            if row.get("label", "cover") != "cover":
                continue
            raw = read(media_path(path.parent, row.get("path")), MAX_FILE)
            if sha(raw) != row["sha256"]:
                raise ValueError("prior original bytes changed")
            fingerprint, _ = decoded(raw)
            prior_decoded.add(fingerprint)
            prior_covers += 1
    hashes, pixels, all_groups, records, datasets = set(), set(), {}, [], set()
    for source_path in sources:
        source, source_sha = document(source_path)
        dataset = source.get("dataset")
        if dataset not in EXPECTED or dataset in datasets:
            raise ValueError("audit source identity")
        datasets.add(dataset)
        count, expected_splits = EXPECTED[dataset]
        grouped = source.get("split_policy") == "esc-original-group-max-fold-v1"
        if grouped:
            if dataset != "esc50" or source.get("archive_sha256") != GROUP_ARCHIVE_SHA:
                raise ValueError("audit grouped source binding")
            expected_splits = {"train": 1200, "validation": 398, "test": 402}
        if (
            source.get("status") != "completed"
            or source.get("schema_version") != "media-diversity-acquisition-v1"
            or len(source["samples"]) != count
            or source.get("prior_manifest_sha256") != sorted(prior_shas)
            or source.get("model_trained") is not False
            or source.get("cover_cleanliness_verified") is not False
        ):
            raise ValueError("audit incomplete acquisition/provenance")
        root = source_path.parent
        evidence = read(root / "license-evidence.txt", 1024**2)
        if sha(evidence) != source.get("license_evidence_sha256"):
            raise ValueError("license evidence changed")
        original_csv = {}
        highest_folds = {}
        if dataset == "esc50":
            if (
                source.get("source_commit") != COMMIT
                or b"creativecommons.org/licenses/by-nc/3.0/" not in evidence
            ):
                raise ValueError("audit ESC source/license")
            csv_raw = read(root / "upstream-metadata.csv", 1024**2)
            if grouped and sha(csv_raw) != GROUP_CSV_SHA:
                raise ValueError("audit grouped CSV binding")
            for row in csv.DictReader(io.StringIO(csv_raw.decode())):
                if row["filename"] in original_csv:
                    raise ValueError("audit duplicate CSV name")
                original_csv[row["filename"]] = row
                highest_folds[row["src_file"]] = max(
                    int(row["fold"]), highest_folds.get(row["src_file"], 0)
                )
            if len(original_csv) != 2000:
                raise ValueError("audit CSV coverage")
            expected_names = set(original_csv)
        else:
            if b"academic research purpose only" not in evidence:
                raise ValueError("audit DIV2K license")
            first = 1 if dataset == "div2k-train" else 801
            expected_names = {f"{i:04d}.png" for i in range(first, first + count)}
        archive_digest, archive_bytes = hashlib.sha256(), 0
        with open_regular(root / "archive.zip") as archive_stream:
            while chunk := archive_stream.read(1024**2):
                deadline()
                archive_bytes += len(chunk)
                if archive_bytes > 4 * 1024**3:
                    raise ValueError("audit archive byte limit")
                archive_digest.update(chunk)
        if archive_bytes != source.get("archive_bytes") or archive_digest.hexdigest() != source.get(
            "archive_sha256"
        ):
            raise ValueError("audit original archive changed")
        names, splits, cells, bytes_total = set(), Counter(), Counter(), 0
        for row in source["samples"]:
            deadline()
            name = row.get("path")
            if name not in expected_names or name in names:
                raise ValueError("audit original membership")
            names.add(name)
            raw = read(media_path(root, name), 32 * 1024**2)
            digest = sha(raw)
            fingerprint, decoded_data = decoded(raw)
            if (
                digest != row.get("sha256")
                or digest != row.get("lineage")
                or len(raw) != row.get("size")
                or digest in hashes
                or digest in prior_hashes
                or fingerprint in pixels
                or fingerprint in prior_decoded
                or sha(decoded_data) != row.get("decoded_sha256")
                or row.get("label") != "cover"
                or row.get("method") is not None
            ):
                raise ValueError("audit original integrity/decoded overlap")
            if dataset == "esc50":
                meta = original_csv[name]
                fold, target, group = (
                    int(meta["fold"]),
                    int(meta["target"]),
                    "ESC-50:" + meta["src_file"],
                )
                assigned = highest_folds[meta["src_file"]] if grouped else fold
                split = "train" if assigned <= 3 else "validation" if assigned == 4 else "test"
                if grouped and row.get("assigned_group_fold") != assigned:
                    raise ValueError("audit grouped role mismatch")
                if (
                    fingerprint[:5] != ("WAV", 1, 2, 44100, 220500)
                    or row.get("source_group") != "ESC-50"
                    or row.get("fold") != fold
                    or row.get("upstream_recording_id") != meta["src_file"]
                    or row.get("group_key") != group
                    or row.get("target") != target
                    or row.get("category") != meta["category"]
                    or row.get("esc10") != (meta["esc10"] == "True")
                    or all_groups.setdefault(group, split) != split
                ):
                    raise ValueError("audit ESC original-group leakage or metadata")
                cells[(fold, target)] += 1
            else:
                split = "train" if dataset == "div2k-train" else "validation"
                with Image.open(io.BytesIO(raw)) as image:
                    if (
                        image.mode != "RGB"
                        or image.format != "PNG"
                        or min(image.size) < 256
                        or image.width * image.height > 8_000_000
                        or row.get("width") != image.width
                        or row.get("height") != image.height
                        or row.get("source_group") != "DIV2K"
                    ):
                        raise ValueError("audit RGB original geometry")
            if row.get("split") != split:
                raise ValueError("audit role mismatch")
            hashes.add(digest)
            pixels.add(fingerprint)
            splits[split] += 1
            bytes_total += len(raw)
        if (
            names != expected_names
            or dict(splits) != expected_splits
            or source.get("splits") != expected_splits
            or bytes_total != source.get("total_media_bytes")
            or (dataset == "esc50" and (len(cells) != 250 or set(cells.values()) != {8}))
        ):
            raise ValueError("audit complete-source accounting")
        records.append(
            {
                "dataset": dataset,
                "manifest_sha256": source_sha,
                "originals": count,
                "splits": dict(splits),
                "archive_sha256": archive_digest.hexdigest(),
                "archive_bytes": archive_bytes,
                "media_bytes": bytes_total,
                "license": source["license"],
                "license_evidence_sha256": sha(evidence),
                "split_policy": source.get("split_policy", "fixed-upstream-roles-v1"),
                "original_recording_groups": len({r.get("group_key") for r in source["samples"]})
                if original_csv
                else None,
            }
        )
    report = {
        "schema_version": "media-diversity-independent-audit-v1",
        "status": "completed",
        "sources": records,
        "prior_manifest_sha256": sorted(prior_shas),
        "prior_covers_reread": prior_covers,
        "originals_audited": sum(r["originals"] for r in records),
        "prior_exact_overlap": 0,
        "prior_decoded_overlap": 0,
        "new_duplicate_originals": 0,
        "original_group_split_overlap": 0,
        "camera_scene_independence_verified": False,
        "cover_cleanliness_verified": False,
        "model_trained": False,
        "accuracy_qualification": "unavailable",
        "audit_source_sha256": sha(Path(__file__).read_bytes()),
        "seconds": time.monotonic() - started,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.source, args.reserved_manifest, args.out)
    except Exception as exc:
        parser.exit(
            2, f"Independent media audit failed ({type(exc).__name__}); no success report\n"
        )
    print(f"Independent media audit: {report['originals_audited']} originals; no accuracy measured")


if __name__ == "__main__":
    main()
