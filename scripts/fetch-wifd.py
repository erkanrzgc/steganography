"""Explicit pinned WIFD SDR covers; whole origin reserved, no accuracy claim."""

import argparse
import hashlib
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

COMMIT = "3f577edf0b14c686aa08e8d0d8ae07a83ba44f26"
PROTOCOL_SHA = "29f706034106f8e8706909afebb47d17d3879de2ef41c4b899a3113675c630a7"
RETRY_PROTOCOL_SHA = "8e01274b87c73610f0a76d6852260cd9761e12c5c17a95e695e3f3a77a6ae7ed"
TREE = "d674ddf9516c3044b2627224e19b7a0361e093a9"
LICENSE_BLOB = "99108f413927f4937261f1e672454fcea4ddceea"
README_BLOB = "34a107995e0675485948759dfcf2045b9883725f"
SOURCE = "https://github.com/CSCRC-SCREED/WIFD"
RAW = f"https://raw.githubusercontent.com/CSCRC-SCREED/WIFD/{COMMIT}/"
API = f"https://api.github.com/repos/CSCRC-SCREED/WIFD/git/trees/{TREE}?recursive=1"
MAX_FILE = 32 * 1024**2
MAX_META = 8 * 1024**2
MAX_TOTAL = 16 * 1024**3
MEMBER = re.compile(r"([a-z0-9_]+)/sdr_image/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+\.jpg\Z")


def blob_hash(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324


def safe(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink path forbidden")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Fetcher:
    def __init__(self):
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.requests = self.budget_bytes = self.received_bytes = 0

    def get(self, url, limit):
        if url != API and not url.startswith(RAW):
            raise ValueError("unapproved download origin")
        if not 0 < limit <= MAX_FILE:
            raise ValueError("invalid network byte limit")
        for attempt in range(3):
            with self.lock:
                remaining = 1800 - (time.monotonic() - self.started)
                if remaining <= 0 or self.requests >= 6100 or self.budget_bytes + limit > MAX_TOTAL:
                    raise ValueError("acquisition network/time budget exceeded")
                self.requests += 1
                self.budget_bytes += limit
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "steganography-research"})  # noqa: S310
                with urllib.request.build_opener(NoRedirect).open(
                    req, timeout=min(20, remaining)
                ) as response:
                    if (
                        response.status != 200
                        or int(response.headers.get("Content-Length", "0")) > limit
                    ):
                        raise ValueError("unexpected bounded download response")
                    raw = bytearray()
                    while True:
                        part = response.read(min(65536, limit + 1 - len(raw)))
                        if not part:
                            break
                        raw.extend(part)
                        if len(raw) > limit or time.monotonic() - self.started > 1800:
                            raise ValueError("download byte/time limit exceeded")
                with self.lock:
                    self.received_bytes += len(raw)
                return bytes(raw)
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise ValueError(f"upstream download failed: HTTP {status}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise ValueError("upstream network unavailable") from None
            time.sleep(2**attempt)
        raise ValueError("retries exhausted")  # pragma: no cover


def select(document, per_device, excluded):
    entries = document.get("tree")
    if (
        document.get("sha") != TREE
        or document.get("truncated") is not False
        or not isinstance(entries, list)
        or len(entries) > 30000
    ):
        raise ValueError("unexpected or incomplete pinned source tree")
    cameras = {}
    names = set()
    for row in entries:
        path = row.get("path", "")
        if not isinstance(path, str):
            raise ValueError("invalid source path")
        match = MEMBER.fullmatch(path)
        if not match:
            continue
        if (
            path in names
            or any(p in (".", "..") for p in path.split("/"))
            or row.get("type") != "blob"
            or row.get("mode") != "100644"
            or type(row.get("size")) is not int
            or not 0 < row["size"] <= MAX_FILE
            or not re.fullmatch(r"[a-f0-9]{40}", str(row.get("sha", "")))
        ):
            raise ValueError("unsafe, duplicate or oversized source member")
        names.add(path)
        if path not in excluded:
            cameras.setdefault(match.group(1), []).append(row)
    if not 1 <= len(cameras) <= 32:
        raise ValueError("invalid source camera count")
    selected = []
    for camera in sorted(cameras):
        ordered = sorted(
            cameras[camera],
            key=lambda r: hashlib.sha256(("wifd:20261008:" + r["path"]).encode()).digest(),
        )
        selected.extend(ordered[:per_device])
    if (
        not 1 <= len(selected) <= 2000
        or sum(r["size"] for r in selected) > MAX_TOTAL - MAX_META * 3
    ):
        raise ValueError("selection exceeds acquisition limits")
    return selected


def acquire(out, reserved_paths, *, per_device=20, allow_bounded_mpo=False):
    if type(per_device) is not int or per_device not in (20, 120) or not reserved_paths:
        raise ValueError("explicit selection and reserved manifests required")
    safe(out)
    if out.exists():
        raise FileExistsError("fresh output required")
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = Path(__file__).resolve().parents[1] / "docs/DATA_EXPANSION_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("frozen acquisition protocol changed")
    if type(allow_bounded_mpo) is not bool:
        raise ValueError("explicit native-format policy required")
    if allow_bounded_mpo:
        retry = protocol.parent / "WIFD_RETRY_PROTOCOL.md"
        if hashlib.sha256(retry.read_bytes()).hexdigest() != RETRY_PROTOCOL_SHA:
            raise ValueError("frozen retry protocol changed")
    reserved, excluded, documents = set(), set(), []
    for path in reserved_paths:
        safe(path)
        if not path.is_file() or path.stat().st_size > MAX_META:
            raise ValueError("unbounded reserved manifest")
        with path.open("rb") as stream:
            raw = stream.read(MAX_META + 1)
        if len(raw) > MAX_META:
            raise ValueError("reserved manifest grew beyond byte limit")
        d = json.loads(raw)
        if not d.get("samples"):
            raise ValueError("empty reserved manifest")
        documents.append(hashlib.sha256(raw).hexdigest())
        for row in d["samples"]:
            if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", ""))):
                raise ValueError("invalid reserved identity")
            reserved.update(row[k] for k in ("sha256", "lineage") if row.get(k))
            if row.get("source_group") == "WIFD" and row.get("upstream_member"):
                excluded.add(row["upstream_member"])
    fetcher = Fetcher()
    license_raw = fetcher.get(RAW + "LICENSE", 32768)
    readme = fetcher.get(RAW + "README.md", 32768)
    if (
        blob_hash(license_raw) != LICENSE_BLOB
        or blob_hash(readme) != README_BLOB
        or b"MIT License" not in license_raw
        or b"data and code" not in readme
        or b"MIT License" not in readme
    ):
        raise ValueError("pinned data license evidence mismatch")
    tree_raw = fetcher.get(API, MAX_META)
    selected = select(json.loads(tree_raw), per_device, excluded)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    for name, raw in (("LICENSE.upstream", license_raw), ("README.upstream", readme)):
        with (out / name).open("xb") as f:
            f.write(raw)
    selection = {
        "commit": COMMIT,
        "tree": TREE,
        "per_device": per_device,
        "allow_bounded_mpo": allow_bounded_mpo,
        "reserved_manifest_sha256": sorted(documents),
        "members": [{k: r[k] for k in ("path", "sha", "size")} for r in selected],
    }
    encoded = json.dumps(selection, indent=2, sort_keys=True).encode() + b"\n"
    with (out / "selection.json").open("xb") as f:
        f.write(encoded)

    def download(item):
        i, row = item
        raw = fetcher.get(RAW + "dataset/" + urllib.parse.quote(row["path"], safe="/"), row["size"])
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != row["size"] or blob_hash(raw) != row["sha"] or digest in reserved:
            raise ValueError("image integrity or reserved-overlap failure")
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            native_format = image.format
            frames = getattr(image, "n_frames", 1)
            if (
                not (
                    (native_format == "JPEG" and frames == 1)
                    or (allow_bounded_mpo and native_format == "MPO" and 2 <= frames <= 4)
                )
                or image.mode not in ("RGB", "L")
                or min(width, height) < 256
                or max(width, height) > 16384
                or width * height > 32_000_000
            ):
                raise ValueError("unexpected or unbounded JPEG geometry")
            image.load()
        target = out / f"{i:04d}.{'mpo' if native_format == 'MPO' else 'jpg'}"
        safe(target)
        with target.open("xb") as f:
            f.write(raw)
        camera = row["path"].split("/")[0]
        return {
            "path": target.name,
            "upstream_member": row["path"],
            "git_blob_sha1": row["sha"],
            "sha256": digest,
            "size": len(raw),
            "lineage": digest,
            "source_group": "WIFD",
            "label": "cover",
            "method": None,
            "split": "test",
            "format": native_format,
            "declared_frames": frames,
            "decoded_frames": 1,
            "decode_policy": "primary frame only; original bytes preserved",
            "camera": camera,
            "device": camera,
            "device_identity_status": "declared upstream directory; not independently verified",
            "scene": None,
            "width": width,
            "height": height,
        }

    samples = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download, enumerate(selected)):
            samples.append(row)
            if len(samples) % 20 == 0:
                print(f"verified {len(samples)}/{len(selected)} WIFD covers", flush=True)
    if len({r["sha256"] for r in samples}) != len(samples):
        raise ValueError("duplicate cover content")
    report = {
        "schema_version": "wifd-acquisition-v1",
        "protocol_sha256": PROTOCOL_SHA,
        "retry_protocol_sha256": RETRY_PROTOCOL_SHA if allow_bounded_mpo else None,
        "allow_bounded_mpo": allow_bounded_mpo,
        "acquisition_script_sha256": source_sha,
        "purpose": "whole WIFD origin reserved from training/calibration; evaluation covers only",
        "source_url": SOURCE,
        "source_commit": COMMIT,
        "license": "MIT",
        "license_url": SOURCE + "/blob/" + COMMIT + "/LICENSE",
        "license_sha256": hashlib.sha256(license_raw).hexdigest(),
        "readme_sha256": hashlib.sha256(readme).hexdigest(),
        "tree_sha256": hashlib.sha256(tree_raw).hexdigest(),
        "selection_sha256": hashlib.sha256(encoded).hexdigest(),
        "reserved_manifest_sha256": sorted(documents),
        "received_bytes": fetcher.received_bytes,
        "requests": fetcher.requests,
        "total_bytes": sum(r["size"] for r in samples),
        "scene_independence_verified": False,
        "accuracy_qualification": "unavailable",
        "samples": samples,
    }
    with (out / "source.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    parser.add_argument("--per-device", type=int, choices=(20, 120), default=20)
    parser.add_argument("--allow-bounded-mpo", action="store_true")
    args = parser.parse_args(argv)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        report = acquire(
            args.out,
            args.reserved_manifest,
            per_device=args.per_device,
            allow_bounded_mpo=args.allow_bounded_mpo,
        )
    except Exception as exc:
        print(
            f"WIFD acquisition failed ({type(exc).__name__}); no success manifest; "
            "partial output retained"
        )
        return 2
    print(f"WIFD acquired: {len(report['samples'])} covers; no evaluation or training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
