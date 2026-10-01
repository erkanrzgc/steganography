"""Explicit small Kodak source check; keep images local, never redistribute."""

# Fixed HTTPS image URLs; no caller-controlled network target.
# ruff: noqa: S310
import argparse
import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

SOURCE_URL = "https://r0k.us/graphics/kodak/"
MAX_BYTES = 2 * 1024 * 1024


def fetch(out: Path, reserved_path: Path) -> dict:
    reserved = json.loads(reserved_path.read_text())
    reserved_hashes = {
        str(s.get(key, "")) for s in reserved["samples"] for key in ("sha256", "lineage")
    }
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("symlink output is not accepted")
    out.mkdir(parents=True, exist_ok=False)

    def one(index):
        name = f"kodim{index:02d}.png"
        url = SOURCE_URL + "kodak/" + name
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.geturl() != url or response.status != 200:
                raise ValueError("unexpected image response or redirect")
            data = response.read(MAX_BYTES + 1)
            etag = response.headers.get("ETag")
        if len(data) > MAX_BYTES:
            raise ValueError("image exceeds download limit")
        sha = hashlib.sha256(data).hexdigest()
        if sha in reserved_hashes:
            raise ValueError("image overlaps reserved data")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size not in {(768, 512), (512, 768)}:
                raise ValueError("unexpected Kodak image format/dimensions")
            image.verify()
        with (out / name).open("xb") as stream:
            stream.write(data)
        return {"path": name, "sha256": sha, "source_url": url, "etag": etag}

    with ThreadPoolExecutor(max_workers=4) as executor:
        samples = list(executor.map(one, range(1, 25)))
    if len({s["sha256"] for s in samples}) != 24:
        raise ValueError("duplicate source images")
    manifest = {
        "source_group": "Kodak-PCD0992-r0k",
        "source_url": SOURCE_URL,
        "license": (
            "Curator states belief in unrestricted Kodak use; no formal grant verified. "
            "Local evaluation only; no redistribution."
        ),
        "selection": "all 24 numbered suite images",
        "samples": samples,
        "upstream_sha256_verified": False,
        "reserved_sha256_overlap": 0,
        "reserved_manifest_sha256": hashlib.sha256(reserved_path.read_bytes()).hexdigest(),
    }
    with (out / "source.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(f"downloaded {len(fetch(args.out, args.reserved_manifest)['samples'])} source images")


if __name__ == "__main__":
    main()
