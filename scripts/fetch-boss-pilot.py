"""Explicit, bounded BOSSbase subset download; never redistributes source images.

Uses ZIP range reads to avoid downloading the entire 1.67 GB archive. Selection
is archive-order, not a representative random sample or independent source.
"""

# Fixed official HTTPS URL; no caller-controlled scheme.
# ruff: noqa: S310

import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"


class RemoteZip(io.RawIOBase):
    def __init__(self, template=None):
        if template is not None:
            self.size, self.etag = template.size, template.etag
            self.position = 0
            self.cache = template.cache.copy()
            return
        with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=30) as r:
            self.size = int(r.headers["Content-Length"])
            self.etag = r.headers.get("ETag", "")
        self.position = 0
        self.cache: dict[tuple[int, int], bytes] = {}

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= self.position <= self.size:
            raise ValueError("invalid archive seek")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if not size:
            return b""
        start, end = self.position, self.position + size
        for (left, right), value in self.cache.items():
            if left <= start and end <= right:
                self.position = end
                return value[start - left : end - left]
        if size > 300 * 1024 * 1024:
            raise ValueError("range exceeds download budget")
        request = urllib.request.Request(
            URL, headers={"Range": f"bytes={start}-{end - 1}", "If-Match": self.etag}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("server did not honor bounded range")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("archive changed or incomplete download")
        self.cache[(start, end)] = data
        self.position = end
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        parser.error("count must be 1..1000")
    if args.out.is_symlink():
        raise ValueError("symlink output is not accepted")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    remote = RemoteZip()
    records = []
    with zipfile.ZipFile(remote) as archive:
        members = sorted(
            (i for i in archive.infolist() if i.filename.lower().endswith(".pgm")),
            key=lambda i: i.header_offset,
        )[: args.count]
        if len(members) != args.count:
            raise ValueError("insufficient images")

        def fetch(item):
            index, member = item
            if member.file_size > 2 * 1024 * 1024:
                raise ValueError("oversized image")
            name = f"{index:04d}.pgm"
            target = args.out / name
            if target.is_symlink():
                raise ValueError("symlink sample is not accepted")
            if target.exists():
                data = target.read_bytes()
                if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
                    raise ValueError("existing sample does not match upstream ZIP CRC")
                return {
                    "path": name,
                    "upstream_member": member.filename,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            local = RemoteZip(remote)
            # Prefetch header, filename, extra fields and compressed bytes in one request.
            for attempt in range(3):
                try:
                    local.seek(member.header_offset)
                    local.read(min(member.compress_size + 4096, local.size - member.header_offset))
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            with zipfile.ZipFile(local) as reader:
                data = reader.read(member)  # zipfile checks the upstream CRC
            with target.open("xb") as stream:
                stream.write(data)
            return {
                "path": name,
                "upstream_member": member.filename,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        with ThreadPoolExecutor(max_workers=4) as executor:
            for index, record in enumerate(executor.map(fetch, enumerate(members))):
                records.append(record)
                if index % 25 == 0:
                    print(f"downloaded {index + 1}/{args.count}", flush=True)
    manifest = {
        "source_url": URL,
        "source_group": "BOSSbase-1.01",
        "license": "unspecified on download page; local research only; do not redistribute",
        "upstream_etag": remote.etag,
        "selection": "first members in archive order",
        "upstream_sha256_verified": False,
        "samples": records,
    }
    (args.out / "source.json").write_text(json.dumps(manifest, indent=2))
    print(f"complete: {len(records)} images", flush=True)


if __name__ == "__main__":
    main()
