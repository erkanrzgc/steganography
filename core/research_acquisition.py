"""Explicit, bounded Commons/Unicode development acquisition; never training."""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter
from pathlib import Path

from PIL import Image

MAX_DOWNLOAD = 1024**3
MAX_FILE = 32 * 1024**2
MAX_SECONDS = 1800
TARGETS = {"png": 40, "gif": 20, "wav": 20, "mp3": 20}
MIMES = {"png": "image/png", "gif": "image/gif", "wav": "audio/wav", "mp3": "audio/mpeg"}
UNICODE_ROOT = "https://www.unicode.org/Public/17.0.0/ucd/"
UNICODE_FILES = {
    "GraphemeBreakTest.txt": UNICODE_ROOT + "auxiliary/GraphemeBreakTest.txt",
    "WordBreakTest.txt": UNICODE_ROOT + "auxiliary/WordBreakTest.txt",
    "SentenceBreakTest.txt": UNICODE_ROOT + "auxiliary/SentenceBreakTest.txt",
    "NormalizationTest.txt": UNICODE_ROOT + "NormalizationTest.txt",
    "BidiCharacterTest.txt": UNICODE_ROOT + "BidiCharacterTest.txt",
    "emoji-test.txt": "https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink output")
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def check_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname
        not in {"commons.wikimedia.org", "upload.wikimedia.org", "www.unicode.org"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
    ):
        raise ValueError("acquisition requires fixed HTTPS hosts")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Budget:
    def __init__(self, journal: Path | None = None):
        self.bytes = 0
        self.started = time.monotonic()
        self.journal = journal
        if journal is not None:
            if any(p.is_symlink() for p in (journal, *journal.parents)):
                raise ValueError("symlink budget journal")
            journal.open("x").close()

    def checkpoint(self):
        if self.journal is not None:
            with self.journal.open("a") as stream:
                stream.write(
                    json.dumps(
                        {"downloaded_bytes": self.bytes, "seconds": time.monotonic() - self.started}
                    )
                    + "\n"
                )

    def deadline(self):
        if time.monotonic() - self.started >= MAX_SECONDS:
            raise TimeoutError("acquisition deadline")

    def fetch(self, url: str, limit: int = MAX_FILE) -> bytes:
        self.checkpoint()
        try:
            return self._fetch(url, limit)
        finally:
            self.checkpoint()

    def _fetch(self, url: str, limit: int) -> bytes:
        check_url(url)
        self.deadline()
        limit = min(limit, MAX_DOWNLOAD - self.bytes - 1)
        if limit <= 0:
            raise ValueError("acquisition byte budget exhausted")
        request = urllib.request.Request(  # noqa: S310 - validated fixed HTTPS hosts
            url, headers={"User-Agent": "SteganographyResearch/0.6"}
        )
        opener = urllib.request.build_opener(SafeRedirect)
        result = bytearray()
        with opener.open(request, timeout=15) as response:  # noqa: S310 - fixed HTTPS allowlist
            check_url(response.url)
            size = response.headers.get("Content-Length")
            if response.status != 200 or (size is not None and not 0 <= int(size) <= limit):
                raise ValueError("acquisition response/size bound")
            while True:
                self.deadline()
                chunk = response.read(min(65536, limit - len(result) + 1))
                if not chunk:
                    break
                self.bytes += len(chunk)
                result.extend(chunk)
                if len(result) > limit:
                    raise ValueError("acquisition byte bound")
            if not result or (size is not None and len(result) != int(size)):
                raise ValueError("empty/truncated acquisition")
        return bytes(result)


def inspect(data: bytes, family: str) -> dict:
    if family in {"png", "gif"}:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != family.upper() or image.width * image.height > 8_000_000:
                raise ValueError("pilot image format/extent")
            frames = getattr(image, "n_frames", 1)
            if family == "png" and (image.mode != "RGB" or min(image.size) < 256 or frames != 1):
                raise ValueError("pilot requires native single RGB PNG >=256")
            if frames > 256 or frames * image.width * image.height > 32_000_000:
                raise ValueError("pilot GIF frame/pixel budget")
            decoded = hashlib.sha256()
            duration = 0
            for index in range(frames):
                image.seek(index)
                decoded.update(image.convert("RGB").tobytes())
                duration += image.info.get("duration", 0)
            return {
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "frames": frames,
                "decoded_sha256": decoded.hexdigest(),
                "duration_ms": duration,
                "benign_comment": bool(image.info.get("comment")),
                "loop": image.info.get("loop"),
                "application_extension": bool(image.info.get("extension")),
            }
    if family == "wav":
        with wave.open(io.BytesIO(data)) as audio:
            if (
                audio.getnchannels() != 2
                or audio.getsampwidth() != 2
                or audio.getcomptype() != "NONE"
                or not 160 <= audio.getnframes() <= 4_410_000
                or not 8000 <= audio.getframerate() <= 192000
            ):
                raise ValueError("pilot requires native stereo PCM16")
            frames = audio.readframes(audio.getnframes())
            if len(frames) != audio.getnframes() * 4:
                raise ValueError("pilot truncated stereo")
            if frames[0::4] == frames[2::4] and frames[1::4] == frames[3::4]:
                raise ValueError("pilot duplicated mono")
            return {
                "channels": 2,
                "frames": audio.getnframes(),
                "sample_rate": audio.getframerate(),
                "decoded_sha256": digest(frames),
                "left_bytes_sha256": digest(
                    b"".join(frames[i : i + 2] for i in range(0, len(frames), 4))
                ),
                "right_bytes_sha256": digest(
                    b"".join(frames[i : i + 2] for i in range(2, len(frames), 4))
                ),
            }
    if family != "mp3":
        raise ValueError("unknown pilot family")
    # Inventory validation only: two native MPEG Layer III frame headers.
    offset = 0
    if data[:3] == b"ID3":
        if len(data) < 10 or any(v & 128 for v in data[6:10]):
            raise ValueError("invalid ID3 extent")
        offset = 10 + sum(v << shift for v, shift in zip(data[6:10], (21, 14, 7, 0), strict=True))
        offset += 10 if data[5] & 16 else 0
    begin = offset
    for _ in range(2):
        header = data[offset : offset + 4]
        if len(header) != 4 or header[0] != 255 or header[1] & 224 != 224:
            raise ValueError("native MPEG frame required")
        version, layer = (header[1] >> 3) & 3, (header[1] >> 1) & 3
        bitrate, rate = header[2] >> 4, (header[2] >> 2) & 3
        if version == 1 or layer != 1 or bitrate in (0, 15) or rate == 3:
            raise ValueError("native MP3 Layer III required")
        table = (
            (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
            if version == 3
            else (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
        )
        hz = (44100, 48000, 32000)[rate] // (1 if version == 3 else 2 if version == 2 else 4)
        length = (144 if version == 3 else 72) * table[bitrate] * 1000 // hz
        offset += length + ((header[2] >> 1) & 1)
        if offset > len(data):
            raise ValueError("truncated native MP3")
    return {
        "decoded_sha256": None,
        "audio_bytes_sha256": digest(data[begin:]),
        "native_layer3_headers": 2,
        "decoder_validation": "unavailable",
    }


def metadata(page: dict, family: str) -> dict:
    info = page["imageinfo"][0]
    ext = info.get("extmetadata", {})

    def value(key):
        return html.unescape(re.sub("<[^>]+>", "", str(ext.get(key, {}).get("value", ""))))[:4096]

    license_name, license_url = value("LicenseShortName"), value("LicenseUrl")
    if not license_name or not (
        license_url.startswith("https://creativecommons.org/")
        or "public domain" in license_name.lower()
    ):
        raise ValueError("missing file-level license evidence")
    if info["size"] > MAX_FILE or info["size"] < 32:
        raise ValueError("pilot source byte bound")
    title = page["title"]
    if not title.lower().endswith("." + family):
        raise ValueError("pilot extension mismatch")
    author = value("Artist")
    return {
        "source_id": f"commons:{page['pageid']}",
        "title": title,
        "source_url": info["descriptionurl"],
        "original_url": info["url"],
        "license": license_name,
        "license_url": license_url,
        "author": author or None,
        "credit": value("Credit"),
        "object_name": value("ObjectName"),
        "known_group": author or None,
        "recording_group": None,
        "ancestry_independence": "unavailable",
        "declared_size": info["size"],
        "metadata_sha256": digest(json.dumps(page, sort_keys=True).encode()),
    }


def acquire(out: Path, *, unicode_only: bool = False) -> dict:
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("fresh nonsymlink pilot output required")
    out.mkdir(parents=True, mode=0o700)
    budget = Budget(out / "download-budget.jsonl")
    result: dict = {
        "schema_version": "nonjpeg-native-pilot-v1",
        "targets": TARGETS,
        "families": {},
        "unicode": {},
        "model_trained": False,
        "test_media_opened": False,
        "jpeg_data_added": False,
    }
    identities: set[str] = set()
    commons_available = not unicode_only
    for family, target in TARGETS.items():
        rows: list[dict] = []
        failures: Counter[str] = Counter()
        authors: Counter[str] = Counter()
        for page_index in range(8):
            if len(rows) >= target or not commons_available:
                break
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filemime:{MIMES[family]}",
                "gsrnamespace": 6,
                "gsrlimit": 50,
                "gsroffset": page_index * 50,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iilimit": 1,
            }
            try:
                raw = budget.fetch(
                    "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params),
                    4 * 1024**2,
                )
                document = json.loads(raw)
                write_json(out / f"{family}-metadata-{page_index}.json", document)
                pages = sorted(
                    document.get("query", {}).get("pages", {}).values(),
                    key=lambda p: (p.get("index", 0), p.get("pageid", 0)),
                )
                if not pages:
                    failures["metadata_empty"] += 1
                    break
            except urllib.error.HTTPError as exc:
                failures[f"http_{exc.code}"] += 1
                commons_available = False
                break
            except (OSError, ValueError, TimeoutError):
                failures["metadata_access_unavailable"] += 1
                break
            for page in pages:
                if len(rows) >= target:
                    break
                try:
                    row = metadata(page, family)
                    if authors[row["author"] or "unknown"] >= 2:
                        failures["author_diversity_limit"] += 1
                        continue
                    data = budget.fetch(row["original_url"])
                    if len(data) != row["declared_size"]:
                        raise ValueError("original size mismatch")
                    representation = inspect(data, family)
                    identity = representation.get("decoded_sha256") or representation.get(
                        "audio_bytes_sha256"
                    )
                    if not isinstance(identity, str):
                        raise ValueError("missing carrier identity")
                    if identity in identities:
                        failures["duplicate_carrier"] += 1
                        continue
                    identities.add(identity)
                    filename = f"{family}-{len(rows):03d}.{family}"
                    (out / filename).open("xb").write(data)
                    row.update(
                        path=filename,
                        original_sha256=digest(data),
                        size=len(data),
                        representation=representation,
                        split="development_unassigned",
                    )
                    rows.append(row)
                    write_json(out / f"{filename}.source.json", row)
                    authors[row["author"] or "unknown"] += 1
                    time.sleep(1)  # respect the public service, including small pilots
                except urllib.error.HTTPError as exc:
                    failures[f"http_{exc.code}"] += 1
                    if exc.code in {403, 429, 503}:
                        commons_available = False
                        break
                except (OSError, ValueError, KeyError, EOFError, wave.Error):
                    failures["source_rejected_or_unavailable"] += 1
        result["families"][family] = {
            "status": "ready"
            if len(rows) == target
            else ("incomplete" if commons_available else "unavailable"),
            "count": len(rows),
            "missing": target - len(rows),
            "samples": rows,
            "failed_checks": dict(failures),
            "statistical_detection": "unavailable",
        }
        write_json(out / f"{family}-source.json", result["families"][family])
    for name, url in UNICODE_FILES.items():
        try:
            data = budget.fetch(url, 16 * 1024**2)
            text = data.decode("utf-8-sig")
            if "17.0" not in text[:4096]:
                raise ValueError("Unicode version mismatch")
            (out / name).open("xb").write(data)
            result["unicode"][name] = {
                "status": "ready",
                "version": "17.0.0",
                "url": url,
                "sha256": digest(data),
                "bytes": len(data),
            }
        except (OSError, ValueError):
            result["unicode"][name] = {"status": "unavailable", "url": url}
    result.update(
        downloaded_bytes=budget.bytes,
        raw_download_limit_bytes=MAX_DOWNLOAD,
        seconds=time.monotonic() - budget.started,
    )
    write_json(out / "report.json", result)
    return result
