"""Explicit small development fixtures from independent encoders, never blind data."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
import urllib.parse
import wave
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import piexif
from PIL import Image

from core.tools import ToolRunner


def generate_matrix(output: Path, *, external: bool = False) -> dict[str, Any]:
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("matrix output must not use symlinks")
    output.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    pixels = np.random.default_rng(71011).integers(0, 256, (128, 128, 3), dtype=np.uint8)

    def save(method: str, suffix: str, encoder: Any, tools: tuple[str, ...] = ()) -> None:
        identity = f"case-{len(cases) + 1:02d}"
        payload = f"flag{{matrix_{identity}}}".encode()
        path = output / (identity + suffix)
        encoder(path, payload)
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024**2:
            raise RuntimeError("independent fixture generation failed")
        cases.append({
            "id": identity, "source_group": "generated-matrix-v1",
            "method_family": method, "input": path.name,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(), "required_tools": list(tools),
        })

    def image(path: Path, payload: bytes) -> None:
        Image.fromarray(pixels).save(path)
        with path.open("ab") as stream:
            stream.write(payload)

    def metadata(path: Path, payload: bytes) -> None:
        data = piexif.dump({"0th": {piexif.ImageIFD.ImageDescription: payload}})
        Image.fromarray(pixels).save(path, exif=data)

    def comment(path: Path, payload: bytes) -> None:
        Image.fromarray(pixels).save(path, comment=payload)

    def archive(path: Path, payload: bytes) -> None:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                member_zip = zipfile.ZipInfo(
                    "evidence/payload.txt", date_time=(1980, 1, 1, 0, 0, 0)
                )
                member_zip.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(member_zip, payload)
        else:
            # Fixed metadata and gzip mtime make manifest identities reproducible.
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                member = tarfile.TarInfo("evidence/payload.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
            path.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))

    def audio(path: Path, payload: bytes, *, width: int, channel: int | None) -> None:
        channels = 1 if channel is None else 2
        samples = np.random.default_rng(814).integers(-100, 100, (4096, channels), dtype=np.int32)
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        chosen = 0 if channel is None else channel
        samples[:len(bits), chosen] = (samples[:len(bits), chosen] & ~1) | bits
        raw = (samples + 128).astype(np.uint8).tobytes() if width == 1 else samples.astype(
            "<i2" if width == 2 else "<i4"
        ).tobytes()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(width)
            wav.setframerate(8000)
            wav.writeframes(raw)

    def whitespace(path: Path, payload: bytes) -> None:
        bits = "".join(format(byte, "08b") for byte in payload)
        path.write_text("".join("cover line" + ("\t" if bit == "1" else " ") + "\n"
                                for bit in bits), encoding="utf-8")

    def zerowidth(path: Path, payload: bytes) -> None:
        bits = "".join(format(byte, "08b") for byte in payload)
        path.write_text("Cover text." + "".join("\u200c" if bit == "1" else "\u200b"
                                               for bit in bits) + "\u200d", encoding="utf-8")

    if not external:
        for method, encoder in (
            ("base32", base64.b32encode), ("base85", base64.b85encode),
            ("url", lambda p: urllib.parse.quote_from_bytes(p).encode()),
            ("base64-gzip", lambda p: base64.b64encode(gzip.compress(p, mtime=0))),
        ):
            save(method, ".txt", lambda path, p, enc=encoder: path.write_bytes(enc(p)))
        save("zip", ".zip", archive)
        save("gzip-tar", ".gz", archive)
        save("png-appended", ".png", image)
        save("bmp-appended", ".bmp", image)
        save("jpeg-exif", ".jpg", metadata)
        save("gif-comment", ".gif", comment)
        for width, channel in ((1, None), (2, None), (4, None), (2, 0), (2, 1)):
            save("wav-lsb-mono" if channel is None else "wav-lsb-channel", ".wav",
                 lambda path, p, w=width, c=channel: audio(path, p, width=w, channel=c))
        save("text-whitespace", ".txt", whitespace)
        save("text-zerowidth", ".txt", zerowidth)
    else:
        runner = ToolRunner(timeout=30)

        def embed(path: Path, payload: bytes, *, tool: str, suffix: str) -> None:
            cover = output / ("cover" + suffix)
            Image.fromarray(pixels).save(cover)
            message = output / "message.bin"
            message.write_bytes(payload)
            if tool == "steghide":
                args = ["embed", "-cf", cover.name, "-ef", message.name,
                        "-sf", path.name, "-p", "", "-f"]
            elif tool == "outguess":
                args = ["-d", message.name, cover.name, path.name]
            else:
                args = ["embed", "-a", "randomlsb", "-cf", cover.name,
                        "-mf", message.name, "-sf", path.name, "-E"]
            result = runner.run(tool, args, cwd=output)
            if result.status != "completed":
                raise RuntimeError(f"{tool} independent fixture generation unavailable")

        for tool, suffix in (("steghide", ".bmp"), ("steghide", ".jpg"),
                             ("outguess", ".jpg"), ("openstego", ".png")):
            save(tool, suffix, lambda path, p, t=tool, s=suffix: embed(path, p, tool=t, suffix=s),
                 tools=(tool,))
    document = {
        "schema_version": "ctf-evaluation-manifest-v1",
        "suite_id": "matrix-external-v1" if external else "matrix-native-v1",
        "mode": "balanced" if external else "quick",
        "limits": dict(max_depth=3, max_artifacts=256, max_bytes=1024**3,
                       tool_timeout=30, job_timeout=180),
        "cases": cases,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
    return document
