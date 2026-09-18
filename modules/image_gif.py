"""Bounded GIF palette, frame, extension, delay, and trailer analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence

from core.analyzer import Analyzer
from core.context import AnalysisContext
from core.result import AnalysisResult, Signal

_FLAG_PATTERN = re.compile(rb"([a-zA-Z0-9_]{3,24}\{[ -~]{4,120}\})")


class ImageGifAnalyzer(Analyzer):
    name = "image_gif"

    def analyze(self, src: Path) -> AnalysisResult:
        return self.analyze_context(AnalysisContext(src))

    def analyze_context(self, context: AnalysisContext) -> AnalysisResult:
        if context.path.suffix.lower() != ".gif":
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        data = context.data
        if not data.startswith((b"GIF87a", b"GIF89a")):
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        signals: list[Signal] = []

        parsed = parse_gif_stream(data)
        comments_list: list[bytes] = parsed.get("comments", [])
        delays_list: list[int] = parsed.get("delays", [])
        gct_colors: list[tuple[int, int, int]] = parsed.get("gct_colors", [])
        structural_end = parsed.get("structural_end")

        comments, applications = _extension_counts(data)
        if comments:
            signals.append(
                Signal(
                    "gif_comment_extensions",
                    min(55, 15 + comments * 5),
                    f"GIF contains {comments} comment extension(s)",
                    category="gif_extensions",
                    evidence="heuristic" if comments >= 4 else "informational",
                )
            )
        if applications:
            signals.append(
                Signal(
                    "gif_application_extensions",
                    0,
                    f"GIF contains {applications} application extension(s)",
                    category="gif_extensions",
                    evidence="informational",
                )
            )

        for comment in comments_list:
            flag_match = _FLAG_PATTERN.search(comment)
            if flag_match:
                flag_str = flag_match.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "gif_comment_flag",
                        98,
                        f"flag detected in GIF comment: {flag_str}",
                        category="gif_comment",
                        evidence="verified",
                    )
                )
                break
            if len(comment) >= 6 and sum(32 <= b <= 126 for b in comment) / len(comment) >= 0.80:
                signals.append(
                    Signal(
                        "gif_comment_payload",
                        65,
                        f"GIF comment contains text payload ({len(comment)} bytes)",
                        category="gif_comment",
                        evidence="heuristic",
                    )
                )
                break

        if structural_end is not None and len(data) > structural_end:
            trailer = data[structural_end:]
            flag_match = _FLAG_PATTERN.search(trailer[:4096])
            if flag_match:
                flag_str = flag_match.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "gif_appended_flag",
                        98,
                        f"flag detected in GIF trailer ({len(trailer)} bytes): {flag_str}",
                        category="gif_trailer",
                        evidence="verified",
                    )
                )
            else:
                signals.append(
                    Signal(
                        "gif_appended_data",
                        70,
                        f"{len(trailer)} bytes after GIF trailer",
                        category="gif_trailer",
                        evidence="strong",
                    )
                )

        delay_payload = extract_gif_delays_payload(delays_list)
        if delay_payload is not None:
            raw_delay_bytes, desc = delay_payload
            flag_match = _FLAG_PATTERN.search(raw_delay_bytes)
            if flag_match:
                flag_str = flag_match.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "gif_delay_flag",
                        98,
                        f"flag detected in frame delays: {flag_str}",
                        category="gif_delay",
                        evidence="verified",
                    )
                )
            else:
                signals.append(
                    Signal(
                        "gif_delay_payload",
                        80,
                        desc,
                        category="gif_delay",
                        evidence="strong",
                    )
                )

        used_indices: set[int] = set()
        with Image.open(context.path) as image:
            frames: list[np.ndarray] = []
            palette_sizes: list[int] = []
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                if index >= 512:
                    signals.append(
                        Signal(
                            "gif_frame_limit",
                            70,
                            "analysis stopped at the 512-frame safety limit",
                            category="gif_structure",
                            evidence="strong",
                        )
                    )
                    break
                palette = frame.getpalette()
                palette_sizes.append(len(palette) // 3 if palette else 0)
                if frame.mode == "P":
                    used_indices.update(int(idx) for idx in np.unique(np.array(frame)))
                frames.append(np.asarray(frame.convert("RGB"), dtype=np.int16))

        if gct_colors and used_indices:
            used_colors = [gct_colors[idx] for idx in used_indices if idx < len(gct_colors)]
            duplicates = len(used_colors) - len(set(used_colors))
            if duplicates > 0:
                signals.append(
                    Signal(
                        "gif_palette_duplicates",
                        85,
                        f"GIF uses {duplicates} duplicate palette entries across active pixels",
                        category="gif_palette",
                        evidence="strong",
                    )
                )

        if len(set(palette_sizes)) > 1:
            signals.append(
                Signal(
                    "gif_palette_changes",
                    min(60, 25 + len(set(palette_sizes)) * 4),
                    f"palette sizes vary across frames: {sorted(set(palette_sizes))}",
                    category="gif_palette",
                    evidence="heuristic",
                )
            )
        if len(frames) > 1:
            differences = [
                float(np.mean(np.abs(right - left)))
                for left, right in zip(frames, frames[1:], strict=False)
            ]
            near_duplicates = sum(value < 0.01 for value in differences)
            signals.append(
                Signal(
                    "gif_frame_differences",
                    min(50, near_duplicates * 5),
                    f"frames={len(frames)}, near-identical transitions={near_duplicates}, "
                    f"mean delta={float(np.mean(differences)):.4f}",
                    category="gif_frames",
                    evidence="heuristic" if near_duplicates >= 6 else "informational",
                )
            )
        suspicion = max(
            (signal.score for signal in signals if signal.evidence != "informational"),
            default=0,
        )
        return AnalysisResult(self.name, suspicion, tuple(signals), None)


def _extension_counts(data: bytes) -> tuple[int, int]:
    comments = 0
    applications = 0
    offset = 0
    while True:
        offset = data.find(b"\x21", offset)
        if offset < 0 or offset + 1 >= len(data):
            break
        label = data[offset + 1]
        comments += label == 0xFE
        applications += label == 0xFF
        offset += 2
    return comments, applications


def parse_gif_stream(data: bytes) -> dict[str, Any]:
    """Parse GIF blocks to extract comments, frame delays, palette, and trailer offset."""
    if not (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")) or len(data) < 13:
        return {}
    packed = data[10]
    gct_flag = bool(packed & 0x80)
    gct_size = 3 * (1 << ((packed & 7) + 1)) if gct_flag else 0
    offset = 13 + gct_size
    if offset > len(data):
        return {}
    gct_colors = [
        (data[13 + i * 3], data[13 + i * 3 + 1], data[13 + i * 3 + 2])
        for i in range(gct_size // 3)
    ]
    comments: list[bytes] = []
    delays: list[int] = []
    structural_end: int | None = None

    max_blocks = 100_000
    blocks = 0
    while offset < len(data) and blocks < max_blocks:
        blocks += 1
        b = data[offset]
        offset += 1
        if b == 0x3B:  # Trailer
            structural_end = offset
            break
        if b == 0x21:  # Extension
            if offset >= len(data):
                break
            label = data[offset]
            offset += 1
            if label == 0xF9:  # Graphic Control
                if offset + 5 <= len(data):
                    sz = data[offset]
                    delay = int.from_bytes(data[offset + 2 : offset + 4], "little")
                    delays.append(delay)
                    offset += sz + 1
                    if offset < len(data) and data[offset] == 0:
                        offset += 1
                else:
                    break
            elif label == 0xFE:  # Comment
                sub_bytes = bytearray()
                while offset < len(data):
                    sl = data[offset]
                    offset += 1
                    if sl == 0:
                        break
                    if offset + sl > len(data):
                        sub_bytes.extend(data[offset:])
                        offset = len(data)
                        break
                    sub_bytes.extend(data[offset : offset + sl])
                    offset += sl
                comments.append(bytes(sub_bytes))
            else:  # Other extension (0x01 Plain Text, 0xFF Application, etc.)
                while offset < len(data):
                    sl = data[offset]
                    offset += 1
                    if sl == 0:
                        break
                    offset += sl
        elif b == 0x2C:  # Image Descriptor
            if offset + 9 > len(data):
                break
            img_packed = data[offset + 8]
            offset += 9
            if img_packed & 0x80:
                lct_sz = 3 * (1 << ((img_packed & 7) + 1))
                offset += lct_sz
            if offset >= len(data):
                break
            offset += 1  # LZW minimum code size
            while offset < len(data):
                sl = data[offset]
                offset += 1
                if sl == 0:
                    break
                offset += sl
        else:
            break

    return {
        "structural_end": structural_end,
        "comments": comments,
        "delays": delays,
        "gct_colors": gct_colors,
    }


def gif_structural_end(data: bytes) -> int | None:
    """Find the byte offset directly following the terminating 0x3B GIF trailer."""
    parsed = parse_gif_stream(data)
    end = parsed.get("structural_end")
    if end is not None:
        return int(end)
    idx = data.rfind(b";")
    return idx + 1 if idx >= 0 else None


def extract_gif_comments(data: bytes) -> bytes:
    """Extract and concatenate comment sub-blocks from a GIF file."""
    parsed = parse_gif_stream(data)
    comments: list[bytes] = parsed.get("comments", [])
    return b"\n".join(comments) if comments else b""


def extract_gif_delays_payload(delays: list[int]) -> tuple[bytes, str] | None:
    """Decode ASCII or 2-state binary steganography embedded in frame delays."""
    if len(delays) < 4:
        return None
    if all(32 <= d <= 126 for d in delays):
        return bytes(delays), "GIF frame delay ASCII payload"
    unique_delays = sorted(set(delays))
    if len(unique_delays) == 2 and len(delays) >= 16:
        d0, d1 = unique_delays
        bits = [0 if d == d0 else 1 for d in delays]
        octets = bytearray(
            int("".join(str(b) for b in bits[i : i + 8]), 2)
            for i in range(0, len(bits) - 7, 8)
        )
        if _FLAG_PATTERN.search(octets) or (
            len(octets) >= 2 and sum(32 <= b <= 126 for b in octets) / len(octets) >= 0.75
        ):
            return bytes(octets), "GIF 2-state frame delay binary payload"
    return None
