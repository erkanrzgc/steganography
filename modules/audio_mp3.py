"""MP3 audio steganalysis, ID3v2 frame/padding extraction, and trailer carving."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.carrier import Carrier
from core.context import AnalysisContext
from core.result import AnalysisResult, EmbedResult, Signal

_FLAG_PATTERN = re.compile(rb"([a-zA-Z0-9_]{3,24}\{[ -~]{4,120}\})")

# Bitrate lookup tables (in kbps)
_MPEG1_L3_BITRATES = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_MPEG1_L2_BITRATES = (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384)
_MPEG1_L1_BITRATES = (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448)
_MPEG2_L1_BITRATES = (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256)
_MPEG2_L23_BITRATES = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)

_MPEG1_SAMPLERATES = (44100, 48000, 32000)
_MPEG2_SAMPLERATES = (22050, 24000, 16000)
_MPEG25_SAMPLERATES = (11025, 12000, 8000)


def parse_synchsafe(b: bytes) -> int:
    """Parse 4 synchsafe bytes into an integer."""
    if len(b) < 4:
        return 0
    return (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]


def get_mpeg_frame_length(b: bytes) -> tuple[int, int] | None:
    """Parse MPEG audio frame header and return (frame_length_bytes, private_bit)."""
    if len(b) < 4:
        return None
    b1, b2, b3, _ = b[:4]
    if b1 != 0xFF or (b2 & 0xE0) != 0xE0:
        return None
    ver_bits = (b2 >> 3) & 3
    layer_bits = (b2 >> 1) & 3
    if ver_bits == 1 or layer_bits == 0:
        return None

    br_idx = (b3 >> 4) & 0x0F
    sr_idx = (b3 >> 2) & 3
    if br_idx in (0, 15) or sr_idx == 3:
        return None

    if ver_bits == 3:  # MPEG-1
        sr = _MPEG1_SAMPLERATES[sr_idx]
        if layer_bits == 3:  # Layer I
            br = _MPEG1_L1_BITRATES[br_idx] * 1000
        elif layer_bits == 2:  # Layer II
            br = _MPEG1_L2_BITRATES[br_idx] * 1000
        else:  # Layer III
            br = _MPEG1_L3_BITRATES[br_idx] * 1000
    else:  # MPEG-2 / MPEG-2.5
        sr = (_MPEG2_SAMPLERATES if ver_bits == 2 else _MPEG25_SAMPLERATES)[sr_idx]
        if layer_bits == 3:  # Layer I
            br = _MPEG2_L1_BITRATES[br_idx] * 1000
        else:  # Layer II & III
            br = _MPEG2_L23_BITRATES[br_idx] * 1000

    padding = (b3 >> 1) & 1
    private = b3 & 1

    if layer_bits == 3:
        frame_len = int((12 * br / sr) + padding) * 4
    elif layer_bits == 1 and ver_bits != 3:
        frame_len = int(72 * br / sr) + padding
    else:
        frame_len = int(144 * br / sr) + padding

    if frame_len < 4:
        return None
    return frame_len, private


def parse_id3v2(data: bytes) -> dict[str, Any]:
    """Parse ID3v2 tags, extracting frames, padding, and embedded artwork."""
    if not data.startswith(b"ID3") or len(data) < 10:
        return {}
    major_ver = data[3]
    rev = data[4]
    flags = data[5]
    tag_size = parse_synchsafe(data[6:10])
    total_tag_len = 10 + tag_size + (10 if (flags & 0x10) else 0)

    frames: list[dict[str, Any]] = []
    padding = b""
    offset = 10
    limit = min(len(data), 10 + tag_size)

    while offset + 10 <= limit:
        # Check for padding start (null bytes)
        if data[offset : offset + 4] == b"\x00\x00\x00\x00":
            padding = data[offset:limit]
            break
        frame_id = data[offset : offset + 4]
        try:
            id_str = frame_id.decode("ascii")
            if not id_str.isalnum():
                padding = data[offset:limit]
                break
        except UnicodeDecodeError:
            padding = data[offset:limit]
            break

        if major_ver == 4:
            frame_sz = parse_synchsafe(data[offset + 4 : offset + 8])
        else:
            frame_sz = int.from_bytes(data[offset + 4 : offset + 8], "big")

        if frame_sz <= 0 or offset + 10 + frame_sz > limit:
            padding = data[offset:limit]
            break

        content = data[offset + 10 : offset + 10 + frame_sz]
        frames.append({"id": id_str, "size": frame_sz, "content": content})
        offset += 10 + frame_sz

    return {
        "version": (major_ver, rev),
        "flags": flags,
        "tag_size": tag_size,
        "total_tag_len": total_tag_len,
        "frames": frames,
        "padding": padding,
    }


def mp3_structural_end(data: bytes) -> int | None:
    """Find the byte offset where MPEG audio frames and ID3v1 end."""
    if not data:
        return None

    offset = 0
    id3 = parse_id3v2(data)
    if id3:
        offset = id3["total_tag_len"]

    last_valid_end = offset
    max_frames = 100_000
    frames_count = 0

    while offset + 4 <= len(data) and frames_count < max_frames:
        parsed = get_mpeg_frame_length(data[offset : offset + 4])
        if parsed is None:
            break
        frame_len, _ = parsed
        if offset + frame_len > len(data):
            break
        offset += frame_len
        last_valid_end = offset
        frames_count += 1

    # Check for ID3v1 tag at EOF (starts with b"TAG" and is 128 bytes)
    if last_valid_end + 128 <= len(data) and data[last_valid_end : last_valid_end + 3] == b"TAG":
        last_valid_end += 128
    elif len(data) >= 128 and data[-128:-125] == b"TAG" and last_valid_end <= len(data) - 128:
        return last_valid_end

    return last_valid_end if frames_count > 0 or id3 else None


def extract_mp3_payloads(data: bytes) -> list[tuple[str, bytes, str, bool]]:
    """Extract candidate CTF artifacts from ID3 frames, padding, artwork, and trailer."""
    candidates: list[tuple[str, bytes, str, bool]] = []
    id3 = parse_id3v2(data)

    if id3:
        # 1. Inspect ID3 Frames
        for frame in id3.get("frames", []):
            content: bytes = frame["content"]
            fid = frame["id"]
            if fid in {"COMM", "TXXX", "PRIV", "TIT2", "TPE1", "USER"}:
                # Strip null/encoding prefix if present
                clean = content.lstrip(b"\x00\x01\x02\x03eng\x00")
                is_flag = bool(_FLAG_PATTERN.search(content))
                if is_flag or (len(clean) >= 6 and any(32 <= b <= 126 for b in clean)):
                    candidates.append((
                        f"id3-{fid.lower()}.txt",
                        clean if clean else content,
                        f"ID3v2 {fid} text payload",
                        is_flag,
                    ))
            elif fid == "APIC":
                # Embedded album art: locate JPEG or PNG magic
                jpg_idx = content.find(b"\xff\xd8\xff")
                png_idx = content.find(b"\x89PNG\r\n\x1a\n")
                if jpg_idx >= 0:
                    candidates.append((
                        "mp3-artwork.jpg",
                        content[jpg_idx:],
                        "ID3v2 embedded JPEG album art",
                        False,
                    ))
                elif png_idx >= 0:
                    candidates.append((
                        "mp3-artwork.png",
                        content[png_idx:],
                        "ID3v2 embedded PNG album art",
                        False,
                    ))

        # 2. Inspect ID3 Padding
        padding: bytes = id3.get("padding", b"")
        non_zero = padding.strip(b"\x00")
        if non_zero:
            is_flag = bool(_FLAG_PATTERN.search(non_zero))
            candidates.append((
                "id3-padding.bin",
                non_zero,
                "ID3v2 padding area hidden payload",
                is_flag,
            ))

    # 3. Trailer Carving
    end = mp3_structural_end(data)
    if end is not None and len(data) > end:
        trailer = data[end:]
        is_flag = bool(_FLAG_PATTERN.search(trailer))
        candidates.append((
            "trailer.bin",
            trailer,
            "data appended after MP3 audio stream",
            is_flag,
        ))

    return candidates


class AudioMp3(Carrier):
    name = "audio_mp3"
    method_id = "audio_mp3"
    extensions = (".mp3",)
    can_embed = False
    can_extract = False
    priority = 150
    requires_explicit = False

    def capacity(self, src: Path) -> int:
        del src
        return 0

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        del src, payload, out
        raise NotImplementedError("audio_mp3 is analysis-only")

    def extract(self, src: Path) -> bytes:
        del src
        raise NotImplementedError("audio_mp3 is analysis-only")

    def analyze(self, src: Path) -> AnalysisResult:
        if not src.is_file():
            return AnalysisResult(
                self.name, 0, (), None, status="unsupported", error="file does not exist"
            )
        return self.analyze_context(AnalysisContext(src))

    def analyze_context(self, context: AnalysisContext) -> AnalysisResult:
        if context.path.suffix.lower() != ".mp3":
            return AnalysisResult(self.name, 0, (), None, status="unsupported")

        data = context.data
        is_mp3 = data.startswith(b"ID3") or (
            len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
        )
        if not is_mp3:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")

        signals: list[Signal] = []
        id3 = parse_id3v2(data)

        # 1. ID3v2 Analysis
        if id3:
            frames = id3.get("frames", [])
            v_maj = id3["version"][0]
            signals.append(
                Signal(
                    "id3_metadata_present",
                    0,
                    f"ID3v2.{v_maj} tag with {len(frames)} frames ({id3['tag_size']} bytes)",
                    category="id3_metadata",
                    evidence="informational",
                )
            )
            for frame in frames:
                fid = frame["id"]
                content = frame["content"]
                m = _FLAG_PATTERN.search(content)
                if m:
                    flag_str = m.group(1).decode("latin1", errors="replace")
                    signals.append(
                        Signal(
                            "id3_frame_flag",
                            98,
                            f"flag detected in ID3 frame {fid}: {flag_str}",
                            category="id3_metadata",
                            evidence="verified",
                        )
                    )
                elif fid in {"COMM", "TXXX", "PRIV"} and len(content) >= 8:
                    clean = content.lstrip(b"\x00\x01\x02\x03eng\x00")
                    if any(32 <= b <= 126 for b in clean):
                        signals.append(
                            Signal(
                                "id3_frame_payload",
                                60,
                                f"ID3 frame {fid} contains text payload ({len(clean)} bytes)",
                                category="id3_metadata",
                                evidence="heuristic",
                            )
                        )

            # Padding analysis
            padding = id3.get("padding", b"")
            non_zero = padding.strip(b"\x00")
            if non_zero:
                m_pad = _FLAG_PATTERN.search(non_zero)
                if m_pad:
                    flag_str = m_pad.group(1).decode("latin1", errors="replace")
                    signals.append(
                        Signal(
                            "id3_padding_flag",
                            98,
                            f"flag detected in ID3 padding: {flag_str}",
                            category="id3_padding",
                            evidence="verified",
                        )
                    )
                else:
                    signals.append(
                        Signal(
                            "id3_padding_payload",
                            75,
                            f"{len(non_zero)} non-zero bytes hidden in ID3 padding",
                            category="id3_padding",
                            evidence="strong",
                        )
                    )

        # 2. MPEG Audio Frames & Private Bit Analysis
        offset = id3["total_tag_len"] if id3 else 0
        private_bits: list[int] = []
        max_frames = 50_000
        frames_count = 0

        while offset + 4 <= len(data) and frames_count < max_frames:
            parsed = get_mpeg_frame_length(data[offset : offset + 4])
            if parsed is None:
                break
            frame_len, private = parsed
            if offset + frame_len > len(data):
                break
            private_bits.append(private)
            offset += frame_len
            frames_count += 1

        if len(private_bits) >= 32:
            raw_bytes = bytearray(
                int("".join(str(b) for b in private_bits[i : i + 8]), 2)
                for i in range(0, len(private_bits) - 7, 8)
            )
            m_priv = _FLAG_PATTERN.search(raw_bytes)
            if m_priv:
                flag_str = m_priv.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "mp3_private_bit_flag",
                        98,
                        f"flag detected in frame private bits: {flag_str}",
                        category="mp3_header",
                        evidence="verified",
                    )
                )
            elif sum(32 <= b <= 126 for b in raw_bytes) / len(raw_bytes) >= 0.85:
                signals.append(
                    Signal(
                        "mp3_private_bit_payload",
                        80,
                        f"frame private bits encode ASCII string: {bytes(raw_bytes[:24])!r}",
                        category="mp3_header",
                        evidence="strong",
                    )
                )

        # 3. Trailer / Appended Data Analysis
        end = mp3_structural_end(data)
        if end is not None and len(data) > end:
            trailer = data[end:]
            m_trl = _FLAG_PATTERN.search(trailer[:4096])
            if m_trl:
                flag_str = m_trl.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "mp3_appended_flag",
                        98,
                        f"flag detected in MP3 trailer ({len(trailer)} bytes): {flag_str}",
                        category="mp3_trailer",
                        evidence="verified",
                    )
                )
            else:
                signals.append(
                    Signal(
                        "mp3_appended_data",
                        70,
                        f"{len(trailer)} bytes appended after MP3 stream",
                        category="mp3_trailer",
                        evidence="strong",
                    )
                )

        suspicion = max(
            (s.score for s in signals if s.evidence != "informational"),
            default=0,
        )
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
