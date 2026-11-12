"""Builders for hidden-data video fixtures (shared by tests and the private benchmark)."""

from __future__ import annotations

from modules.video_container import _walk, boxes


def hide_in_mdat(data: bytes, payload: bytes) -> bytes:
    """TCSteg-style: insert ``payload`` at the start of mdat and shift every chunk offset."""
    mdat = next((body, stop) for kind, body, stop in boxes(data, 0, len(data)) if kind == b"mdat")
    at = mdat[0]
    out = bytearray(data[:at] + payload + data[at:])
    if data[at - 12 : at - 8] == b"mdat":  # 64-bit largesize header
        size = int.from_bytes(data[at - 8 : at], "big") + len(payload)
        out[at - 8 : at] = size.to_bytes(8, "big")
    else:
        out[at - 8 : at - 4] = (int.from_bytes(data[at - 8 : at - 4], "big")
                                + len(payload)).to_bytes(4, "big")
    shift = len(payload)
    for kind, body, _stop in _walk(bytes(out), 0, len(out), [10**6]):
        if kind not in (b"stco", b"co64"):
            continue
        width = 4 if kind == b"stco" else 8
        count = int.from_bytes(out[body + 4 : body + 8], "big")
        for i in range(count):
            pos = body + 8 + i * width
            value = int.from_bytes(out[pos : pos + width], "big")
            if value >= at:
                out[pos : pos + width] = (value + shift).to_bytes(width, "big")
    return bytes(out)


def avi_with_chunk(data: bytes, kind: bytes, payload: bytes) -> bytes:
    """Append a chunk inside the RIFF body and fix the RIFF size."""
    chunk = kind + len(payload).to_bytes(4, "little") + payload + b"\x00" * (len(payload) & 1)
    riff_end = 8 + int.from_bytes(data[4:8], "little")
    out = data[:riff_end] + chunk + data[riff_end:]
    return out[:4] + (riff_end - 8 + len(chunk)).to_bytes(4, "little") + out[8:]
