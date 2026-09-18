from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from core.ctf import CTFService, _trailer
from modules.filestruct_appended import FilestructAppended
from modules.image_gif import (
    ImageGifAnalyzer,
    extract_gif_comments,
    extract_gif_delays_payload,
    gif_structural_end,
    parse_gif_stream,
)


def test_gif_comment_flag_detection_and_ctf_extraction(tmp_path: Path) -> None:
    im = Image.new("P", (10, 10), 0)
    im.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    gif_path = tmp_path / "comment.gif"
    flag = b"flag{gif_comment_payload_extracted}"
    im.save(gif_path, format="GIF", comment=flag)

    res = ImageGifAnalyzer().analyze(gif_path)
    signals = {s.name: s for s in res.signals}
    assert "gif_comment_flag" in signals
    assert signals["gif_comment_flag"].evidence == "verified"
    assert signals["gif_comment_flag"].score == 98

    # CTF solve extracts the comment and confirms
    out_dir = tmp_path / "ctf_comment_out"
    report = CTFService().solve(gif_path, out_dir, mode="quick")
    assert report.verdict == "confirmed"
    assert any("gif-comment" in a.name for a in report.artifacts)


def test_gif_comment_printable_text_payload(tmp_path: Path) -> None:
    im = Image.new("P", (10, 10), 0)
    im.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    gif_path = tmp_path / "comment_text.gif"
    text_comment = b"Secret note: the password is hidden elsewhere!"
    im.save(gif_path, format="GIF", comment=text_comment)

    res = ImageGifAnalyzer().analyze(gif_path)
    signals = {s.name: s for s in res.signals}
    assert "gif_comment_payload" in signals
    assert signals["gif_comment_payload"].score == 65


def test_gif_trailer_detection_and_carving(tmp_path: Path) -> None:
    im = Image.new("P", (10, 10), 0)
    im.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    buf = io.BytesIO()
    im.save(buf, format="GIF")
    raw = buf.getvalue()
    appended = b"CARVED_DATA_WITH;EXTRA;SEMICOLONS;flag{gif_trailer_carved}"
    gif_file = tmp_path / "trailer.gif"
    gif_file.write_bytes(raw + appended)

    res = ImageGifAnalyzer().analyze(gif_file)
    signals = {s.name: s for s in res.signals}
    assert "gif_appended_flag" in signals
    assert signals["gif_appended_flag"].score == 98

    fs_res = FilestructAppended().analyze(gif_file)
    assert fs_res.suspicion >= 70

    trailer = _trailer(gif_file.read_bytes(), ".gif")
    assert trailer == appended


def test_gif_palette_duplicates(tmp_path: Path) -> None:
    palette = [10, 20, 30, 10, 20, 30] + [0] * 762
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[0, 0] = 1
    im = Image.fromarray(arr, mode="P")
    im.putpalette(palette)
    gif_path = tmp_path / "dup_palette.gif"
    im.save(gif_path, format="GIF")

    res = ImageGifAnalyzer().analyze(gif_path)
    signals = {s.name: s for s in res.signals}
    assert "gif_palette_duplicates" in signals
    assert signals["gif_palette_duplicates"].score == 85


def test_gif_frame_delays_ascii_and_2state(tmp_path: Path) -> None:
    flag = "flag{gif_frame_delays_123}"
    delays = [ord(c) for c in flag]
    frames = [Image.new("P", (10, 10), i % 2) for i in range(len(delays))]
    for f in frames:
        f.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)

    buf = io.BytesIO()
    durations = [d * 10 for d in delays]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=durations)
    gif_path = tmp_path / "delays.gif"
    gif_path.write_bytes(buf.getvalue())

    res = ImageGifAnalyzer().analyze(gif_path)
    signals = {s.name: s for s in res.signals}
    assert "gif_delay_flag" in signals

    out_dir = tmp_path / "ctf_delays"
    report = CTFService().solve(gif_path, out_dir, mode="quick")
    assert report.verdict == "confirmed"
    assert any("gif-delays" in a.name for a in report.artifacts)

    # 2-state binary delays test
    payload = b"flag{2state_delays}"
    bits = [int(b) for b in "".join(f"{byte:08b}" for byte in payload)]
    two_state_delays = [20 if b == 1 else 10 for b in bits]
    decoded = extract_gif_delays_payload(two_state_delays)
    assert decoded is not None
    assert decoded[0] == payload
    assert "2-state" in decoded[1]


def test_text_whitespace_and_zerowidth_ctf_raw_extraction(tmp_path: Path) -> None:
    flag_ws = b"flag{whitespace_raw_payload_ctf}"
    bits = [(b >> (7 - i)) & 1 for b in flag_ws for i in range(8)]
    lines = [f"Line {idx}" + ("\t" if bit else " ") for idx, bit in enumerate(bits)]
    lines.extend([f"Padding line {i}" for i in range(10)])
    ws_file = tmp_path / "whitespace.txt"
    ws_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_ws = tmp_path / "out_ws"
    report_ws = CTFService().solve(ws_file, out_ws, mode="quick")
    assert report_ws.verdict == "confirmed"
    assert any("whitespace-extracted" in a.name for a in report_ws.artifacts)

    flag_zw = b"flag{zerowidth_raw_payload_ctf}"
    zw_bits = "".join(f"{b:08b}" for b in flag_zw)
    zw_text = "Here is cover text " + "".join("‌" if c == "1" else "​" for c in zw_bits) + "‍"
    zw_file = tmp_path / "zerowidth.txt"
    zw_file.write_text(zw_text, encoding="utf-8")

    out_zw = tmp_path / "out_zw"
    report_zw = CTFService().solve(zw_file, out_zw, mode="quick")
    assert report_zw.verdict == "confirmed"
    assert any("zerowidth-extracted" in a.name for a in report_zw.artifacts)


def test_gif_structural_end_and_malformed() -> None:
    assert gif_structural_end(b"not_gif") is None
    assert extract_gif_comments(b"not_gif") == b""
    assert extract_gif_delays_payload([]) is None
    assert extract_gif_delays_payload([10, 20]) is None

    parsed = parse_gif_stream(b"GIF89a\x01\x00\x01\x00\x00\x00\x00")
    assert parsed == {"comments": [], "delays": [], "gct_colors": [], "structural_end": None}
