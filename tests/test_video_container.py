"""Video container hiding rules on hand-built MP4, Matroska and AVI structures."""

import shutil
import subprocess

import numpy as np
import pytest

from modules import video_container as vc
from modules.video_container import VideoContainerAnalyzer
from tests.video_fixtures import avi_with_chunk, hide_in_mdat

RANDOM = np.random.default_rng(0).bytes(200_000)


def box(kind, body):
    return (len(body) + 8).to_bytes(4, "big") + kind + body


def full(kind, body):
    return box(kind, b"\x00\x00\x00\x00" + body)


def mp4(samples=(1000, 1200, 800), chunk_table=b"stco", fixed=0, large_mdat=False):
    sizes = b"".join(s.to_bytes(4, "big") for s in samples)
    stsz = full(b"stsz", fixed.to_bytes(4, "big") + len(samples).to_bytes(4, "big")
                + (b"" if fixed else sizes))
    stsc = full(b"stsc", (1).to_bytes(4, "big") + (1).to_bytes(4, "big")
                + (1).to_bytes(4, "big") + (1).to_bytes(4, "big"))
    ftyp = box(b"ftyp", b"isom\x00\x00\x02\x00")

    def build(first):
        width = 4 if chunk_table == b"stco" else 8
        offsets, pos = [], first
        for s in samples:
            offsets.append(pos.to_bytes(width, "big"))
            pos += fixed or s
        table = full(chunk_table, len(samples).to_bytes(4, "big") + b"".join(offsets))
        stbl = box(b"stbl", stsc + stsz + table)
        return box(b"moov", box(b"trak", box(b"mdia", box(b"minf", stbl))))

    media = bytes(sum(fixed or s for s in samples))
    moov = build(0)
    head = len(ftyp) + len(moov) + (16 if large_mdat else 8)
    moov = build(head)
    if large_mdat:
        mdat = (1).to_bytes(4, "big") + b"mdat" + (len(media) + 16).to_bytes(8, "big") + media
    else:
        mdat = box(b"mdat", media)
    return ftyp + moov + mdat


def analyze(tmp_path, data):
    path = tmp_path / "v.bin"
    path.write_bytes(data)
    result = VideoContainerAnalyzer().analyze(path)
    return result.status, {s.name: s for s in result.signals}


@pytest.mark.parametrize("kwargs", [{}, {"chunk_table": b"co64"}, {"fixed": 900},
                                    {"large_mdat": True}])
def test_mp4_sample_tables_cover_clean_media(tmp_path, kwargs):
    data = mp4(**kwargs)
    assert sum(e - s for s, e in vc.sample_ranges(data)) == len(data) - data.find(b"mdat") - 4 - (
        8 if kwargs.get("large_mdat") else 0)
    assert analyze(tmp_path, data) == ("ok", {})
    hidden = analyze(tmp_path, hide_in_mdat(data, RANDOM))[1]
    assert hidden["mp4_unreferenced_media_data"].evidence == "strong"


def test_mp4_small_gaps_and_missing_tables_are_clean(tmp_path):
    assert analyze(tmp_path, hide_in_mdat(mp4(), RANDOM[:1000]))[1] == {}
    ftyp = box(b"ftyp", b"isom")
    assert analyze(tmp_path, ftyp + box(b"mdat", RANDOM[:100_000]))[1] == {}
    assert analyze(tmp_path, ftyp + box(b"free", b""))[1] == {}
    tables = full(b"stsz", bytes(8)) + full(b"stsc", bytes(4))
    no_offsets = ftyp + box(b"moov", box(b"trak", box(b"stbl", tables))) + box(b"mdat", b"x")
    assert analyze(tmp_path, no_offsets)[1] == {}
    truncated = full(b"stco", (5).to_bytes(4, "big") + (8).to_bytes(4, "big"))
    assert vc._table(truncated, 8, len(truncated), 4, 1) == [(8,)]
    assert list(vc.boxes(b"\x00\x00\x00\x04abcd", 0, 8)) == []
    assert list(vc.boxes(b"\x00\x00\x00\x00mdatXYZ", 0, 11)) == [(b"mdat", 8, 11)]


def test_mp4_walk_budget(monkeypatch):
    assert list(vc._walk(mp4(), 0, len(mp4()), [1])) == [next(vc.boxes(mp4(), 0, len(mp4())))]


def vint(value, length):
    return ((1 << (7 * length)) | value).to_bytes(length, "big")


def element(element_id, body):
    head = element_id.to_bytes((element_id.bit_length() + 7) // 8, "big")
    return head + vint(len(body), 4) + body


def mkv(*children):
    header = element(0x1A45DFA3, element(0x4282, b"matroska"))
    return header + element(vc.SEGMENT, b"".join(children))


def attachment(mime):
    return element(vc.ATTACHMENTS, element(vc.ATTACHED_FILE, element(vc.MIME, mime)
                                           + element(vc.FILE_DATA, RANDOM[:5000])))


def test_matroska_rules(tmp_path):
    assert analyze(tmp_path, mkv(element(0x1549A966, b"info"))) == ("ok", {})
    found = analyze(tmp_path, mkv(attachment(b"application/octet-stream")))[1]
    assert found["mkv_attachment"].evidence == "strong"
    assert analyze(tmp_path, mkv(attachment(b"application/x-truetype-font")))[1] == {}
    bare = element(vc.ATTACHMENTS, element(vc.ATTACHED_FILE, element(vc.FILE_DATA, b"x" * 10)))
    assert "unknown" in analyze(tmp_path, mkv(bare))[1]["mkv_attachment"].detail
    tag = element(vc.TAGS, element(vc.TAG, element(vc.SIMPLE_TAG,
                                                   element(vc.TAG_BINARY, RANDOM[:600]))))
    assert analyze(tmp_path, mkv(tag))[1]["mkv_binary_tag"].score == 72
    void = element(vc.VOID, b"words " * 100)
    assert analyze(tmp_path, mkv(void))[1]["mkv_void_payload"].score == 50
    assert analyze(tmp_path, mkv(element(vc.VOID, bytes(600))))[1] == {}
    unknown_size = b"\x1a\x45\xdf\xa3" + vint(0, 1) + vc.SEGMENT.to_bytes(4, "big") + b"\x01" \
        + b"\xff" * 7 + attachment(b"application/zip")
    assert "mkv_attachment" in analyze(tmp_path, unknown_size)[1]
    assert analyze(tmp_path, b"\x1a\x45\xdf\xa3\x00")[1] == {}
    assert list(vc.ebml(b"\x40", 0, 1)) == []


def test_matroska_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "MAX_ITEMS", 1)
    assert analyze(tmp_path, mkv(attachment(b"application/octet-stream")))[1] == {}


def avi(*chunks):
    body = b"AVI " + b"".join(k + len(b).to_bytes(4, "little") + b + b"\x00" * (len(b) & 1)
                              for k, b in chunks)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def test_avi_rules(tmp_path):
    clean = avi((b"LIST", b"hdrl" + bytes(100)), (b"JUNK", bytes(3000)), (b"idx1", bytes(32)))
    assert analyze(tmp_path, clean) == ("ok", {})
    found = analyze(tmp_path, avi_with_chunk(clean, b"JUNK", RANDOM[:5001]))[1]
    assert found["avi_junk_payload"].evidence == "strong"
    assert "avi_unknown_chunk" in analyze(tmp_path, avi_with_chunk(clean, b"zzzz", RANDOM[:500]))[1]
    assert "avi_trailing_data" in analyze(tmp_path, clean + b"extra")[1]


def test_scope(tmp_path, monkeypatch):
    assert analyze(tmp_path, b"plain")[0] == "unsupported"
    monkeypatch.setattr(VideoContainerAnalyzer, "max_bytes", 1)
    assert analyze(tmp_path, mp4())[0] == "unsupported"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_real_encoder_output_stays_playable_and_is_flagged(tmp_path):
    clean, ffmpeg = tmp_path / "c.mp4", shutil.which("ffmpeg")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",  # noqa: S603
                    "testsrc2=size=160x120:rate=10:duration=2", "-c:v", "libx264", str(clean)],
                   check=True)
    assert VideoContainerAnalyzer().analyze(clean).signals == ()
    hidden = tmp_path / "h.mp4"
    hidden.write_bytes(hide_in_mdat(clean.read_bytes(), RANDOM))
    decode = subprocess.run([ffmpeg, "-v", "error", "-i", str(hidden), "-f", "null", "-"],  # noqa: S603
                            capture_output=True, text=True)
    assert decode.returncode == 0 and decode.stderr == ""
    assert VideoContainerAnalyzer().analyze(hidden).signals[0].name == "mp4_unreferenced_media_data"
