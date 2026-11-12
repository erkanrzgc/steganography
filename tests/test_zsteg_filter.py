"""zsteg output filtering: natural-image noise is ignored, real low-plane content counts."""

from modules.external_tools import ZstegAdapter, _zsteg_candidate

# Lines observed from `zsteg --all` on a clean 512x512 DIV2K crop (with zsteg's \r separators).
CLEAN = "\r".join([
    'imagedata           .. text: "\' /)$7\\t\\t"',
    "chunk:0:IHDR        .. file: Lotus unknown worksheet or configuration, revision 0",
    'b2,r,lsb,xy         .. text: "QP\\rg-`N^"',
    "b3,bgr,msb,xy       .. file: OpenPGP Public Key",
    "b6p,g,msb,xy        .. file: zlib compressed data",
    'b6p,rgb,lsb,xy      .. text: "NJE@AAAFFFFFAAFKGGGFAAFFJFFFFLLKFKFFKAAAAFLKLGEST[[ZZV[V_a_[t"',
    'b7p,bgr,lsb,xY      .. text: "quvvsrx{}{zvxz{z|zuvuqqprqosonqurpvvutyz}"',
    'b4,r,msb,xy         .. text: ["U" repeated 10 times]',
    'b2,r,msb,xY         .. text: "jUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"',
    'b1,g,lsb,xy         .. text: "abababababababababababab"',
    "b1,r,lsb,xy         .. b1,r,msb,xy         .. ",
])


def test_clean_image_noise_is_not_a_candidate():
    assert ZstegAdapter().interpret(CLEAN, 0) == ()


def test_low_plane_text_files_flags_and_appended_data_count():
    assert _zsteg_candidate('b1,rgb,lsb,xy       .. text: "3268:Vjt+dsA1Xhjx7UiEr4MibvpOV"')
    assert _zsteg_candidate("b1,rgb,lsb,xy .. text: 'flag{zsteg}'")
    assert _zsteg_candidate("b2,r,lsb .. file: zip")
    assert _zsteg_candidate("extradata:0          .. file: Zip archive data, at least v2.0")
    assert _zsteg_candidate('chunk:1:tEXt         .. text: "CTF{in_a_text_chunk}"')
    assert not _zsteg_candidate('b1,r,lsb,xy         .. text: "short text"')
    assert not _zsteg_candidate("b1,r,lsb,xy         .. file: OpenPGP Public Key")
    assert not _zsteg_candidate('b8,b,lsb,xy         .. text: "abcdefghijklmnopqrstuvwxyz{abc}"')
    assert not _zsteg_candidate("garbage without a row")
    signals = ZstegAdapter().interpret(CLEAN + "\r" + "b1,rgb,lsb,xy .. text: 'flag{x1}'", 0)
    assert len(signals) == 1 and signals[0].evidence == "strong"
