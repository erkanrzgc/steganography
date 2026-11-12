"""zsteg output filtering: natural-image noise is ignored, real low-plane content counts."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from modules import external_tools as et
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


def test_metadata_and_scanline_rows_with_spaces_in_their_names():
    assert _zsteg_candidate('meta Comment        .. text: "ctf{hidden_flag_here}"')
    assert _zsteg_candidate('meta Creation Time  .. text: "flag{in_a_spaced_key}"')
    assert not _zsteg_candidate('meta Software       .. text: "Adobe ImageReady"')
    assert _zsteg_candidate("scanline extradata  .. file: Zip archive data, at least v2.0")
    assert not _zsteg_candidate('scanline extradata  .. text: "abababab"')


def test_zsteg_runs_default_checks_and_exhaustive_low_planes():
    argv = ZstegAdapter().commands(Path("x.png"))
    assert argv == [["zsteg", "x.png"], ["zsteg", "--all", "--bits", "1,2", "x.png"]]


def fake_runs(monkeypatch, outputs):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        result = outputs[len(calls) - 1]
        if isinstance(result, Exception):
            raise result
        kwargs["stdout"].write(result.encode())
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(et.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(et.subprocess, "run", run)
    return calls


def test_outputs_of_both_runs_are_merged_and_deduplicated(tmp_path, monkeypatch):
    flag = "b1,rgb,lsb,xy       .. text: 'flag{seen_twice}'"
    meta = 'meta Comment .. text: "ctf{meta}"'
    calls = fake_runs(monkeypatch, [CLEAN + "\r" + flag, flag + "\r" + meta])
    result = ZstegAdapter().analyze(tmp_path / "x.png")
    assert [argv[0] for argv in calls] == ["/usr/bin/zsteg", "/usr/bin/zsteg"]
    assert result.suspicion == 70 and "2 candidate" in result.signals[0].detail
    calls = fake_runs(monkeypatch, [CLEAN, subprocess.TimeoutExpired("zsteg", 20)])
    timed_out = ZstegAdapter().analyze(tmp_path / "x.png")
    assert timed_out.status == "error" and len(calls) == 2


@pytest.mark.skipif(not shutil.which("zsteg"), reason="zsteg unavailable")
def test_real_zsteg_flags_a_text_chunk_but_not_a_clean_image(tmp_path):
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:96, 0:128]
    smooth = (120 + 50 * np.sin(x / 9.0)[..., None] + rng.normal(0, 2, (96, 128, 3)))
    image = Image.fromarray(np.clip(smooth, 0, 255).astype(np.uint8))
    image.save(tmp_path / "clean.png")
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "ctf{zsteg_meta_row}")
    image.save(tmp_path / "flag.png", pnginfo=info)
    assert ZstegAdapter().analyze(tmp_path / "clean.png").signals == ()
    assert ZstegAdapter().analyze(tmp_path / "flag.png").suspicion == 70
