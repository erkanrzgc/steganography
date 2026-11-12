"""Lossless-video LSB analyzer: mocked control flow plus an optional real ffmpeg run."""

import shutil
import subprocess

import numpy as np
import pytest

from modules import video_frames_lsb as vf
from modules.video_frames_lsb import VideoFramesLsbAnalyzer


def frame(seed, size=(64, 96)):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0 : size[0], 0 : size[1]]
    base = 120 + 60 * np.sin(x / 7.0) * np.cos(y / 9.0)
    rgb = base[:, :, None] + rng.normal(0, 2, (*size, 3)) + [0, 5, -5]
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


def replace_lsb(rgb, rate, seed):
    rng = np.random.default_rng(seed)
    flat = rgb.reshape(-1).copy()
    pos = rng.choice(flat.size, int(rate * flat.size), replace=False)
    flat[pos] = (flat[pos] & 0xFE) | rng.integers(0, 2, len(pos)).astype(np.uint8)
    return flat.reshape(rgb.shape)


@pytest.fixture
def fake_tools(monkeypatch):
    monkeypatch.setattr(vf.shutil, "which", lambda name: f"/usr/bin/{name}")


def run(tmp_path, suffix=".mkv"):
    path = tmp_path / f"v{suffix}"
    path.write_bytes(b"video")
    return VideoFramesLsbAnalyzer().analyze(path)


def test_scope_and_missing_tools(tmp_path, monkeypatch):
    assert run(tmp_path, ".png").status == "unsupported"
    monkeypatch.setattr(vf.shutil, "which", lambda name: None)
    assert run(tmp_path).status == "unavailable"


def test_lossy_large_and_failing_inputs(tmp_path, monkeypatch, fake_tools):
    monkeypatch.setattr(vf, "probe", lambda *a: {"codec_name": "h264", "width": 8, "height": 8})
    assert run(tmp_path).status == "unsupported"
    monkeypatch.setattr(vf, "probe", lambda *a: {"codec_name": "ffv1", "width": 99999,
                                                  "height": 99999})
    assert run(tmp_path).status == "unsupported"

    def boom(*args):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(vf, "probe", boom)
    result = run(tmp_path)
    assert result.status == "error" and "decoder failed" in result.error
    monkeypatch.setattr(vf, "probe", lambda *a: {"codec_name": "ffv1", "width": 96, "height": 64})
    monkeypatch.setattr(vf, "frames", lambda *a: np.zeros((0, 64, 96, 3), np.uint8))
    assert run(tmp_path).error == "no frames"


def test_clean_and_replaced_frames_are_scored(tmp_path, monkeypatch, fake_tools):
    monkeypatch.setattr(vf, "probe", lambda *a: {"codec_name": "ffv1", "width": 96, "height": 64,
                                                  "nb_read_packets": "6"})
    clean = np.stack([frame(i) for i in range(5)])
    monkeypatch.setattr(vf, "frames", lambda *a: clean)
    quiet = run(tmp_path)
    assert quiet.status == "ok" and quiet.signals[0].evidence == "informational"
    stego = np.stack([replace_lsb(f, 1.0, i) for i, f in enumerate(clean)])
    monkeypatch.setattr(vf, "frames", lambda *a: stego)
    loud = run(tmp_path)
    assert loud.suspicion == 70 and loud.signals[0].evidence == "heuristic"


def test_runner_and_frame_decoding(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        out = b'{"streams": [{"codec_name": "ffv1"}]}' if "ffprobe" in argv[0] else bytes(100)
        return subprocess.CompletedProcess(argv, 0, out, b"")

    monkeypatch.setattr(vf.subprocess, "run", fake_run)
    assert vf.probe("ffprobe", "x.mkv") == {"codec_name": "ffv1"}
    decoded = vf.frames("ffmpeg", "x.mkv", 4, 2, 60)
    assert decoded.shape == (4, 2, 4, 3) and "mod(n\\,10)" in " ".join(calls[-1])
    monkeypatch.setattr(vf.subprocess, "run",
                        lambda argv, **k: subprocess.CompletedProcess(argv, 1, b"", b"bad"))
    with pytest.raises(RuntimeError, match="bad"):
        vf._run(["ffmpeg"])


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    reason="ffmpeg unavailable")
def test_real_ffv1_video(tmp_path):
    frames = [frame(i, (96, 128)) for i in range(4)]

    def encode(images, path):
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo",  # noqa: S603
                        "-pix_fmt", "rgb24", "-s", "128x96", "-r", "2", "-i", "-", "-c:v", "ffv1",
                        "-pix_fmt", "bgr0", str(path)], input=np.stack(images).tobytes(),
                       check=True)

    encode(frames, tmp_path / "c.mkv")
    encode([replace_lsb(f, 1.0, i) for i, f in enumerate(frames)], tmp_path / "s.mkv")
    analyzer = VideoFramesLsbAnalyzer()
    assert analyzer.analyze(tmp_path / "c.mkv").suspicion == 0
    assert analyzer.analyze(tmp_path / "s.mkv").suspicion == 70
