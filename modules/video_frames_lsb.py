"""LSB replacement in losslessly coded video frames (optional ffmpeg/ffprobe).

Lossy codecs (H.264, VP9, MJPEG, ...) destroy pixel LSBs, so only lossless
codecs are examined: FFV1, HuffYUV, UtVideo, PNG/QuickTime RLE and raw video. Up
to ``MAX_FRAMES`` evenly spaced frames are decoded to RGB by a resource-limited
ffmpeg child process. The Weighted-Stego estimator
(``modules.lsb_replacement``) runs on each frame and the median estimate is
scored on the same 0..70 heuristic scale as still images.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal
from modules import lsb_replacement

SUFFIXES = (".mkv", ".avi", ".mov", ".mp4", ".webm", ".nut")
LOSSLESS = {"ffv1", "huffyuv", "ffvhuff", "utvideo", "png", "rawvideo", "qtrle", "magicyuv"}
MAX_FRAMES = 6
MAX_FRAME_PIXELS = 4096 * 2160
TIMEOUT_SECONDS = 60
MEMORY_BYTES = 1024 * 1024 * 1024


def _limits() -> None:  # pragma: no cover - runs in the child before exec
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SECONDS, TIMEOUT_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _run(argv: list[str]) -> bytes:
    completed = subprocess.run(  # noqa: S603 - resolved executable, fixed argv
        argv, capture_output=True, check=False, timeout=TIMEOUT_SECONDS,
        stdin=subprocess.DEVNULL, env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"},
        preexec_fn=_limits if os.name == "posix" else None,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace")[-200:])
    return completed.stdout


def probe(ffprobe: str, src: Path) -> dict:
    """Codec, size and frame count of the first video stream."""
    out = _run([ffprobe, "-v", "error", "-threads", "1", "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=codec_name,width,height,nb_read_packets",
                "-of", "json", str(src)])
    streams = json.loads(out or b"{}").get("streams") or [{}]
    return streams[0]


def frames(ffmpeg: str, src: Path, width: int, height: int, count: int) -> np.ndarray:
    """Up to ``MAX_FRAMES`` evenly spaced RGB frames, shape (n, height, width, 3)."""
    step = max(1, count // MAX_FRAMES)
    raw = _run([ffmpeg, "-v", "error", "-threads", "1", "-filter_threads", "1",
                "-i", str(src), "-map", "0:v:0",
                "-vf", f"select=not(mod(n\\,{step}))", "-vsync", "0",
                "-frames:v", str(MAX_FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    size = width * height * 3
    usable = len(raw) // size * size
    return np.frombuffer(raw[:usable], np.uint8).reshape(-1, height, width, 3)


class VideoFramesLsbAnalyzer(Analyzer):
    """Weighted-Stego LSB replacement estimate on lossless video frames."""

    name = "video_frames_lsb"

    def analyze(self, src: Path) -> AnalysisResult:
        if src.suffix.lower() not in SUFFIXES:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            return AnalysisResult(self.name, 0, (), "ffmpeg/ffprobe not installed",
                                  status="unavailable")
        try:
            info = probe(ffprobe, src)
            codec, width, height = info.get("codec_name"), info.get("width"), info.get("height")
            if codec not in LOSSLESS:
                return AnalysisResult(self.name, 0, (), f"lossy or unknown codec {codec!r}",
                                      status="unsupported")
            if not width or not height or width * height > MAX_FRAME_PIXELS:
                return AnalysisResult(self.name, 0, (), "frame size out of bounds",
                                      status="unsupported")
            decoded = frames(ffmpeg, src, width, height, int(info.get("nb_read_packets") or 1))
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            return AnalysisResult(self.name, 0, (), None, status="error", error=str(exc)[:200])
        if not len(decoded):
            return AnalysisResult(self.name, 0, (), None, status="error", error="no frames")
        estimates = [lsb_replacement.weighted_stego(frame) for frame in decoded]
        median = sorted(estimates, key=lambda e: e.payload)[len(estimates) // 2]
        score = lsb_replacement.score(median)
        signal = Signal("video_frame_ws_payload", score,
                        f"{codec}: median WS payload {median.payload:.4f} over "
                        f"{len(estimates)} frame(s); prediction RMS {median.prediction_rms:.2f}",
                        category="video_lsb", evidence="heuristic" if score >= 45 else
                        "informational")
        return AnalysisResult(self.name, score, (signal,), None)
