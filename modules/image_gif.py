"""Bounded GIF palette, frame, extension, and frame-difference analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from core.analyzer import Analyzer
from core.context import AnalysisContext
from core.result import AnalysisResult, Signal


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
                frames.append(np.asarray(frame.convert("RGB"), dtype=np.int16))
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
