"""Multi-channel/bit-order lossless-image steganalysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.analyzer import Analyzer
from core.payload import MAGIC
from core.result import AnalysisResult, Signal


class ImageBitplaneAnalyzer(Analyzer):
    name = "image_bitplane"

    def analyze(self, src: Path) -> AnalysisResult:
        if src.suffix.lower() not in {".png", ".bmp"}:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        with Image.open(src) as image:
            if image.width * image.height > 50_000_000:
                raise ValueError("image exceeds the 50 megapixel analysis limit")
            array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        rgb = array[:, :, :3]
        signals: list[Signal] = []

        # Chi-square pairs-of-values: embedding tends to equalize even/odd bins.
        for channel_index, channel_name in enumerate("rgb"):
            values = rgb[:, :, channel_index].reshape(-1)
            histogram = np.bincount(values, minlength=256).astype(np.float64)
            pairs = histogram.reshape(128, 2)
            totals = pairs.sum(axis=1)
            valid = totals > 0
            expected = totals[valid] / 2.0
            chi = float(
                np.sum(
                    ((pairs[valid, 0] - expected) ** 2) / expected
                    + ((pairs[valid, 1] - expected) ** 2) / expected
                )
            )
            normalized = chi / max(1, int(valid.sum()))
            score = max(0, min(65, round((1.0 - min(normalized, 1.0)) * 65)))
            signals.append(
                Signal(
                    f"chi_square_{channel_name}_lsb",
                    score,
                    f"pairs-of-values normalized chi-square={normalized:.4f}",
                    category="image_bitplane_statistics",
                    evidence="heuristic" if score >= 45 else "informational",
                )
            )

        # Approximate RS/sample-pair indicators, kept independently categorized.
        lsb = rgb & 1
        horizontal_agreement = float(np.mean(lsb[:, 1:, :] == lsb[:, :-1, :]))
        spa_delta = abs(horizontal_agreement - 0.5)
        spa_score = max(0, min(55, round((0.04 - spa_delta) * 900)))
        signals.append(
            Signal(
                "sample_pair_balance",
                spa_score,
                f"adjacent LSB agreement={horizontal_agreement:.4f}",
                category="image_bitplane_statistics",
                evidence="heuristic" if spa_score >= 45 else "informational",
            )
        )
        differences = np.abs(np.diff(rgb.astype(np.int16), axis=1))
        regular = float(np.mean((differences & 1) == 0))
        rs_score = max(0, min(55, round((0.03 - abs(regular - 0.5)) * 1100)))
        signals.append(
            Signal(
                "rs_regular_singular_balance",
                rs_score,
                f"regular/singular parity balance={regular:.4f}",
                category="image_bitplane_statistics",
                evidence="heuristic" if rs_score >= 45 else "informational",
            )
        )

        marker = self._search_known_marker(rgb)
        if marker:
            signals.append(
                Signal(
                    "known_payload_marker",
                    98,
                    marker,
                    category="known_marker",
                    evidence="verified",
                )
            )
        suspicion = max(
            (signal.score for signal in signals if signal.evidence != "informational"),
            default=0,
        )
        return AnalysisResult(self.name, suspicion, tuple(signals), None)

    @staticmethod
    def _search_known_marker(rgb: np.ndarray) -> str | None:
        orders = (("row-major", rgb), ("column-major", rgb.transpose(1, 0, 2)))
        channel_sets = (("r", (0,)), ("g", (1,)), ("b", (2,)), ("rgb", (0, 1, 2)))
        for order_name, ordered in orders:
            for channel_name, indexes in channel_sets:
                selected = ordered[:, :, indexes].reshape(-1)
                for bit_depth in range(2):
                    bits = ((selected >> bit_depth) & 1).astype(np.uint8)
                    for bit_order in ("big", "little"):
                        packed = np.packbits(bits, bitorder=bit_order).tobytes()
                        offset = packed.find(MAGIC)
                        if offset >= 0:
                            return (
                                f"STEG marker at packed offset {offset}; channels={channel_name}, "
                                f"plane={bit_depth}, pixels={order_name}, bits={bit_order}"
                            )
        return None
