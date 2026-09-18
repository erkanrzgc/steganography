"""Multi-channel/bit-order lossless-image steganalysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.analyzer import Analyzer
from core.context import AnalysisContext
from core.payload import MAGIC
from core.result import AnalysisResult, Signal


class ImageBitplaneAnalyzer(Analyzer):
    name = "image_bitplane"

    def analyze(self, src: Path) -> AnalysisResult:
        return self.analyze_context(AnalysisContext(src))

    def analyze_context(self, context: AnalysisContext) -> AnalysisResult:
        src = context.path
        if src.suffix.lower() not in {".png", ".bmp"}:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        array = context.image_rgba
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
                    category="image_chi_square",
                    evidence="heuristic" if score >= 45 else "informational",
                )
            )

        # Approximate RS/sample-pair indicators, kept independently categorized.
        lsb = rgb & 1
        plane2 = (rgb >> 2) & 1
        higher_plane_structured = float(np.mean(plane2[:, 1:, :] == plane2[:, :-1, :])) >= 0.54
        horizontal_agreement = float(np.mean(lsb[:, 1:, :] == lsb[:, :-1, :]))
        spa_delta = abs(horizontal_agreement - 0.5)
        if higher_plane_structured:
            spa_score = max(0, min(70, round((0.10 - spa_delta) / 0.10 * 70)))
        else:
            spa_score = 0
        signals.append(
            Signal(
                "sample_pair_balance",
                spa_score,
                f"adjacent LSB agreement={horizontal_agreement:.4f}",
                category="image_sample_pair",
                evidence="heuristic" if spa_score >= 25 else "informational",
            )
        )

        # Bit-plane entropy makes plane-specific anomalies visible without
        # treating naturally noisy least-significant planes as proof.
        for plane in range(4):
            bits = ((rgb >> plane) & 1).reshape(-1)
            probability = float(np.mean(bits))
            if probability in {0.0, 1.0}:
                entropy = 0.0
            else:
                entropy = -probability * np.log2(probability) - (1.0 - probability) * np.log2(
                    1.0 - probability
                )
            signals.append(
                Signal(
                    f"bit_plane_{plane}_entropy",
                    0,
                    f"RGB plane {plane} entropy={entropy:.5f}; one-ratio={probability:.5f}",
                    category="image_bitplane_entropy",
                    evidence="informational",
                )
            )

        # Weighted-stego approximation: compare LSB changes with local edge
        # energy. Embedding concentrated in smooth regions is more unusual.
        luminance = np.mean(rgb.astype(np.float32), axis=2)
        edge = np.abs(np.diff(luminance, axis=1))
        lsb_changes = np.mean(lsb[:, 1:, :] != lsb[:, :-1, :], axis=2)
        smooth = edge < np.percentile(edge, 35)
        smooth_flip_rate = float(np.mean(lsb_changes[smooth])) if np.any(smooth) else 0.0
        textured_flip_rate = float(np.mean(lsb_changes[~smooth])) if np.any(~smooth) else 0.0
        weighted_delta = smooth_flip_rate - textured_flip_rate
        weighted_score = max(0, min(55, round((weighted_delta - 0.01) * 900)))
        signals.append(
            Signal(
                "weighted_stego_smooth_region_bias",
                weighted_score,
                f"smooth/textured LSB flip rates={smooth_flip_rate:.4f}/{textured_flip_rate:.4f}",
                category="image_weighted_stego",
                evidence="heuristic" if weighted_score >= 45 else "informational",
            )
        )
        differences = np.abs(np.diff(rgb.astype(np.int16), axis=1))
        regular = float(np.mean((differences & 1) == 0))
        rs_delta = abs(regular - 0.5)
        if higher_plane_structured:
            rs_score = max(0, min(70, round((0.10 - rs_delta) / 0.10 * 70)))
        else:
            rs_score = 0
        signals.append(
            Signal(
                "rs_regular_singular_balance",
                rs_score,
                f"regular/singular parity balance={regular:.4f}",
                category="image_rs_analysis",
                evidence="heuristic" if rs_score >= 25 else "informational",
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
