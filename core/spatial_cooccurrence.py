"""Bounded experimental spatial residual co-occurrences, not full SRM/SPAM."""

from __future__ import annotations

import io
import itertools

import numpy as np
from PIL import Image

from core.features import MAX_IMAGE_BYTES, MAX_PIXELS

FEATURE_VERSION = "spatial-cooccurrence-v1"
FILTERS = ("h1", "v1", "h2", "v2", "d1", "a1")


def canonical(triple: tuple[int, ...]) -> tuple[int, ...]:
    reversed_triple = triple[::-1]
    return min(
        triple, reversed_triple, tuple(-v for v in triple), tuple(-v for v in reversed_triple)
    )


BINS = tuple(sorted({canonical(t) for t in itertools.product(range(-2, 3), repeat=3)}))
LOOKUP = np.asarray([BINS.index(canonical(t)) for t in itertools.product(range(-2, 3), repeat=3)])
FEATURE_NAMES = tuple(
    f"{region}_{filter_name}_{a}_{b}_{c}"
    for region in ("full", "top")
    for filter_name in FILTERS
    for a, b, c in BINS
)


def residual_histogram(pixels: np.ndarray, filter_name: str) -> list[float]:
    if filter_name == "h1":
        residual, axis = np.diff(pixels, axis=1), 1
    elif filter_name == "v1":
        residual, axis = np.diff(pixels, axis=0), 0
    elif filter_name == "h2":
        residual, axis = np.diff(pixels, n=2, axis=1), 1
    elif filter_name == "v2":
        residual, axis = np.diff(pixels, n=2, axis=0), 0
    elif filter_name == "d1":
        residual, axis = pixels[1:, 1:] - pixels[:-1, :-1], 1
    elif filter_name == "a1":
        residual, axis = pixels[1:, :-1] - pixels[:-1, 1:], 1
    else:
        raise ValueError("unknown residual filter")
    residual = np.clip(residual, -2, 2).astype(np.int16) + 2
    if axis == 0:
        residual = np.swapaxes(residual, 0, 1)
    encoded = residual[:, :-2] * 25 + residual[:, 1:-1] * 5 + residual[:, 2:]
    histogram = np.bincount(LOOKUP[encoded.reshape(-1)], minlength=len(BINS))
    return (histogram / histogram.sum()).tolist()


def spatial_cooccurrence_features(data: bytes) -> list[float]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("feature image exceeds byte limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "BMP"}:
            raise ValueError("spatial co-occurrences require PNG/BMP")
        if min(image.size) < 8 or image.width * image.height > MAX_PIXELS:
            raise ValueError("feature image dimensions outside limits")
        pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    return cooccurrence_from_pixels(pixels)


def cooccurrence_from_pixels(pixels: np.ndarray) -> list[float]:
    """Shared internal descriptor for already bounded decoded signed pixels."""
    result = []
    for region in (pixels, pixels[: max(6, pixels.shape[0] // 4)]):
        for filter_name in FILTERS:
            result.extend(residual_histogram(region, filter_name))
    return result
