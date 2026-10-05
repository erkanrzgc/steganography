"""Custom parity-conditioned residual histograms; controlled research only."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from core.features import MAX_IMAGE_BYTES, MAX_PIXELS
from core.spatial_cooccurrence import FEATURE_NAMES as BASE_NAMES
from core.spatial_cooccurrence import cooccurrence_from_pixels

FEATURE_VERSION = "spatial-parity-residual-v1"
DIRECTIONS = ("h", "v", "d", "a")
FEATURE_NAMES = (
    *BASE_NAMES,
    *(
        f"parity_{direction}_bit{bit}_{parity}_{residual}"
        for direction in DIRECTIONS
        for bit in range(3)
        for parity in range(2)
        for residual in range(-4, 5)
    ),
)


def parity_histogram(pixels: np.ndarray, direction: str, bit: int) -> list[float]:
    if bit not in (0, 1, 2):
        raise ValueError("unsupported parity bit plane")
    if direction == "h":
        center, neighbor = pixels[:, :-1], pixels[:, 1:]
    elif direction == "v":
        center, neighbor = pixels[:-1], pixels[1:]
    elif direction == "d":
        center, neighbor = pixels[:-1, :-1], pixels[1:, 1:]
    elif direction == "a":
        center, neighbor = pixels[:-1, 1:], pixels[1:, :-1]
    else:
        raise ValueError("unknown parity direction")
    bins = ((center >> bit) & 1) * 9 + np.clip(neighbor - center, -4, 4) + 4
    counts = np.bincount(bins.reshape(-1), minlength=18)
    return (counts / counts.sum()).tolist()


def spatial_parity_features(data: bytes) -> list[float]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("feature image exceeds byte limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "BMP"}:
            raise ValueError("spatial parity features require PNG/BMP")
        if min(image.size) < 8 or image.width * image.height > MAX_PIXELS:
            raise ValueError("feature image dimensions outside limits")
        # Pool identically replicated grayscale once; RGB carriers pool all
        # channels. No interchannel label shortcut or resize is introduced.
        pixels = (
            np.asarray(image, dtype=np.int16)[..., None]
            if image.mode == "L"
            else np.asarray(image.convert("RGB"), dtype=np.int16)
        )
    values = cooccurrence_from_pixels(pixels)
    for direction in DIRECTIONS:
        for bit in range(3):
            values.extend(parity_histogram(pixels, direction, bit))
    # Versioned 1e-8 quantization keeps the complete 5,712-row artifact bounded.
    return [round(v, 8) for v in values]
