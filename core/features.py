"""Versioned, bounded exploratory spatial features; not a calibrated detector."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

FEATURE_VERSION = "spatial-summary-v1"
FEATURE_NAMES = tuple(
    f"{channel}_{stat}"
    for channel in "rgb"
    for stat in ("residual_mean_abs", "residual_std", "lsb_one_ratio", "lsb_agreement")
)
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 4_000_000


def spatial_features(data: bytes) -> list[float]:
    """RGB/no resize; pooled horizontal+vertical differences, normalized by 255.

    Deliberately simple baseline features, not RS, SRM, or SRNet. Replicated
    grayscale channels and ordinary image processing can confound them.
    """
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("feature image exceeds byte limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "BMP"}:
            raise ValueError("spatial features require PNG/BMP")
        if min(image.size) < 2 or image.width * image.height > MAX_PIXELS:
            raise ValueError("feature image dimensions outside limits")
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    values: list[float] = []
    for index in range(3):
        channel = rgb[:, :, index]
        differences = np.concatenate(
            (np.diff(channel, axis=0).ravel(), np.diff(channel, axis=1).ravel())
        )
        values.extend(
            (
                float(np.abs(differences).mean() / 255),
                float(differences.std() / 255),
                float((channel & 1).mean()),
                float(((differences & 1) == 0).mean()),
            )
        )
    return values
