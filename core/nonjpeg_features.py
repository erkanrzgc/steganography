"""Versioned native spatial/PCM descriptors for explicit development research."""

from __future__ import annotations

import io
import wave

import numpy as np
from PIL import Image

from core.features import MAX_IMAGE_BYTES
from core.spatial_parity import DIRECTIONS, parity_histogram, spatial_parity_features
from core.spatial_parity import FEATURE_NAMES as BASE_NAMES

LOCAL_VERSION = "spatial-local-parity-v2"
RGB_VERSION = "spatial-rgb-parity-v2"
AUDIO_VERSION = "audio-pcm16-parity-residual-v1"
QUARTERS = ("tl", "tr", "bl", "br")


def parity_names(prefix: str) -> tuple[str, ...]:
    return tuple(
        f"{prefix}_{direction}_bit{bit}_{parity}_{residual}"
        for direction in DIRECTIONS
        for bit in range(3)
        for parity in range(2)
        for residual in range(-4, 5)
    )


LOCAL_NAMES = (*BASE_NAMES, *(n for q in QUARTERS for n in parity_names(q)))
RGB_NAMES = (*BASE_NAMES, *(n for c in "rgb" for n in parity_names(c)))
AUDIO_NAMES = tuple(
    f"{region}_lag{lag}_bit{bit}_{kind}"
    for region in ("full", "q0", "q1", "q2", "q3")
    for lag in (1, 2)
    for bit in (0, 1)
    for kind in (
        *(f"parity{p}_delta{d}" for p in (0, 1) for d in range(-8, 9)),
        *(f"transition{a}{b}" for a in (0, 1) for b in (0, 1)),
    )
)


def spatial_pixels(data: bytes, *, rgb: bool = False) -> np.ndarray:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("feature image exceeds byte limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "BMP"}:
            raise ValueError("native PNG/BMP required")
        if image.size != (256, 256) or getattr(image, "n_frames", 1) != 1:
            raise ValueError("prepared 256x256 single image required; no resize")
        if image.mode not in (("RGB",) if rgb else ("L", "RGB")):
            raise ValueError("native RGB required" if rgb else "native L/RGB required")
        pixels = np.asarray(image, dtype=np.int16)
    return pixels[..., None] if pixels.ndim == 2 else pixels


def spatial_features(data: bytes, *, rgb: bool = False) -> list[float]:
    pixels = spatial_pixels(data, rgb=rgb)
    regions = (
        [pixels[..., c : c + 1] for c in range(3)]
        if rgb
        else [pixels[y : y + 128, x : x + 128] for y in (0, 128) for x in (0, 128)]
    )
    values = spatial_parity_features(data)
    for region in regions:
        for direction in DIRECTIONS:
            for bit in range(3):
                values.extend(parity_histogram(region, direction, bit))
    return [round(v, 8) for v in values]


def rgb_features(data: bytes) -> list[float]:
    return spatial_features(data, rgb=True)


def pcm16_samples(data: bytes) -> np.ndarray:
    """Reject truncated/ambiguous RIFF before decoding native mono PCM16."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("audio exceeds byte limit")
    if (
        data[:4] != b"RIFF"
        or data[8:12] != b"WAVE"
        or int.from_bytes(data[4:8], "little") + 8 != len(data)
    ):
        raise ValueError("invalid RIFF extent")
    offset = 12
    chunks: list[tuple[bytes, int]] = []
    while offset + 8 <= len(data) and len(chunks) < 128:
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        end = offset + 8 + size
        if end + size % 2 > len(data):
            raise ValueError("truncated RIFF chunk")
        chunks.append((data[offset : offset + 4], size))
        offset = end + size % 2
    if (
        offset != len(data)
        or sum(k == b"data" for k, _ in chunks) != 1
        or sum(k == b"fmt " for k, _ in chunks) != 1
    ):
        raise ValueError("ambiguous RIFF chunks")
    with wave.open(io.BytesIO(data)) as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getcomptype() != "NONE"
            or not 160 <= audio.getnframes() <= 220_500
            or not 8_000 <= audio.getframerate() <= 192_000
            or audio.getnframes() / audio.getframerate() > 30
        ):
            raise ValueError("native mono PCM16 frame/rate/duration contract")
        frames = audio.readframes(audio.getnframes())
        if len(frames) != audio.getnframes() * 2 or dict(chunks)[b"data"] != len(frames):
            raise ValueError("truncated PCM extent")
    return np.frombuffer(frames, dtype="<i2").astype(np.int32)


def audio_histogram(samples: np.ndarray, lag: int, bit: int) -> list[float]:
    if lag not in (1, 2) or bit not in (0, 1):
        raise ValueError("unsupported PCM lag/bit")
    # Conversion precedes subtraction, including -32768 -> 32767 transitions.
    values = samples.astype(np.int32)
    center, neighbor = values[:-lag], values[lag:]
    parity, following = (center >> bit) & 1, (neighbor >> bit) & 1
    bins = parity * 17 + np.clip(neighbor - center, -8, 8) + 8
    residuals = np.bincount(bins, minlength=34) / len(center)
    transitions = np.bincount(parity * 2 + following, minlength=4) / len(center)
    return [*residuals.tolist(), *transitions.tolist()]


def audio_features(data: bytes) -> list[float]:
    samples = pcm16_samples(data)
    regions = [samples, *np.split(samples[: len(samples) // 4 * 4], 4)]
    # Remainder belongs to the final quarter, never discard native samples.
    regions[-1] = samples[3 * (len(samples) // 4) :]
    return [
        round(value, 8)
        for region in regions
        for lag in (1, 2)
        for bit in (0, 1)
        for value in audio_histogram(region, lag, bit)
    ]


CONTRACTS = {
    LOCAL_VERSION: (LOCAL_NAMES, spatial_features),
    RGB_VERSION: (RGB_NAMES, rgb_features),
    AUDIO_VERSION: (AUDIO_NAMES, audio_features),
}
