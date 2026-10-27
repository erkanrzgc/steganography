"""Bounded generated-only spatial pairs and independently decoded scattered LSBs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from core import colab_artifacts as a
from core import nonjpeg_models as models

RECIPE = "generated-even-spatial-scattered-40p0-v1"
SIDE = 256
MAX_ORIGINALS = 128


def seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("spatial uint32 seed required")
    return value


@dataclass(frozen=True)
class GeneratedSpec:
    mode: str
    seed: int
    split: str
    originals: int

    def __post_init__(self) -> None:
        seed(self.seed)
        if (
            self.mode not in ("L", "RGB")
            or self.split not in ("train", "development")
            or type(self.originals) is not int
            or not 1 <= self.originals <= MAX_ORIGINALS
        ):
            raise ValueError("bounded generated spatial specification required")

    @property
    def domain(self) -> str:
        return models.GRAY_DOMAIN if self.mode == "L" else models.RGB_DOMAIN


def pixels(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[:2] != (SIDE, SIDE)
        or value.shape[2] not in (1, 3)
    ):
        raise ValueError("native 256x256 HWC uint8 samples required")
    return value


def positions(samples: int, scatter_seed: int) -> np.ndarray:
    seed(scatter_seed)
    if type(samples) is not int or samples not in (SIDE * SIDE, SIDE * SIDE * 3):
        raise ValueError("bounded spatial sample count required")
    bits = (samples * 2 // 5) // 8 * 8
    return np.random.Generator(np.random.PCG64(scatter_seed)).permutation(samples)[:bits]


def embed(value: np.ndarray, payload: bytes, scatter_seed: int) -> tuple[np.ndarray, np.ndarray]:
    value = pixels(value)
    selected = positions(value.size, scatter_seed)
    if not isinstance(payload, bytes) or len(payload) != len(selected) // 8:
        raise ValueError("exact forty-percent byte-rounded payload required")
    result = value.copy()
    flat = result.reshape(-1)
    bits = np.unpackbits(np.frombuffer(payload, dtype="u1"), bitorder="big")
    flat[selected] = (flat[selected] & 254) | bits
    return result, selected


def recover(value: np.ndarray, selected: np.ndarray) -> bytes:
    value = pixels(value)
    expected = (value.size * 2 // 5) // 8 * 8
    if (
        not isinstance(selected, np.ndarray)
        or selected.dtype != np.dtype("int64")
        or selected.shape != (expected,)
        or np.any(selected < 0)
        or np.any(selected >= value.size)
        or len(np.unique(selected)) != expected
    ):
        raise ValueError("bounded unique scattered positions required")
    flat = value.reshape(-1)
    # Deliberately independent scalar decoder, not packbits/unpackbits inversion.
    return bytes(
        sum((int(flat[int(selected[start + k])]) & 1) << (7 - k) for k in range(8))
        for start in range(0, expected, 8)
    )


def pair(spec: GeneratedSpec, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if type(spec) is not GeneratedSpec or type(index) is not int or not 0 <= index < spec.originals:
        raise ValueError("generated source/index required")
    # Count is not part of source identity: expanding a spec cannot invent new groups
    # for the same seed/split/index. Split and mode use disjoint seed namespaces.
    descriptor = {
        "recipe": RECIPE,
        "mode": spec.mode,
        "seed": spec.seed,
        "split": spec.split,
        "index": index,
    }
    identity = a.digest(a.encode(descriptor))
    rng = np.random.Generator(np.random.PCG64(int(identity[:16], 16)))
    channels = 1 if spec.mode == "L" else 3
    cover = rng.integers(0, 128, (SIDE, SIDE, channels), dtype="u1") * np.uint8(2)
    payload = rng.integers(0, 256, (cover.size * 2 // 5) // 8, dtype="u1").tobytes()
    scatter_seed = int(identity[16:24], 16)
    control, selected = embed(cover, payload, scatter_seed)
    if recover(control, selected) != payload:
        raise ValueError("generated spatial payload recovery failed")
    return (
        cover,
        control,
        {
            "id": "generated-" + identity,
            "group": "generated-" + identity,
            "split": spec.split,
            "cover_sha256": a.digest(cover.tobytes()),
            "control_sha256": a.digest(control.tobytes()),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "positions_sha256": a.digest(selected.astype("<i8").tobytes()),
            "payload_bytes": len(payload),
        },
    )


def manifest(spec: GeneratedSpec) -> dict:
    if type(spec) is not GeneratedSpec:
        raise ValueError("generated specification required")
    return {
        "recipe": RECIPE,
        "generated_only": True,
        "spec": asdict(spec),
        "sources": [pair(spec, index)[2] for index in range(spec.originals)],
    }


def batch(spec: GeneratedSpec, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(spec) is not GeneratedSpec
        or not isinstance(indices, tuple)
        or not 1 <= len(indices) <= 4
        or any(type(i) is not int or not 0 <= i < spec.originals for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("one to four distinct generated pairs required")
    values: list[np.ndarray] = []
    for index in indices:
        cover, control, _ = pair(spec, index)
        values.extend((cover.transpose(2, 0, 1), control.transpose(2, 0, 1)))
    result = np.stack(values).astype("<f4")
    if not np.array_equal(result.astype("u1").astype("<f4"), result):
        raise ValueError("generated FP32 single-LSB preservation failed")
    return result, np.tile(np.array([0, 1], dtype="<i8"), len(indices))
