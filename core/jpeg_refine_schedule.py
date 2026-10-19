"""Frozen paired prefix and exposure-preserving partner-free continuation."""

from __future__ import annotations

import hashlib

import numpy as np

from core import jpeg_study as study
from core import jpeg_study_data as data

ARMS = ("paired", "unpaired")
WARMUP = 2048
FINAL = 8192
INTERVAL = 512
ROWS = 16
TOTAL = 28672


def segments():
    """Round-robin warm-ups first, then both arms across both seeds."""
    return [
        ("warmup", seed, stop)
        for stop in range(INTERVAL, WARMUP + 1, INTERVAL)
        for seed in study.SEEDS
    ] + [
        (arm, seed, stop)
        for stop in range(WARMUP + INTERVAL, FINAL + 1, INTERVAL)
        for seed in study.SEEDS
        for arm in ARMS
    ]


def schedule(samples, seed, arm):
    if arm not in ("warmup", *ARMS):
        raise ValueError("refine schedule arm invalid")
    # Reuse the public historical validator before extending its bounded prefix.
    study.schedule(samples, seed, ROWS, ROWS)
    groups = [
        [i for i in range(0, len(samples), 2) if samples[i]["source_group"] == o]
        for o in data.ORIGINS
    ]
    count = len(groups[0])
    if count < 8 or count % 4 or len({r["lineage"] for r in samples}) != len(samples) // 2:
        raise ValueError("refine requires distinct originals and complete four-pair blocks")
    rng = np.random.default_rng(seed)
    exposures: dict[str, int] = {}
    indices: list[int] = []
    angles: list[int] = []
    views: list[int] = []
    for epoch in range(FINAL * ROWS // len(samples)):
        shuffled = [rng.permutation(group) for group in groups]
        rows: list[int] = []
        rotations: list[int] = []
        crops: list[int] = []
        for a, b in zip(*shuffled, strict=True):
            for pair in (int(a), int(b)):
                angle = int(rng.integers(8))
                lineage = samples[pair]["lineage"]
                offset = (
                    int.from_bytes(
                        hashlib.sha256(f"jpeg-five-view:{seed}:{lineage}".encode()).digest()[:8],
                        "little",
                    )
                    % 5
                )
                exposure = exposures.get(lineage, 0)
                view = (offset + exposure) % 5
                exposures[lineage] = exposure + 1
                rows.extend((pair, pair + 1))
                rotations.extend((angle, angle))
                crops.extend((view, view))
        values = [np.asarray(x, dtype=np.int64) for x in (rows, rotations, crops)]
        if arm == "unpaired" and epoch * len(samples) // ROWS >= WARMUP:
            for source_index, origin in enumerate(data.ORIGINS):
                slots = np.arange(1 + source_index * 2, len(samples), 4)
                # A separate RNG cannot perturb the historical order or D4 draws.
                shift_rng = np.random.default_rng(
                    int.from_bytes(
                        hashlib.sha256(
                            f"jpeg-refine-shift:{seed}:{epoch}:{origin}".encode()
                        ).digest()[:8],
                        "little",
                    )
                )
                shift = 4 * int(shift_rng.integers(1, count // 4))
                for value in values:
                    value[slots] = np.roll(value[slots].copy(), shift)
        for target, value in zip((indices, angles, views), values, strict=True):
            target.extend(value.tolist())
    result = tuple(
        np.asarray(x, dtype=np.int64).reshape(-1, ROWS) for x in (indices, angles, views)
    )
    for value in result:
        value.setflags(write=False)
    return result


def digest(values):
    return hashlib.sha256(b"".join(v.tobytes() for v in values)).hexdigest()


def exposure_digest(values, epoch_rows):
    """Canonical per-epoch image/view/D4 multisets, independent of batch grouping."""
    flat = np.stack([v.ravel() for v in values], axis=1)
    checksum = hashlib.sha256()
    for start in range(0, len(flat), epoch_rows):
        epoch = flat[start : start + epoch_rows]
        checksum.update(epoch[np.lexsort(epoch.T[::-1])].tobytes())
    return checksum.hexdigest()
