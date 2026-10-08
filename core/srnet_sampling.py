"""Deterministic train-only cover/stego pairs, balanced by declared context."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

import numpy as np

RECIPE = "srnet-source-quality-method-pairs-v1"
MAX_ROWS = 4000
MAX_PAIRS = 40_000
METHODS = {"JUNIWARD", "UERD"}


def source_id(source: str) -> str:
    return "source-" + hashlib.sha256(source.encode()).hexdigest()[:16]


def buckets(samples):
    return _buckets(samples, row_limit=MAX_ROWS)


def _buckets(samples, *, row_limit):
    if not isinstance(samples, list) or not 1 <= len(samples) <= row_limit:
        raise ValueError("SRNet sampling row limit")
    groups: dict[tuple[str, str, int | None], list[tuple[int, dict[str, Any]]]] = {}
    hashes = set()
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("SRNet sampling requires training JPEG rows")
        source, lineage = sample.get("source_group"), sample.get("lineage")
        quality, digest = sample.get("quality_factor"), sample.get("sha256")
        if (
            sample.get("split") != "train"
            or sample.get("format") != "JPEG"
            or not isinstance(sample.get("label"), str)
            or sample["label"] not in {"cover", "stego"}
            or not isinstance(source, str)
            or not source.strip()
            or not isinstance(lineage, str)
            or not lineage.strip()
            or (quality is not None and (type(quality) is not int or not 1 <= quality <= 100))
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or digest in hashes
        ):
            raise ValueError("SRNet sampling requires unique bound training JPEG rows")
        hashes.add(digest)
        groups.setdefault((source, lineage, quality), []).append((i, sample))
    result: dict[tuple[str, int | None, str], list[tuple[int, int]]] = {}
    for (source, _, quality), members in groups.items():
        covers = [(i, s) for i, s in members if s["label"] == "cover"]
        stegos = [(i, s) for i, s in members if s["label"] == "stego"]
        if (
            len(covers) != 1
            or covers[0][1].get("method") is not None
            or len(stegos) != 2
            or any(not isinstance(s.get("method"), str) for _, s in stegos)
            or {s["method"] for _, s in stegos} != METHODS
        ):
            raise ValueError("SRNet sampling requires complete matched-family pairs")
        for i, sample in stegos:
            result.setdefault((source_id(source), quality, sample["method"]), []).append(
                (covers[0][0], i)
            )
    return result


def epoch_pairs(samples, *, seed: int, epoch: int):
    """Equal sources, then equal quality/method cells; no discarded train pairs."""
    return _epoch_pairs(samples, seed=seed, epoch=epoch, row_limit=MAX_ROWS)


def _epoch_pairs(samples, *, seed, epoch, row_limit):
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(epoch) is not int
        or not 0 <= epoch < 50
    ):
        raise ValueError("SRNet sampling seed/epoch outside limits")
    cells = _buckets(samples, row_limit=row_limit)
    sources = sorted({key[0] for key in cells})
    if len(sources) > 8:
        raise ValueError("SRNet sampling source limit")
    keys = {s: sorted((k for k in cells if k[0] == s), key=repr) for s in sources}
    multiple = math.lcm(*(len(keys[s]) for s in sources))
    # Each bucket must exhaust every original pair at least once. Balance can
    # oversample small contexts, never silently truncate a large context.
    required = max(max(len(cells[k]) for k in keys[s]) * len(keys[s]) for s in sources)
    quota = ((required + multiple - 1) // multiple) * multiple
    if quota * len(sources) > MAX_PAIRS:
        raise ValueError("SRNet balanced sampling exceeds pair limit")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    streams: dict[str, list[tuple[int, int]]] = {}
    for s in sources:
        streams[s] = []
        ordered_cells = [keys[s][i] for i in rng.permutation(len(keys[s]))]
        draws = quota // len(ordered_cells)
        sequences = {}
        for key in ordered_cells:
            original = cells[key]
            sequence: list[tuple[int, int]] = []
            while len(sequence) < draws:
                sequence.extend(original[i] for i in rng.permutation(len(original)))
            sequences[key] = sequence[:draws]
        for j in range(draws):
            streams[s].extend(sequences[key][j] for key in ordered_cells)
    order = [sources[i] for i in rng.permutation(len(sources))]
    pairs = np.asarray([streams[s][j] for j in range(quota) for s in order], dtype="<i8")
    pairs.flags.writeable = False
    counts = Counter(
        (
            source_id(samples[j]["source_group"]),
            samples[j].get("quality_factor"),
            samples[j]["method"],
        )
        for _, j in pairs
    )
    identity = [[samples[i]["sha256"], samples[j]["sha256"]] for i, j in pairs]
    record = {
        "recipe": RECIPE,
        "epoch": epoch,
        "pairs": len(pairs),
        "rows": 2 * len(pairs),
        "pairs_per_source": quota,
        "ordered_pair_sha256": hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode()
        ).hexdigest(),
        "cells": [
            {
                "source_id": k[0],
                "quality_factor": k[1],
                "method": k[2],
                "original_pairs": len(cells[k]),
                "sampled_pairs": counts[k],
            }
            for k in sorted(cells, key=repr)
        ],
    }
    return pairs, record
