"""Bounded complete validation replay, separate from train-only block readers."""

import hashlib
from contextlib import ExitStack
from typing import Any, cast

import numpy as np

from core import jpeg_float256, srnet, srnet_reference
from core.jpeg_scale import safe_name
from core.srnet_sampling import source_id
from core.srnet_stream import (
    CHUNK_BYTES,
    MAX_BYTES,
    ROW_BYTES,
    TrainBlocks,
    document,
    regular_open,
    signature,
)


class ValidationBlocks:
    """Not a TrainBlocks subclass: it cannot be accepted by the training service."""

    _finite = staticmethod(TrainBlocks._finite)

    def batch(self, indices):
        return TrainBlocks.batch(cast(Any, self), indices)

    def close(self):
        self._closed = True
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __init__(self, root, *, index_sha256, audit, audit_sha256, deadline):
        if not callable(deadline):
            raise ValueError("validation requires a whole-job deadline")
        self._stack, self._deadline, self._closed = ExitStack(), deadline, False
        self.samples: list[dict[str, Any]] = []
        self._locations: list[tuple] = []
        self.train_bytes, self.max_batch_bytes = 0, 0
        try:
            # Shared catalog/lineage/decoder audit, without changing TrainBlocks.
            with TrainBlocks(
                root,
                index_sha256=index_sha256,
                audit=audit,
                audit_sha256=audit_sha256,
                deadline=deadline,
            ):
                pass
            index = document(root / "index.json", index_sha256)
            for block in index["blocks"]:
                deadline()
                manifest = document(root / block["path"], block["manifest_sha256"])
                samples = [s for s in manifest["samples"] if s["split"] == "validation"]
                if not samples:
                    continue
                bound = block["caches"]["validation"]
                safe_name(bound["path"])
                path = root / bound["path"]
                if path != (root / block["path"]).parent / "validation/cache.json":
                    raise ValueError("validation cache belongs to another block")
                cache = document(path, bound["sha256"])
                if (
                    cache.get("schema_version") != "research-float256-cache-v1"
                    or cache.get("feature_version") != jpeg_float256.FEATURE_VERSION
                    or cache.get("manifest_sha256") != block["manifest_sha256"]
                    or cache.get("split") != "validation"
                    or cache.get("dtype") != "<f4"
                    or cache.get("shape") != [len(samples), 1, 256, 256]
                    or cache.get("decoder") != jpeg_float256.decoder_contract()
                    or cache.get("data_sha256") != bound["data_sha256"]
                    or bound["rows"] != len(samples)
                    or not isinstance(cache.get("rows"), list)
                    or len(cache["rows"]) != len(samples)
                ):
                    raise ValueError("validation cache contract mismatch")
                for row, sample in zip(cache["rows"], samples, strict=True):
                    size = row.get("image_size") if isinstance(row, dict) else None
                    if (
                        not isinstance(size, list)
                        or len(size) != 2
                        or any(type(n) is not int or n < 256 for n in size)
                        or size[0] * size[1] > 4_000_000
                        or row.get("region") != jpeg_float256.crop_box(*size)
                        or any(row.get(k) != sample[k] for k in ("sha256", "lineage", "label"))
                    ):
                        raise ValueError("validation row identity/region mismatch")
                stream = self._stack.enter_context(regular_open(path.parent / "pixels.f32"))
                initial = signature(stream)
                remaining = len(samples) * ROW_BYTES
                self.train_bytes += remaining
                if initial[2] != remaining or self.train_bytes > MAX_BYTES:
                    raise ValueError("validation byte bound/length mismatch")
                digest = hashlib.sha256()
                while remaining:
                    deadline()
                    raw = stream.read(min(CHUNK_BYTES, remaining))
                    if not raw:
                        raise ValueError("validation tensor truncated")
                    self._finite(raw)
                    digest.update(raw)
                    remaining -= len(raw)
                if digest.hexdigest() != bound["data_sha256"] or signature(stream) != initial:
                    raise ValueError("validation tensor checksum/mutation mismatch")
                self.samples.extend(samples)
                self._locations.extend(
                    (stream, i * ROW_BYTES, initial) for i in range(len(samples))
                )
            if not self.samples or len(self.samples) != index["splits"].get("validation"):
                raise ValueError("validation corpus incomplete")
            deadline()
        except BaseException:
            self.close()
            raise


def evaluate(model, reader, deadline, *, metrics):
    """Fixed stored-BN CPU inference and independent float64 forward audit."""
    if not isinstance(reader, ValidationBlocks) or not callable(deadline):
        raise ValueError("complete validation reader required")
    arrays = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    rows: list[np.ndarray] = []
    audits = []
    selected: dict[tuple, int] = {}
    for i, sample in enumerate(reader.samples):
        key = (sample["source_group"], sample["quality_factor"], sample["label"], sample["method"])
        selected.setdefault(key, i)
    if len(selected) > 24:
        raise ValueError("validation forward-audit context bound exceeded")
    for first in range(0, len(reader.samples), 4):
        deadline()
        indices = np.arange(first, min(first + 4, len(reader.samples)), dtype="<i8")
        values = reader.batch(indices)
        logits = srnet.float_logits(model, values)
        for local, index in enumerate(indices):
            if int(index) in selected.values():
                reference = srnet_reference.reference_logits(arrays, values[local : local + 1])
                audits.append(
                    {
                        "sha256": reader.samples[int(index)]["sha256"],
                        **srnet_reference.compare(logits[local : local + 1], reference),
                    }
                )
                deadline()
        rows.extend(logits)
    native = np.asarray(rows, dtype="<f4")
    margins = native[:, 1].astype(np.float64) - native[:, 0]
    scores = np.exp(-np.logaddexp(0, -margins))
    contexts = {
        (s["source_group"], s["quality_factor"], s["method"])
        for s in reader.samples
        if s["label"] == "stego"
    }
    cells = []
    for source, quality, method in sorted(contexts, key=str):
        observations = [
            (s["label"] == "stego", float(score * 100))
            for s, score in zip(reader.samples, scores, strict=True)
            if s["source_group"] == source
            and s["quality_factor"] == quality
            and (s["label"] == "cover" or s["method"] == method)
        ]
        cells.append(
            {
                "source_id": source_id(source),
                "quality_factor": quality,
                "method": method,
                "metrics": metrics(observations, threshold=50, recommend_threshold=False),
                "support": "experimental",
            }
        )
    deadline()
    if any(
        not np.array_equal(v, model.state_dict()[k].detach().cpu().numpy())
        for k, v in arrays.items()
    ):
        raise ValueError("stored-normalization evaluation mutated model")
    return {
        "status": "completed" if all(a["passed"] for a in audits) else "failed_numerical_gate",
        "validation_rows": len(reader.samples),
        "cells": cells,
        "independent_forward_audit": audits,
        "threshold": 0.5,
        "threshold_tuned": False,
        "normalization_refreshed": False,
        "evaluation_device": "cpu",
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
