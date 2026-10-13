"""Two bounded, source-bound halves; never qualify an incomplete logical fit."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from core.jpeg_scale import identity
from core.jpeg_study_data import fresh
from core.jpeg_study_process import WorkerFailure
from core.jpeg_timing import write_document
from core.srnet_checkpoint import MAX_BYTES
from core.srnet_stream import regular_open

MODE = "bounded-half-v1"
CONTROLS_SECONDS = 6 * 3600


def checkpoint(path, checksum):
    identity(checksum)
    digest, size = hashlib.sha256(), 0
    with regular_open(path) as stream:
        while block := stream.read(1024**2):
            size += len(block)
            if size > MAX_BYTES:
                raise ValueError("study segment checkpoint exceeds bounds")
            digest.update(block)
    if not size or digest.hexdigest() != checksum:
        raise ValueError("study segment checkpoint checksum mismatch")


def trial(config, out, *, arm, seed, rows, sources, recipe, launch, budget_left):
    if config.get("resume"):
        raise ValueError("segmented controls do not import external resumes")
    fresh(out)
    out.mkdir()
    segments: list[dict[str, Any]] = []
    first = None
    for index, stop in enumerate((16384, 32768)):
        if budget_left() <= 0:
            return {"status": "unavailable", "reason": "controls_budget_exhausted",
                    "segments": segments, "execution_mode": MODE, "real_gpu_fit": False}
        payload = dict(config)
        # Each isolated worker executes one bounded phase; only the parent owns
        # the segmented policy. External segmented configurations cannot resume.
        payload.pop("execution_mode", None)
        if first is not None:
            entry = first["checkpoints"][0]
            payload["resume"] = [{"arm": arm, "seed": seed,
                                  "path": str(out / "segment-0" / entry["path"]),
                                  "sha256": entry["sha256"]}]
        started = time.monotonic()
        try:
            value = launch(payload, out / f"segment-{index}", stop,
                           min(1830, max(1, budget_left())))
        except WorkerFailure as exc:
            value = exc.unavailable()
        except TimeoutError:
            value = {"status": "unavailable", "reason": "worker_timeout"}
        except InterruptedError:
            value = {"status": "cancelled", "reason": "user_cancelled"}
        if not isinstance(value, dict) or value.get("status") not in {
            "checkpointed", "completed", "unavailable", "cancelled"
        }:
            raise ValueError("study segment response invalid")
        if value.get("source_sha256", sources) != sources:
            raise ValueError("study segment source binding mismatch")
        segments.append({"index": index, "start_rows": index * 16384, "stop_rows": stop,
                         "status": value["status"], "reason": value.get("reason"),
                         "seconds": value.get("seconds"),
                         "worker_wall_seconds": time.monotonic() - started,
                         **({"worker_execution": value["worker_execution"]}
                            if "worker_execution" in value else {})})
        if value["status"] in {"unavailable", "cancelled"}:
            write_document(out / f"receipt-{stop}.json", value)
            return {**value, "segments": segments, "execution_mode": MODE,
                    "real_gpu_fit": False}
        expected_status = "checkpointed" if index == 0 else "completed"
        if (value["status"] != expected_status or value.get("source_sha256") != sources
            or value.get("presented_rows") != stop
            or value.get("resumed_from_rows") != index * 16384
            or value.get("optimizer_updates") != 16384 // rows
            or value.get("total_optimizer_updates") != stop // rows
            or value.get("recipe") != recipe
            or not isinstance(value.get("execution"), dict)
            or value.get("execution", {}).get("device") != "cuda:0"
            or value.get("real_gpu_segment") is not True):
            raise ValueError("incomplete/non-GPU study segment cannot complete")
        for key in ("binding", "initial_learned_sha256", "schedule_sha256"):
            identity(value.get(key))
        for key in ("seconds", "mean_cross_entropy"):
            number = value.get(key)
            if (not isinstance(number, (int, float)) or isinstance(number, bool)
                or not math.isfinite(number) or number < 0):
                raise ValueError("study segment numeric receipt invalid")
        checkpoints = value.get("checkpoints")
        if (not isinstance(checkpoints, list) or len(checkpoints) != 1
            or not isinstance(checkpoints[0], dict)
            or checkpoints[0].get("path") != f"rows-{stop}.npz"
            or checkpoints[0].get("presented_rows") != stop):
            raise ValueError("study segment checkpoint provenance invalid")
        checkpoint(out / f"segment-{index}" / checkpoints[0]["path"],
                   checkpoints[0].get("sha256"))
        if index == 0:
            if value.get("real_gpu_fit") or "scores" in value or "cells" in value:
                raise ValueError("partial study segment cannot report evaluation")
            first = value
        elif (first is None or value.get("binding") != first.get("binding")
              or value.get("recipe") != first.get("recipe")
              or value.get("initial_learned_sha256") != first.get("initial_learned_sha256")
              or value.get("real_gpu_fit") is not True):
            raise ValueError("study segment recipe/initialization/binding mismatch")
        write_document(out / f"receipt-{stop}.json", value)
    if first is None:
        raise ValueError("study segments missing first checkpoint")
    result = dict(value)
    result.update(execution_mode=MODE, segments=segments,
                  seconds=sum(s["seconds"] for s in segments),
                  optimizer_updates=32768 // rows, resumed_from_rows=0,
                  mean_cross_entropy=(first["mean_cross_entropy"]
                                      + value["mean_cross_entropy"]) / 2,
                  mean_cross_entropy_aggregation="update-weighted_segment_means",
                  checkpoints=[{**first["checkpoints"][0],
                                "path": "segment-0/rows-16384.npz"},
                               {**value["checkpoints"][0],
                                "path": "segment-1/rows-32768.npz"}])
    return result
