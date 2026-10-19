"""Explicit AudioMNIST PCM pilot training primitives (never invoked by imports)."""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from core import nonjpeg_features, nonjpeg_linear, nonjpeg_models
from core.wav_pilot import RECIPES, materialize, rows_for_role

SEEDS = (20261016, 20261017)
EPOCHS = 20
VIRTUAL_BATCH = 64
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001


def balanced_row_weight(label: int, *, row_count: int = 3600, originals: int = 400) -> float:
    if (
        type(label) is not int
        or label not in (0, 1)
        or type(row_count) is not int
        or type(originals) is not int
        or row_count != 9 * originals
        or originals <= 0
    ):
        raise ValueError("fixed AudioMNIST class-balance contract")
    class_rows = originals if label == 0 else 8 * originals
    return row_count / (2 * class_rows)


def epoch_schedule(row_count: int, batch_size: int = VIRTUAL_BATCH) -> list[int]:
    if (
        type(row_count) is not int
        or type(batch_size) is not int
        or row_count <= 0
        or batch_size <= 0
    ):
        raise ValueError("positive integer row/batch count required")
    return [min(batch_size, row_count - start) for start in range(0, row_count, batch_size)]


def _model_inputs(torch, root: Path, row: dict[str, Any], device: str):
    data = materialize(root, row["source"], row["recipe"])
    value = nonjpeg_models.inputs(data, nonjpeg_models.AUDIO_DOMAIN)
    return torch.from_numpy(value).to(device=device, dtype=torch.float32)


def _linear_row(row: dict[str, Any]) -> dict[str, Any]:
    recipe = row["recipe"]
    return {
        "sha256": row["source"]["sha256"],
        "lineage": row["lineage"],
        "source_group": "AudioMNIST",
        "speaker": row["group"],
        "split": row["source"]["role"],
        "label": row["label"],
        "format": "WAV",
        "method": None if recipe is None else recipe[0],
        "rate_percent": None if recipe is None else recipe[1],
        "channel": None if recipe is None else "mono",
        "plane": None if recipe is None else recipe[2],
    }


def feature_matrix(
    root: Path,
    rows: list[dict[str, Any]],
    role: str,
    *,
    deadline: Callable[[], None] = lambda: None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    expected_count = 3600 if role == "train" else 1800
    if role not in ("train", "development") or len(rows) != expected_count:
        raise ValueError("fixed WAV feature role/count required")
    if any(row.get("source", {}).get("role") != role for row in rows):
        raise ValueError("WAV feature split mismatch")
    values = np.empty((len(rows), len(nonjpeg_features.AUDIO_NAMES)), dtype=np.float64)
    labels = np.empty(len(rows), dtype=np.int64)
    linear_rows = []
    for index, row in enumerate(rows):
        deadline()
        data = materialize(root, row["source"], row["recipe"])
        features = np.asarray(nonjpeg_features.audio_features(data), dtype=np.float64)
        if (
            features.shape != (760,)
            or not np.isfinite(features).all()
            or np.any((features < 0) | (features > 1))
        ):
            raise ValueError("audio feature contract")
        values[index] = features
        labels[index] = int(row["label"] == "stego")
        linear_rows.append(_linear_row(row))
    if values.shape[1] != 760:
        raise ValueError("unexpected PCM feature version/dimension")
    return values, labels, linear_rows


def fit_linear_seed(
    root: Path,
    rows: list[dict[str, Any]],
    seed: int,
    *,
    deadline: Callable[[], None] = lambda: None,
    on_update: Callable[[], None] = lambda: None,
):
    """Fit the matched 760-feature reference, with no OOF or threshold search."""
    if seed not in SEEDS or len(rows) != 3600:
        raise ValueError("frozen AudioMNIST linear seed/exposure contract")
    x, y, linear_rows = feature_matrix(root, rows, "train", deadline=deadline)
    model = nonjpeg_linear.fit(
        x, y, linear_rows, seed, epochs=EPOCHS, deadline=deadline, on_update=on_update
    )
    if model["optimizer_updates"] != 1140 or model["row_presentations"] != 72000:
        raise ValueError("matched AudioMNIST linear exposure mismatch")
    return model


def predict_linear(
    root: Path,
    rows: list[dict[str, Any]],
    model,
    *,
    deadline: Callable[[], None] = lambda: None,
) -> np.ndarray:
    x, _y, _metadata = feature_matrix(root, rows, "development", deadline=deadline)
    scores = nonjpeg_linear.predict(model, x)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("invalid fixed-threshold linear development scores")
    return scores


def fit_seed(
    root: Path,
    rows: list[dict[str, Any]],
    seed: int,
    *,
    device: str = "cuda",
    deadline_seconds: float = 3600,
    deadline_at: float | None = None,
    on_update: Callable[[], None] = lambda: None,
):
    """Run one seed with isolated RNG and strict FP32/deterministic backend policy."""
    if seed not in SEEDS or len(rows) != 3600 or deadline_seconds <= 0:
        raise ValueError("frozen AudioMNIST seed/exposure contract")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AudioMNIST CNN unavailable: optional Torch missing") from exc
    if not device.startswith("cuda"):
        raise ValueError("frozen AudioMNIST CNN fit requires the owned CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("AudioMNIST CNN unavailable: requested CUDA device missing")
    selected = torch.device(device)
    device_index = selected.index if selected.index is not None else torch.cuda.current_device()
    prior = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[device_index]):
            return _fit_seed_impl(
                torch,
                root,
                rows,
                seed,
                device=device,
                deadline_seconds=deadline_seconds,
                deadline_at=deadline_at,
                on_update=on_update,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior[0]
        torch.backends.cudnn.allow_tf32 = prior[1]
        torch.backends.cudnn.deterministic = prior[2]
        torch.backends.cudnn.benchmark = prior[3]
        torch.use_deterministic_algorithms(prior[4])


def _fit_seed_impl(
    torch: Any,
    root: Path,
    rows: list[dict[str, Any]],
    seed: int,
    *,
    device: str = "cuda",
    deadline_seconds: float = 3600,
    deadline_at: float | None = None,
    on_update: Callable[[], None] = lambda: None,
):
    """Fit one frozen seed; caller must separately authorize a real experiment."""
    if any(row.get("source", {}).get("role") != "train" for row in rows):
        raise ValueError("training rows must all use frozen train speakers")
    family_counts: dict[str, int] = {}
    family_recipes: dict[str, set[tuple[str, int, int] | None]] = {}
    family_groups: dict[str, str] = {}
    for row in rows:
        family_counts[row["lineage"]] = family_counts.get(row["lineage"], 0) + 1
        family_recipes.setdefault(row["lineage"], set()).add(row["recipe"])
        if family_groups.setdefault(row["lineage"], row["group"]) != row["group"]:
            raise ValueError("one source file has multiple speaker groups")
    expected_recipes = {None, *RECIPES}
    if (
        len(family_counts) != 400
        or set(family_counts.values()) != {9}
        or len(set(family_groups.values())) != 20
        or any(recipes != expected_recipes for recipes in family_recipes.values())
    ):
        raise ValueError("expected 400 complete nine-row source families")
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = nonjpeg_models.network(nonjpeg_models.AUDIO_DOMAIN).to(
        device=device, dtype=torch.float32
    )
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim > 1 and not name.endswith("bias") else no_decay).append(parameter)
    optimizer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )
    started = time.monotonic()
    expires = started + deadline_seconds
    if deadline_at is not None:
        expires = min(expires, deadline_at)
    _check_deadline(expires, "AudioMNIST shared GPU ceiling reached")
    updates = presentations = 0
    losses: list[float] = []
    model.train()
    # Keep row order identical to core.nonjpeg_linear.fit for this seed.
    rng = np.random.default_rng(seed ^ 0x4E4A5047)
    for _epoch in range(EPOCHS):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), VIRTUAL_BATCH):
            _check_deadline(expires, "AudioMNIST shared GPU ceiling reached")
            batch = [rows[int(i)] for i in order[start : start + VIRTUAL_BATCH]]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for row in batch:
                values = _model_inputs(torch, root, row, device)
                label = int(row["label"] == "stego")
                logits = model(values)
                raw_loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([label], dtype=torch.long, device=device), reduction="sum"
                )
                weight = balanced_row_weight(label)
                (raw_loss * (weight / len(batch))).backward()
                batch_loss += float(raw_loss.detach()) / len(batch)
                presentations += 1
                del values, logits, raw_loss
                _check_deadline(expires, "AudioMNIST shared GPU ceiling reached")
            if not all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
            ):
                raise ValueError("non-finite AudioMNIST gradient")
            optimizer.step()
            updates += 1
            losses.append(batch_loss)
            on_update()
    expected_updates = EPOCHS * len(epoch_schedule(3600))
    if updates != expected_updates or presentations != EPOCHS * len(rows):
        raise ValueError("AudioMNIST exposure accounting mismatch")
    return model, {
        "status": "completed_seed_fit",
        "seed": seed,
        "epochs": EPOCHS,
        "row_presentations": presentations,
        "optimizer_updates": updates,
        "virtual_batch": VIRTUAL_BATCH,
        "partial_batch_rows": epoch_schedule(3600)[-1],
        "seconds": time.monotonic() - started,
        "mean_batch_loss": float(np.mean(losses)),
        "checkpoint_written": False,
        "device": device,
    }


def development_scores(
    root: Path,
    rows: list[dict[str, Any]],
    model,
    *,
    device: str = "cuda",
    deadline_at: float | None = None,
):
    """Score only fixed development rows at the prespecified probability threshold."""
    if len(rows) != 1800 or any(row.get("source", {}).get("role") != "development" for row in rows):
        raise ValueError("closed development role required")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AudioMNIST scoring unavailable: optional Torch missing") from exc
    previous = model.training
    previous_tf32 = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    scores = []
    try:
        with torch.inference_mode():
            for row in rows:
                if deadline_at is not None and time.monotonic() >= deadline_at:
                    raise TimeoutError("AudioMNIST shared GPU ceiling reached")
                values = _model_inputs(torch, root, row, device)
                probability = torch.softmax(model(values), dim=1)[0, 1]
                score = float(probability.cpu())
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("invalid development score")
                scores.append(score)
    finally:
        model.train(previous)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32[0]
        torch.backends.cudnn.allow_tf32 = previous_tf32[1]
    return np.asarray(scores, dtype=np.float64)


def development_metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    """Compute the eight prespecified matched cells with speaker-cluster intervals."""
    from core import nonjpeg_compare_metrics as compare_metrics

    if (
        len(rows) != 1800
        or scores.shape != (len(rows),)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or any(row.get("source", {}).get("role") != "development" for row in rows)
    ):
        raise ValueError("development-only score contract")
    covers: dict[str, tuple[int, str]] = {}
    cells: dict[tuple[str, int, int], list[tuple[int, str, str]]] = {}
    all_lineages = set()
    for index, row in enumerate(rows):
        lineage, group = row.get("lineage"), row.get("group")
        if not isinstance(lineage, str) or not isinstance(group, str):
            raise ValueError("missing bounded development identity")
        all_lineages.add(lineage)
        recipe = row.get("recipe")
        if row.get("label") == "cover":
            if recipe is not None or lineage in covers:
                raise ValueError("duplicate/invalid development cover")
            covers[lineage] = (index, group)
        elif row.get("label") == "stego" and recipe in RECIPES:
            method, rate, plane = recipe
            cells.setdefault((method, rate, plane), []).append((index, lineage, group))
        else:
            raise ValueError("unknown development row recipe")
    if len(covers) != 200 or len(cells) != 8 or set(covers) != all_lineages:
        raise ValueError("incomplete development source families")
    result = []
    for (method, rate, plane), stegos in sorted(cells.items()):
        if len(stegos) != 200 or {lineage for _, lineage, _ in stegos} != set(covers):
            raise ValueError("incomplete matched development cell")
        pairs, groups = [], []
        for index, lineage, group in stegos:
            cover_index, cover_group = covers[lineage]
            if group != cover_group:
                raise ValueError("cover/stego speaker group mismatch")
            pairs.append((scores[cover_index], scores[index]))
            groups.append(group)
        pair_array = np.asarray(pairs, dtype=np.float64)
        if not 2 <= len(set(groups)) <= 32:
            raise ValueError("speaker-cluster count outside confidence interval contract")
        draws = compare_metrics.bootstrap_weights(groups)
        point = compare_metrics.measured(pair_array, np.ones((1, len(pair_array))), 0.5)
        interval = compare_metrics.measured(pair_array, draws, 0.5)
        result.append(
            {
                "method": method,
                "rate_percent": rate,
                "plane": plane,
                "pairs": len(pair_array),
                "independent_groups": len(set(groups)),
                "uncertainty": "speaker-cluster bootstrap; ten development speakers",
                "bootstrap_draws": compare_metrics.DRAWS,
                "threshold": 0.5,
                "metrics": {key: float(value[0]) for key, value in point.items()},
                "metric_ci95": {
                    key: np.quantile(value, [0.025, 0.975]).tolist()
                    for key, value in interval.items()
                },
                "confusion": {
                    "tp": int(np.sum(pair_array[:, 1] >= 0.5)),
                    "tn": int(np.sum(pair_array[:, 0] < 0.5)),
                    "fp": int(np.sum(pair_array[:, 0] >= 0.5)),
                    "fn": int(np.sum(pair_array[:, 1] < 0.5)),
                },
            }
        )
    return result


def _check_deadline(expires: float, message: str) -> None:
    if time.monotonic() >= expires:
        raise TimeoutError(message)


def run_linear_pair(
    root: Path,
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    *,
    ceiling_seconds: float = 600,
) -> dict[str, Any]:
    """Fit/evaluate the matched linear reference under one CPU ceiling."""
    if ceiling_seconds <= 0:
        raise ValueError("positive CPU ceiling required")
    started = time.monotonic()
    expires = started + ceiling_seconds
    results = []
    for seed in SEEDS:

        def deadline() -> None:
            _check_deadline(expires, "AudioMNIST CPU ceiling reached")

        model = fit_linear_seed(root, train_rows, seed, deadline=deadline)
        scores = predict_linear(root, development_rows, model, deadline=deadline)
        results.append(
            {"seed": seed, "fit": model, "cells": development_metrics(development_rows, scores)}
        )
    deadline()
    return {
        "status": "completed_linear_pair",
        "seconds": time.monotonic() - started,
        "ceiling_seconds": ceiling_seconds,
        "seeds": results,
        "optimizer_updates": sum(row["fit"]["optimizer_updates"] for row in results),
        "checkpoint_written": False,
    }


def run_cnn_pair(
    root: Path,
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    *,
    device: str = "cuda",
    ceiling_seconds: float = 3600,
) -> dict[str, Any]:
    """Fit/evaluate both CNN seeds within one shared wall-clock ceiling."""
    if ceiling_seconds <= 0:
        raise ValueError("positive GPU ceiling required")
    started = time.monotonic()
    expires = started + ceiling_seconds
    results = []
    for seed in SEEDS:
        remaining = expires - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("AudioMNIST shared GPU ceiling reached")
        model, fit = fit_seed(
            root,
            train_rows,
            seed,
            device=device,
            deadline_seconds=remaining,
            deadline_at=expires,
        )
        scores = development_scores(
            root, development_rows, model, device=device, deadline_at=expires
        )
        results.append(
            {"seed": seed, "fit": fit, "cells": development_metrics(development_rows, scores)}
        )
        del model
    _check_deadline(expires, "AudioMNIST shared GPU ceiling reached")
    return {
        "status": "completed_cnn_pair",
        "seconds": time.monotonic() - started,
        "ceiling_seconds": ceiling_seconds,
        "seeds": results,
        "optimizer_updates": sum(row["fit"]["optimizer_updates"] for row in results),
        "checkpoint_written": False,
    }


def generated_cnn_preflight(
    train_lengths: list[int],
    development_lengths: list[int],
    *,
    device: str = "cuda",
    ceiling_seconds: float = 3600,
) -> dict[str, Any]:
    """Exercise the complete CNN schedule on generated WAVs at real file lengths."""
    if (
        len(train_lengths) != 400
        or len(development_lengths) != 200
        or any(
            type(n) is not int or not 160 <= n <= 220500
            for n in train_lengths + development_lengths
        )
        or device == "cpu"
        or ceiling_seconds <= 0
    ):
        raise ValueError("generated WAV preflight bounds")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AudioMNIST preflight unavailable: optional Torch missing") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("AudioMNIST preflight unavailable: CUDA missing")
    total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
    allocator_limit = 4 * 1024**3
    torch.cuda.set_per_process_memory_fraction(min(allocator_limit / total_memory, 1.0), 0)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="wav-generated-preflight-") as directory:
        root = Path(directory)
        records = []
        for role, lengths in (
            ("train", train_lengths),
            ("development", development_lengths),
        ):
            for index, frame_count in enumerate(lengths):
                speaker = f"generated-{role}-{index // 20:02d}"
                sample_seed = hashlib.sha256(f"wav-preflight:{role}:{index}".encode()).digest()
                samples = np.random.default_rng(int.from_bytes(sample_seed[:8], "little")).integers(
                    -32768, 32768, size=frame_count, dtype=np.int16
                )
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(48000)
                    audio.writeframes(samples.astype("<i2", copy=False).tobytes())
                raw = buffer.getvalue()
                relative = f"data/{speaker}/{index % 10}_{speaker}_{(index // 10) % 2}.wav"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(raw)
                records.append(
                    {
                        "relative_source_path": relative,
                        "role": role,
                        "speaker_group": speaker,
                        "parent_recording_id": None,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        if (
            len({row["speaker_group"] for row in records if row["role"] == "train"}) != 20
            or len({row["speaker_group"] for row in records if row["role"] == "development"}) != 10
        ):
            raise ValueError("generated WAV group schedule")
        train_rows = rows_for_role(records, "train")
        development_rows = rows_for_role(records, "development")
        # Preserve the two-seed RNG behavior while bounding GPU allocation.
        cnn = run_cnn_pair(
            root,
            train_rows,
            development_rows,
            device=device,
            ceiling_seconds=ceiling_seconds,
        )
        linear = run_linear_pair(root, train_rows, development_rows, ceiling_seconds=600)
        result: dict[str, Any] = {}
        cnn_seeds = []
        for row in cnn["seeds"]:
            fit = row["fit"]
            cnn_seeds.append(
                {
                    key: fit[key]
                    for key in (
                        "seed",
                        "epochs",
                        "row_presentations",
                        "optimizer_updates",
                        "virtual_batch",
                        "partial_batch_rows",
                        "seconds",
                    )
                }
            )
        linear_seeds = []
        for row in linear["seeds"]:
            fit = row["fit"]
            linear_seeds.append(
                {
                    key: fit[key]
                    for key in (
                        "seed",
                        "epochs",
                        "row_presentations",
                        "optimizer_updates",
                        "exposure_sha256",
                    )
                }
            )
        result.update(
            {
                "status": "passed_generated_only",
                "generated_cnn_optimizer_updates": cnn["optimizer_updates"],
                "generated_linear_optimizer_updates": linear["optimizer_updates"],
                "generated_optimizer_updates": cnn["optimizer_updates"]
                + linear["optimizer_updates"],
                "real_source_optimizer_updates": 0,
                "real_media_opened": False,
                "reserved_media_opened": False,
                "checkpoint_written": False,
                "cnn_pair_seconds": cnn["seconds"],
                "linear_pair_seconds": linear["seconds"],
                "cnn_seed_receipts": cnn_seeds,
                "linear_seed_receipts": linear_seeds,
                "generated_file_count": len(records),
                "generated_bytes": sum(path.stat().st_size for path in root.rglob("*.wav")),
                "frame_count_min": min(train_lengths + development_lengths),
                "frame_count_max": max(train_lengths + development_lengths),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "allocator_ceiling_bytes": allocator_limit,
                "seconds_including_imports": time.monotonic() - started,
            }
        )
        if (
            result["generated_cnn_optimizer_updates"] != 2280
            or result["generated_linear_optimizer_updates"] != 2280
            or result["peak_cuda_allocated_bytes"] > allocator_limit
        ):
            raise ValueError("generated WAV preflight gate failed")
        return result


def generated_schedule_smoke() -> dict[str, Any]:
    schedule = epoch_schedule(3600)
    if len(schedule) != 57 or schedule[-1] != 16 or sum(schedule) != 3600:
        raise ValueError("virtual batch schedule contract")
    return {
        "status": "ready",
        "rows_per_epoch": 3600,
        "updates_per_epoch": len(schedule),
        "rows_last_batch": schedule[-1],
        "updates_two_seeds": 2 * EPOCHS * len(schedule),
        "optimizer_updates": 0,
    }
