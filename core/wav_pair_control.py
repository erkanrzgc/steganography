"""Source-bound fresh generated CPU pair-control preparation and explicit fitting."""

from __future__ import annotations

import hashlib
import io
import math
import resource
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core import wav_control_runner as old
from core import wav_pair_candidate as candidate
from core import wav_pilot
from core.nonjpeg_data import regular_read, write

PROTOCOL = "wav-pair-fresh-cpu-control-20261022-v1"
SEED = 2026102201
MIB = 1024**2
SOURCES = (
    *old.SOURCES,
    "core/wav_pair_candidate.py",
    "tests/test_wav_pair_candidate.py",
    "core/wav_pair_control.py",
    "steganography/research_wav_pair_control.py",
    "tests/test_wav_pair_control.py",
    "docs/WAV_PAIR_CONTROL_PROTOCOL_20261022.md",
)
SPEC: dict[str, Any] = {
    "domain": candidate.DOMAIN,
    "generator_seed": SEED,
    "seeds": [20261016, 20261017],
    "epochs": 20,
    "batch": 64,
    "train_rows": 3600,
    "development_rows": 1800,
    "features": 10,
    "updates_per_seed": 1140,
    "presentations_per_seed": 72000,
    "optimizer": "AdamW",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "foreach": False,
    "fused": False,
    "learning_rate": 0.001,
    "weight_decay_weights_only": 0.0001,
    "loss": "sum(class_balanced_unreduced_cross_entropy)/actual_batch_rows",
    "dtype": "float32",
    "device": "cpu",
    "threads": 2,
    "numpy": "2.4.6",
    "torch": "2.14.0+cpu",
    "profile_seconds_max": 300,
    "projection_factor": 10,
    "projection_scoring_reserve_seconds": 60,
    "total_seconds_max": 1800,
    "host_bytes_max": 2 * 1024**3,
    "swap_bytes_max": 0,
    "output_bytes_max": 64 * MIB,
    "file_bytes_max": MIB,
    "train_auc_min": 0.60,
    "development_auc_min": 0.70,
}


def digest(path: Path) -> str:
    return hashlib.sha256(regular_read(path, MIB)).hexdigest()


def load(root: Path, path: Path):
    value, sha = old.gate.read_document(path)
    if (
        value.get("schema_version") != "wav-pair-control-protocol-v1"
        or value.get("protocol_id") != PROTOCOL
        or old.encoded({"spec": value.get("spec")}) != old.encoded({"spec": SPEC})
        or value.get("real_fit_authorized") is not False
        or old.encoded({"attempts": value.get("attempts_allowed")})
        != old.encoded({"attempts": {"profile": 1, "fit": 1}})
        or set(value.get("source_sha256", {})) != set(SOURCES)
    ):
        raise ValueError("pair protocol contract")
    for name, expected in value["source_sha256"].items():
        if old.gate.source_hash(root / name) != expected:
            raise ValueError("pair source binding")
    return value, sha


def runtime():
    import torch

    if np.__version__ != SPEC["numpy"] or torch.__version__ != SPEC["torch"]:
        raise RuntimeError("frozen CPU runtime unavailable")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return torch


def resources():
    folder = old.cgroup()
    if int(regular_read(folder / "memory.max", 64)) > SPEC["host_bytes_max"]:
        raise ValueError("pair memory cap")
    measured = {
        "host_peak_bytes": int(regular_read(folder / "memory.peak", 64)),
        "swap_bytes": int(regular_read(folder / "memory.swap.current", 64)),
        "process_rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    if not 0 < measured["host_peak_bytes"] <= SPEC["host_bytes_max"] or measured["swap_bytes"]:
        raise ValueError("pair resource usage exceeded")
    return measured


def new_output(path: Path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("pair output symlink")
    path.mkdir(mode=0o700)


def array_save(path: Path, values):
    out = io.BytesIO()
    np.save(out, values, allow_pickle=False)
    raw = out.getvalue()
    if len(raw) > MIB:
        raise ValueError("pair array bound")
    write(path, raw)


def generated(root: Path, schedule, expires):
    rng = np.random.default_rng(SEED)
    records, decoded, total = [], set(), 0
    for role in ("train", "development"):
        for index, frames in enumerate(schedule[role]):
            old.check_deadline(expires)
            pcm = old.speech_like(rng, frames).tobytes()
            pcm_sha = hashlib.sha256(pcm).hexdigest()
            if pcm_sha in decoded:
                raise ValueError("duplicate fresh PCM")
            decoded.add(pcm_sha)
            out = io.BytesIO()
            with wave.open(out, "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(48000)
                stream.writeframes(pcm)
            raw = out.getvalue()
            total += len(raw)
            if total > 48 * MIB:
                raise ValueError("fresh corpus storage bound")
            group = f"pair-fresh-{role}-{index // 20:02d}"
            rel = f"data/{group}/{index:04d}.wav"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, raw)
            records.append(
                {
                    "role": role,
                    "speaker_group": group,
                    "relative_source_path": rel,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "frames": frames,
                    "decoded_pcm_sha256": pcm_sha,
                    "parent_recording_id": None,
                }
            )
    return records, total


def row_sha(rows):
    values = [
        {
            "source_sha256": r["lineage"],
            "recipe": r["recipe"],
            "group": r["group"],
            "role": r["source"]["role"],
        }
        for r in rows
    ]
    return hashlib.sha256(old.encoded({"rows": values})).hexdigest()


def validate_records(records, previous):
    if not isinstance(records, list) or len(records) != 600:
        raise ValueError("fresh corpus count")
    groups: dict[str, set[str]] = {role: set() for role in ("train", "development")}
    for r in records:
        if (
            not isinstance(r, dict)
            or r.get("role") not in groups
            or not isinstance(r.get("speaker_group"), str)
            or type(r.get("frames")) is not int
            or not 160 <= r["frames"] <= 220500
        ):
            raise ValueError("fresh record domain")
        groups[r["role"]].add(r["speaker_group"])
    if groups["train"] & groups["development"]:
        raise ValueError("fresh group overlap")
    for key in ("sha256", "decoded_pcm_sha256"):
        values = [r[key] for r in records]
        if len(set(values)) != 600 or set(values) & {r[key] for r in previous["files"]}:
            raise ValueError("fresh source/PCM overlap")
    return {role: wav_pilot.rows_for_role(records, role) for role in ("train", "development")}


def profile(root: Path, protocol: Path, out: Path):
    started = old.process_started()
    _, sha = load(root, protocol)
    resources()
    new_output(out)
    old.save(out / "profile-started.json", {"protocol_sha256": sha, "optimizer_updates": 0})
    expires = started + SPEC["profile_seconds_max"] - 5
    torch = runtime()
    previous, _ = old.gate.read_document(root / old.CORPUS)
    extraction_started = time.monotonic()
    oracles = 0
    with tempfile.TemporaryDirectory(prefix="generated-only-", dir=out) as name:
        folder = Path(name)
        records, total = generated(folder, old.lengths(root), expires)
        rows = validate_records(records, previous)
        for role, selected in rows.items():
            values = []
            for row in selected:
                old.check_deadline(expires)
                raw = wav_pilot.materialize(folder, row["source"], row["recipe"])
                oracles += int(row["recipe"] is not None)
                values.append(candidate.inputs(raw)[0])
            array_save(out / f"{role}.npy", np.asarray(values, dtype=np.float32))
            read_array(out / f"{role}.npy", len(selected))
    manifest = {
        "generator_seed": SEED,
        "generated_bytes": total,
        "files": records,
        "payload_oracles_passed": oracles,
        "old_raw_and_pcm_overlap": 0,
        "train_development_raw_and_pcm_overlap": 0,
    }
    old.save(out / "manifest.json", manifest)
    extraction_seconds = time.monotonic() - extraction_started
    train = read_array(out / "train.npy", 3600)
    with torch.random.fork_rng():
        torch.manual_seed(2026102202)
        model = candidate.network()
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.from_numpy(train[:64])
    smoke_started = time.monotonic()
    for _ in range(20):
        old.check_deadline(expires)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), torch.arange(64) % 2).backward()
        finite_gradients(torch, model)
    smoke_seconds = time.monotonic() - smoke_started
    if not all(torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True)):
        raise ValueError("zero-update profile changed parameters")
    load(root, protocol)
    old.check_deadline(expires)
    projection = time.monotonic() - started + smoke_seconds / 20 * 2280 * 10 + 60
    if projection > SPEC["total_seconds_max"]:
        raise ValueError("profile time projection exceeds budget")
    receipt = {
        "status": "passed_zero_update_profile",
        "protocol_sha256": sha,
        "profile_wall_seconds": time.monotonic() - started,
        "extraction_seconds": extraction_seconds,
        "smoke_seconds": smoke_seconds,
        "conservative_total_projection_seconds": projection,
        "backward_batches": 20,
        "optimizer_updates": 0,
        "parameters_unchanged": True,
        "manifest_sha256": digest(out / "manifest.json"),
        "arrays_sha256": {r: digest(out / f"{r}.npy") for r in rows},
        "row_order_sha256": {r: row_sha(v) for r, v in rows.items()},
        "resources": resources(),
        "real_media_opened": False,
    }
    old.save(out / "profile.json", receipt)
    return receipt


def read_array(path: Path, count: int):
    values = np.load(io.BytesIO(regular_read(path, MIB)), allow_pickle=False)
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float32
        or values.shape != (count, 10)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("pair cache dtype/shape/range")
    return values


def finite_gradients(torch, model):
    if not all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    ):
        raise ValueError("nonfinite pair gradients")


def fit_seed(torch, x, labels, seed, expires, on_update):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = candidate.network()
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": 0.0001},
            {"params": [p for p in model.parameters() if p.ndim == 1], "weight_decay": 0.0},
        ],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    rng = np.random.default_rng(seed ^ 0x4E4A5047)
    updates = presentations = 0
    started = time.monotonic()
    for _ in range(20):
        order = rng.permutation(len(x))
        for start in range(0, len(order), 64):
            old.check_deadline(expires)
            batch = order[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(x[batch]), labels[batch], reduction="none"
            )
            weights = torch.where(labels[batch] == 0, 4.5, 0.5625).to(dtype=torch.float32)
            objective = (loss * weights).sum() / len(batch)
            objective.backward()
            finite_gradients(torch, model)
            optimizer.step()
            updates += 1
            presentations += len(batch)
            on_update()
    return model, {
        "seed": seed,
        "epochs": 20,
        "optimizer_updates": updates,
        "row_presentations": presentations,
        "fit_seconds": time.monotonic() - started,
    }


def predict(torch, model, values):
    model.eval()
    with torch.inference_mode():
        scores = torch.softmax(model(torch.from_numpy(values)), 1)[:, 1].numpy()
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite pair scores")
    return scores.astype(np.float64)


def fit(root: Path, protocol: Path, cache: Path, binding: Path):
    started = old.process_started()
    _, sha = load(root, protocol)
    resources()
    profile_receipt, profile_sha = old.gate.read_document(cache / "profile.json")
    bound, _ = old.gate.read_document(binding)
    debit = bound.get("profile_debit_seconds")
    expected = {
        "protocol_sha256": sha,
        "profile_sha256": profile_sha,
        "fit_attempts_allowed": 1,
        "profile_debit_seconds": debit,
    }
    if old.encoded(bound) != old.encoded(expected):
        raise ValueError("fit binding")
    if (
        profile_receipt.get("status") != "passed_zero_update_profile"
        or profile_receipt.get("protocol_sha256") != sha
        or not isinstance(debit, (int, float))
        or isinstance(debit, bool)
        or not math.isfinite(debit)
        or not 0 < debit <= SPEC["profile_seconds_max"]
        or not number(profile_receipt.get("profile_wall_seconds"), 0.000001, debit)
        or type(profile_receipt.get("optimizer_updates")) is not int
        or profile_receipt["optimizer_updates"] != 0
        or profile_receipt.get("parameters_unchanged") is not True
        or not number(profile_receipt.get("conservative_total_projection_seconds"), 0.000001, 1800)
    ):
        raise ValueError("profile prerequisite/debit")
    manifest, manifest_sha = old.gate.read_document(cache / "manifest.json")
    if (
        manifest_sha != profile_receipt["manifest_sha256"]
        or manifest["generator_seed"] != SEED
        or manifest["payload_oracles_passed"] != 4800
    ):
        raise ValueError("fresh manifest binding")
    previous, _ = old.gate.read_document(root / old.CORPUS)
    rows = validate_records(manifest["files"], previous)
    arrays = {}
    for role, selected in rows.items():
        if (
            digest(cache / f"{role}.npy") != profile_receipt["arrays_sha256"][role]
            or row_sha(selected) != profile_receipt["row_order_sha256"][role]
        ):
            raise ValueError("cache source/split binding")
        arrays[role] = read_array(cache / f"{role}.npy", len(selected))
    out = cache / "fit"
    new_output(out)
    old.save(out / "fit-started.json", {"protocol_sha256": sha, "automatic_retry": False})
    expires = started + SPEC["total_seconds_max"] - debit - 5
    torch = runtime()
    x = torch.from_numpy(arrays["train"])
    labels = torch.tensor([int(r["recipe"] is not None) for r in rows["train"]])
    state: dict[str, Any] = {"confirmed_updates_lower_bound": 0, "phase": "fit", "seed": 20261016}
    runs = []
    for seed in SPEC["seeds"]:
        state["seed"] = seed
        old.progress(out, state)

        def on_update():
            state["confirmed_updates_lower_bound"] += 1
            old.progress(out, state)

        model, run = fit_seed(torch, x, labels, seed, expires, on_update)
        old.save(out / f"fit-{seed}.json", run)
        old.check_deadline(expires)
        for role, values in arrays.items():
            scores = predict(torch, model, values)
            auc, cells = old.metrics(rows[role], scores)
            run[role] = {
                "auc_all": auc,
                "cells": cells,
                "scores_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
            }
        old.save(out / f"seed-{seed}.json", run)
        runs.append(run)
        del model
    old.check_deadline(expires)
    load(root, protocol)
    baseline = {}
    for role, values in arrays.items():
        auc, cells = old.metrics(rows[role], values[:, 0].astype(np.float64))
        baseline[role] = {"auc_all": auc, "cells": cells}
    old.check_deadline(expires)
    result = {
        "status": "completed_generated_pair_control",
        "domain": candidate.DOMAIN,
        "generator_seed": SEED,
        "generated_only": True,
        "reserved_media_opened": False,
        "protocol_sha256": sha,
        "profile_sha256": profile_sha,
        "profile_debit_seconds": debit,
        "seeds": runs,
        "fit_wall_seconds": time.monotonic() - started,
        "total_wall_seconds": debit + time.monotonic() - started,
        "resources": resources(),
        "real_fit_authorized": False,
        "production_qualified": False,
        "real_media_opened": False,
        "checkpoint_written": False,
    }
    result["fixed_global_plane0"] = baseline
    result["decision"] = decide(result, sha)
    old.save(out / "result.json", result)
    old.progress(out, {**state, "phase": "completed"})
    return result


def number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def decide(result, expected_sha, external_total_seconds=None):
    if external_total_seconds is not None and not number(external_total_seconds, 0.000001, 1800):
        return "unavailable"
    if not isinstance(result, dict):
        return "unavailable"
    resources_ = result.get("resources")
    runs = result.get("seeds")
    if (
        result.get("status") != "completed_generated_pair_control"
        or result.get("protocol_sha256") != expected_sha
        or result.get("domain") != candidate.DOMAIN
        or type(result.get("generator_seed")) is not int
        or result["generator_seed"] != SEED
        or result.get("generated_only") is not True
        or not number(result.get("total_wall_seconds"), 0.000001, 1800)
        or any(
            result.get(k) is not False
            for k in (
                "real_fit_authorized",
                "production_qualified",
                "real_media_opened",
                "reserved_media_opened",
                "checkpoint_written",
            )
        )
        or not isinstance(resources_, dict)
        or type(resources_.get("swap_bytes")) is not int
        or resources_["swap_bytes"] != 0
        or not number(resources_.get("host_peak_bytes"), 1, 2 * 1024**3)
        or not isinstance(runs, list)
        or len(runs) != 2
    ):
        return "unavailable"
    outcomes = []
    for run, seed in zip(runs, SPEC["seeds"], strict=True):
        if not isinstance(run, dict) or any(
            type(run.get(k)) is not int or run[k] != value
            for k, value in {
                "seed": seed,
                "epochs": 20,
                "optimizer_updates": 1140,
                "row_presentations": 72000,
            }.items()
        ):
            return "unavailable"
        for role in ("train", "development"):
            metrics = run.get(role)
            if not isinstance(metrics, dict) or not number(metrics.get("auc_all"), 0, 1):
                return "unavailable"
            cells = metrics.get("cells")
            if not isinstance(cells, list) or len(cells) != 8:
                return "unavailable"
            keys = []
            for cell in cells:
                if (
                    not isinstance(cell, dict)
                    or type(cell.get("rate_percent")) is not int
                    or type(cell.get("plane")) is not int
                    or not isinstance(cell.get("method"), str)
                    or not number(cell.get("roc_auc"), 0, 1)
                ):
                    return "unavailable"
                keys.append((cell["method"], cell["rate_percent"], cell["plane"]))
            if set(keys) != set(wav_pilot.RECIPES):
                return "unavailable"
        outcomes.append(
            run["train"]["auc_all"] >= 0.60
            and all(
                c["roc_auc"] >= 0.70
                for c in run["development"]["cells"]
                if c["rate_percent"] == 40 and c["plane"] == 0
            )
        )
    return "passed_generated_only" if all(outcomes) else "failed_learning_gate"
