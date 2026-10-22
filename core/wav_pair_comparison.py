"""Fresh, source-bound raw/z-score PCM pair-head research comparison."""

from __future__ import annotations

import hashlib
import io
import math
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core import wav_pair_control as base
from core.nonjpeg_data import write

PROTOCOL = "wav-pair-scaling-comparison-20261022-v1"
GENERATOR_SEED = 2026102204
PREVIOUS = "benchmarks/wav-pair-fresh-manifest-20261022.json"
PARENT = "benchmarks/wav-pair-control-protocol-20261022.json"
ARMS = ("raw", "train_zscore")
DOMAINS = {"raw": base.candidate.DOMAIN, "train_zscore": "audio-pcm16-pair-zscore-mlp-v2"}
POINTS = (1, 57, 570, 1140)
SOURCES = (
    *base.SOURCES,
    PARENT,
    PREVIOUS,
    "benchmarks/wav-pair-diagnosis-results-20261022.json",
    "docs/WAV_PAIR_NEXT_INTERVENTION_DRAFT_20261022.md",
    "core/wav_pair_comparison.py",
    "steganography/research_wav_pair_comparison.py",
    "tests/test_wav_pair_comparison.py",
    "docs/WAV_PAIR_COMPARISON_PROTOCOL_20261022.md",
)
SPEC: dict[str, Any] = {
    "generator_seed": GENERATOR_SEED,
    "arms": list(ARMS),
    "domains": DOMAINS,
    "seeds": [20261016, 20261017],
    "order": "seed-major; raw then train_zscore",
    "epochs": 20,
    "batch": 64,
    "updates_per_fit": 1140,
    "presentations_per_fit": 72000,
    "total_updates": 4560,
    "total_presentations": 288000,
    "parameters": 418,
    "recipe": {
        k: base.SPEC[k]
        for k in (
            "optimizer",
            "betas",
            "eps",
            "foreach",
            "fused",
            "learning_rate",
            "weight_decay_weights_only",
            "loss",
            "dtype",
            "device",
            "threads",
            "numpy",
            "torch",
        )
    },
    "class_weights": {"cover": 4.5, "stego": 0.5625},
    "amsgrad": False,
    "permutation_seed_xor": 0x4E4A5047,
    "normalizer": "all_train_rows_float64_population_mean_std",
    "std_floor": 1e-6,
    "transform": "float64_center_divide_then_single_float32_cast; no_clip",
    "telemetry_updates": list(POINTS),
    "epoch_train_telemetry": "all_20_epochs",
    "profile_seconds_max": 300,
    "total_seconds_max": 1800,
    "host_bytes_max": 2 * 1024**3,
    "swap_bytes_max": 0,
    "output_bytes_max": 64 * base.MIB,
    "file_bytes_max": base.MIB,
    "projection_factor": 10,
    "scoring_reserve_seconds": 60,
    "train_auc_min": 0.60,
    "development_auc_min": 0.70,
}


def load(root: Path, protocol: Path):
    document, sha = base.old.gate.read_document(protocol)
    if (
        document.get("schema_version") != "wav-pair-comparison-protocol-v1"
        or document.get("protocol_id") != PROTOCOL
        or base.old.encoded({"spec": document.get("spec")}) != base.old.encoded({"spec": SPEC})
        or document.get("real_fit_authorized") is not False
        or base.old.encoded({"attempts": document.get("attempts_allowed")})
        != base.old.encoded({"attempts": {"profile": 1, "fit": 1}})
        or set(document.get("source_sha256", {})) != set(SOURCES)
    ):
        raise ValueError("comparison protocol")
    for name, expected in document["source_sha256"].items():
        if base.old.gate.source_hash(root / name) != expected:
            raise ValueError("comparison source binding")
    return document, sha


def normalization(values):
    if (
        values.dtype != np.float32
        or values.shape != (3600, 10)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("normalizer training domain")
    raw = values.astype(np.float64)
    return {
        "mean": raw.mean(axis=0).tolist(),
        "std": np.maximum(raw.std(axis=0), 1e-6).tolist(),
        "rows": 3600,
        "ddof": 0,
        "std_floor": 1e-6,
        "input_domain": base.candidate.DOMAIN,
        "output_domain": DOMAINS["train_zscore"],
    }


def transform(values, state, arm):
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != 10
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("raw comparison input domain")
    if arm == "raw":
        return values.copy()
    if arm != "train_zscore":
        raise ValueError("unknown comparison arm")
    mean, std = np.asarray(state["mean"]), np.asarray(state["std"])
    if (
        mean.shape != (10,)
        or std.shape != (10,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std < 1e-6)
    ):
        raise ValueError("normalizer parameters")
    result = ((values.astype(np.float64) - mean) / std).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite standardized inputs")
    return result


def generate(folder, schedule, expires):
    rng = np.random.default_rng(GENERATOR_SEED)
    records, total = [], 0
    for role in ("train", "development"):
        for index, frames in enumerate(schedule[role]):
            base.old.check_deadline(expires)
            pcm = base.old.speech_like(rng, frames).tobytes()
            output = io.BytesIO()
            with wave.open(output, "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(48000)
                stream.writeframes(pcm)
            raw = output.getvalue()
            total += len(raw)
            if total > 48 * base.MIB:
                raise ValueError("comparison generated storage")
            group = f"pair-scale-{role}-{index // 20:02d}"
            relative = f"data/{group}/{index:04d}.wav"
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, raw)
            records.append(
                {
                    "role": role,
                    "speaker_group": group,
                    "relative_source_path": relative,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "frames": frames,
                    "decoded_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "parent_recording_id": None,
                }
            )
    return records, total


def records(root, manifest):
    rows = None
    for name in (base.old.CORPUS, PREVIOUS):
        previous, _ = base.old.gate.read_document(root / name)
        rows = base.validate_records(manifest["files"], previous)
    return rows


def parameter_vector(model):
    return np.concatenate(
        [p.detach().cpu().numpy().ravel().astype(np.float64) for p in model.parameters()]
    )


def parameter_sha(model):
    return hashlib.sha256(parameter_vector(model).astype("<f4").tobytes()).hexdigest()


def profile(root: Path, protocol: Path, out: Path):
    started = base.old.process_started()
    _, sha = load(root, protocol)
    base.resources()
    base.new_output(out)
    base.old.save(out / "profile-started.json", {"protocol_sha256": sha, "optimizer_updates": 0})
    expires = started + 295
    torch = base.runtime()
    with tempfile.TemporaryDirectory(prefix="fresh-comparison-", dir=out) as name:
        folder = Path(name)
        items, total = generate(folder, base.old.lengths(root), expires)
        manifest = {
            "generator_seed": GENERATOR_SEED,
            "files": items,
            "generated_bytes": total,
            "payload_oracles_passed": 0,
            "prior_and_cross_split_overlap": 0,
        }
        selected = records(root, manifest)
        for role, rows in selected.items():
            values = []
            for row in rows:
                base.old.check_deadline(expires)
                raw = base.wav_pilot.materialize(folder, row["source"], row["recipe"])
                manifest["payload_oracles_passed"] += int(row["recipe"] is not None)
                values.append(base.candidate.inputs(raw)[0])
            base.array_save(out / f"{role}.npy", np.asarray(values, dtype=np.float32))
    if manifest["payload_oracles_passed"] != 4800:
        raise ValueError("comparison payload count")
    base.old.save(out / "manifest.json", manifest)
    arrays = {
        role: base.read_array(out / f"{role}.npy", len(rows)) for role, rows in selected.items()
    }
    state = normalization(arrays["train"])
    base.old.save(out / "normalization.json", state)
    smoke = []
    for arm in ARMS:
        with torch.random.fork_rng():
            torch.manual_seed(2026102205)
            model = base.candidate.network()
        before = parameter_sha(model)
        x = torch.from_numpy(transform(arrays["train"], state, arm)[:64])
        smoke_started = time.monotonic()
        for _ in range(20):
            base.old.check_deadline(expires)
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x), torch.arange(64) % 2).backward()
            base.finite_gradients(torch, model)
        if parameter_sha(model) != before:
            raise ValueError("profile parameter mutation")
        smoke.append(
            {
                "arm": arm,
                "backward_batches": 20,
                "seconds": time.monotonic() - smoke_started,
                "parameters_unchanged": True,
                "initial_parameter_sha256": before,
            }
        )
    projection = time.monotonic() - started + max(s["seconds"] for s in smoke) / 20 * 4560 * 10 + 60
    if projection > 1800:
        raise ValueError("comparison time projection")
    load(root, protocol)
    base.old.check_deadline(expires)
    result = {
        "status": "passed_zero_update_profile",
        "protocol_sha256": sha,
        "optimizer_updates": 0,
        "smoke": smoke,
        "profile_wall_seconds": time.monotonic() - started,
        "conservative_total_projection_seconds": projection,
        "manifest_sha256": base.digest(out / "manifest.json"),
        "normalization_sha256": base.digest(out / "normalization.json"),
        "arrays_sha256": {r: base.digest(out / f"{r}.npy") for r in selected},
        "row_order_sha256": {r: base.row_sha(rows) for r, rows in selected.items()},
        "resources": base.resources(),
        "real_media_opened": False,
    }
    base.old.save(out / "profile.json", result)
    return result


def telemetry(torch, model, x, labels, rows):
    prior = model.training
    before = parameter_sha(model)
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        losses = (
            torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            .numpy()
            .astype(np.float64)
        )
        scores = torch.softmax(logits, 1)[:, 1].numpy().astype(np.float64)
    model.train(prior)
    if not np.isfinite(losses).all() or parameter_sha(model) != before:
        raise ValueError("nonfinite or mutating telemetry")
    auc, cells = base.old.metrics(rows, scores)
    cover_loss = float(losses[::9].mean())
    for index, cell in enumerate(cells, 1):
        cell["stego_loss"] = float(losses[index::9].mean())
        cell["balanced_loss"] = (cover_loss + cell["stego_loss"]) / 2
    return {
        "auc_all": auc,
        "cells": cells,
        "cover_loss": cover_loss,
        "unweighted_loss": float(losses.mean()),
        "weighted_objective": float((losses * np.where(labels.numpy() == 0, 4.5, 0.5625)).mean()),
    }


def fit_one(torch, values, rows, seed, arm, expires, on_update, on_epoch):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = base.candidate.network()
    model.architecture = DOMAINS[arm]
    initial = parameter_vector(model)
    initial_sha = parameter_sha(model)
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
    x = torch.from_numpy(values)
    labels = torch.tensor([int(r["recipe"] is not None) for r in rows])
    rng = np.random.default_rng(seed ^ 0x4E4A5047)
    updates = presentations = 0
    steps, epochs = [], []
    order_hash = hashlib.sha256()
    for epoch in range(1, 21):
        order = rng.permutation(len(x))
        order_hash.update(order.astype("<i4").tobytes())
        for start in range(0, len(x), 64):
            base.old.check_deadline(expires)
            batch = order[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(x[batch]), labels[batch], reduction="none"
            )
            weights = torch.where(labels[batch] == 0, 4.5, 0.5625).to(torch.float32)
            objective = (loss * weights).sum() / len(batch)
            objective.backward()
            base.finite_gradients(torch, model)
            measured = updates + 1 in POINTS
            if measured:
                before = parameter_vector(model)
                grad_norm = math.sqrt(
                    sum(float(p.grad.detach().double().square().sum()) for p in model.parameters())
                )
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("nonfinite updated parameters")
            updates += 1
            presentations += len(batch)
            on_update()
            if measured:
                steps.append(
                    {
                        "update": updates,
                        "objective": float(objective.detach()),
                        "gradient_norm": grad_norm,
                        "parameter_step_norm": float(
                            np.linalg.norm(parameter_vector(model) - before)
                        ),
                    }
                )
        base.old.check_deadline(expires)
        record = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "row_presentations": presentations,
            **telemetry(torch, model, x, labels, rows),
        }
        on_epoch(record)
        epochs.append(record)
    return model, {
        "seed": seed,
        "arm": arm,
        "domain": DOMAINS[arm],
        "epochs": 20,
        "optimizer_updates": updates,
        "row_presentations": presentations,
        "initial_parameter_sha256": initial_sha,
        "final_parameter_sha256": parameter_sha(model),
        "parameter_change_norm": float(np.linalg.norm(parameter_vector(model) - initial)),
        "order_sha256": order_hash.hexdigest(),
        "step_telemetry": steps,
        "epoch_telemetry": epochs,
    }


def fit(root: Path, protocol: Path, cache: Path, binding: Path):
    started = base.old.process_started()
    _, sha = load(root, protocol)
    base.resources()
    profile_receipt, profile_sha = base.old.gate.read_document(cache / "profile.json")
    bound, _ = base.old.gate.read_document(binding)
    debit = bound.get("profile_debit_seconds")
    if (
        base.old.encoded(bound)
        != base.old.encoded(
            {
                "protocol_sha256": sha,
                "profile_sha256": profile_sha,
                "profile_debit_seconds": debit,
                "fit_attempts_allowed": 1,
            }
        )
        or not isinstance(debit, (int, float))
        or not base.number(debit, 0.000001, 300)
        or profile_receipt.get("status") != "passed_zero_update_profile"
        or profile_receipt.get("protocol_sha256") != sha
        or type(profile_receipt.get("optimizer_updates")) is not int
        or profile_receipt["optimizer_updates"] != 0
        or not base.number(profile_receipt.get("profile_wall_seconds"), 0.000001, debit)
        or not base.number(
            profile_receipt.get("conservative_total_projection_seconds"), 0.000001, 1800
        )
        or not profile_proof(profile_receipt)
    ):
        raise ValueError("comparison fit prerequisite")
    for name in ("manifest", "normalization"):
        if base.digest(cache / f"{name}.json") != profile_receipt[f"{name}_sha256"]:
            raise ValueError("comparison manifest/normalizer binding")
    manifest, _ = base.old.gate.read_document(cache / "manifest.json")
    if (
        manifest.get("generator_seed") != GENERATOR_SEED
        or manifest.get("payload_oracles_passed") != 4800
    ):
        raise ValueError("comparison manifest provenance")
    selected = records(root, manifest)
    arrays = {}
    for role, rows in selected.items():
        if (
            base.digest(cache / f"{role}.npy") != profile_receipt["arrays_sha256"][role]
            or base.row_sha(rows) != profile_receipt["row_order_sha256"][role]
        ):
            raise ValueError("comparison cache/order binding")
        arrays[role] = base.read_array(cache / f"{role}.npy", len(rows))
    state, _ = base.old.gate.read_document(cache / "normalization.json")
    if base.old.encoded(state) != base.old.encoded(normalization(arrays["train"])):
        raise ValueError("normalizer must use train only")
    out = cache / "fit"
    base.new_output(out)
    base.old.save(out / "fit-started.json", {"protocol_sha256": sha, "automatic_retry": False})
    expires = started + 1800 - debit - 5
    torch = base.runtime()
    progress: dict[str, Any] = {"confirmed_updates_lower_bound": 0, "phase": "fit"}
    runs = []
    for seed in SPEC["seeds"]:
        for arm in ARMS:
            destination = out / f"{arm}-{seed}"
            base.new_output(destination)
            progress.update({"seed": seed, "arm": arm})
            base.old.progress(out, progress)

            def update():
                progress["confirmed_updates_lower_bound"] += 1
                base.old.progress(out, progress)

            def epoch(record, target=destination):
                base.old.save(target / f"epoch-{record['epoch']:02d}.json", record)

            train = transform(arrays["train"], state, arm)
            model, run = fit_one(torch, train, selected["train"], seed, arm, expires, update, epoch)
            base.old.save(destination / "fit.json", run)
            for role, values in arrays.items():
                base.old.check_deadline(expires)
                scores = base.predict(torch, model, transform(values, state, arm))
                base.array_save(destination / f"{role}-scores.npy", scores)
                auc, cells = base.old.metrics(selected[role], scores)
                run[role] = {
                    "auc_all": auc,
                    "cells": cells,
                    "scores_sha256": base.digest(destination / f"{role}-scores.npy"),
                }
            base.old.save(destination / "seed.json", run)
            runs.append(run)
            del model
    fixed = {}
    for role, values in arrays.items():
        auc, cells = base.old.metrics(selected[role], values[:, 0].astype(np.float64))
        fixed[role] = {"auc_all": auc, "cells": cells}
    load(root, protocol)
    base.old.check_deadline(expires)
    result = {
        "status": "completed_generated_comparison",
        "protocol_sha256": sha,
        "generator_seed": GENERATOR_SEED,
        "profile_sha256": profile_sha,
        "profile_debit_seconds": debit,
        "runs": runs,
        "fixed_global_plane0": fixed,
        "fit_wall_seconds": time.monotonic() - started,
        "total_wall_seconds": debit + time.monotonic() - started,
        "resources": base.resources(),
        "real_media_opened": False,
        "reserved_media_opened": False,
        "checkpoint_written": False,
        "real_fit_authorized": False,
        "production_qualified": False,
    }
    result["decisions"] = decide(result, sha)
    base.old.save(out / "result.json", result)
    base.old.progress(out, {**progress, "phase": "completed"})
    return result


def profile_proof(value):
    smoke = value.get("smoke")
    resources = value.get("resources")
    if (
        not isinstance(smoke, list)
        or len(smoke) != 2
        or value.get("real_media_opened") is not False
        or not isinstance(resources, dict)
        or type(resources.get("swap_bytes")) is not int
        or resources["swap_bytes"] != 0
        or not base.number(resources.get("host_peak_bytes"), 1, 2 * 1024**3)
    ):
        return False
    for point, arm in zip(smoke, ARMS, strict=True):
        if (
            not isinstance(point, dict)
            or point.get("arm") != arm
            or type(point.get("backward_batches")) is not int
            or point["backward_batches"] != 20
            or point.get("parameters_unchanged") is not True
            or not base.number(point.get("seconds"), 0.000000001, 300)
            or not is_sha(point.get("initial_parameter_sha256"))
        ):
            return False
    return smoke[0]["initial_parameter_sha256"] == smoke[1]["initial_parameter_sha256"]


def is_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def valid_metrics(value):
    if not isinstance(value, dict) or not base.number(value.get("auc_all"), 0, 1):
        return False
    cells = value.get("cells")
    return (
        isinstance(cells, list)
        and len(cells) == 8
        and all(
            isinstance(c, dict)
            and type(c.get("rate_percent")) is int
            and type(c.get("plane")) is int
            and isinstance(c.get("method"), str)
            and base.number(c.get("roc_auc"), 0, 1)
            for c in cells
        )
        and {(c["method"], c["rate_percent"], c["plane"]) for c in cells}
        == set(base.wav_pilot.RECIPES)
    )


def decide(result, sha, external_total_seconds=None):
    unavailable = {arm: "unavailable" for arm in ARMS}
    if not isinstance(result, dict):
        return unavailable
    resources = result.get("resources")
    if (
        result.get("status") != "completed_generated_comparison"
        or result.get("protocol_sha256") != sha
        or type(result.get("generator_seed")) is not int
        or result["generator_seed"] != GENERATOR_SEED
        or not base.number(result.get("total_wall_seconds"), 0.000001, 1800)
        or external_total_seconds is not None
        and not base.number(external_total_seconds, 0.000001, 1800)
        or any(
            result.get(k) is not False
            for k in (
                "real_media_opened",
                "reserved_media_opened",
                "checkpoint_written",
                "real_fit_authorized",
                "production_qualified",
            )
        )
        or not isinstance(resources, dict)
        or type(resources.get("swap_bytes")) is not int
        or resources["swap_bytes"] != 0
        or not base.number(resources.get("host_peak_bytes"), 1, 2 * 1024**3)
    ):
        return unavailable
    runs = result.get("runs")
    expected = [(seed, arm) for seed in SPEC["seeds"] for arm in ARMS]
    if not isinstance(runs, list) or len(runs) != 4:
        return unavailable
    outcomes: dict[str, list[bool]] = {arm: [] for arm in ARMS}
    for run, (seed, arm) in zip(runs, expected, strict=True):
        if (
            not isinstance(run, dict)
            or run.get("arm") != arm
            or run.get("domain") != DOMAINS[arm]
            or any(
                type(run.get(k)) is not int or run[k] != n
                for k, n in {
                    "seed": seed,
                    "epochs": 20,
                    "optimizer_updates": 1140,
                    "row_presentations": 72000,
                }.items()
            )
            or not base.number(run.get("parameter_change_norm"), 1e-30, 1e30)
            or any(
                not is_sha(run.get(k))
                for k in ("initial_parameter_sha256", "final_parameter_sha256", "order_sha256")
            )
            or run["initial_parameter_sha256"] == run["final_parameter_sha256"]
            or not all(valid_metrics(run.get(role)) for role in ("train", "development"))
            or any(not is_sha(run[role].get("scores_sha256")) for role in ("train", "development"))
        ):
            return unavailable
        epochs = run.get("epoch_telemetry")
        points = run.get("step_telemetry")
        if (
            not isinstance(epochs, list)
            or len(epochs) != 20
            or not isinstance(points, list)
            or len(points) != 4
        ):
            return unavailable
        for index, epoch in enumerate(epochs, 1):
            if (
                not valid_metrics(epoch)
                or any(
                    type(epoch.get(k)) is not int or epoch[k] != n
                    for k, n in {
                        "epoch": index,
                        "optimizer_updates": 57 * index,
                        "row_presentations": 3600 * index,
                    }.items()
                )
                or any(
                    not base.number(epoch.get(k), 0, 1e30)
                    for k in ("cover_loss", "unweighted_loss", "weighted_objective")
                )
                or any(
                    not base.number(cell.get(k), 0, 1e30)
                    for cell in epoch["cells"]
                    for k in ("stego_loss", "balanced_loss")
                )
            ):
                return unavailable
        for point, number in zip(points, POINTS, strict=True):
            if (
                not isinstance(point, dict)
                or type(point.get("update")) is not int
                or point["update"] != number
                or any(
                    not base.number(point.get(k), 0, 1e30)
                    for k in ("objective", "gradient_norm", "parameter_step_norm")
                )
            ):
                return unavailable
        outcomes[arm].append(
            run["train"]["auc_all"] >= 0.60
            and all(
                c["roc_auc"] >= 0.70
                for c in run["development"]["cells"]
                if c["plane"] == 0 and c["rate_percent"] == 40
            )
        )
    for left, right in ((runs[0], runs[1]), (runs[2], runs[3])):
        if any(left[k] != right[k] for k in ("initial_parameter_sha256", "order_sha256")):
            return unavailable
    return {
        arm: "passed_generated_only" if all(values) else "failed_learning_gate"
        for arm, values in outcomes.items()
    }
