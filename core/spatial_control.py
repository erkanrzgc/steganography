"""One generated spatial profile/control, with frozen sources and scalar evidence."""

from __future__ import annotations

import hashlib
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from core import colab_artifacts as a
from core import spatial_colab as capacity
from core import spatial_colab_state as numeric
from core import spatial_control_io as transport
from core import spatial_generated_training as engine
from core import spatial_training_data as data

ID = "spatial-generated-control-20261028-v1"
CORPUS = "benchmarks/png-bmp-generated-corpus-20261028.json"
DOCUMENT = "docs/PNG_BMP_GENERATED_CONTROL_PROTOCOL_20261028.md"
SECONDS, RESERVE_SECONDS, MAX_OUTPUT = 1800, 20, 512 * a.MIB
SEEDS = (2026102301, 2026102302)
CELLS = tuple((mode, seed) for mode in ("L", "RGB") for seed in SEEDS)
SETTINGS: dict[str, Any] = {
    "seconds": SECONDS,
    "reserve_seconds": RESERVE_SECONDS,
    "host_bytes": 8 * 1024**3,
    "swap_bytes": 0,
    "allocator_bytes": 4 * 1024**3,
    "output_bytes": MAX_OUTPUT,
    "transport_bytes": transport.MAX_SCRATCH_BYTES,
    "batch_images": 8,
    "epochs": 20,
    "train_originals": 80,
    "development_originals": 24,
    "profile_updates": 16,
    "fit_updates": 1600,
    "model_seeds": list(SEEDS),
    "train_auc": 0.85,
    "development_auc": 0.70,
    "final_loss_fraction": 0.80,
    "projection_safety_factor": 1.5,
    "real_fit_authorized": False,
    "automatic_retry": False,
}


def sources(root: Path) -> tuple[str, ...]:
    names = {
        str(p.relative_to(root))
        for folder in ("core", "steganography", "modules", "api", "ui", "report")
        for p in (root / folder).rglob("*.py")
    }
    names.update(
        (
            "registry.py",
            "config.py",
            "cli.py",
            CORPUS,
            DOCUMENT,
            "tests/test_spatial_control.py",
            "tests/test_spatial_control_io.py",
        )
    )
    return tuple(sorted(names))


def checksum(path: Path) -> str:
    a.safe(path)
    return a.digest(a.read(path, 4 * a.MIB) if path.stat().st_size else b"")


def freeze(root: Path, path: Path, environment: dict) -> dict:
    if set(environment) != set(capacity.ENVIRONMENT) - {"runtime"} | {"cudnn", "gpu", "driver"}:
        raise ValueError("explicit complete spatial CUDA environment required")
    value = {
        "schema": "spatial-generated-control-protocol-v1",
        "id": ID,
        "settings": SETTINGS,
        "environment": environment,
        "corpus_sha256": checksum(root / CORPUS),
        "source_sha256": {name: checksum(root / name) for name in sources(root)},
    }
    a.put(path, a.encode(value), 4 * a.MIB)
    return value


def load(root: Path, path: Path) -> tuple[dict, str, dict]:
    raw = a.read(path, 4 * a.MIB)
    value = a.document(path, 4 * a.MIB)
    if (
        set(value) != {"schema", "id", "settings", "environment", "corpus_sha256", "source_sha256"}
        or value["schema"] != "spatial-generated-control-protocol-v1"
        or value["id"] != ID
        or value["settings"] != SETTINGS
        or set(value["source_sha256"]) != set(sources(root))
    ):
        raise ValueError("spatial control protocol mismatch")
    for name, expected in value["source_sha256"].items():
        if checksum(root / name) != expected:
            raise ValueError("spatial control source mismatch")
    if checksum(root / CORPUS) != value["corpus_sha256"]:
        raise ValueError("spatial control corpus mismatch")
    corpus = a.document(root / CORPUS, 4 * a.MIB)
    if corpus["recipe"] != data.RECIPE or corpus["originals"] != 220:
        raise ValueError("spatial generated corpus contract")
    expected = {
        (phase, mode, split): (seed, count)
        for phase, seed, train, dev in (
            ("control", 2026102801, 80, 24),
            ("profile", 2026102802, 4, 2),
        )
        for mode in ("L", "RGB")
        for split, count in (("train", train), ("development", dev))
    }
    found, identities, hashes = {}, set(), set()
    for phase in ("control", "profile"):
        for manifest in corpus[phase]:
            spec = data.GeneratedSpec(**manifest["spec"])
            key = phase, spec.mode, spec.split
            if (
                key in found
                or expected.get(key) != (spec.seed, spec.originals)
                or len(manifest["sources"]) != spec.originals
                or manifest["recipe"] != data.RECIPE
            ):
                raise ValueError("spatial corpus exposure/duplicate specification")
            found[key] = True
            for row in manifest["sources"]:
                if (
                    row["id"] != row["group"]
                    or row["id"] in identities
                    or row["cover_sha256"] in hashes
                ):
                    raise ValueError("spatial corpus group/hash leakage")
                identities.add(row["id"])
                hashes.add(row["cover_sha256"])
    if set(found) != set(expected) or len(identities) != 220:
        raise ValueError("spatial corpus incomplete")
    return value, a.digest(raw), corpus


def environment() -> dict:
    result = capacity.environment()
    run = subprocess.run(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        timeout=5,
        text=True,
    )
    result["driver"] = run.stdout.strip()
    return result


def manifest_for(corpus: dict, phase: str, mode: str, split: str) -> dict:
    return next(
        m for m in corpus[phase] if m["spec"]["mode"] == mode and m["spec"]["split"] == split
    )


def plan(corpus: dict, phase: str, cell: int, binding: str) -> engine.Plan:
    mode, seed = CELLS[cell]
    spec = data.GeneratedSpec(**manifest_for(corpus, phase, mode, "train")["spec"])
    return engine.Plan(spec, seed, 4 if phase == "profile" else 20, 8, binding)


def model_sha(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0" + value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def remaining(expires: float) -> float:
    value = expires - time.monotonic()
    if value <= 0:
        raise TimeoutError("spatial generated control deadline")
    return min(1800.0, value)


def progress(out: Path, state: dict) -> None:
    temporary = out / "progress.next.json"
    a.put(temporary, a.encode(state))
    a.safe(out / "progress.json")
    temporary.replace(out / "progress.json")


def event(out: Path, state: dict, phase: str, cell: int, step: int, result=None) -> None:
    sequence = state["intents"] if result is not None else state["intents"] + 1
    kind = "2-commit" if result is not None else "1-intent"
    value = {
        "sequence": sequence,
        "phase": phase,
        "cell": cell,
        "step": step,
        "previous_sha256": state["last_sha256"],
        "monotonic": time.monotonic(),
    }
    if result is not None:
        value["result"] = result
    state["last_sha256"] = a.put(out / "events" / f"{sequence:06d}-{kind}.json", a.encode(value))
    state["intents" if result is None else "confirmed_updates"] += 1
    progress(out, state)


def projected_seconds(profiles: list[dict]) -> float:
    if len(profiles) != 4 or [r["cell"] for r in profiles] != list(range(4)):
        raise ValueError("complete four-cell spatial profile required")
    total = 0.0
    for row in profiles:
        timings = (
            row["step_seconds"]
            + row["transport_seconds"]
            + [
                row[k]
                for k in ("initialization_seconds", "evaluation_seconds", "checkpoint_seconds")
            ]
        )
        if (
            len(row["step_seconds"]) != 4
            or len(row["transport_seconds"]) != 4
            or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in timings)
        ):
            raise ValueError("spatial profile positive finite intervals required")
        # I/O is a separately measured component. Adding it to the in-memory
        # workload is conservative and is not a native-streaming timing claim.
        total += (
            400 * (max(row["step_seconds"][1:]) + max(row["transport_seconds"]))
            + 208 * row["evaluation_seconds"] / 2
            + row["checkpoint_seconds"]
            + 20 * row["initialization_seconds"]
        )
    return total * float(SETTINGS["projection_safety_factor"])


def checkpoint(experiment, out: Path, phase: str, cell: int, corpus: dict, binding: str) -> dict:
    started = time.monotonic()
    name = f"{phase}-{cell}.numeric"
    checksum_value = experiment.checkpoint(out / name)
    meta, arrays = numeric.load(
        out / name,
        checksum=checksum_value,
        environment=capacity.environment(),
        source_sha256=experiment.binding,
        data_sha256=experiment.data_sha256,
    )
    # Exercise the same-state numeric read/restore path without restarting any fit.
    numeric.restore(experiment.model, experiment.optimizer, meta, arrays)
    return {
        "name": name,
        "sha256": checksum_value,
        "bytes": (out / name).stat().st_size,
        "seconds": time.monotonic() - started,
    }


def worker(root: Path, protocol: Path, out: Path, expires: float) -> None:
    import torch

    contract, binding, corpus = load(root, protocol)
    if environment() != contract["environment"]:
        raise ValueError("spatial CUDA environment mismatch")
    state = {"phase": "transport", "intents": 0, "confirmed_updates": 0, "last_sha256": "0" * 64}
    (out / "events").mkdir()
    progress(out, state)
    for phase in ("control", "profile"):
        for expected in corpus[phase]:
            remaining(expires)
            if data.manifest(data.GeneratedSpec(**expected["spec"])) != expected:
                raise ValueError("spatial regenerated corpus mismatch")
    images = transport.prepare(out / "images", corpus)
    transport_checks = transport.verify(out / "images", images, corpus)
    profiles, fits = [], []
    for cell, (mode, _) in enumerate(CELLS):
        state["phase"] = "profile"
        p = plan(corpus, "profile", cell, binding)
        started = time.monotonic()
        with engine.session(p, device="cuda:0", max_seconds=remaining(expires)) as experiment:
            init = model_sha(experiment.model)
            initialization = time.monotonic() - started
            torch.cuda.reset_peak_memory_stats(0)
            steps, io_times = [], []
            for step in range(1, 5):
                remaining(expires)
                before = time.monotonic()
                transport.batch(
                    out / "images", images, p.spec, (0, 1, 2, 3), "PNG" if step % 2 else "BMP"
                )
                io_times.append(time.monotonic() - before)
                event(out, state, "profile", cell, step)
                before = time.monotonic()
                record = experiment.step()
                steps.append(time.monotonic() - before)
                event(out, state, "profile", cell, step, record)
            before = time.monotonic()
            experiment.scores(
                data.GeneratedSpec(**manifest_for(corpus, "profile", mode, "development")["spec"])
            )
            evaluation = time.monotonic() - before
            saved = checkpoint(experiment, out, "profile", cell, corpus, binding)
            row = {
                "cell": cell,
                "initial_model_sha256": init,
                "step_seconds": steps,
                "transport_seconds": io_times,
                "evaluation_seconds": evaluation,
                "checkpoint_seconds": saved["seconds"],
                "initialization_seconds": initialization,
                "checkpoint": saved,
                "allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
                "reserved_peak_bytes": torch.cuda.max_memory_reserved(0),
            }
        profiles.append(row)
        a.put(out / f"profile-{cell}.json", a.encode(row))
        del experiment
    projection = projected_seconds(profiles)
    a.put(
        out / "profile.json",
        a.encode(
            {
                "cells": profiles,
                "projected_control_seconds": projection,
                "remaining_seconds": remaining(expires),
            }
        ),
    )
    if projection >= remaining(expires):
        a.put(
            out / "worker.json",
            a.encode(
                {
                    "status": "unavailable",
                    "reason": "profile_budget_gate",
                    "profiles": profiles,
                    "confirmed_updates": 16,
                }
            ),
        )
        return
    for cell, (mode, _) in enumerate(CELLS):
        state["phase"] = "control"
        p = plan(corpus, "control", cell, binding)
        dev = data.GeneratedSpec(**manifest_for(corpus, "control", mode, "development")["spec"])
        with engine.session(p, device="cuda:0", max_seconds=remaining(expires)) as experiment:
            init = model_sha(experiment.model)
            initial = {
                "train": experiment.scores(p.spec)["scores"].tolist(),
                "development": experiment.scores(dev)["scores"].tolist(),
            }
            losses = []
            for step in range(1, 401):
                remaining(expires)
                event(out, state, "control", cell, step)
                record = experiment.step()
                losses.append(record["loss"])
                event(out, state, "control", cell, step, record)
            final = {
                "train": experiment.scores(p.spec)["scores"].tolist(),
                "development": experiment.scores(dev)["scores"].tolist(),
            }
            saved = checkpoint(experiment, out, "control", cell, corpus, binding)
            row = {
                "cell": cell,
                "initial_model_sha256": init,
                "initial": initial,
                "final": final,
                "losses": losses,
                "checkpoint": saved,
                "optimizer_updates": experiment.updates,
                "presentations": experiment.presentations,
                "allocated_peak_bytes": torch.cuda.max_memory_allocated(0),
                "reserved_peak_bytes": torch.cuda.max_memory_reserved(0),
            }
        fits.append(row)
        a.put(out / f"control-{cell}.json", a.encode(row))
        del experiment
    remaining(expires)
    load(root, protocol)
    a.put(
        out / "worker.json",
        a.encode(
            {
                "status": "completed_generated_control",
                "protocol_sha256": binding,
                "profiles": profiles,
                "fits": fits,
                "transport_checks": transport_checks,
                "generated_only": True,
                "native_media_opened": False,
                "confirmed_updates": state["confirmed_updates"],
            }
        ),
    )
    state["phase"] = "completed"
    progress(out, state)


def ledger(out: Path, *, strict: bool = True) -> dict:
    previous, updates, intents = "0" * 64, 0, 0
    valid = True
    losses: dict[str, list[float]] = {}
    try:
        for path in sorted((out / "events").glob("*")):
            row = a.document(path)
            sequence = updates + 1
            phase = "profile" if sequence <= 16 else "control"
            offset, per_cell = (sequence - 1, 4) if phase == "profile" else (sequence - 17, 400)
            commit = "result" in row
            expected = f"{sequence:06d}-{'2-commit' if commit else '1-intent'}.json"
            if (
                path.name != expected
                or row["previous_sha256"] != previous
                or row["sequence"] != sequence
                or row["phase"] != phase
                or row["cell"] != offset // per_cell
                or row["step"] != offset % per_cell + 1
                or sequence > 1616
                or (commit and intents != updates + 1)
                or (not commit and intents != updates)
            ):
                raise ValueError("spatial immutable update ledger mismatch")
            if commit:
                r = row["result"]
                if (
                    r["optimizer_updates"] != offset % per_cell + 1
                    or r["presentations"] != 8 * (offset % per_cell + 1)
                    or not math.isfinite(r["loss"])
                    or r["loss"] < 0
                ):
                    raise ValueError("spatial committed update evidence mismatch")
                updates += 1
                losses.setdefault(f"{phase}-{row['cell']}", []).append(r["loss"])
            else:
                intents += 1
            previous = checksum(path)
    except (OSError, ValueError, KeyError, TypeError):
        if strict:
            raise
        valid = False
    return {
        "confirmed_updates_lower_bound": updates,
        "step_intents": intents,
        "uncertain_final_step": intents > updates,
        "last_sha256": previous,
        "ledger_valid": valid,
        "cell_losses": losses,
    }


def checkpoint_proof(out: Path, row: dict, p: engine.Plan, corpus: dict, phase: str) -> None:
    spec_manifest = manifest_for(corpus, phase, p.spec.mode, "train")
    binding = a.digest(
        a.encode(
            {
                "plan": asdict(p),
                "device": "cuda:0",
                "recipe": data.RECIPE,
                "optimizer": "scalar-adamw-1e-4-v1",
            }
        )
    )
    name = row["checkpoint"]["name"]
    if name != f"{phase}-{row['cell']}.numeric":
        raise ValueError("spatial checkpoint name")
    meta, arrays = numeric.load(
        out / name,
        checksum=row["checkpoint"]["sha256"],
        environment=capacity.environment(),
        source_sha256=binding,
        data_sha256=a.digest(a.encode(spec_manifest)),
    )
    updates = 4 if phase == "profile" else 400
    if (
        not np.array_equal(arrays["data/order"], engine.order(p))
        or meta["optimizer"]
        != {
            "lr": 0.0001,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        }
        or meta["domain"] != p.spec.domain
        or meta["training"] is not True
        or not 0 < meta["consumption"]["elapsed_seconds"] < SECONDS
        or meta["consumption"]["epoch"] != p.epochs
        or meta["consumption"]["optimizer_updates"] != updates
        or meta["consumption"]["presentations"] != updates * 8
        or meta["consumption"]["cursor"] != len(arrays["data/order"])
        or (out / name).stat().st_size != row["checkpoint"]["bytes"]
        or not (row["checkpoint"]["bytes"] <= meta["consumption"]["output_bytes"] <= MAX_OUTPUT)
    ):
        raise ValueError("spatial final checkpoint evidence mismatch")


def decide(root: Path, protocol: Path, out: Path, result: dict, resources: dict) -> dict:
    import torch

    from core.jpeg_study import auc
    from core.wav_control_runner import auc as independent_auc

    contract, binding, corpus = load(root, protocol)
    if environment() != contract["environment"]:
        raise ValueError("spatial closing environment mismatch")
    log = ledger(out)
    base = {
        "status": "unavailable",
        "generated_only": True,
        "real_fit_authorized": False,
        "protocol_sha256": binding,
        **log,
        "resources": resources,
    }
    if (
        result.get("status") != "completed_generated_control"
        or result.get("protocol_sha256") != binding
        or log["confirmed_updates_lower_bound"] != 1616
        or log["uncertain_final_step"]
        or result.get("native_media_opened") is not False
        or not math.isfinite(resources["wall_seconds"])
        or not 0 < resources["wall_seconds"] <= SECONDS
        or not 0 < resources["output_bytes"] <= MAX_OUTPUT
        or resources["memory_peak_bytes"] > SETTINGS["host_bytes"]
        or resources["memory_peak_bytes"] <= 0
        or not 0 < resources["memory_max_bytes"] <= SETTINGS["host_bytes"]
        or resources["memory_swap_max_bytes"] != 0
        or resources["swap_peak_bytes"] != 0
    ):
        return base
    profiles, fits = result["profiles"], result["fits"]
    projected_seconds(profiles)
    if len(fits) != 4 or [r["cell"] for r in fits] != list(range(4)):
        return base
    checks, learning = [], True
    for phase, rows in (("profile", profiles), ("control", fits)):
        for row in rows:
            p = plan(corpus, phase, row["cell"], binding)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(p.model_seed)
                model = capacity.models.network(p.spec.domain)
                if model_sha(model) != row["initial_model_sha256"]:
                    raise ValueError("spatial fresh initialization mismatch")
                del model
            if any(
                type(row[k]) is not int or not 0 < row[k] <= SETTINGS["allocator_bytes"]
                for k in ("allocated_peak_bytes", "reserved_peak_bytes")
            ):
                return base
            checkpoint_proof(out, row, p, corpus, phase)
            if phase == "profile":
                continue
            losses = row["losses"]
            if len(losses) != 400 or any(not math.isfinite(v) or v < 0 for v in losses):
                return base
            if losses != log["cell_losses"].get(f"control-{row['cell']}"):
                raise ValueError("spatial losses do not match committed optimizer events")
            if row["optimizer_updates"] != 400 or row["presentations"] != 3200:
                return base
            scores = {}
            for stage in ("initial", "final"):
                for split, count in (("train", 80), ("development", 24)):
                    values = np.asarray(row[stage][split], dtype=np.float64)
                    if (
                        values.shape != (count, 2)
                        or not np.isfinite(values).all()
                        or np.any((values < 0) | (values > 1))
                    ):
                        raise ValueError("spatial bounded complete score arrays required")
                    score = auc(values)
                    if abs(score - independent_auc(values[:, 0], values[:, 1])) > 1e-12:
                        raise ValueError("spatial independent AUC mismatch")
                    scores[stage + "_" + split] = score
            first, last = math.fsum(losses[:20]) / 20, math.fsum(losses[-20:]) / 20
            passed = (
                scores["final_train"] >= 0.85
                and scores["final_development"] >= 0.70
                and last <= 0.8 * first
            )
            learning = learning and passed
            checks.append(
                {
                    "cell": row["cell"],
                    **scores,
                    "first_epoch_loss": first,
                    "last_epoch_loss": last,
                    "passed": passed,
                }
            )
    actual_transport = transport.verify(
        out / "images", a.document(out / "images/manifest.json"), corpus
    )
    if actual_transport != result["transport_checks"]:
        raise ValueError("spatial final transport evidence mismatch")
    return {
        **base,
        "status": "passed_generated_only" if learning else "failed_learning_gate",
        "cells": checks,
        "independent_score_replays": 16,
        "transport_checks": actual_transport,
        "checkpoints_verified": 8,
    }
