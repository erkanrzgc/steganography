"""Separately frozen RGB pilot: train-only measurement and approval-bound fits."""

from __future__ import annotations

import gc
import hashlib
import io
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core import colab_artifacts as a
from core import spatial_control as historical
from core import spatial_native_data as data
from core import spatial_native_training as engine

ID = "spatial-rgb-native-pilot-20261028-v1"
PROPOSAL = "benchmarks/png-bmp-rgb-pilot-proposal-20261028.json"
DOCUMENT = "docs/PNG_BMP_RGB_PILOT_PROTOCOL_20261028.md"
ACTOR = ".benchmark/png-bmp-rgb-pilot-20261028/launch_phase.py"
DEPLOYMENT = ".benchmark/png-bmp-rgb-pilot-20261028/deployment.json"
SEEDS = (2026102301, 2026102302)
PROFILE_SECONDS = 240
PROFILE_OUTPUT = 256 * a.MIB
FIT_SECONDS_CEILING = 1800
CUMULATIVE_OUTPUT_CEILING = 768 * a.MIB
SETTINGS: dict[str, Any] = {
    "profile_seconds": PROFILE_SECONDS,
    "profile_output_bytes": PROFILE_OUTPUT,
    "host_bytes": 8 * 1024**3,
    "swap_bytes": 0,
    "allocator_bytes": 4 * 1024**3,
    "native_optimizer_updates": 0,
    "generated_profile_updates": 6,
    "profile_passes_per_seed": 3,
    "counts": [64, 20, 21],
    "mode": "RGB",
    "input_shape": [256, 256, 3],
    "precision": "float32-ieee",
    "mixed_precision": False,
    "tf32": False,
    "epochs": 20,
    "batch_images": 8,
    "model_seeds": list(SEEDS),
    "payload_seed": 2026102805,
    "native_profile_payload_seed": 2026102806,
    "generated_pixel_seed": 2026102807,
    "generated_payload_seed": 2026102808,
    "fit_updates": 640,
    "fit_presentations": 5120,
    "projection_factor": 1.5,
    "fit_requires_later_bound_approval": True,
    "automatic_retry": False,
}


def checksum(path: Path) -> str:
    a.safe(path)
    return a.digest(a.read(path, 4 * a.MIB) if path.stat().st_size else b"")


def source_names(root: Path) -> tuple[str, ...]:
    names = {
        str(p.relative_to(root))
        for folder in ("core", "steganography", "modules", "api", "ui", "report")
        for p in (root / folder).rglob("*.py")
    }
    names.update(
        (
            "cli.py",
            "config.py",
            "registry.py",
            PROPOSAL,
            DOCUMENT,
            ACTOR,
            DEPLOYMENT,
            "tests/test_spatial_native_training.py",
            "tests/test_spatial_native_pilot.py",
        )
    )
    return tuple(sorted(names))


def freeze(root: Path, out: Path, environment: dict) -> dict:
    proposal = a.document(root / PROPOSAL)
    split = engine.split_document(proposal["split"])
    if (
        proposal["status"] != "proposal_only"
        or split.mode != "RGB"
        or tuple(map(len, (split.train, split.development, split.future_reserved))) != (64, 20, 21)
        or proposal["models"]["total_fit_updates"] != 640
    ):
        raise ValueError("RGB pilot proposal mismatch")
    if set(environment) != set(engine.capacity.environment()) | {"driver"}:
        raise ValueError("complete explicit WSL environment required")
    value = {
        "schema": "spatial-rgb-native-pilot-protocol-v1",
        "id": ID,
        "settings": SETTINGS,
        "environment": environment,
        "source_sha256": {n: checksum(root / n) for n in source_names(root)},
        "split": asdict(split),
        "proposal_sha256": checksum(root / PROPOSAL),
    }
    a.put(out, a.encode(value), 4 * a.MIB)
    return value


def verify(root: Path, protocol: Path, cache: data.Cache) -> tuple[dict, str, engine.Split]:
    value = a.document(protocol, 4 * a.MIB)
    if (
        set(value)
        != {"schema", "id", "settings", "environment", "source_sha256", "split", "proposal_sha256"}
        or value["schema"] != "spatial-rgb-native-pilot-protocol-v1"
        or value["id"] != ID
        or value["settings"] != SETTINGS
        or set(value["source_sha256"]) != set(source_names(root))
    ):
        raise ValueError("native pilot protocol mismatch")
    if any(checksum(root / n) != sha for n, sha in value["source_sha256"].items()):
        raise ValueError("native pilot source binding")
    if checksum(root / PROPOSAL) != value["proposal_sha256"]:
        raise ValueError("native pilot proposal hash")
    if value["split"] != a.document(root / PROPOSAL)["split"]:
        raise ValueError("native pilot proposed split mismatch")
    split = engine.split_document(value["split"])
    split.validate(cache)
    return value, checksum(protocol), split


def sync(device: str) -> None:
    if device == "cuda:0":
        import torch

        torch.cuda.synchronize()


def memory(device: str) -> dict:
    if device != "cuda:0":
        return {"allocated_bytes": None, "reserved_bytes": None}
    import torch

    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def release(device: str) -> None:
    gc.collect()
    if device == "cuda:0":
        import torch

        torch.cuda.empty_cache()


def parameters(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        h.update(name.encode() + b"\0" + value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def generated_cache(out: Path, mode: str, pairs: int) -> tuple[data.Cache, engine.Split]:
    inputs = out / "generated-inputs"
    a.directory(inputs)
    refs = []
    rng = np.random.Generator(np.random.PCG64(SETTINGS["generated_pixel_seed"]))
    for i in range(pairs + 2):
        channels = 1 if mode == "L" else 3
        pixels = rng.integers(0, 128, (256, 256, channels), dtype="u1") * np.uint8(2)
        name = f"generated-profile-{i}"
        buffer = io.BytesIO()
        Image.fromarray(pixels[:, :, 0] if mode == "L" else pixels).save(buffer, format="PNG")
        raw = buffer.getvalue()
        path = inputs / (name + "-cover.png")
        a.put(path, raw, data.admission.MAX_FILE)
        source = data.Source(
            name,
            mode,
            name,
            "Generated profile fixture " + str(i),
            "https://commons.wikimedia.org/wiki/File:Generated_profile_fixture_" + str(i) + ".png",
            a.digest(raw),
            a.digest(pixels.tobytes()),
            "CC0-1.0",
            data.native.LICENSES["CC0-1.0"],
            a.digest(name.encode()),
        )
        refs.append(data.Input(source, path, a.digest(raw)))
    folder = out / "generated-cache"
    data.prepare(tuple(refs), folder)
    cache = data.load(folder, checksum(folder / "cache.json"))
    ids = tuple(row.id for row in cache.rows)
    return cache, engine.Split(
        cache.manifest_sha256, mode, 2026102807, ids[:pairs], (ids[pairs],), (ids[pairs + 1],), ()
    )


def remaining(deadline: float) -> float:
    result = deadline - time.monotonic()
    if result <= 0:
        raise TimeoutError("native pilot deadline")
    return min(1800.0, result)


def record_event(out: Path, sequence: int, kind: str, value: dict) -> None:
    a.put(out / f"{sequence:03d}-{kind}.json", a.encode(value))


def measure(
    cache: data.Cache,
    split: engine.Split,
    out: Path,
    binding: str,
    *,
    device: str = "cuda:0",
    seconds: float = 220,
) -> dict:
    """CPU execution is software QA only; the physical runner requires CUDA."""
    import torch

    split.validate(cache)
    a.identity(binding)
    if (
        device not in ("cpu", "cuda:0")
        or type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 220
    ):
        raise ValueError("native measurement device/deadline")
    a.directory(out)
    started = time.monotonic()
    deadline = started + seconds
    if device == "cuda:0":
        torch.cuda.reset_peak_memory_stats()
    pairs = next(n for n in (4, 2, 1) if n <= len(split.train))
    fake, fake_split = generated_cache(out, split.mode, pairs)
    cells = []
    confirmed = 0
    for cell, seed in enumerate(SEEDS):
        release(device)
        begin = time.monotonic()
        native_plan = engine.Plan(
            split, seed, SETTINGS["native_profile_payload_seed"], 1, 2 * pairs, binding
        )
        with engine.session(
            cache, native_plan, device=device, max_seconds=remaining(deadline)
        ) as s:
            initialized = time.monotonic() - begin
            before = parameters(s.model)
            passes = []
            for _ in range(3):
                s.check()
                sync(device)
                begin = time.monotonic()
                x, labels = data.batch(
                    cache,
                    split.mode,
                    tuple(s.lookup[i] for i in split.train[:pairs]),
                    native_plan.payload_seed,
                )
                values = torch.from_numpy(x).to(device)
                targets = torch.from_numpy(labels).to(device)
                s.optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(s.model(values), targets)
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(
                    p.grad is None or not bool(torch.isfinite(p.grad).all())
                    for p in s.model.parameters()
                ):
                    raise ValueError("native profile nonfinite gradients")
                sync(device)
                passes.append(time.monotonic() - begin)
                s.check()
            evaluation = s.scores("train", record_scores=False)
            if before != parameters(s.model) or s.updates or s.optimizer.state:
                raise ValueError("native profile changed parameters/optimizer")
            native_memory = memory(device)
        del s, values, targets, x, labels, loss
        release(device)
        plan = engine.Plan(
            fake_split, seed, SETTINGS["generated_payload_seed"], 3, 2 * pairs, binding
        )
        begin = time.monotonic()
        with engine.session(fake, plan, device=device, max_seconds=remaining(deadline)) as s:
            generated_initialization = time.monotonic() - begin
            steps = []
            for step in range(3):
                record_event(
                    out,
                    confirmed + 1,
                    "intent",
                    {"cell": cell, "step": step, "generated_only": True},
                )
                sync(device)
                begin = time.monotonic()
                result = s.step()
                sync(device)
                steps.append(time.monotonic() - begin)
                confirmed += 1
                record_event(
                    out,
                    confirmed,
                    "commit",
                    {"cell": cell, "step": step, "result": result, "generated_only": True},
                )
            expected = historical.model_sha(s.model)
            begin = time.monotonic()
            path = out / f"generated-{cell}.numeric"
            sha = s.checkpoint(path)
            write_seconds = time.monotonic() - begin
            generated_memory = memory(device)
        del s
        release(device)
        begin = time.monotonic()
        with engine.session(fake, plan, device=device, max_seconds=remaining(deadline)) as restored:
            restored.resume(path, sha)
            if restored.updates != 3 or historical.model_sha(restored.model) != expected:
                raise ValueError("generated profile restore mismatch")
        restore_seconds = time.monotonic() - begin
        del restored
        cells.append(
            {
                "seed": seed,
                "native_initialization_seconds": initialized,
                "native_forward_backward_seconds": passes,
                "native_train_evaluation": evaluation,
                "native_parameters_unchanged": True,
                "native_optimizer_updates": 0,
                "generated_initialization_seconds": generated_initialization,
                "generated_step_seconds": steps,
                "generated_optimizer_updates": 3,
                "checkpoint_write_seconds": write_seconds,
                "checkpoint_restore_seconds": restore_seconds,
                "checkpoint_sha256": sha,
                "checkpoint_bytes": path.stat().st_size,
                "native_memory": native_memory,
                "generated_memory": generated_memory,
            }
        )
        release(device)
        remaining(deadline)
    result = {
        "schema": "spatial-rgb-native-measurement-v1",
        "status": "measured",
        "device": device,
        "physical_cuda": device == "cuda:0",
        "binding": binding,
        "manifest_sha256": cache.manifest_sha256,
        "split": asdict(split),
        "cells": cells,
        "native_optimizer_updates": 0,
        "generated_optimizer_updates": confirmed,
        "native_backward_presentations": len(SEEDS) * 3 * pairs * 2,
        "native_evaluation_presentations": len(SEEDS) * len(split.train) * 2,
        "generated_fixture_only_cache": fake.manifest_sha256,
        "worker_seconds": time.monotonic() - started,
        "cuda_peak": memory(device),
    }
    a.put(out / "measurement.json", a.encode(result), 4 * a.MIB)
    return result


def projection(result: dict) -> dict:
    if (
        result.get("schema") != "spatial-rgb-native-measurement-v1"
        or [cell["seed"] for cell in result["cells"]] != list(SEEDS)
        or result["generated_optimizer_updates"] != 6
        or result["native_optimizer_updates"] != 0
    ):
        raise ValueError("complete two-seed native profile required")
    total = 0.0
    for row in result["cells"]:
        timings = (
            row["native_forward_backward_seconds"]
            + row["generated_step_seconds"]
            + [
                row["native_train_evaluation"]["max_pair_seconds"],
                row["native_initialization_seconds"],
                row["generated_initialization_seconds"],
                row["checkpoint_write_seconds"],
                row["checkpoint_restore_seconds"],
            ]
        )
        if (
            len(row["native_forward_backward_seconds"]) != 3
            or len(row["generated_step_seconds"]) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in timings)
            or row["native_optimizer_updates"] != 0
            or row["generated_optimizer_updates"] != 3
            or not row["native_parameters_unchanged"]
        ):
            raise ValueError("native profile timing/update evidence")
        total += 320 * (
            max(row["native_forward_backward_seconds"][1:]) + max(row["generated_step_seconds"][1:])
        )
        total += 168 * row["native_train_evaluation"]["max_pair_seconds"]
        total += 4 * (row["checkpoint_write_seconds"] + row["checkpoint_restore_seconds"])
        total += row["native_initialization_seconds"] + row["generated_initialization_seconds"] + 20
    return {
        "fit_projection_seconds": 1.5 * total,
        "factor": 1.5,
        "native_step_estimate": (
            "native I/O+forward/backward plus a complete generated "
            "optimizer step; conservative double charge"
        ),
        "candidate_max_cumulative_seconds": 1800,
        "candidate_max_cumulative_output_bytes": 768 * a.MIB,
        "fit_updates": 640,
        "fit_presentations": 5120,
        "native_accuracy_inferred": False,
        "guaranteed_upper_bound": False,
        "approval_required": True,
    }


def approved_limits(
    approval: dict, profile: dict, profile_sha: str, binding: str, split: engine.Split
) -> tuple[float, int]:
    expected = {
        "schema",
        "user_approved",
        "protocol_sha256",
        "profile_sha256",
        "split_sha256",
        "max_cumulative_seconds",
        "max_cumulative_output_bytes",
    }
    if (
        not isinstance(approval, dict)
        or set(approval) != expected
        or approval["schema"] != "spatial-rgb-native-approval-v1"
        or approval["user_approved"] is not True
        or approval["protocol_sha256"] != binding
        or approval["profile_sha256"] != profile_sha
        or approval["split_sha256"] != a.digest(a.encode(asdict(split)))
    ):
        raise ValueError("explicit profile-bound native approval required")
    seconds = approval["max_cumulative_seconds"]
    output = approval["max_cumulative_output_bytes"]
    if (
        profile.get("status") != "profile_ready"
        or profile.get("protocol_sha256") != binding
        or profile.get("physical_cuda") is not True
        or type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 1800
        or type(output) is not int
        or not 0 < output <= CUMULATIVE_OUTPUT_CEILING
        or profile["charged_seconds"] + profile["projection"]["fit_projection_seconds"] >= seconds
        or profile["logical_output_bytes"] >= output
        or type(profile.get("max_checkpoint_bytes")) is not int
        or not 0 < profile["max_checkpoint_bytes"] <= 129 * a.MIB
        or profile["logical_output_bytes"] + 8 * profile["max_checkpoint_bytes"] + 8 * a.MIB
        > output
    ):
        raise ValueError("native profile/budget is ineligible")
    return float(seconds - profile["charged_seconds"]), output - profile["logical_output_bytes"]


def fit(
    root: Path, protocol: Path, cache: data.Cache, out: Path, approval: dict, profile_path: Path
) -> dict:
    """Only a later explicit approval may reach actual native optimizer steps."""
    _, binding, split = verify(root, protocol, cache)
    profile = a.document(profile_path, 4 * a.MIB)
    seconds, byte_limit = approved_limits(approval, profile, checksum(profile_path), binding, split)
    a.directory(out)
    deadline = time.monotonic() + seconds - 20
    if remaining(deadline) <= 0:
        raise TimeoutError("native fit closing reserve")
    commits = 0
    cells = []
    for cell, seed in enumerate(SEEDS):
        plan = engine.Plan(split, seed, SETTINGS["payload_seed"], 20, 8, binding)
        losses = []
        checkpoints = []
        with engine.session(cache, plan, device="cuda:0", max_seconds=remaining(deadline)) as s:
            initial = {"train": s.scores("train"), "development": s.scores("development")}
            initial_model = historical.model_sha(s.model)
            for epoch in range(20):
                for step in range(16):
                    sequence = commits + 1
                    record_event(
                        out,
                        sequence,
                        "intent",
                        {"cell": cell, "epoch": epoch, "step": step, "native": True},
                    )
                    result = s.step()
                    commits += 1
                    losses.append(result["loss"])
                    record_event(
                        out,
                        commits,
                        "commit",
                        {
                            "cell": cell,
                            "epoch": epoch,
                            "step": step,
                            "native": True,
                            "result": result,
                        },
                    )
                    historical.progress(
                        out,
                        {
                            "confirmed_fit_updates": commits,
                            "cell": cell,
                            "epoch": epoch,
                            "cursor": s.cursor,
                            "loss": result["loss"],
                        },
                    )
                    remaining(deadline)
                    if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > byte_limit:
                        raise ValueError("native cumulative output allowance")
                if (epoch + 1) % 5 == 0:
                    used = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
                    if used + profile["max_checkpoint_bytes"] + 65536 > byte_limit:
                        raise ValueError("native checkpoint reservation exceeds allowance")
                    path = out / f"seed-{seed}-epoch-{epoch + 1:02d}.numeric"
                    sha = s.checkpoint(path)
                    checkpoints.append(
                        {
                            "name": path.name,
                            "sha256": sha,
                            "bytes": path.stat().st_size,
                            "optimizer_updates": s.updates,
                        }
                    )
                    if (
                        path.stat().st_size > profile["max_checkpoint_bytes"]
                        or sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > byte_limit
                    ):
                        raise ValueError("native cumulative checkpoint allowance")
            final = {"train": s.scores("train"), "development": s.scores("development")}
            if s.updates != 320 or s.presentations != 2560 or s.cursor != 1280:
                raise ValueError("native complete exposure mismatch")
            first = float(np.mean(losses[:16]))
            last = float(np.mean(losses[-16:]))
            passed = (
                final["train"]["auc"] >= 0.85
                and final["development"]["auc"] >= 0.70
                and last <= 0.80 * first
            )

            def serial_scores(values):
                return {role: {**v, "scores": v["scores"].tolist()} for role, v in values.items()}

            cells.append(
                {
                    "seed": seed,
                    "initial_model_sha256": initial_model,
                    "losses": losses,
                    "initial": serial_scores(initial),
                    "final": serial_scores(final),
                    "first_epoch_loss": first,
                    "last_epoch_loss": last,
                    "passed": passed,
                    "optimizer_updates": s.updates,
                    "presentations": s.presentations,
                    "checkpoints": checkpoints,
                }
            )
        del s
        release("cuda:0")
        a.put(out / f"cell-{cell}.json", a.encode(cells[-1]), 4 * a.MIB)
    result = {
        "schema": "spatial-rgb-native-fit-v1",
        "status": "passed_exploratory_native_gate"
        if all(c["passed"] for c in cells)
        else "failed_learning_gate",
        "protocol_sha256": binding,
        "profile_sha256": checksum(profile_path),
        "split": asdict(split),
        "fit_updates": commits,
        "presentations": 5120,
        "cells": cells,
        "qualified": False,
        "deployment": False,
        "future_reserved_opened": False,
    }
    remaining(deadline)
    a.put(out / "fit.json", a.encode(result), 4 * a.MIB)
    return result
