"""Colab native pilot preparation; execution requires a measured, approved contract."""

from __future__ import annotations

import math
import os
import random
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from core import colab_artifacts as a
from core import spatial_colab_profile as profile
from core import spatial_colab_state as numeric
from core import spatial_native_data as data
from core import spatial_native_training as engine

if TYPE_CHECKING:
    from core.colab_execution import Limits

SCHEMA = "colab-rgb-fit-data-v1"
RECIPE: dict[str, Any] = {
    "epochs": 20,
    "batch_images": 8,
    "seeds": list(profile.pilot.SEEDS),
    "payload_seed": 2026102805,
    "checkpoint_epochs": [5, 10, 15, 20],
    "development_every_epoch": True,
    "fit_updates": 640,
    "presentations": 5120,
    "train_auc": 0.85,
    "development_auc": 0.70,
    "last_loss_fraction": 0.8,
    "precision": "float32-ieee",
    "amp": False,
    "tf32": False,
}
ENV_KEYS = {"python", "numpy", "pillow", "torch", "cuda", "cudnn", "gpu"}


def finite(value, *, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError("finite nonnegative debit required")
    return float(value)


def contract(payload: dict, limits) -> engine.Split:
    from core import spatial_colab_budget as budget
    from core.colab_execution import Limits

    amended = payload.get("schema") == budget.SCHEMA
    if (
        set(payload)
        != {
            "schema",
            "split",
            "recipe",
            "environment",
            "profile_contract_sha256",
            "profile_receipt_sha256",
            "prior_seconds",
            "prior_output_bytes",
            "projection_seconds",
            "max_checkpoint_bytes",
            "setup_debit",
        } | ({"resource_amendment"} if amended else set())
        or payload["schema"] not in (SCHEMA, budget.SCHEMA)
        or payload["recipe"] != RECIPE
    ):
        raise ValueError("Colab fit contract fields/recipe")
    env = payload["environment"]
    if (
        set(env) != profile.ENVIRONMENT_KEYS | {"cudnn"}
        or type(env["cudnn"]) is not int
        or env["cudnn"] <= 0
    ):
        raise ValueError("complete measured fit environment required")
    split = profile.contract(
        {
            "schema": profile.PROFILE_SCHEMA,
            "split": payload["split"],
            "recipe": profile.RECIPE,
            "expected_environment": {k: env[k] for k in profile.ENVIRONMENT_KEYS},
        },
        Limits(240, 0, 0, 256 * a.MIB, 128 * a.MIB, closing_seconds=10),
    )
    for key in ("profile_contract_sha256", "profile_receipt_sha256"):
        a.identity(payload[key])
    prior = finite(payload["prior_seconds"], positive=True)
    projected = finite(payload["projection_seconds"], positive=True)
    used, checkpoint = payload["prior_output_bytes"], payload["max_checkpoint_bytes"]
    if (
        type(used) is not int
        or not 0 < used < 768 * a.MIB
        or type(checkpoint) is not int
        or not 0 < checkpoint <= 128 * a.MIB
        or limits.seconds != (budget.SECONDS if amended else 1800) - prior
        or limits.closing_seconds != 20
        or limits.output_bytes != 768 * a.MIB - used
        or limits.data_bytes
        or limits.resident_watch_bytes
        or limits.address_bytes
        or limits.file_bytes != 128 * a.MIB
        or projected + 30 >= limits.seconds
        or 8 * checkpoint + 16 * a.MIB >= limits.output_bytes
    ):
        raise ValueError("measured Colab fit budget ineligible")
    setup_debit(payload["setup_debit"])
    if amended:
        budget.validate(payload)
    return split


def setup_debit(value: dict) -> None:
    if (
        set(value) != {"schema", "seconds", "output_bytes", "complete"}
        or value["schema"] != "colab-fit-external-debit-v1"
        or value["complete"] is not True
        or type(value["output_bytes"]) is not int
        or value["output_bytes"] < 0
    ):
        raise ValueError("explicit complete setup/review debit required")
    finite(value["seconds"])


def accepted_profile(root: Path, out: Path) -> tuple[dict, str]:
    from core import colab_execution as controller

    result = controller.audit(root, out / "contract.json", out)
    if (
        result["status"] != "completed"
        or result["kind"] != "native_profile"
        or result["worker"]["physical_cuda"] is not True
        or result["worker"]["native_optimizer_updates"] != 0
        or result["worker"]["generated_optimizer_updates"] != 6
        or result["worker"]["recipe"] != profile.RECIPE
    ):
        raise ValueError("accepted physical Colab profile required")
    return result, a.digest(a.read(out / "closed.json", 60000))


def proposal(root: Path, profile_out: Path, debit: dict, *, amended=False) -> tuple[dict, Limits]:
    """Source-bound proposal only; never create fitting permission."""
    from core import spatial_colab_budget as budget
    from core.colab_execution import Limits

    setup_debit(debit)
    debit = dict(debit)
    result, checksum = (budget.accepted_profile if amended else accepted_profile)(root, profile_out)
    worker = result["worker"]
    # The original estimate has initial/final evaluation only. Add epoch-dev I/O/inference.
    projected = profile.pilot.projection({**worker, "schema": "spatial-rgb-native-measurement-v1"})
    extra_pairs = (RECIPE["epochs"] - 1) * len(worker["split"]["development"])
    projection = projected["fit_projection_seconds"] + 1.5 * extra_pairs * sum(
        row["native_train_evaluation"]["max_pair_seconds"] for row in worker["cells"]
    )
    prior = finite(result["charged_seconds"], positive=True) + debit["seconds"]
    used = sum(v["bytes"] for v in result["output_files"].values())
    used += (len(a.encode(result)) if amended else (profile_out / "closed.json").stat().st_size)
    used += debit["output_bytes"]
    maximum = max(row["checkpoint_bytes"] for row in worker["cells"]) + a.MIB
    payload = {
        "schema": budget.SCHEMA if amended else SCHEMA,
        "split": worker["split"],
        "recipe": RECIPE,
        "environment": worker["environment"],
        "profile_contract_sha256": result["contract_sha256"],
        "profile_receipt_sha256": checksum,
        "prior_seconds": prior,
        "prior_output_bytes": used,
        "projection_seconds": projection,
        "max_checkpoint_bytes": maximum,
        "setup_debit": debit,
    }
    if amended:
        payload["resource_amendment"] = budget.AMENDMENT
    limits = Limits((budget.SECONDS if amended else 1800) - prior, 0, 0,
                    768 * a.MIB - used, 128 * a.MIB, closing_seconds=20)
    contract(payload, limits)
    return payload, limits


def freeze(root: Path, profile_out: Path, debit: dict, target: Path, *, amended=False) -> dict:
    from core import colab_execution as controller

    payload, limits = proposal(root, profile_out, debit, amended=amended)
    return controller.freeze(
        root,
        target,
        "native_fit",
        payload,
        limits,
        target_environment={"python": "3.12.13", "implementation": "CPython", "platform": "linux"},
    )


def approval(job: dict, binding: str, path: Path | None) -> None:
    from core import spatial_colab_budget as budget

    value = a.document(path) if path is not None else None
    amended = job["payload"]["schema"] == budget.SCHEMA
    extra = ({"resource_amendment": budget.AMENDMENT, "cumulative_seconds": budget.SECONDS}
             if amended else {})
    if (
        value is None
        or value.get("user_approved") is not True
        or value
        != {
            "schema": "colab-fit-permission-v2" if amended else "colab-fit-permission-v1",
            "user_approved": True,
            "kind": "native_fit",
            "contract_sha256": binding,
            "profile_receipt_sha256": job["payload"]["profile_receipt_sha256"],
            "resource_acknowledgement": profile.RESOURCE_ACK,
        } | extra
    ):
        raise ValueError("separate explicit profile-bound fitting permission required")


def bind_profile(root: Path, job: dict, out: Path) -> None:
    from core import spatial_colab_budget as budget

    expected, _ = proposal(root, out, job["payload"]["setup_debit"],
                           amended=job["payload"]["schema"] == budget.SCHEMA)
    if expected != job["payload"]:
        raise ValueError("fitting profile/debit binding changed")


def auc(values) -> float:
    """Bounded average-rank ROC AUC; audit uses an independent pairwise calculation."""
    pairs = np.asarray(values, dtype=np.float64)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not 1 <= len(pairs) <= 128
        or not np.isfinite(pairs).all()
        or np.any((pairs < 0) | (pairs > 1))
    ):
        raise ValueError("bounded finite paired scores required")
    scores = pairs.ravel()
    ordered = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[ordered[stop]] == scores[ordered[start]]:
            stop += 1
        ranks[ordered[start:stop]] = (start + 1 + stop) / 2
        start = stop
    n = len(pairs)
    return float((ranks[1::2].sum() - n * (n + 1) / 2) / (n * n))


class _Session(engine._Session):
    def scores(self, role, *, record_scores=True):
        import torch

        if role not in ("train", "development") or record_scores is not True:
            raise ValueError("fit scoring is restricted to train/development")
        previous, values = self.model.training, []
        try:
            self.model.eval()
            with torch.inference_mode():
                for source_id in getattr(self.plan.split, role):
                    self.check()
                    x, _ = data.batch(
                        self.cache,
                        self.plan.split.mode,
                        (self.lookup[source_id],),
                        self.plan.payload_seed,
                    )
                    logits = self.model(torch.from_numpy(x).to(self.device))
                    values.append(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
        finally:
            self.model.train(previous)
        self.check()
        return {"scores": values, "auc": auc(values), "originals": len(values)}


@contextmanager
def session(cache, plan, *, device="cpu", max_seconds=120):
    import torch

    if device not in ("cpu", "cuda:0") or not 0 < finite(max_seconds) <= 1800:
        raise ValueError("explicit fitting device/deadline required")
    state = random.getstate(), np.random.get_state(), torch.get_num_threads()
    policy = profile.cuda_policy() if device == "cuda:0" else engine.capacity.fp32_policy()
    try:
        with policy, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.set_num_threads(2)
            torch.random.default_generator.manual_seed(plan.model_seed)
            if device == "cuda:0":
                torch.cuda.manual_seed(plan.model_seed)
            random.seed(plan.model_seed)
            np.random.seed(plan.model_seed)
            yield _Session(cache, plan, device, max_seconds)
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_num_threads(state[2])


def train(
    cache,
    split,
    out,
    binding,
    *,
    device="cuda:0",
    seconds=1700,
    epochs=20,
    max_checkpoint_bytes=128 * a.MIB,
):
    """Shared loop. Reduced CPU schedules are restricted to generated QA namespaces."""
    if (
        type(epochs) is not int
        or not 1 <= epochs <= 20
        or (
            epochs != 20
            and (
                device != "cpu"
                or any(not row.id.startswith("generated-profile-") for row in cache.rows)
            )
        )
    ):
        raise ValueError("reduced exposure is generated CPU QA only")
    split.validate(cache)
    a.identity(binding)
    a.directory(out)
    deadline, commits, cells = time.monotonic() + finite(seconds, positive=True), 0, []
    steps = math.ceil(len(split.train) / 4)
    for cell, seed in enumerate(profile.pilot.SEEDS):
        plan = engine.Plan(split, seed, RECIPE["payload_seed"], epochs, 8, binding)
        with session(
            cache, plan, device=device, max_seconds=profile.pilot.remaining(deadline)
        ) as s:
            initial = {role: s.scores(role) for role in ("train", "development")}
            rows, checkpoints, losses = [], [], []
            for epoch in range(1, epochs + 1):
                epoch_losses = []
                for step in range(steps):
                    intent = {"cell": cell, "epoch": epoch, "step": step, "native": True}
                    profile.pilot.record_event(out, commits + 1, "intent", intent)
                    result = s.step()
                    commits += 1
                    epoch_losses.append(result["loss"])
                    losses.append(result["loss"])
                    profile.pilot.record_event(out, commits, "commit", {**intent, "result": result})
                development = s.scores("development")
                row = {
                    "seed": seed,
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "development": development,
                    "optimizer_updates": s.updates,
                }
                rows.append(row)
                a.put(out / f"seed-{seed}-epoch-{epoch:02d}.json", a.encode(row))
                profile.pilot.historical.progress(out, {"contract_sha256": binding, **row})
                print(jsonline(row), flush=True)
                if epoch % 5 == 0 or epoch == epochs:
                    path = out / f"seed-{seed}-epoch-{epoch:02d}.numeric"
                    sha = s.checkpoint(path)
                    if path.stat().st_size > max_checkpoint_bytes:
                        raise ValueError("measured checkpoint bound exceeded")
                    checkpoints.append(
                        {
                            "name": path.name,
                            "sha256": sha,
                            "epoch": epoch,
                            "bytes": path.stat().st_size,
                        }
                    )
            final = {"train": s.scores("train"), "development": development}
            expected = epochs * len(split.train)
            if (
                s.cursor != expected
                or s.updates != steps * epochs
                or s.presentations != 2 * expected
            ):
                raise ValueError("complete fitting exposure mismatch")
            first, last = float(np.mean(losses[:steps])), float(np.mean(losses[-steps:]))
            passed = (
                final["train"]["auc"] >= 0.85
                and final["development"]["auc"] >= 0.70
                and last <= 0.8 * first
            )
            cells.append(
                {
                    "seed": seed,
                    "initial": initial,
                    "final": final,
                    "epochs": rows,
                    "losses": losses,
                    "checkpoints": checkpoints,
                    "passed": passed,
                    "updates": s.updates,
                    "presentations": s.presentations,
                }
            )
        del s
        profile.pilot.release(device)
    result = {
        "schema": "colab-rgb-fit-result-v1",
        "contract_sha256": binding,
        "status": "passed_exploratory_native_gate"
        if all(c["passed"] for c in cells)
        else "failed_learning_gate",
        "split": asdict(split),
        "cells": cells,
        "fit_updates": commits,
        "presentations": sum(c["presentations"] for c in cells),
        "physical_cuda": device == "cuda:0",
        "device": device,
        "epochs": epochs,
        "qualified": False,
        "deployment": False,
        "future_reserved_opened": False,
    }
    a.put(out / "fit.json", a.encode(result), 4 * a.MIB)
    return a.document(out / "fit.json", 4 * a.MIB)


def jsonline(row):
    import json

    return json.dumps(
        {k: row[k] for k in ("seed", "epoch", "loss", "optimizer_updates")}
        | {"development_auc": row["development"]["auc"]},
        allow_nan=False,
    )


def execute(job: dict, binding: str, out: Path) -> dict:
    from core.colab_execution import Limits

    limits = Limits(**job["limits"])
    split = contract(job["payload"], limits)
    for key, maximum in (
        (resource.RLIMIT_CPU, math.ceil(limits.seconds)),
        (resource.RLIMIT_FSIZE, limits.file_bytes),
    ):
        soft, hard = resource.getrlimit(key)
        if not 0 < soft <= hard <= maximum:
            raise ValueError("hard fitting CPU/file limits unavailable")
    observed = profile.environment()
    if observed != job["payload"]["environment"]:
        raise ValueError("profile/fitting environment changed")
    a.put(out / "environment-before-model.json", a.encode(observed))
    result = train(
        data.load(Path(os.environ["STEGANOGRAPHY_COLAB_CACHE"]), split.manifest_sha256),
        split,
        out / "fit",
        binding,
        seconds=limits.seconds - 30,
        max_checkpoint_bytes=job["payload"]["max_checkpoint_bytes"],
    )
    result.update(
        environment=observed,
        recipe=RECIPE,
        profile_receipt_sha256=job["payload"]["profile_receipt_sha256"],
    )
    return result


def audit_training(out: Path, result: dict, binding: str, observed: dict) -> dict:
    """Independent pairwise AUC, update ledger and numeric-state replay; no pixel reads."""
    split = engine.split_document(result["split"])
    if (
        result["contract_sha256"] != binding
        or result["epochs"] not in range(1, 21)
        or [c["seed"] for c in result["cells"]] != list(profile.pilot.SEEDS)
        or result["future_reserved_opened"] is not False
        or result["qualified"] is not False
        or result["deployment"] is not False
    ):
        raise ValueError("fit result binding/scope mismatch")
    commits = aucs = states = presentations = 0
    steps = math.ceil(len(split.train) / 4)
    for cell, row in enumerate(result["cells"]):
        plan = engine.Plan(split, row["seed"], RECIPE["payload_seed"], result["epochs"], 8, binding)
        if (
            len(row["epochs"]) != plan.epochs
            or len(row["losses"]) != steps * plan.epochs
            or row["updates"] != steps * plan.epochs
            or row["presentations"] != 2 * plan.epochs * len(split.train)
        ):
            raise ValueError("fit exposure/epoch mismatch")
        presentations += row["presentations"]
        for epoch, record in enumerate(row["epochs"], 1):
            if (
                record["seed"] != plan.model_seed
                or record["epoch"] != epoch
                or record["optimizer_updates"] != steps * epoch
                or record != a.document(out / f"seed-{plan.model_seed}-epoch-{epoch:02d}.json")
            ):
                raise ValueError("epoch record mismatch")
            losses = []
            for step in range(steps):
                commits += 1
                intent = {"cell": cell, "epoch": epoch, "step": step, "native": True}
                actual = a.document(out / f"{commits:03d}-commit.json")
                expected_cursor = (epoch - 1) * len(split.train) + min(
                    4 * (step + 1), len(split.train)
                )
                if (
                    a.document(out / f"{commits:03d}-intent.json") != intent
                    or {k: v for k, v in actual.items() if k != "result"} != intent
                    or actual["result"]["cursor"] != expected_cursor
                    or actual["result"]["optimizer_updates"] != (epoch - 1) * steps + step + 1
                    or actual["result"]["presentations"] != 2 * expected_cursor
                ):
                    raise ValueError("fit commit/consumption mismatch")
                losses.append(finite(actual["result"]["loss"]))
            if losses != row["losses"][(epoch - 1) * steps : epoch * steps] or record[
                "loss"
            ] != float(np.mean(losses)):
                raise ValueError("epoch loss mismatch")
        scores = [
            (row["initial"][role], len(getattr(split, role))) for role in ("train", "development")
        ]
        scores += [(r["development"], len(split.development)) for r in row["epochs"]]
        scores += [
            (row["final"][role], len(getattr(split, role))) for role in ("train", "development")
        ]
        for score, count in scores:
            pairs = np.asarray(score["scores"], dtype=np.float64)
            auc(pairs)  # Validate shape/range independently of the stored scalar.
            pairwise = float(
                np.mean(
                    (pairs[:, 1, None] > pairs[None, :, 0])
                    + 0.5 * (pairs[:, 1, None] == pairs[None, :, 0])
                )
            )
            if (
                pairs.shape != (count, 2)
                or score["originals"] != count
                or abs(pairwise - finite(score["auc"])) > 1e-12
            ):
                raise ValueError("independent score/AUC mismatch")
            aucs += 1
        first, last = float(np.mean(row["losses"][:steps])), float(np.mean(row["losses"][-steps:]))
        passed = (
            row["final"]["train"]["auc"] >= 0.85
            and row["final"]["development"]["auc"] >= 0.70
            and last <= 0.8 * first
        )
        if (
            row["passed"] is not passed
            or row["final"]["development"] != row["epochs"][-1]["development"]
        ):
            raise ValueError("independent learning rule mismatch")
        epochs = [e for e in range(1, plan.epochs + 1) if e % 5 == 0 or e == plan.epochs]
        if [c["epoch"] for c in row["checkpoints"]] != epochs:
            raise ValueError("checkpoint schedule mismatch")
        source = a.digest(
            a.encode(
                {
                    "plan": asdict(plan),
                    "device": result["device"],
                    "recipe": data.RECIPE,
                    "optimizer": "scalar-adamw-1e-4-v1",
                }
            )
        )
        datum = a.digest(
            a.encode(
                {
                    "manifest": split.manifest_sha256,
                    "split": asdict(split),
                    "recipe": data.RECIPE,
                    "payload_seed": plan.payload_seed,
                }
            )
        )
        for checkpoint in row["checkpoints"]:
            name = f"seed-{plan.model_seed}-epoch-{checkpoint['epoch']:02d}.numeric"
            if checkpoint["name"] != name or (out / name).stat().st_size != checkpoint["bytes"]:
                raise ValueError("numeric checkpoint name/extent mismatch")
            meta, arrays = numeric.load(
                out / name,
                checksum=checkpoint["sha256"],
                environment={k: observed[k] for k in ENV_KEYS},
                source_sha256=source,
                data_sha256=datum,
            )
            count = meta["consumption"]
            epoch = checkpoint["epoch"]
            if (
                count["epoch"] != epoch
                or count["cursor"] != epoch * len(split.train)
                or count["optimizer_updates"] != epoch * steps
                or count["presentations"] != 2 * epoch * len(split.train)
                or not np.array_equal(arrays["data/order"], engine.order(plan))
            ):
                raise ValueError("numeric fit debit/order mismatch")
            states += 1
            del meta, arrays
    status = (
        "passed_exploratory_native_gate"
        if all(c["passed"] for c in result["cells"])
        else "failed_learning_gate"
    )
    if (
        result["status"] != status
        or result["fit_updates"] != commits
        or result["presentations"] != presentations
    ):
        raise ValueError("aggregate fitting result mismatch")
    checked = {
        "schema": "colab-native-fit-independent-v1",
        "contract_sha256": binding,
        "commits_verified": commits,
        "numeric_states_verified": states,
        "aucs_verified": aucs,
        "native_pixels_opened": False,
    }
    a.put(out.parent / "independent-fit-audit.json", a.encode(checked))
    return checked
