"""Separate Colab RGB profile adapter; no native optimizer or fitting entry point."""

from __future__ import annotations

import math
import os
import random
import re
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core import colab_artifacts as a
from core import spatial_native_data as data
from core import spatial_native_pilot as pilot
from core import spatial_native_training as engine

PROFILE_SCHEMA = "colab-rgb-profile-data-v1"
SPLIT_SHA256 = "852b36d9bfee693c3be175ab5365d160a3e9d5c6da08e9fa1a2a0add552abefa"
RESOURCE_ACK = "hard_cgroup_ram_and_swap_not_enforced"
RECIPE = {
    "mode": "RGB", "input_shape": [256, 256, 3], "batch_images": 8,
    "seeds": list(pilot.SEEDS), "native_optimizer_updates": 0,
    "generated_optimizer_updates": 6, "passes_per_seed": 3,
    "numeric_restores": 2, "precision": "float32-ieee", "amp": False, "tf32": False,
}
ENVIRONMENT_KEYS = {"python", "numpy", "pillow", "torch", "cuda", "gpu", "driver"}


def contract(payload: dict, limits) -> engine.Split:
    if (set(payload) != {"schema", "split", "recipe", "expected_environment"}
            or payload["schema"] != PROFILE_SCHEMA or payload["recipe"] != RECIPE):
        raise ValueError("Colab profile recipe/schema mismatch")
    split = engine.split_document(payload["split"])
    if split.mode != "RGB" or tuple(map(len, (
        split.train, split.development, split.future_reserved, split.unassigned,
    ))) != (64, 20, 21, 13):
        raise ValueError("Colab profile requires the exact RGB proposal counts")
    if a.digest(a.encode(asdict(split))) != SPLIT_SHA256:
        raise ValueError("frozen RGB proposal binding mismatch")
    env = payload["expected_environment"]
    if (not isinstance(env, dict) or set(env) != ENVIRONMENT_KEYS
            or any(not isinstance(v, str) or not 1 <= len(v) <= 128
                   or not all(32 <= ord(c) < 127 for c in v) for v in env.values())
            or env["python"] != "3.12.13" or env["numpy"] != "2.0.2"
            or env["pillow"] != "11.3.0" or env["torch"] != "2.11.0+cu128"
            or env["cuda"] != "12.8" or env["gpu"] != "Tesla T4"
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", env["driver"]) is None):
        raise ValueError("explicit Colab 2026.07 T4 environment expectations required")
    if (limits.seconds != 240 or limits.closing_seconds != 10
            or limits.data_bytes != 0 or limits.resident_watch_bytes != 0
            or limits.output_bytes != 256 * a.MIB or limits.file_bytes != 128 * a.MIB
            or limits.address_bytes != 0):
        raise ValueError("separate Colab profile limits required")
    return split


def cache_binding(cache: data.Cache, split: engine.Split) -> None:
    split.validate(cache)
    lookup = {row.id: row for row in cache.rows}
    counts = []
    for role in (split.train, split.development, split.future_reserved, split.unassigned):
        ids = set(role)
        counts.append(sum(bool(ids.intersection(group)) for group in data.components(cache.rows)))
    if counts != [38, 12, 19, 6] or any(lookup[i].mode != "L" for i in split.unassigned):
        raise ValueError("exact metadata ancestry/mode binding required")


def driver_version(path: Path = Path("/proc/driver/nvidia/version")) -> str:
    text = a.read(path, 4096).decode("ascii")
    found = re.search(
        r"Kernel Module(?: for [A-Za-z0-9_-]+)?\s+([0-9]+(?:\.[0-9]+){1,3})\s", text
    )
    if found is None:
        raise ValueError("NVIDIA kernel version unavailable")
    return found[1]


def environment() -> dict:
    return {**engine.capacity.environment(), "driver": driver_version()}


def inspect_cuda() -> dict:
    """A new prospective host policy; never invoke or bypass the frozen WSL guard."""
    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("Colab CUDA unavailable; no CPU fallback")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("deterministic CUBLAS environment required")
    matmul, conv = torch.backends.cuda.matmul, getattr(torch.backends.cudnn, "conv", None)
    if not hasattr(matmul, "fp32_precision") or not hasattr(conv, "fp32_precision"):
        raise RuntimeError("modern IEEE precision API required")
    capability = torch.cuda.get_device_capability(0)
    suffix = "_" + "".join(str(n) for n in capability)
    if not {"sm" + suffix, "compute" + suffix}.intersection(torch.cuda.get_arch_list()):
        raise RuntimeError("GPU architecture absent from Torch build")
    props = torch.cuda.get_device_properties(0)
    budget = min(4 * 1024**3, int(props.total_memory * 0.7))
    if budget < 1024**3 or torch.cuda.memory_reserved(0) > budget:
        raise RuntimeError("GPU allocator budget unavailable")
    return {"torch_allocator_limit_bytes": budget, "total_gpu_bytes": props.total_memory,
            "hard_cgroup_ram_or_swap_enforced": False, "cpu_fallback": False,
            "amp": False, "tf32": False, "precision": "float32-ieee"}


@contextmanager
def cuda_policy():
    import torch

    info = inspect_cuda()
    cudnn = torch.backends.cudnn
    matmul, conv = torch.backends.cuda.matmul, cudnn.conv  # type: ignore[attr-defined]
    previous = (matmul.fp32_precision, conv.fp32_precision, cudnn.benchmark, cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.cuda.get_per_process_memory_fraction(0))
    try:
        torch.cuda.set_per_process_memory_fraction(
            info["torch_allocator_limit_bytes"] / info["total_gpu_bytes"], 0
        )
        matmul.fp32_precision = conv.fp32_precision = "ieee"
        cudnn.benchmark, cudnn.deterministic = False, True
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.cuda.device(0), torch.autocast(device_type="cuda", enabled=False):
            yield info
    finally:
        matmul.fp32_precision, conv.fp32_precision = previous[:2]
        cudnn.benchmark, cudnn.deterministic = previous[2:4]
        torch.use_deterministic_algorithms(previous[4], warn_only=previous[5])
        torch.cuda.set_per_process_memory_fraction(previous[6], 0)


class _ProfileSession(engine._Session):
    """Narrower profiling view: train timing only and generated namespace updates."""

    def __init__(self, cache, plan, device, max_seconds, generated_optimizer):
        if type(generated_optimizer) is not bool or (generated_optimizer and any(
            not row.id.startswith("generated-profile-")
            or not row.author.startswith("Generated profile fixture ") for row in cache.rows
        )):
            raise ValueError("optimizer is restricted to generated profile fixtures")
        self.generated_optimizer = generated_optimizer
        super().__init__(cache, plan, device, max_seconds)

    def step(self):
        if not self.generated_optimizer:
            raise ValueError("native profile optimizer updates are forbidden")
        return super().step()

    def checkpoint(self, path):
        if not self.generated_optimizer:
            raise ValueError("only generated profile checkpoints are permitted")
        return super().checkpoint(path)

    def scores(self, role, *, record_scores=False):
        """Measure train pairs without importing the application's JPEG/AUC stack."""
        import torch

        if role != "train" or record_scores is not False:
            raise ValueError("profile permits train timing only, without saved scores")
        self.check()
        previous, durations = self.model.training, []
        try:
            self.model.eval()
            with torch.inference_mode():
                for source_id in self.plan.split.train:
                    self.check()
                    begin = time.monotonic()
                    x, _ = data.batch(self.cache, self.plan.split.mode,
                                      (self.lookup[source_id],), self.plan.payload_seed)
                    logits = self.model(torch.from_numpy(x).to(self.device))
                    outputs = torch.softmax(logits, 1)[:, 1]
                    pair = outputs.cpu().numpy()
                    if not np.isfinite(pair).all() or np.any((pair < 0) | (pair > 1)):
                        raise ValueError("nonfinite profile timing inference")
                    pilot.sync(self.device)
                    durations.append(time.monotonic() - begin)
        finally:
            self.model.train(previous)
        self.check()
        return {"originals": len(durations), "max_pair_seconds": max(durations),
                "seconds": sum(durations), "role": "train", "optimizer_updates": self.updates}


@contextmanager
def session(cache, plan, *, device="cpu", max_seconds=120, generated_optimizer=False):
    if (device not in ("cpu", "cuda:0") or type(max_seconds) not in (int, float)
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 220):
        raise ValueError("explicit Colab profile device/deadline required")
    import torch

    python, numpy, threads = random.getstate(), np.random.get_state(), torch.get_num_threads()
    try:
        policy = cuda_policy() if device == "cuda:0" else engine.capacity.fp32_policy()
        with policy, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.set_num_threads(2)
            torch.random.default_generator.manual_seed(plan.model_seed)
            if device == "cuda:0":
                torch.cuda.manual_seed(plan.model_seed)
            random.seed(plan.model_seed)
            np.random.seed(plan.model_seed)
            # Reuse the pinned optimizer/data/numeric engine under this distinct policy.
            yield _ProfileSession(cache, plan, device, max_seconds, generated_optimizer)
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_num_threads(threads)


def execute(job: dict, binding: str, out: Path) -> dict:
    """Only the permission-checked process worker invokes physical CUDA profiling."""
    from core.colab_execution import Limits

    limits = Limits(**job["limits"])
    split = contract(job["payload"], limits)
    for key, maximum in ((resource.RLIMIT_CPU, 240), (resource.RLIMIT_FSIZE, limits.file_bytes)):
        soft, hard = resource.getrlimit(key)
        if not 0 < soft <= hard <= maximum:
            raise ValueError("hard process profile limits unavailable")
    cache = data.load(Path(os.environ["STEGANOGRAPHY_COLAB_CACHE"]), split.manifest_sha256)
    cache_binding(cache, split)
    observed = environment()
    expected = job["payload"]["expected_environment"]
    if (any(observed[k] != value for k, value in expected.items())
            or type(observed.get("cudnn")) is not int or observed["cudnn"] <= 0):
        raise ValueError("Colab profile environment mismatch")
    a.put(out / "environment-before-model.json", a.encode(observed))
    result = measure(cache, split, out / "profile", binding, seconds=220)
    projected = pilot.projection({**result, "schema": "spatial-rgb-native-measurement-v1"})
    result.update(contract_sha256=binding, environment=observed, recipe=RECIPE,
                  projection=projected,
                  native_fit_authorized=False, future_reserved_opened=False,
                  hard_cgroup_ram_or_swap_enforced=False)
    return result


def audit_measurement(out: Path, result: dict, binding: str, observed: dict) -> dict:
    """Independent NumPy/numeric replay; no Torch import or native pixel reads."""
    from core import spatial_colab_state as numeric

    pilot.projection({**result, "schema": "spatial-rgb-native-measurement-v1"})
    fake = data.load(out / "generated-cache", result["generated_fixture_only_cache"])
    pairs = len(fake.rows) - 2
    if pairs not in (1, 2, 4) or any(not r.id.startswith("generated-profile-") for r in fake.rows):
        raise ValueError("generated audit namespace/count")
    ids = tuple(row.id for row in fake.rows)
    split = engine.Split(fake.manifest_sha256, result["split"]["mode"], 2026102807,
                         ids[:pairs], (ids[pairs],), (ids[pairs + 1],), ())
    split.validate(fake)
    environment_keys = {"python", "numpy", "pillow", "torch", "cuda", "cudnn", "gpu"}
    env = {key: observed[key] for key in environment_keys}
    checks = []
    for cell, seed in enumerate(pilot.SEEDS):
        plan = engine.Plan(split, seed, pilot.SETTINGS["generated_payload_seed"], 3, 2 * pairs,
                           binding)
        source_sha = a.digest(a.encode({"plan": asdict(plan), "device": result["device"],
                                        "recipe": data.RECIPE,
                                        "optimizer": "scalar-adamw-1e-4-v1"}))
        data_sha = a.digest(a.encode({"manifest": fake.manifest_sha256, "split": asdict(split),
                                      "recipe": data.RECIPE, "payload_seed": plan.payload_seed}))
        meta, arrays = numeric.load(out / f"generated-{cell}.numeric",
                                    checksum=result["cells"][cell]["checkpoint_sha256"],
                                    environment=env, source_sha256=source_sha, data_sha256=data_sha)
        debit = meta["consumption"]
        if (debit["optimizer_updates"] != 3 or debit["presentations"] != 6 * pairs
                or debit["cursor"] != 3 * pairs or not np.array_equal(arrays["data/order"],
                                                                             engine.order(plan))):
            raise ValueError("generated numeric profile debit/order mismatch")
        for step in range(3):
            sequence = cell * 3 + step + 1
            intent = a.document(out / f"{sequence:03d}-intent.json")
            commit = a.document(out / f"{sequence:03d}-commit.json")
            if (intent != {"cell": cell, "step": step, "generated_only": True}
                    or commit["cell"] != cell or commit["step"] != step
                    or commit["generated_only"] is not True
                    or commit["result"]["optimizer_updates"] != step + 1
                    or commit["result"]["presentations"] != 2 * pairs * (step + 1)
                    or commit["result"]["cursor"] != pairs * (step + 1)):
                raise ValueError("generated commit profile debit mismatch")
        checks.append({"seed": seed, "arrays_verified": len(arrays), "optimizer_updates": 3,
                       "checkpoint_sha256": result["cells"][cell]["checkpoint_sha256"]})
        del arrays, meta
    receipt = {"schema": "colab-profile-independent-numeric-v1", "contract_sha256": binding,
               "generated_optimizer_updates_verified": 6, "native_optimizer_updates": 0,
               "numeric_checkpoints_verified": 2, "commits_verified": 6, "cells": checks,
               "native_pixels_opened_by_auditor": False, "torch_imported_by_auditor": False}
    a.put(out.parent / "independent-numeric-audit.json", a.encode(receipt))
    return receipt


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
    fake, fake_split = pilot.generated_cache(out, split.mode, pairs)
    cells = []
    confirmed = 0
    for cell, seed in enumerate(pilot.SEEDS):
        pilot.release(device)
        begin = time.monotonic()
        native_plan = engine.Plan(
            split, seed, pilot.SETTINGS["native_profile_payload_seed"], 1, 2 * pairs, binding
        )
        with session(
            cache, native_plan, device=device, max_seconds=pilot.remaining(deadline)
        ) as s:
            initialized = time.monotonic() - begin
            before = pilot.parameters(s.model)
            passes = []
            for _ in range(3):
                s.check()
                pilot.sync(device)
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
                pilot.sync(device)
                passes.append(time.monotonic() - begin)
                s.check()
            evaluation = s.scores("train", record_scores=False)
            if before != pilot.parameters(s.model) or s.updates or s.optimizer.state:
                raise ValueError("native profile changed parameters/optimizer")
            native_memory = pilot.memory(device)
        del s, values, targets, x, labels, loss
        pilot.release(device)
        plan = engine.Plan(
            fake_split, seed, pilot.SETTINGS["generated_payload_seed"], 3, 2 * pairs, binding
        )
        begin = time.monotonic()
        with session(fake, plan, device=device, max_seconds=pilot.remaining(deadline),
                     generated_optimizer=True) as s:
            generated_initialization = time.monotonic() - begin
            steps = []
            for step in range(3):
                pilot.record_event(
                    out,
                    confirmed + 1,
                    "intent",
                    {"cell": cell, "step": step, "generated_only": True},
                )
                pilot.sync(device)
                begin = time.monotonic()
                result = s.step()
                pilot.sync(device)
                steps.append(time.monotonic() - begin)
                confirmed += 1
                pilot.record_event(
                    out,
                    confirmed,
                    "commit",
                    {"cell": cell, "step": step, "result": result, "generated_only": True},
                )
            expected = pilot.historical.model_sha(s.model)
            begin = time.monotonic()
            path = out / f"generated-{cell}.numeric"
            sha = s.checkpoint(path)
            write_seconds = time.monotonic() - begin
            generated_memory = pilot.memory(device)
        del s
        pilot.release(device)
        begin = time.monotonic()
        with session(fake, plan, device=device, max_seconds=pilot.remaining(deadline),
                     generated_optimizer=True) as restored:
            restored.resume(path, sha)
            if restored.updates != 3 or pilot.historical.model_sha(restored.model) != expected:
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
        pilot.release(device)
        pilot.remaining(deadline)
    result = {
        "schema": "spatial-colab-native-measurement-v1",
        "status": "measured",
        "device": device,
        "physical_cuda": device == "cuda:0",
        "binding": binding,
        "manifest_sha256": cache.manifest_sha256,
        "split": asdict(split),
        "cells": cells,
        "native_optimizer_updates": 0,
        "generated_optimizer_updates": confirmed,
        "native_backward_presentations": len(pilot.SEEDS) * 3 * pairs * 2,
        "native_evaluation_presentations": len(pilot.SEEDS) * len(split.train) * 2,
        "generated_fixture_only_cache": fake.manifest_sha256,
        "worker_seconds": time.monotonic() - started,
        "cuda_peak": pilot.memory(device),
    }
    a.put(out / "measurement.json", a.encode(result), 4 * a.MIB)
    return result

