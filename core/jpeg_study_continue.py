"""Explicit continuation of an owned cancelled study; frozen fitting sources."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from core import jpeg_study as study
from core import jpeg_study_data as data
from core import jpeg_study_segments as segments
from core import jpeg_study_state as state
from core import jpeg_study_views as views
from core.jpeg_scale import identity
from core.jpeg_timing import write_document
from core.srnet_checkpoint import MAX_BYTES, PARAMETERS
from core.srnet_stream import document, regular_open

SCHEMA = "jpeg-owned-continuation-v1"


def digest(path):
    with regular_open(path) as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("continuation metadata exceeds bounds")
    return hashlib.sha256(raw).hexdigest()


def prepare(plan, sources, *, torch_version):
    """Validate all inherited evidence before opening any fresh GPU worker."""
    if (not isinstance(plan, dict) or set(plan) != {
        "schema_version", "parent", "parent_report_sha256", "config", "config_sha256",
        "segments",
    } or plan["schema_version"] != SCHEMA):
        raise ValueError("continuation plan schema invalid")
    if any(not isinstance(plan[k], str) or not plan[k] for k in ("parent", "config")):
        raise ValueError("continuation private paths invalid")
    parent = Path(plan["parent"])
    prior = document(parent / "report.json", plan["parent_report_sha256"])
    config = document(Path(plan["config"]), plan["config_sha256"])
    spent = prior.get("seconds")
    if (prior.get("schema_version") != "jpeg-learning-study-v1"
        or prior.get("action") != "controls" or prior.get("status") != "cancelled"
        or prior.get("reason") != "user_cancelled"
        or prior.get("source_sha256") != sources
        or prior.get("config_sha256") != plan["config_sha256"]
        or prior.get("accuracy_qualification") != "unavailable"
        or prior.get("deployed") is not False
        or set(config) != {"schema_version", "action", "root", "manifest_sha256",
                           "audit", "audit_sha256", "execution_mode"}
        or config.get("schema_version") != "jpeg-study-config-v1"
        or config.get("action") != "controls"
        or config.get("execution_mode") != segments.MODE
        or type(spent) not in (int, float) or not math.isfinite(spent)
        or not 0 <= spent < segments.CONTROLS_SECONDS):
        raise ValueError("continuation parent/source/config/budget mismatch")
    manifest = document(Path(config["root"]) / "manifest.json", config["manifest_sha256"])
    data.validate_manifest(manifest)
    proof = document(Path(config["audit"]), config["audit_sha256"])
    if (manifest.get("schema_version") != views.SCHEMA
        or proof.get("schema_version") != "jpeg-study-independent-audit-v2"
        or proof.get("status") != "completed" or proof.get("source_sha256") != sources
        or proof.get("manifest_sha256") != config["manifest_sha256"]
        or proof.get("representation") != views.contract()
        or proof.get("idct_rows") != {r: len(v) for r, v in manifest["samples"].items()}
        or proof.get("idct_views") != {r: 5 * len(v) for r, v in manifest["samples"].items()}
        or manifest["originals_per_source"] != {"fit": 256, "development": 64}):
        raise ValueError("continuation independent audit mismatch")
    inherited = plan["segments"]
    ordered = [(a, s, stop) for s in study.SEEDS for a in study.ARMS
               for stop in (16384, 32768)]
    if not isinstance(inherited, list) or not 1 <= len(inherited) < len(ordered):
        raise ValueError("continuation inherited prefix invalid")
    cached: dict[tuple[str, int, int], dict[str, Any]] = {}
    for entry, expected in zip(inherited, ordered, strict=False):
        required = {
            "arm", "seed", "stop_rows", "receipt_sha256"
        } | ({"trial_sha256"} if expected[2] == 32768 else set())
        if (not isinstance(entry, dict) or set(entry) != required
            or type(entry["seed"]) is not int or type(entry["stop_rows"]) is not int
            or (entry["arm"], entry["seed"], entry["stop_rows"]) != expected):
            raise ValueError("continuation must inherit an exact owned prefix")
        arm, seed, stop = expected
        phase = stop // 16384 - 1
        receipt_path = parent / f"{arm}-{seed}" / f"receipt-{stop}.json"
        value = document(receipt_path, entry["receipt_sha256"])
        rows = value.get("recipe", {}).get("rows")
        recipe = study.recipe(arm, seed, rows, representation=views.FIVE)
        binding = hashlib.sha256(json.dumps({
            "recipe": recipe, "manifest_sha256": config["manifest_sha256"],
            "sources": sources, "torch_version": torch_version,
        }, sort_keys=True).encode()).hexdigest()
        checks = value.get("checkpoints")
        if (value.get("status") != ("checkpointed" if phase == 0 else "completed")
            or value.get("source_sha256") != sources or value.get("recipe") != recipe
            or value.get("binding") != binding
            or value.get("manifest_sha256") != config["manifest_sha256"]
            or value.get("execution", {}).get("device") != "cuda:0"
            or value.get("execution", {}).get("torch_version") != torch_version
            or value.get("execution", {}).get("cpu_fallback") is not False
            or value.get("presented_rows") != stop
            or value.get("resumed_from_rows") != phase * 16384
            or value.get("optimizer_updates") != 16384 // rows
            or value.get("total_optimizer_updates") != stop // rows
            or value.get("real_gpu_segment") is not True
            or value.get("real_gpu_fit") is not bool(phase)
            or value.get("accuracy_qualification") != "unavailable"
            or value.get("deployed") is not False
            or not isinstance(checks, list) or len(checks) != 1
            or not isinstance(checks[0], dict)
            or checks[0].get("path") != f"rows-{stop}.npz"
            or checks[0].get("presented_rows") != stop):
            raise ValueError("continuation inherited receipt binding invalid")
        for key in ("initial_learned_sha256", "schedule_sha256"):
            identity(value.get(key))
        for key in ("seconds", "mean_cross_entropy"):
            number = value.get(key)
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError("continuation inherited numeric receipt invalid")
        if phase == 0 and ("scores" in value or "cells" in value):
            raise ValueError("continuation partial receipt cannot contain evaluation")
        if phase and not isinstance(value.get("scores"), list):
            raise ValueError("continuation complete receipt requires evaluation")
        path = parent / f"{arm}-{seed}" / f"segment-{phase}" / checks[0]["path"]
        snapshot = state.load(path, checks[0]["sha256"], binding)
        if (snapshot["metadata"]["presented_rows"] != stop or any(
            float(snapshot["arrays"][f"optimizer/{name}/step"]) != stop // rows
            for name in PARAMETERS
        )):
            raise ValueError("continuation snapshot exposure/batch mismatch")
        if phase:
            first = cached[(arm, seed, 16384)]["value"]
            if any(value[k] != first[k] for k in ("binding", "initial_learned_sha256")):
                raise ValueError("continuation inherited halves differ")
            if f"trial-{arm}-{seed}.json" not in prior.get("trial_reports", []):
                raise ValueError("continuation complete trial not owned by parent")
            trial = document(parent / f"trial-{arm}-{seed}.json", entry["trial_sha256"])
            if (trial.get("status") != "completed" or trial.get("arm") != arm
                or trial.get("seed") != seed or trial.get("execution_mode") != segments.MODE
                or any(trial.get(k) != value.get(k) for k in (
                    "source_sha256", "binding", "recipe", "scores", "schedule_sha256",
                    "initial_learned_sha256", "presented_rows", "total_optimizer_updates",
                ))):
                raise ValueError("continuation complete parent trial mismatch")
        cached[expected] = {"value": value, "path": path,
                            "receipt_sha256": entry["receipt_sha256"]}
    if sum(c["value"]["seconds"] for c in cached.values()) > spent:
        raise ValueError("continuation prior budget omits inherited computation")
    return config, manifest, prior, cached


def execute(plan, out, *, sources, torch_version, launch, probe,
            current_sources, controller_sources, current_controller_sources,
            clock=time.monotonic):
    """Replay verified owned receipts, train only missing halves, retain budget."""
    started = clock()
    config, manifest, prior, cached = prepare(plan, sources, torch_version=torch_version)
    for protected in (Path(plan["parent"]), Path(config["root"])):
        if out.absolute().is_relative_to(protected.absolute()):
            raise ValueError("continuation output cannot modify parent or dataset")
    data.fresh(out)
    out.mkdir(parents=True)
    spent = prior["seconds"]
    trials: list[dict[str, Any]] = []
    capacities: list[dict[str, Any]] = []
    report = {"schema_version": "jpeg-learning-study-v1", "action": "controls",
              "source_sha256": sources, "controller_sha256": controller_sources,
              "config_sha256": plan["config_sha256"], "execution_mode": segments.MODE,
              "manifest_sha256": config["manifest_sha256"],
              "audit_sha256": config["audit_sha256"],
              "continuation_schema": SCHEMA,
              "parent_report_sha256": plan["parent_report_sha256"],
              "prior_controls_seconds": spent, "controls_budget_seconds": segments.CONTROLS_SECONDS,
              "accuracy_qualification": "unavailable", "deployed": False,
              "cpu_fallback": False, "cloud_spending": 0,
              "baseline_stage": "not_started", "adaptation_stage": "not_started",
              "independent_source_evaluation": "not_started", "learning_gate": False,
              "passing_arms": [], "representation": views.contract(),
              "inherited_segments": [{**e, "checkpoint_sha256": cached[
                  (e["arm"], e["seed"], e["stop_rows"])]["value"]["checkpoints"][0]["sha256"]}
                  for e in plan["segments"]]}
    write_document(out / "continuation-start.json", {**report, "status": "running"})

    def remaining():
        return segments.CONTROLS_SECONDS - spent - (clock() - started)

    def cancelled():
        marker = out / "CANCEL"
        if marker.is_symlink():
            raise ValueError("continuation cancellation cannot use symlink")
        return marker.exists()

    try:
        for rows in (8, 16, 32):
            if cancelled():
                raise InterruptedError
            if remaining() <= 0:
                raise TimeoutError
            value = probe(rows, min(180, remaining()), cancelled)
            if (not isinstance(value, dict) or value.get("status") not in {
                "completed", "unavailable"
            } or value.get("rows") != rows):
                raise ValueError("continuation capacity receipt invalid")
            capacities.append(value)
            write_document(out / f"capacity-{rows}.json", value)
        available = [v["rows"] for v in capacities if v["status"] == "completed"]
        for seed in study.SEEDS:
            for arm in study.ARMS:
                if cancelled():
                    raise InterruptedError
                if remaining() <= 0:
                    raise TimeoutError
                rows = max(available) if arm == "genuine-batch" and available else 4
                if arm == "genuine-batch" and not available:
                    value = {"status": "unavailable", "reason": "no_genuine_batch_fits"}
                else:
                    def phase(payload, phase_out, stop, timeout, *, arm=arm, seed=seed, rows=rows):
                        if cancelled():
                            raise InterruptedError
                        inherited = cached.get((arm, seed, stop))
                        if inherited is None:
                            return launch(payload, phase_out, arm, seed, rows, stop,
                                          timeout, cancelled)
                        value = copy.deepcopy(inherited["value"])
                        if value["recipe"]["rows"] != rows:
                            raise ValueError("continuation current batch differs from parent")
                        phase_out.mkdir()
                        target = phase_out / f"rows-{stop}.npz"
                        segments.checkpoint(inherited["path"], value["checkpoints"][0]["sha256"])
                        with regular_open(inherited["path"]) as src, target.open("xb") as dst:
                            size = 0
                            while block := src.read(1024**2):
                                size += len(block)
                                if size > MAX_BYTES:
                                    raise ValueError("continuation copy exceeds byte bound")
                                dst.write(block)
                        segments.checkpoint(target, value["checkpoints"][0]["sha256"])
                        value["inherited_from"] = {
                            "parent_report_sha256": plan["parent_report_sha256"],
                            "receipt_sha256": inherited["receipt_sha256"],
                        }
                        return value

                    value = segments.trial(config, out / f"{arm}-{seed}", arm=arm,
                                           seed=seed, rows=rows, sources=sources,
                                           recipe=study.recipe(arm, seed, rows,
                                                               representation=views.FIVE),
                                           launch=phase, budget_left=remaining)
                if value["status"] == "completed":
                    baseline = next((t for t in trials if t["seed"] == seed
                                     and t["arm"] == "legacy" and t["status"] == "completed"), None)
                    if baseline and any(value[k] != baseline[k] for k in (
                        "initial_learned_sha256", "schedule_sha256"
                    )):
                        raise ValueError("continuation unmatched initialization/schedule")
                    value["cells"] = study.statistics(manifest["samples"]["development"],
                                                       value["scores"], baseline=baseline[
                                                           "scores"] if baseline else None)
                value.update(arm=arm, seed=seed)
                value["inherited_segment_indices"] = [i for i, stop in enumerate((16384, 32768))
                                                       if (arm, seed, stop) in cached]
                trials.append(value)
                write_document(out / f"trial-{arm}-{seed}.json", value)
                if value["status"] == "cancelled":
                    raise InterruptedError
        winners = [arm for arm in study.ARMS[1:] if all(any(
            t["arm"] == arm and t["seed"] == seed and t["status"] == "completed"
            and all(c["learning_gate"] for c in t["cells"]) for t in trials
        ) for seed in study.SEEDS)]
        complete = len(trials) == 8 and all(t["status"] == "completed" for t in trials)
        report.update(status="completed" if complete else "unavailable",
                      reason=None if complete else "incomplete_required_controls",
                      passing_arms=winners, learning_gate=bool(winners))
    except (InterruptedError, KeyboardInterrupt):
        report.update(status="cancelled", reason="user_cancelled")
    except TimeoutError:
        report.update(status="unavailable", reason="controls_budget_exhausted")
    finally:
        report.update(trials=trials, capacity=capacities,
                      trial_reports=[f"trial-{t['arm']}-{t['seed']}.json" for t in trials
                                     if t["status"] == "completed"],
                      new_controls_seconds=clock() - started,
                      seconds=spent + clock() - started)
        if (current_sources() != sources
            or current_controller_sources() != controller_sources):
            report.update(status="unavailable", reason="source_mutation",
                          learning_gate=False, passing_arms=[])
        if "status" not in report:
            report.update(status="unavailable", reason="continuation_contract_failed")
        write_document(out / "report.json", report)
    return report
