"""Frozen six-hour, ten-GiB refinement controller; no retries or deployment."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from core import jpeg_refine_metrics as metrics
from core import jpeg_refine_schedule as schedule
from core import jpeg_refine_state as state
from core import jpeg_study as study
from core import jpeg_study_data as data
from core import jpeg_study_views as views
from core import srnet_samepool
from core.jpeg_scale import identity
from core.jpeg_study_process import WorkerFailure
from core.jpeg_timing import read_bytes, write_document
from core.srnet_checkpoint import MAX_BYTES
from core.srnet_stream import document, regular_open

SECONDS = 21600
BYTES = 10 * 1024**3
EVALUATION_SECONDS = 900
MANIFEST_SHA = "87df666c9896cec3e1b4897ead54321af4b8f79f4a0ec02e397582e725638fb5"
AUDIT_SHA = "e9aae4d6c35199fcefa7e7752b790ec7143fb8a1ebd059bfc2f3867fc1de163b"


def recipe():
    return {
        "architecture": srnet_samepool.ARCHITECTURE,
        "optimizer_rule": study.jpeg_study_optimizer.RULE,
        "learning_rate": 0.001,
        "convolution_only_l2": True,
        "batch_rows": 16,
        "seeds": list(study.SEEDS),
        "warmup_updates": schedule.WARMUP,
        "final_updates": schedule.FINAL,
        "total_updates": schedule.TOTAL,
        "checkpoint_interval": schedule.INTERVAL,
        "representation": views.contract(),
        "gradient_accumulation": False,
        "amp": False,
        "tf32": False,
        "threshold": 0.5,
        "bootstrap": 1000,
        "schedule": "paired-prefix-source-block-cyclic-stego-v1",
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path, maximum=65536):
    with regular_open(path) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("refine metadata exceeds bounds")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def configuration(value, sources):
    required = {
        "schema_version",
        "root",
        "manifest_sha256",
        "audit",
        "audit_sha256",
        "source_sha256",
        "recipe",
        "budget_seconds",
        "budget_bytes",
        "expected_gpu",
    }
    if (
        not isinstance(value, dict)
        or set(value) - {"resume"} != required
        or value["schema_version"] != "jpeg-refine-config-v1"
        or value["source_sha256"] != sources
        or value["recipe"] != recipe()
        or type(value["budget_seconds"]) is not int
        or value["budget_seconds"] != SECONDS
        or type(value["budget_bytes"]) is not int
        or value["budget_bytes"] != BYTES
        or any(not isinstance(value[k], str) or not value[k] for k in ("root", "audit"))
    ):
        raise ValueError("refine config/source/recipe/budget mismatch")
    for key in ("manifest_sha256", "audit_sha256"):
        identity(value[key])
    if value["manifest_sha256"] != MANIFEST_SHA or value["audit_sha256"] != AUDIT_SHA:
        raise ValueError("refine frozen data/audit identity mismatch")
    gpu = value["expected_gpu"]
    if (
        not isinstance(gpu, dict)
        or set(gpu) != {"name", "uuid_sha256"}
        or not isinstance(gpu["name"], str)
        or "RTX 5060" not in gpu["name"]
        or len(gpu["name"]) > 128
    ):
        raise ValueError("refine requires the owned RTX 5060 identity")
    identity(gpu["uuid_sha256"])
    if "resume" in value:
        resume = value["resume"]
        if (
            not isinstance(resume, dict)
            or set(resume) != {"parent", "report_sha256", "lost_seconds", "remaining_seconds"}
            or not isinstance(resume["parent"], str)
            or not resume["parent"]
            or any(
                type(resume[k]) not in (int, float) or not math.isfinite(resume[k]) or resume[k] < 0
                for k in ("lost_seconds", "remaining_seconds")
            )
        ):
            raise ValueError("refine explicit resume budget invalid")
        identity(resume["report_sha256"])
    return {**value, "root": Path(value["root"])}


def protocol(config):
    return {k: v for k, v in config.items() if k not in ("root", "audit", "resume")}


def binding(config, execution, seed):
    return digest({"protocol": protocol(config), "runtime": execution, "seed": seed})


def prepared(config):
    manifest = document(config["root"] / "manifest.json", config["manifest_sha256"])
    data.validate_manifest(manifest)
    audit = document(Path(config["audit"]), config["audit_sha256"])
    if (
        manifest["schema_version"] != views.SCHEMA
        or manifest["originals_per_source"] != {"fit": 256, "development": 64}
        or audit.get("schema_version") != "jpeg-study-independent-audit-v2"
        or audit.get("status") != "completed"
        or audit.get("manifest_sha256") != config["manifest_sha256"]
        or audit.get("representation") != views.contract()
        or audit.get("idct_rows") != {r: len(v) for r, v in manifest["samples"].items()}
        or audit.get("idct_views") != {r: 5 * len(v) for r, v in manifest["samples"].items()}
    ):
        raise ValueError("refine requires the complete checksum-bound five-view audit")
    return manifest


def storage(out):
    """Account only this fresh owned tree; never follow artifact symlinks."""
    used = 0
    if out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise ValueError("refine storage symlink forbidden")
    for path in out.rglob("*"):
        if path.is_symlink():
            raise ValueError("refine artifact symlink forbidden")
        if path.is_file():
            used += path.stat().st_size
    if used > BYTES:
        raise ValueError("refine disk budget exceeded")
    return used


def validate_receipt(value, config, execution, task, parent, out):
    arm, seed, stop = task
    bound = binding(config, execution, seed)
    snapshot = state.load(out / "snapshot.npz", value.get("checkpoint_sha256"), bound)
    meta = snapshot["metadata"]
    expected = state.metadata(
        bound,
        arm,
        seed,
        stop,
        value.get("schedule_sha256"),
        parent["sha256"] if parent else None,
        parent["warmup_sha256"] if parent and arm != "warmup" else None,
    )
    if (
        value.get("status") != "checkpointed"
        or value.get("arm") != arm
        or value.get("seed") != seed
        or value.get("binding") != bound
        or value.get("source_sha256") != config["source_sha256"]
        or value.get("execution") != execution
        or value.get("real_gpu_segment") is not True
        or value.get("metadata") != meta
        or meta != expected
        or value.get("new_updates") != schedule.INTERVAL
        or value.get("total_updates") != stop
    ):
        raise ValueError("refine receipt parent/runtime/update mismatch")
    for key in ("seconds", "mean_cross_entropy", "mean_objective"):
        number = value.get(key)
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError("refine receipt numeric invalid")
    for key in ("initial_sha256", "exposure_sha256", "schedule_sha256"):
        identity(value.get(key))
    return {
        "path": out / "snapshot.npz",
        "sha256": value["checkpoint_sha256"],
        "binding": bound,
        "warmup_sha256": value["checkpoint_sha256"]
        if arm == "warmup" and stop == schedule.WARMUP
        else meta["warmup_sha256"],
    }


def parent_for(latest, arm, seed, stop):
    return latest.get(("warmup" if stop <= schedule.WARMUP + schedule.INTERVAL else arm, seed))


def frozen_schedules(manifest):
    result = {}
    samples = manifest["samples"]["fit"]
    for seed in study.SEEDS:
        for arm in ("warmup", *schedule.ARMS):
            values = schedule.schedule(samples, seed, arm)
            result[(arm, seed)] = (
                schedule.digest(values),
                schedule.exposure_digest(values, len(samples)),
            )
    return result


def validate_exposure(value, task, hashes, initial):
    arm, seed, _ = task
    expected = hashes[(arm, seed)]
    if (value["schedule_sha256"], value["exposure_sha256"]) != expected or (
        value["initial_sha256"] != initial.setdefault(seed, value["initial_sha256"])
    ):
        raise ValueError("refine initialization/schedule/exposure mismatch")


def validate_evaluation(value, config, execution, checkpoint, meta):
    if (
        not isinstance(value, dict)
        or value.get("status") != "completed"
        or value.get("source_sha256") != config["source_sha256"]
        or value.get("execution") != execution
        or value.get("metadata") != meta
        or value.get("checkpoint_sha256") != checkpoint["sha256"]
        or value.get("real_gpu_evaluation") is not True
        or value.get("model_sha256_before") != value.get("model_sha256_after")
    ):
        raise ValueError("refine evaluation runtime/state/checkpoint mismatch")
    identity(value.get("model_sha256_before"))
    roles = ("development",) if meta["arm"] == "warmup" else ("development", "fit")
    for role in roles:
        if value.get(role, {}).get("singleton_batch_parity") is not True:
            raise ValueError("refine evaluation singleton parity missing")
    return value


def estimate(probe, updates, remaining):
    if (
        not isinstance(probe, dict)
        or probe.get("status") != "completed"
        or probe.get("generated_parity") is not True
        or probe.get("save_resume_parity") is not True
    ):
        raise ValueError("refine preflight incomplete")
    for key in ("max_update_seconds", "startup_checkpoint_seconds"):
        value = probe.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (not math.isfinite(value) or value <= 0)
        ):
            raise ValueError("refine timing invalid")
    required = (
        updates * probe["max_update_seconds"] * 1.25
        + math.ceil(updates / schedule.INTERVAL) * probe["startup_checkpoint_seconds"] * 1.25
        + EVALUATION_SECONDS
    )
    return required, required <= remaining


def execute(config, out, *, sources, current_sources, inspect, launch, clock=time.monotonic):
    """Own-host controller also verifies every checkpoint before final qualification."""
    started = clock()
    config = configuration(config, sources)
    for protected in (
        config["root"],
        Path(config["audit"]),
        Path(config.get("resume", {}).get("parent", config["root"])),
    ):
        if out.resolve().is_relative_to(protected.resolve()):
            raise ValueError("refine output cannot modify data or parent")
    data.fresh(out)
    out.mkdir(parents=True)
    receipts: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    inherited_count = 0
    report = {
        "schema_version": "jpeg-refine-report-v1",
        "protocol_sha256": digest(protocol(config)),
        "source_sha256": sources,
        "manifest_sha256": config["manifest_sha256"],
        "audit_sha256": config["audit_sha256"],
        "recipe": recipe(),
        "budget_seconds": SECONDS,
        "budget_bytes": BYTES,
        "prior_seconds": 0.0,
        "prior_output_bytes": 0,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "cpu_fallback": False,
        "learning_gate": False,
        "passing_arms": [],
        "selected_arm": None,
        "unpaired_superiority": False,
        "cloud_spending": 0,
    }

    def remaining():
        return SECONDS - report["prior_seconds"] - (clock() - started)

    def cancelled():
        marker = out / "CANCEL"
        if marker.is_symlink():
            raise ValueError("refine cancellation cannot use symlink")
        return marker.exists()

    def check(reserve=0):
        if cancelled():
            raise InterruptedError
        if remaining() <= reserve:
            raise TimeoutError
        if current_sources() != sources:
            raise ValueError("refine source mutation")
        if storage(out) + report["prior_output_bytes"] > BYTES:
            raise ValueError("refine cumulative disk budget exceeded")

    def call(mode, payload, target, reserve=0):
        check(reserve)
        value = launch(mode, payload, target, min(1830, remaining() - reserve), cancelled)
        if not isinstance(value, dict) or value.get("status") not in {
            "completed",
            "checkpointed",
            "unavailable",
            "cancelled",
        }:
            raise ValueError("refine worker response invalid")
        return value

    try:
        execution = inspect(config["expected_gpu"])
        report["execution"] = execution
        manifest = prepared(config)
        hashes = frozen_schedules(manifest)
        initial: dict[int, str] = {}
        tasks = schedule.segments()
        resume = config.get("resume")
        if resume:
            prior = document(Path(resume["parent"]) / "report.json", resume["report_sha256"])
            if (
                prior.get("schema_version") != report["schema_version"]
                or prior.get("protocol_sha256") != report["protocol_sha256"]
                or prior.get("source_sha256") != sources
                or prior.get("execution") != execution
                or prior.get("status") not in ("cancelled", "unavailable")
                or prior.get("accuracy_qualification") != "unavailable"
                or prior.get("deployed") is not False
                or type(prior.get("seconds")) not in (int, float)
                or not math.isfinite(prior["seconds"])
                or prior["seconds"] < 0
            ):
                raise ValueError("refine resume source/runtime/protocol mismatch")
            report["prior_seconds"] = prior["seconds"] + resume["lost_seconds"]
            prior_bytes = prior.get("cumulative_output_bytes")
            if (
                type(prior_bytes) is not int
                or not 0 <= prior_bytes <= BYTES
                or (prior_bytes < storage(Path(resume["parent"])))
            ):
                raise ValueError("refine resume disk budget invalid")
            report["prior_output_bytes"] = prior_bytes
            if (
                abs(resume["remaining_seconds"] - (SECONDS - report["prior_seconds"])) > 1e-6
                or resume["remaining_seconds"] <= 0
            ):
                raise ValueError("refine resume budget omits spent/lost time")
            inherited = prior.get("receipts")
            if not isinstance(inherited, list) or len(inherited) > len(tasks):
                raise ValueError("refine resume prefix invalid")
            report["parent_report_sha256"] = resume["report_sha256"]
            for index, entry in enumerate(inherited):
                check()
                if storage(out) + prior_bytes + MAX_BYTES + 1024**2 > BYTES:
                    raise ValueError("refine inherited disk budget exceeded")
                task = tasks[index]
                if (
                    not isinstance(entry, dict)
                    or set(entry)
                    != {
                        "index",
                        "arm",
                        "seed",
                        "stop_updates",
                        "receipt_sha256",
                        "checkpoint_sha256",
                    }
                    or (entry["index"], entry["arm"], entry["seed"], entry["stop_updates"])
                    != (index, *task)
                ):
                    raise ValueError("refine inherited order/provenance mismatch")
                folder = Path(resume["parent"]) / f"segment-{index:02d}"
                value = document(folder / "receipt.json", entry["receipt_sha256"])
                parent = parent_for(latest, *task)
                validate_receipt(value, config, execution, task, parent, folder)
                validate_exposure(value, task, hashes, initial)
                if entry["checkpoint_sha256"] != value["checkpoint_sha256"]:
                    raise ValueError("refine inherited snapshot checksum mismatch")
                target = out / f"segment-{index:02d}"
                target.mkdir()
                for name, maximum in (("snapshot.npz", MAX_BYTES), ("receipt.json", 65536)):
                    with regular_open(folder / name) as src, (target / name).open("xb") as dst:
                        size = 0
                        while raw := src.read(1024**2):
                            size += len(raw)
                            if size > maximum:
                                raise ValueError("refine inherited copy exceeds bound")
                            dst.write(raw)
                latest[task[:2]] = validate_receipt(value, config, execution, task, parent, target)
                receipts.append(entry)
            inherited_count = len(receipts)
        write_document(out / "start.json", report)
        audit = call("audit", {"config": config}, out)
        if (
            audit.get("status") != "completed"
            or audit.get("source_sha256") != sources
            or audit.get("execution") != execution
            or audit.get("manifest_sha256") != config["manifest_sha256"]
            or audit.get("idct_rows") != {r: len(v) for r, v in manifest["samples"].items()}
            or audit.get("idct_views") != {r: 5 * len(v) for r, v in manifest["samples"].items()}
        ):
            raise ValueError("refine fresh independent audit incomplete")
        write_document(out / "fresh-audit.json", audit)
        probe = call("preflight", {"config": config}, out)
        write_document(out / "preflight.json", probe)
        if probe.get("status") != "completed":
            report.update(status="unavailable", reason=probe.get("reason", "preflight_unavailable"))
        else:
            if probe.get("source_sha256") != sources or probe.get("execution") != execution:
                raise ValueError("refine preflight source/runtime mismatch")
            required, fits = estimate(
                probe, (len(tasks) - len(receipts)) * schedule.INTERVAL, remaining()
            )
            report["estimated_remaining_seconds"] = required
            if not fits:
                report.update(status="unavailable", reason="whole_matrix_exceeds_remaining_budget")
            else:
                # Replay endpoint evaluations on resume; no new model selection points.
                for index, task in enumerate(tasks):
                    arm, seed, stop = task
                    target = out / f"segment-{index:02d}"
                    if index >= len(receipts):
                        check(EVALUATION_SECONDS)
                        if (
                            storage(out) + report["prior_output_bytes"] + MAX_BYTES + 1024**2
                            > BYTES
                        ) or (shutil.disk_usage(out).free < MAX_BYTES + 1024**2):
                            raise ValueError("refine checkpoint disk budget unavailable")
                        parent = parent_for(latest, *task)
                        payload = {
                            "config": config,
                            "task": task,
                            "execution": execution,
                            "binding": binding(config, execution, seed),
                            "parent": parent,
                        }
                        value = call("train", payload, target, EVALUATION_SECONDS)
                        if value.get("status") != "checkpointed":
                            write_document(out / f"failure-{index:02d}.json", value)
                            report.update(
                                status="unavailable", reason=value.get("reason", "segment_failed")
                            )
                            break
                        checkpoint = validate_receipt(
                            value, config, execution, task, parent, target
                        )
                        validate_exposure(value, task, hashes, initial)
                        checksum = write_document(target / "receipt.json", value)
                        latest[(arm, seed)] = checkpoint
                        receipts.append(
                            {
                                "index": index,
                                "arm": arm,
                                "seed": seed,
                                "stop_updates": stop,
                                "receipt_sha256": checksum,
                                "checkpoint_sha256": checkpoint["sha256"],
                            }
                        )
                        write_document(
                            out / f"progress-{index:02d}.json",
                            {
                                "verified_segments": len(receipts),
                                "verified_updates": len(receipts) * schedule.INTERVAL,
                                "seconds": SECONDS - remaining(),
                                "remaining_seconds": max(0, remaining()),
                            },
                        )
                    else:
                        value, _ = read(target / "receipt.json")
                        checkpoint = {
                            "path": target / "snapshot.npz",
                            "binding": value["binding"],
                            "sha256": value["checkpoint_sha256"],
                        }
                    if stop in (schedule.WARMUP, schedule.FINAL):
                        value = validate_evaluation(
                            call("evaluate", {"config": config, "checkpoint": checkpoint}, target),
                            config,
                            execution,
                            checkpoint,
                            value["metadata"],
                        )
                        evaluations.append(value)
                        write_document(target / "evaluation.json", value)
                if "status" not in report:
                    # Revalidate disk evidence, schedules and ancestry before applying gates.
                    verify(config, out, execution, receipts, evaluations, deadline=check)
                    check()
                    report.update(
                        metrics.compare(manifest, evaluations), status="completed", reason=None
                    )
    except (InterruptedError, KeyboardInterrupt):
        report.update(status="cancelled", reason="user_cancelled")
    except TimeoutError:
        report.update(status="unavailable", reason="budget_or_worker_timeout")
    except WorkerFailure as exc:
        report.update(exc.unavailable())
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        # No host paths, commands, dependency output or secrets in public failures.
        report.update(status="unavailable", reason=getattr(exc, "reason", "refine_contract_failed"))
    finally:
        seconds = report["prior_seconds"] + clock() - started
        report.update(
            receipts=receipts,
            evaluations=evaluations,
            seconds=seconds,
            remaining_seconds=max(0, SECONDS - seconds),
            verified_updates=len(receipts) * schedule.INTERVAL,
            new_verified_updates=(len(receipts) - inherited_count) * schedule.INTERVAL,
            status=report.get("status", "unavailable"),
        )
        if current_sources() != sources:
            report.update(status="unavailable", reason="source_mutation")
        if seconds >= SECONDS:
            report.update(status="unavailable", reason="controls_budget_exhausted")
        if report["status"] != "completed" or current_sources() != sources:
            report.update(
                learning_gate=False, passing_arms=[], selected_arm=None, unpaired_superiority=False
            )
        try:
            used = storage(out) + report["prior_output_bytes"]
            report["cumulative_output_bytes"] = used
            for _ in range(4):
                size = len((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
                report["cumulative_output_bytes"] = used + size
            if report["cumulative_output_bytes"] > BYTES:
                report.update(
                    status="unavailable",
                    reason="disk_budget_exhausted",
                    learning_gate=False,
                    selected_arm=None,
                    passing_arms=[],
                    unpaired_superiority=False,
                )
        except (OSError, ValueError):
            report.update(
                status="unavailable",
                reason="artifact_integrity_failed",
                cumulative_output_bytes=None,
                learning_gate=False,
                selected_arm=None,
                passing_arms=[],
                unpaired_superiority=False,
            )
        write_document(out / "report.json", report)
    return report


def verify_data(config, manifest, deadline=lambda: None):
    for role in ("fit", "development"):
        with data.Reader(
            config["root"], config["manifest_sha256"], role, deadline=deadline
        ) as reader:
            for row in manifest["samples"][role]:
                deadline()
                read_bytes(
                    config["root"] / row["path"], row["sha256"], row["size"], maximum=256 * 1024
                )
            reader.verify()


def verify(config, out, execution, receipts, evaluations, deadline=lambda: None):
    """Read-only verifier; snapshot presence alone can never complete a matrix."""
    if len(receipts) != 56 or len(evaluations) != 6:
        raise ValueError("refine incomplete required matrix")
    manifest = prepared(config)
    verify_data(config, manifest, deadline)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    initial: dict[int, str] = {}
    exposures: dict[int, str] = {}
    hashes = frozen_schedules(manifest)
    for index, task in enumerate(schedule.segments()):
        deadline()
        arm, seed, stop = task
        entry = receipts[index]
        if {k: entry[k] for k in ("index", "arm", "seed", "stop_updates")} != {
            "index": index,
            "arm": arm,
            "seed": seed,
            "stop_updates": stop,
        }:
            raise ValueError("refine receipt prefix/order mismatch")
        folder = out / f"segment-{index:02d}"
        value = document(folder / "receipt.json", entry["receipt_sha256"])
        parent = parent_for(latest, *task)
        checkpoint = validate_receipt(value, config, execution, task, parent, folder)
        validate_exposure(value, task, hashes, initial)
        if (
            entry["checkpoint_sha256"] != checkpoint["sha256"]
            or value["initial_sha256"] != initial.setdefault(seed, value["initial_sha256"])
            or (value["schedule_sha256"], value["exposure_sha256"]) != hashes[(arm, seed)]
            or value["exposure_sha256"] != exposures.setdefault(seed, value["exposure_sha256"])
        ):
            raise ValueError("refine initialization/schedule/exposure mismatch")
        latest[(arm, seed)] = checkpoint
        if stop in (schedule.WARMUP, schedule.FINAL):
            evaluation, _ = read(folder / "evaluation.json", maximum=1024**2)
            validate_evaluation(evaluation, config, execution, checkpoint, value["metadata"])
            if evaluation not in evaluations:
                raise ValueError("refine final evaluation provenance mismatch")
    storage(out)
    return {
        "status": "completed",
        "verified_checkpoints": 56,
        "verified_updates": schedule.TOTAL,
        "verified_evaluations": 6,
    }
