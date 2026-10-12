"""Explicit bounded JPEG study preparation and serial own-GPU controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core import (
    jpeg_study,
    jpeg_study_audit,
    jpeg_study_data,
    jpeg_study_entropy,
    jpeg_study_views,
    srnet_cuda,
)
from core.jpeg_scale import identity
from core.jpeg_timing import write_document
from core.srnet_stream import deadline_after, document, regular_open
from steganography.research_batch_probe import sources as capacity_sources
from steganography.research_srnet_stream import fresh
from steganography.research_srnet_stream import snapshot as previous_sources

ROOT = Path(__file__).resolve().parents[1]
MAX_REPORT = 256 * 1024
MAX_SIMULATION_REPORT = 768 * 1024


def sources():
    result = {**previous_sources(), **capacity_sources()}
    for name in (
        "core/srnet_samepool.py", "core/jpeg_study_optimizer.py", "core/jpeg_study.py",
        "core/jpeg_study_data.py", "core/jpeg_study_state.py", "core/jpeg_study_audit.py",
        "core/jpeg_study_entropy.py", "core/jpeg_study_views.py",
        "core/srnet_checkpoint.py", "core/jpeg_timing.py",
        "steganography/research_jpeg_study.py", "docs/JPEG_LEARNING_STUDY.md",
        "docs/JPEG_FIVE_VIEW_STUDY.md",
        "steganography/benchmarking/metrics.py",
    ):
        with regular_open(ROOT / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("study source closure exceeds size bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def configuration(value):
    if not isinstance(value, dict) or value.get("schema_version") != "jpeg-study-config-v1":
        raise ValueError("study configuration schema invalid")
    action = value.get("action")
    keys = ({"inputs", "reserved_manifests"} if action == "prepare" else
            {"root", "manifest_sha256", "audit", "audit_sha256"})
    optional = {"resume"} if action == "controls" else {"representation"}
    if (action not in {"prepare", "controls"}
        or not keys | {"schema_version", "action"} <= set(value)
        or set(value) - keys - {"schema_version", "action"} - optional):
        raise ValueError("study configuration fields/action invalid")
    if action == "prepare":
        if value.get("representation", jpeg_study_views.CENTER) not in (
            jpeg_study_views.CENTER, jpeg_study_views.FIVE
        ):
            raise ValueError("study representation invalid")
        if (not isinstance(value["inputs"], list) or len(value["inputs"]) != 2
            or not isinstance(value["reserved_manifests"], list)):
            raise ValueError("study configuration input list invalid")
        for entry in value["inputs"]:
            if not isinstance(entry, dict) or set(entry) != {"root", "sha256", "source_group"}:
                raise ValueError("study configuration source fields invalid")
            identity(entry["sha256"])
            if not isinstance(entry["root"], str) or not entry["root"]:
                raise ValueError("study configuration root invalid")
        for entry in value["reserved_manifests"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("study configuration reservation fields invalid")
            identity(entry["sha256"])
    else:
        for key in ("manifest_sha256", "audit_sha256"):
            identity(value[key])
        if any(not isinstance(value[k], str) or not value[k] for k in ("root", "audit")):
            raise ValueError("study configuration data paths invalid")
        resumes = value.get("resume", [])
        if not isinstance(resumes, list) or len(resumes) > 8:
            raise ValueError("study resume list invalid")
        seen = set()
        for entry in resumes:
            if (not isinstance(entry, dict) or set(entry) != {"arm", "seed", "path", "sha256"}
                or entry["arm"] not in jpeg_study.ARMS
                or type(entry["seed"]) is not int or entry["seed"] not in jpeg_study.SEEDS
                or not isinstance(entry["path"], str) or not entry["path"]
                or (entry["arm"], entry["seed"]) in seen):
                raise ValueError("study resume entry invalid")
            identity(entry["sha256"])
            seen.add((entry["arm"], entry["seed"]))
    return value


def child(command, *, timeout, cancelled=lambda: False, payload=None, maximum=MAX_REPORT):
    """Owned process group, bounded output, kill and reap on timeout/cancel."""
    if payload is not None and len(payload) > 1024**2:
        raise ValueError("study child input exceeds bounds")
    if maximum not in (MAX_REPORT, MAX_SIMULATION_REPORT):
        raise ValueError("study child output limit invalid")
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as input_file:
        if payload is not None:
            input_file.write(payload)
            input_file.seek(0)
        process = subprocess.Popen(  # noqa: S603 - fixed module and arguments, no shell
            command, cwd=ROOT, stdin=input_file if payload is not None else subprocess.DEVNULL,
            stdout=output, stderr=subprocess.DEVNULL, start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                 "NUMBA_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError("study cancelled")
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("study worker deadline exceeded")
                if os.fstat(output.fileno()).st_size > maximum:
                    raise ValueError("study worker output exceeded bounds")
                time.sleep(0.1)
        finally:
            # Also kill descendants after a normally exited parent.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output.seek(0)
        raw = output.read(maximum + 1)
        if process.returncode or not 0 < len(raw) <= maximum:
            raise ValueError("study worker failed/output exceeded bounds")
        return raw


def generate(raw, lineage, *, deadline=lambda: None):
    def checkpoint():
        deadline()
        return False

    encoded = child([sys.executable, "-m", "steganography.research_jpeg_study",
                     "--worker", "simulate", "--lineage", lineage], timeout=90, payload=raw,
                    maximum=MAX_SIMULATION_REPORT, cancelled=checkpoint)
    result = json.loads(encoded)
    import base64

    if isinstance(result, dict) and result.get("status") == "unavailable":
        raise ImportError("study simulation dependency unavailable")
    if not isinstance(result, list) or len(result) != 2:
        raise ValueError("study simulation response invalid")
    return [base64.b64decode(value, validate=True) for value in result]


def simulate(raw, lineage):
    import base64

    import conseal
    import jpeglib

    if importlib.metadata.version("conseal") != "2025.11":
        raise ValueError("study requires pinned conseal 2025.11")
    identity(lineage)
    if not 0 < len(raw) <= 1024**2 or hashlib.sha256(raw).hexdigest() != lineage:
        raise ValueError("study simulation original identity/byte bounds")
    with (Image.open(io.BytesIO(raw)) as image,
          tempfile.TemporaryDirectory(prefix="jpeg-study-") as d):
        if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
            raise ValueError("study simulation requires bounded grayscale PGM")
        cover, stego = Path(d) / "cover.jpg", Path(d) / "stego.jpg"
        with cover.open("xb") as stream:
            image.save(stream, format="JPEG", quality=75, optimize=False)
        jpeglib.version.set("6b")
        jpeg = jpeglib.read_dct(str(cover))
        jpeg.load()
        coefficients, qt = jpeg.Y.copy(), jpeg.get_component_qt(0).copy()
        pixels = jpeglib.read_spatial(str(cover), jpeglib.JCS_GRAYSCALE).spatial[..., 0]
        seed = int.from_bytes(hashlib.sha256(f"jpeg-study:{lineage}".encode()).digest()[:4], "big")
        # Same pinned cost maps, probabilities, layout and RNG as the upstream
        # convenience simulator. Its iteration-warning alone is not a residual.
        rhos = conseal.juniward.compute_cost_adjusted(x0=pixels, y0=coefficients, qt=qt)
        maps = tuple(conseal.tools.dct.jpeglib_to_jpegio(rho) for rho in rhos)
        nzac = int(np.count_nonzero(coefficients) - np.count_nonzero(coefficients[..., 0, 0]))
        if nzac <= 0:
            raise ValueError("study embedding requires nonzero AC coefficients")
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always", RuntimeWarning)
            probabilities, _ = conseal.simulate._ternary.probability(maps, 0.4, nzac)
        if any(str(w.message) != "optimization might not have converged" for w in notices):
            raise ValueError("study embedding has an unaudited numerical warning")
        choices = (*probabilities, 1 - probabilities[0] - probabilities[1])
        if any(not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) for p in choices):
            raise ValueError("study embedding probabilities invalid")
        entropy = sum(float(np.sum(-p[p > 0] * np.log2(p[p > 0]))) for p in choices)
        target = int(np.round(0.4 * nzac))
        if target <= 0:
            raise ValueError("study embedding entropy target invalid")
        if abs(entropy - target) > 0.001 * target:
            probabilities = jpeg_study_entropy.refine(maps, target)
        delta_2d = conseal.simulate._ternary.simulate(probabilities, seed=seed)
        modified = coefficients + conseal.tools.dct.jpegio_to_jpeglib(delta_2d)
        delta = modified.astype(np.int64) - coefficients.astype(np.int64)
        if not np.any(delta) or np.max(np.abs(delta)) > 1:
            raise ValueError("study simulation coefficient changes invalid")
        jpeg.Y = modified
        jpeg.write_dct(str(stego))
        check = jpeglib.read_dct(str(stego))
        check.load()
        if (not np.array_equal(check.Y, modified)
            or not np.array_equal(check.get_component_qt(0), qt)):
            raise ValueError("study simulation coefficient round-trip failed")
        result = []
        for path in (cover, stego):
            with regular_open(path) as stream:
                data = stream.read(256 * 1024 + 1)
            if not 0 < len(data) <= 256 * 1024:
                raise ValueError("study simulation encoded output exceeds bounds")
            result.append(base64.b64encode(data).decode("ascii"))
        return result


def worker(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (96 * 1024**2, 96 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    before = sources()
    try:
        if args.worker == "simulate":
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
            resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
            return simulate(sys.stdin.buffer.read(1024**2 + 1), args.lineage)
        raw_config = sys.stdin.buffer.read(65537)
        if len(raw_config) > 65536:
            raise ValueError("study configuration exceeds bounds")
        config = configuration(json.loads(raw_config))
        if args.worker == "audit":
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
            result = jpeg_study_audit.audit(
                Path(config["root"]), config["manifest_sha256"], deadline=deadline_after(1200)
            )
        else:
            if args.out is None:
                raise ValueError("study fit requires a fresh output")
            fresh(args.out)
            srnet_cuda.inspect()
            args.out.mkdir()
            resume = next((r for r in config.get("resume", [])
                           if r["arm"] == args.arm and r["seed"] == args.seed), None)
            result = jpeg_study.fit(
                Path(config["root"]), config["manifest_sha256"], args.out,
                args.arm, args.seed, args.rows, before,
                resume={"path": Path(resume["path"]), "sha256": resume["sha256"]}
                if resume else None,
            )
    except srnet_cuda.CUDAUnavailable as exc:
        result = {"status": "unavailable", "reason": exc.reason}
    except ImportError:
        result = {"status": "unavailable", "reason": "optional_dependency_unavailable"}
    except TimeoutError:
        result = {"status": "unavailable", "reason": "trial_timeout"}
    except InterruptedError:
        result = {"status": "cancelled", "reason": "user_cancelled"}
    except Exception as exc:
        # Invalid input must not import an absent optional backend just to
        # classify the exception. A real CUDA OOM already has Torch loaded.
        cuda = getattr(sys.modules.get("torch"), "cuda", None)
        oom = getattr(cuda, "OutOfMemoryError", None)
        if oom is None or not isinstance(exc, oom):
            raise
        result = {"status": "unavailable", "reason": "gpu_allocator_out_of_memory"}
    if sources() != before:
        raise ValueError("study worker source mutation")
    return {**result, "source_sha256": before}


def run(config_path, out):
    with regular_open(config_path) as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("study configuration exceeds bounds")
    config = configuration(json.loads(raw))
    fresh(out)
    before = sources()
    started = time.monotonic()
    report = {"schema_version": "jpeg-learning-study-v1", "action": config["action"],
              "config_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": before,
              "accuracy_qualification": "unavailable", "deployed": False,
              "max_seconds": 28800, "max_new_bytes": jpeg_study_data.MAX_BYTES,
              "cpu_fallback": False, "cloud_spending": 0}
    if config.get("representation") == jpeg_study_views.FIVE:
        report["representation"] = jpeg_study_views.contract()
    if config["action"] == "prepare":
        limit = deadline_after(1800)
        progress = {"original_sha256": None}

        def cancelled():
            marker = out / "CANCEL"
            if marker.is_symlink():
                raise ValueError("study cancellation marker cannot use symlink")
            return marker.exists()

        def checkpoint():
            limit()
            if cancelled():
                raise InterruptedError("study preparation cancelled")

        def producer(raw, lineage):
            progress["original_sha256"] = lineage
            return generate(raw, lineage, deadline=checkpoint)

        try:
            result = jpeg_study_data.prepare(
                config["inputs"], config["reserved_manifests"], out,
                generator=producer,
                deadline=checkpoint,
                **({"representation": config["representation"]} if "representation" in config
                   else {}),
            )
        except (ImportError, TimeoutError, InterruptedError, KeyboardInterrupt, ValueError) as exc:
            if not out.exists():
                out.mkdir(parents=True, exist_ok=False)
            reasons = {ImportError: "optional_dependency_unavailable",
                       TimeoutError: "preparation_timeout", InterruptedError: "user_cancelled",
                       ValueError: "preparation_contract_failed",
                       KeyboardInterrupt: "user_cancelled"}
            was_cancelled = isinstance(exc, (InterruptedError, KeyboardInterrupt))
            report.update(status="cancelled" if was_cancelled else "unavailable",
                          reason=reasons[type(exc)], optimizer_updates=0, real_model_trained=False,
                          failed_original_sha256=progress["original_sha256"],
                          seconds=time.monotonic() - started)
            write_document(out / "report.json", report)
            return report
        manifest_sha = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
        audit_config = {"schema_version": "jpeg-study-config-v1", "action": "controls",
                        "root": str(out.absolute()), "manifest_sha256": manifest_sha,
                        "audit": "unused", "audit_sha256": "0" * 64}
        try:
            proof = json.loads(child(
                [sys.executable, "-m", "steganography.research_jpeg_study", "--worker", "audit"],
                timeout=1200, payload=json.dumps(audit_config).encode(),
                cancelled=cancelled,
            ))
            audit_sha = write_document(out / "independent-audit.json", proof)
            if proof.get("status") != "completed" or proof.get("source_sha256") != before:
                report.update(status="unavailable", reason="independent_audit_incomplete",
                              manifest_sha256=manifest_sha, audit_sha256=audit_sha,
                              optimizer_updates=0, real_model_trained=False)
            else:
                report.update(status="completed", manifest_sha256=manifest_sha,
                              audit_sha256=audit_sha,
                              selected_originals=sum(
                                  len(v) for v in result["samples"].values()) // 2,
                              optimizer_updates=0, real_model_trained=False)
        except (ImportError, TimeoutError, InterruptedError, KeyboardInterrupt, ValueError) as exc:
            was_cancelled = isinstance(exc, (InterruptedError, KeyboardInterrupt))
            report.update(status="cancelled" if was_cancelled else "unavailable",
                          reason="user_cancelled" if was_cancelled else "independent_audit_failed",
                          manifest_sha256=manifest_sha,
                          optimizer_updates=0, real_model_trained=False)
    else:
        out.mkdir(parents=True, exist_ok=False)
        try:
            srnet_cuda.inspect()
        except (srnet_cuda.CUDAUnavailable, ImportError) as exc:
            report.update(status="unavailable", reason=getattr(
                exc, "reason", "optional_dependency_unavailable"
            ), optimizer_updates=0, real_model_trained=False)
        else:
            try:
                report.update(controls(config, out, before, started))
            except KeyboardInterrupt:
                report.update(status="cancelled", reason="user_cancelled",
                              trial_reports=sorted(p.name for p in out.glob("trial-*.json")))
    if sources() != before:
        raise ValueError("study sources changed")
    report["seconds"] = time.monotonic() - started
    write_document(out / "report.json", report)
    return report


def controls(config, out, before, started):
    proof = document(Path(config["audit"]), config["audit_sha256"])
    manifest = document(Path(config["root"]) / "manifest.json", config["manifest_sha256"])
    jpeg_study_data.validate_manifest(manifest)
    five = manifest.get("schema_version") == jpeg_study_views.SCHEMA
    representation = jpeg_study_views.FIVE if five else jpeg_study_views.CENTER
    if (proof.get("schema_version") != ("jpeg-study-independent-audit-v2" if five
                                        else "jpeg-study-independent-audit-v1")
        or proof.get("status") != "completed"
        or proof.get("manifest_sha256") != config["manifest_sha256"]
        or proof.get("source_sha256") != before
        or proof.get("idct_rows") != {r: len(v) for r, v in manifest["samples"].items()}
        or (five and (proof.get("representation") != jpeg_study_views.contract()
                      or proof.get("idct_views") != {
                          r: 5 * len(v) for r, v in manifest["samples"].items()}))
        or manifest["originals_per_source"] != {"fit": 256, "development": 64}):
        raise ValueError("study controls require complete bound independent audit")
    trials: list[dict[str, Any]] = []

    def cancelled():
        marker = out / "CANCEL"
        if marker.is_symlink():
            raise ValueError("study cancellation marker cannot use symlink")
        return marker.exists()

    probes: list[dict[str, Any]] = []
    for rows in (8, 16, 32):
        try:
            probe = json.loads(child([
                sys.executable, "-m", "steganography.research_batch_probe", "--worker",
                "--rows", str(rows),
            ], timeout=180, cancelled=cancelled))
        except TimeoutError:
            probe = {"status": "unavailable", "reason": "worker_timeout"}
        except InterruptedError:
            return {"status": "cancelled", "trials": trials, "capacity": probes}
        if (not isinstance(probe, dict) or probe.get("status") not in {"completed", "unavailable"}
            or (probe.get("status") == "completed" and (probe.get("rows") != rows
                or probe.get("source_sha256") != capacity_sources()))):
            raise ValueError("study capacity response invalid")
        probe["rows"] = rows
        probes.append(probe)
        write_document(out / f"capacity-{rows}.json", probe)
    available = [p["rows"] for p in probes if p["status"] == "completed"]

    for seed in jpeg_study.SEEDS:
        for arm in jpeg_study.ARMS:
            if cancelled():
                return {"status": "cancelled", "trials": trials}
            if time.monotonic() - started > 4 * 3600:
                return {"status": "unavailable", "reason": "controls_budget_exhausted",
                        "trials": trials}
            if arm == "genuine-batch" and not available:
                trials.append({"arm": arm, "seed": seed, "status": "unavailable",
                               "reason": "no_genuine_batch_fits"})
                continue
            rows = max(available) if arm == "genuine-batch" else 4
            trial_out = out / f"{arm}-{seed}"
            try:
                raw = child([
                    sys.executable, "-m", "steganography.research_jpeg_study", "--worker", "fit",
                    "--out", str(trial_out), "--arm", arm, "--seed", str(seed), "--rows", str(rows),
                ], timeout=min(1830, max(1, 14400 - (time.monotonic() - started))),
                    payload=json.dumps(config).encode(), cancelled=cancelled)
                value = json.loads(raw)
            except TimeoutError:
                value = {"status": "unavailable", "reason": "worker_timeout"}
            except InterruptedError:
                return {"status": "cancelled", "trials": trials}
            if not isinstance(value, dict) or value.get("status") not in {
                "completed", "unavailable", "cancelled"
            }:
                raise ValueError("study trial response invalid")
            if value.get("source_sha256", before) != before:
                raise ValueError("study trial source binding mismatch")
            if value.get("status") == "completed":
                if value.get("recipe") != jpeg_study.recipe(
                    arm, seed, rows, representation=representation
                ) or (
                    value.get("presented_rows") != jpeg_study.PRESENTED_ROWS
                ) or not value.get("real_gpu_fit"):
                    raise ValueError("study incomplete/non-GPU trial cannot complete")
                if value.get("source_sha256") != before:
                    raise ValueError("study completed trial requires source binding")
                baseline = next((t for t in trials if t["seed"] == seed and t["arm"] == "legacy"
                                 and t["status"] == "completed"), None)
                value["cells"] = jpeg_study.statistics(
                    manifest["samples"]["development"], value["scores"],
                    baseline=baseline["scores"] if baseline else None,
                )
                if (baseline and value["initial_learned_sha256"]
                    != baseline["initial_learned_sha256"]):
                    raise ValueError("study initialization not matched")
                if (baseline and value["schedule_sha256"] != baseline["schedule_sha256"]):
                    raise ValueError("study exposure/view/D4 schedule not matched")
            value.update(arm=arm, seed=seed)
            trials.append(value)
            write_document(out / f"trial-{arm}-{seed}.json", value)
    winners = [arm for arm in jpeg_study.ARMS[1:] if all(
        any(t["arm"] == arm and t["seed"] == seed and t["status"] == "completed"
            and all(c["learning_gate"] for c in t["cells"]) for t in trials)
        for seed in jpeg_study.SEEDS
    )]
    completed = all(t["status"] == "completed" for t in trials)
    return {"status": "completed" if completed else "unavailable", "trials": trials,
            "reason": None if completed else "incomplete_required_controls", "capacity": probes,
            "learning_gate": bool(winners), "passing_arms": winners,
            **({"representation": jpeg_study_views.contract()} if five else {}),
            "baseline_stage": "not_started", "adaptation_stage": "not_started",
            "independent_source_evaluation": "not_started"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", choices=("simulate", "audit", "fit"), help=argparse.SUPPRESS)
    parser.add_argument("--lineage", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=jpeg_study.ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=jpeg_study.SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, choices=(4, 8, 16, 32), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return 0
    if args.config is None or args.out is None:
        parser.error("--config and --out are required")
    result = run(args.config, args.out)
    print(json.dumps({"status": result["status"], "action": result["action"]}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
