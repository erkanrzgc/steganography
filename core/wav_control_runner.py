"""Source-bound generated WAV producer and one-shot process supervisor."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core import wav_control_gate as gate
from core import wav_pilot, wav_pilot_training, wav_pilot_v3
from core.nonjpeg_data import regular_read, write

PARENT = "benchmarks/wav-full-control-protocol-20261021.json"
CORPUS = "benchmarks/wav-generated-corpus-20261021.json"
PROTOCOL_ID = "wav-generated-full-control-20261021-runner-v2"
CORPUS_SEED = 2026102101
SECONDS = 3600
RESERVE_SECONDS = 10
MAX_JSON = 1024**2
MAX_SCRATCH = 512 * 1024**2
SOURCES = (
    *gate.SOURCES,
    PARENT,
    CORPUS,
    "core/wav_control_runner.py",
    "steganography/research_wav_control_run.py",
    "tests/test_wav_control_runner.py",
    "docs/WAV_CONTROL_RUNNER_PROTOCOL_20261021.md",
    "docs/WAV_CONTROL_RUNNER_EVIDENCE_AMENDMENT_20261021.md",
)


def encoded(value: dict[str, Any]) -> bytes:
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if len(raw) > MAX_JSON:
        raise ValueError("control JSON size limit")
    return raw


def save(path: Path, value: dict[str, Any]) -> None:
    write(path, encoded(value))


def progress(out: Path, value: dict[str, Any]) -> None:
    """Replace only the owned progress file; its count is a confirmed lower bound."""
    temporary = out / "progress.next.json"
    save(temporary, value)
    target = out / "progress.json"
    if target.is_symlink():
        raise ValueError("progress symlink")
    os.replace(temporary, target)


def check_deadline(expires: float) -> None:
    if time.monotonic() >= expires:
        raise TimeoutError("generated control deadline")


def load_protocol(root: Path, path: Path):
    parent, parent_sha = gate.read_protocol(root, root / PARENT)
    protocol, digest = gate.read_document(path)
    if (
        protocol.get("schema_version") != "wav-control-runner-protocol-v1"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("parent_protocol_sha256") != parent_sha
        or protocol.get("generator_seed") != CORPUS_SEED
        or protocol.get("reserve_seconds") != RESERVE_SECONDS
        or protocol.get("real_fit_authorized") is not False
        or not isinstance(protocol.get("source_closure_sha256"), dict)
        or set(protocol["source_closure_sha256"]) != set(SOURCES)
    ):
        raise ValueError("runner protocol binding")
    for name, expected in protocol["source_closure_sha256"].items():
        if gate.source_hash(root / name) != expected:
            raise ValueError("runner source binding")
    return protocol, digest, parent, parent_sha


def lengths(root: Path) -> dict[str, list[int]]:
    metadata, _ = gate.read_document(root / "benchmarks/wav-audiomnist-preparation-20261017.json")
    rows = metadata.get("files")
    if not isinstance(rows, list) or len(rows) != 600:
        raise ValueError("exact length metadata required")
    result: dict[str, list[int]] = {"train": [], "development": []}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("role") not in result
            or type(row.get("frames")) is not int
            or not 160 <= row["frames"] <= 220500
        ):
            raise ValueError("closed generated length schedule")
        result[row["role"]].append(row["frames"])
    if len(result["train"]) != 400 or len(result["development"]) != 200:
        raise ValueError("generated length role counts")
    return result


def speech_like(rng, frames: int) -> np.ndarray:
    t = np.arange(frames) / 48000
    f0 = rng.uniform(90, 250) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(2, 6) * t))
    phase = 2 * np.pi * np.cumsum(f0) / 48000
    voice = sum(
        rng.uniform(0.2, 1) / k * np.sin(k * phase + rng.uniform(0, 6.3)) for k in range(1, 12)
    )
    start, stop = int(frames * rng.uniform(0.15, 0.3)), int(frames * rng.uniform(0.7, 0.85))
    envelope = np.zeros(frames)
    envelope[start:stop] = np.hanning(stop - start) ** 0.5
    values = rng.uniform(500, 2800) * envelope * voice / np.abs(voice).max()
    values += rng.normal(0, rng.uniform(0.6, 3), frames)
    return np.clip(np.round(values), -32768, 32767).astype("<i2")


def build_corpus(root: Path, schedule: dict[str, list[int]], expires: float):
    if (
        set(schedule) != {"train", "development"}
        or len(schedule["train"]) != 400
        or len(schedule["development"]) != 200
        or any(
            type(n) is not int or not 160 <= n <= 220500
            for values in schedule.values()
            for n in values
        )
    ):
        raise ValueError("generated corpus schedule")
    rng = np.random.default_rng(CORPUS_SEED)
    records, pcm_hashes = [], set()
    total = oracles = 0
    for role in ("train", "development"):
        for index, frames in enumerate(schedule[role]):
            check_deadline(expires)
            pcm = speech_like(rng, frames).tobytes()
            pcm_sha = hashlib.sha256(pcm).hexdigest()
            if pcm_sha in pcm_hashes:
                raise ValueError("duplicate generated PCM")
            pcm_hashes.add(pcm_sha)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(48000)
                audio.writeframes(pcm)
            raw = buffer.getvalue()
            total += len(raw)
            if total > MAX_SCRATCH:
                raise ValueError("generated scratch bound")
            group = f"generated-{role}-{index // 20:02d}"
            relative = f"data/{group}/{index:04d}.wav"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write(path, raw)
            source = {
                "relative_source_path": relative,
                "role": role,
                "speaker_group": group,
                "parent_recording_id": None,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for recipe in wav_pilot.RECIPES:
                check_deadline(expires)
                wav_pilot.materialize(root, source, recipe)
                oracles += 1
            records.append({**source, "frames": frames, "decoded_pcm_sha256": pcm_sha})
    manifest = {
        "schema_version": "wav-generated-control-corpus-v1",
        "generator_seed": CORPUS_SEED,
        "numpy_version": np.__version__,
        "evidence": {**gate.CORPUS_EVIDENCE, "payload_oracles_passed": oracles},
        "generated_bytes": total,
        "files": records,
    }
    save(root / "manifest.json", manifest)
    return records, manifest


def auc(negative: np.ndarray, positive: np.ndarray) -> float:
    if any(
        v.ndim != 1
        or not 1 <= v.size <= 3200
        or not np.isfinite(v).all()
        or np.any((v < 0) | (v > 1))
        for v in (negative, positive)
    ):
        raise ValueError("bounded AUC scores")
    ordered = np.sort(negative)
    wins = np.searchsorted(ordered, positive, side="left")
    ties = np.searchsorted(ordered, positive, side="right") - wins
    # Integer half-win counts avoid per-row rounding at the frozen 0.60/0.70 gates.
    return float(np.sum(2 * wins + ties) / (2 * ordered.size * positive.size))


def score(torch, root: Path, rows, model, expires: float, *, device="cuda:0") -> np.ndarray:
    prior = model.training
    model.eval()
    values = []
    try:
        with torch.inference_mode():
            for row in rows:
                check_deadline(expires)
                raw = wav_pilot.materialize(root, row["source"], row["recipe"])
                x = torch.from_numpy(wav_pilot_v3.inputs(raw)).to(
                    device=device, dtype=torch.float32
                )
                value = float(torch.softmax(model(x), 1)[0, 1].cpu())
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("nonfinite control score")
                values.append(value)
    finally:
        model.train(prior)
    return np.asarray(values, dtype=np.float64)


def metrics(rows, scores):
    covers = scores[[i for i, row in enumerate(rows) if row["recipe"] is None]]
    stegos = scores[[i for i, row in enumerate(rows) if row["recipe"] is not None]]
    cells = []
    for method, rate, plane in wav_pilot.RECIPES:
        chosen = scores[[i for i, row in enumerate(rows) if row["recipe"] == (method, rate, plane)]]
        cells.append(
            {"method": method, "rate_percent": rate, "plane": plane, "roc_auc": auc(covers, chosen)}
        )
    return auc(covers, stegos), cells


def cuda_runtime():
    import torch

    if (
        torch.__version__ != "2.14.0+cu130"
        or np.__version__ != "2.4.6"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5060 Laptop GPU"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise RuntimeError("frozen CUDA runtime unavailable")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(4 * 1024**3 / total, 1.0), 0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch


def worker(root: Path, protocol_path: Path, out: Path, expires: float) -> None:
    protocol, digest, parent, parent_sha = load_protocol(root, protocol_path)
    if not math.isfinite(expires) or not time.monotonic() < expires <= time.monotonic() + SECONDS:
        raise ValueError("finite bounded worker deadline required")
    cgroup()
    save(out / "worker-started.json", {"runner_protocol_sha256": digest})
    torch = cuda_runtime()
    state: dict[str, Any] = {"phase": "generation", "confirmed_updates_lower_bound": 0}
    progress(out, state)
    with tempfile.TemporaryDirectory(prefix="generated-only-", dir=out) as directory:
        data = Path(directory)
        records, manifest = build_corpus(data, lengths(root), expires)
        if (
            hashlib.sha256(encoded(manifest)).hexdigest()
            != protocol["source_closure_sha256"][CORPUS]
        ):
            raise ValueError("generated corpus binding")
        train = wav_pilot.rows_for_role(records, "train")
        dev = wav_pilot.rows_for_role(records, "development")
        runs = []
        for seed in wav_pilot_training.SEEDS:
            state.update(phase="fit", seed=seed)
            progress(out, state)

            def on_update():
                state["confirmed_updates_lower_bound"] += 1
                progress(out, state)

            model, fit = wav_pilot_training.fit_seed(
                data,
                train,
                seed,
                device="cuda:0",
                deadline_at=expires,
                on_update=on_update,
                recipe=wav_pilot_v3.V3_RECIPE,
            )
            save(out / f"fit-{seed}.json", {"runner_protocol_sha256": digest, "fit": fit})
            state["phase"] = "scoring"
            progress(out, state)
            train_scores = score(torch, data, train, model, expires)
            dev_scores = score(torch, data, dev, model, expires)
            train_auc, train_cells = metrics(train, train_scores)
            _, cells = metrics(dev, dev_scores)
            runs.append(
                {
                    "seed": seed,
                    "epochs": fit["epochs"],
                    "optimizer_updates": fit["optimizer_updates"],
                    "row_presentations": fit["row_presentations"],
                    "virtual_batch": fit["virtual_batch"],
                    "final_batch_rows": fit["partial_batch_rows"],
                    "train_files": 400,
                    "development_files": 200,
                    "train_auc_all": train_auc,
                    "train_cells": train_cells,
                    "development_cells": cells,
                    "fit_seconds": fit["seconds"],
                    "mean_batch_loss": fit["mean_batch_loss"],
                    "train_scores_sha256": hashlib.sha256(train_scores.tobytes()).hexdigest(),
                    "development_scores_sha256": hashlib.sha256(dev_scores.tobytes()).hexdigest(),
                }
            )
            save(out / f"seed-{seed}.json", {"runner_protocol_sha256": digest, **runs[-1]})
            del model
        check_deadline(expires)
        load_protocol(root, protocol_path)
        save(
            out / "worker.json",
            {
                "status": "completed_generated_control",
                "schema_version": "wav-full-control-result-v1",
                "protocol_sha256": parent_sha,
                "source_closure_sha256": parent["source_closure_sha256"],
                "runner_protocol_sha256": digest,
                "runner_source_closure_sha256": protocol["source_closure_sha256"],
                "recipe": gate.SPEC["recipe"],
                "generated_only": True,
                "real_media_opened": False,
                "reserved_media_opened": False,
                "checkpoint_written": False,
                "real_optimizer_updates": 0,
                "device": torch.cuda.get_device_name(0),
                "corpus": manifest["evidence"],
                "cuda_peak_bytes": torch.cuda.max_memory_allocated(0),
                "seeds": runs,
            },
        )
    state["phase"] = "completed"
    progress(out, state)


def cgroup(root=Path("/sys/fs/cgroup"), proc=None):
    proc = Path("/proc") / str(os.getpid()) / "cgroup" if proc is None else proc
    text = regular_read(proc, 4096).decode().strip()
    if not text.startswith("0::/") or ".." in text.split("/") or "\n" in text:
        raise ValueError("unified own-process cgroup required")
    folder = root / text[4:]
    maximum = regular_read(folder / "memory.max", 64).decode().strip()
    swap = regular_read(folder / "memory.swap.max", 64).decode().strip()
    if not maximum.isdecimal() or not 0 < int(maximum) <= 8 * 1024**3 or swap != "0":
        raise ValueError("bounded memory and zero swap required")
    return folder


def stop_process(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def process_started(proc=None) -> float:
    proc = Path("/proc") / str(os.getpid()) / "stat" if proc is None else proc
    fields = regular_read(proc, 4096).decode().rsplit(") ", 1)[1].split()
    since_boot = int(fields[19]) / os.sysconf("SC_CLK_TCK")
    elapsed = max(0, time.clock_gettime(time.CLOCK_BOOTTIME) - since_boot)
    return time.monotonic() - elapsed


def execute(root: Path, protocol_path: Path, out: Path) -> dict[str, Any]:
    started = process_started()
    expires = started + SECONDS
    _, digest, parent, parent_sha = load_protocol(root, protocol_path)
    folder = cgroup()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("runner output symlink")
    out.mkdir(mode=0o700)
    save(
        out / "attempt.json",
        {"runner_protocol_sha256": digest, "automatic_retry": False, "real_fit_authorized": False},
    )
    env = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(root),
    }
    command = [
        sys.executable,
        str(root / "steganography/research_wav_control_run.py"),
        "--protocol",
        str(protocol_path),
        "--out",
        str(out),
        "--worker-deadline",
        str(expires - RESERVE_SECONDS),
    ]
    with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.Popen(  # noqa: S603 - frozen local code
                command, cwd=root, env=env, start_new_session=True, stdout=stdout, stderr=stderr
            )
        except OSError:
            process = None
        if process is not None:
            try:
                while process.poll() is None:
                    if time.monotonic() >= expires:
                        stop_process(process)
                        break
                    time.sleep(0.25)
            finally:
                if process.poll() is None:
                    stop_process(process)
    child_exit = process.returncode if process is not None else None
    result: dict[str, Any] = {
        "status": "unavailable",
        "reason": "worker_incomplete",
        "runner_protocol_sha256": digest,
        "real_fit_authorized": False,
    }
    if child_exit == 0 and (out / "worker.json").exists():
        result, _ = gate.read_document(out / "worker.json")
    result.update(
        wall_seconds=time.monotonic() - started,
        host_peak_bytes=int(regular_read(folder / "memory.peak", 64)),
        swap_peak_bytes=int(regular_read(folder / "memory.swap.current", 64)),
    )
    if (out / "progress.json").exists():
        result["partial_progress"], _ = gate.read_document(out / "progress.json")
    decision = gate.evaluate(parent, parent_sha, result)
    save(out / "result.json", result)
    save(
        out / "decision.json",
        {**decision, "runner_protocol_sha256": digest, "child_exit_code": child_exit},
    )
    return decision
