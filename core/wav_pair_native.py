"""Bounded AudioMNIST pair-head preparation; fitting requires separate approval."""

from __future__ import annotations

import ast
import hashlib
import io
import math
import struct
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core import nonjpeg_compare_metrics as metrics
from core import wav_pair_comparison as comparison
from core.nonjpeg_data import regular_read

base = comparison.base
PREPARATION = "benchmarks/wav-audiomnist-preparation-20261017.json"
ASSIGNMENT = "benchmarks/wav-next-experiment-draft-20261016.json"
CONTROL = "benchmarks/wav-pair-comparison-results-20261022.json"
PROTOCOL = "wav-pair-native-audiomnist-20261022-v1"
SOURCES = tuple(
    dict.fromkeys(
        (
            *comparison.SOURCES,
            ASSIGNMENT,
            CONTROL,
            "benchmarks/wav-pair-comparison-protocol-20261022.json",
            "core/wav_pair_native.py",
            "steganography/research_wav_pair_native.py",
            "tests/test_wav_pair_native.py",
            "docs/WAV_PAIR_NATIVE_PROTOCOL_20261022.md",
        )
    )
)
SPEC: dict[str, Any] = {
    "domain": comparison.DOMAINS["train_zscore"],
    "recipe": comparison.SPEC["recipe"],
    "seeds": [20261016, 20261017],
    "epochs": 20,
    "batch": 64,
    "updates_per_seed": 1140,
    "presentations_per_seed": 72000,
    "total_updates": 2280,
    "total_presentations": 144000,
    "normalizer": "all_3600_train_rows_only_float64_population_std_floor_1e-6",
    "threshold": 0.5,
    "bootstrap_draws": 2000,
    "bootstrap_seed": 20261018,
    "profile_seconds_max": 300,
    "total_seconds_max": 1800,
    "host_bytes_max": 2 * 1024**3,
    "swap_bytes_max": 0,
    "output_bytes_max": 64 * base.MIB,
    "file_bytes_max": base.MIB,
    "train_auc_min": 0.60,
    "development_auc_min": 0.70,
}


def load(root: Path, protocol: Path):
    doc, sha = base.old.gate.read_document(protocol)
    if (
        doc.get("schema_version") != "wav-pair-native-protocol-v1"
        or doc.get("protocol_id") != PROTOCOL
        or base.old.encoded({"spec": doc.get("spec")}) != base.old.encoded({"spec": SPEC})
        or doc.get("real_fit_authorized") is not False
        or doc.get("preparation_authorized") is not True
        or set(doc.get("source_sha256", {})) != set(SOURCES)
    ):
        raise ValueError("native protocol contract")
    for name, expected in doc["source_sha256"].items():
        if base.old.gate.source_hash(root / name) != expected:
            raise ValueError("native source binding")
    return doc, sha


def records(root: Path):
    """Inspect pinned metadata only; never traverse a media directory."""
    prep, _ = base.old.gate.read_document(root / PREPARATION)
    draft, _ = base.old.gate.read_document(root / ASSIGNMENT)
    items = prep.get("files")
    expected = [
        (role, group, f"data/{group}/{digit}_{group}_{take}.wav")
        for role in ("train", "development")
        for group in draft["speaker_roles"][role]
        for digit in range(10)
        for take in range(2)
    ]
    if (
        prep["source"]["commit"] != "630d7dab4c040882834de6fa21baf9a60372accd"
        or prep["source"]["repository_license"] != "MIT"
        or not isinstance(items, list)
        or len(items) != 600
        or len(expected) != 600
        or [(r["role"], r["speaker_group"], r["relative_source_path"]) for r in items] != expected
    ):
        raise ValueError("native source/license/assignment")
    groups = {
        role: {r["speaker_group"] for r in items if r["role"] == role}
        for role in ("train", "development")
    }
    if (
        len(groups["train"]) != 20
        or len(groups["development"]) != 10
        or groups["train"] & groups["development"]
        or (groups["train"] | groups["development"]) & set(draft["speaker_roles"]["reserved"])
    ):
        raise ValueError("native speaker split")
    for key in ("sha256", "decoded_pcm_sha256"):
        values = [r.get(key) for r in items]
        if len(set(values)) != 600 or not all(comparison.is_sha(v) for v in values):
            raise ValueError("native duplicate/invalid hash")
    for r in items:
        if (
            r.get("channels") != 1
            or r.get("sample_rate_hz") != 48000
            or r.get("sample_width_bytes") != 2
            or r.get("parent_recording_id") is not None
            or type(r.get("frames")) is not int
            or not 160 <= r["frames"] <= 220500
            or type(r.get("bytes")) is not int
            or not 44 <= r["bytes"] <= 512 * 1024
        ):
            raise ValueError("native PCM metadata domain")
    return prep, {role: base.wav_pilot.rows_for_role(items, role) for role in groups}


def quality(root: Path, protocol: Path, qa: Path):
    _, sha = load(root, protocol)
    proof, _ = base.old.gate.read_document(qa)
    if (
        proof.get("protocol_sha256") != sha
        or proof.get("ruff") != "passed"
        or proof.get("mypy") != "passed"
        or proof.get("diff_check") != "passed"
        or type(proof.get("full_passed")) is not int
        or proof["full_passed"] < 3662
        or not base.number(proof.get("coverage_percent"), 90, 100)
        or set(proof.get("new_module_coverage_percent", {}))
        != {"core/wav_pair_native.py", "steganography/research_wav_pair_native.py"}
        or not all(base.number(v, 95, 100) for v in proof["new_module_coverage_percent"].values())
    ):
        raise ValueError("native software gate")
    control, _ = base.old.gate.read_document(root / CONTROL)
    if control.get("arm_decisions", {}).get("train_zscore") != "passed_generated_only":
        raise ValueError("native generated control prerequisite")
    return sha


def read_array(path: Path, count: int, *, scores=False):
    """Validate the small NPY header and exact bytes before any array allocation."""
    raw = regular_read(path, base.MIB)
    if raw[:8] != b"\x93NUMPY\x01\x00" or len(raw) < 10:
        raise ValueError("native NPY version")
    length = struct.unpack("<H", raw[8:10])[0]
    if not 1 <= length <= 512:
        raise ValueError("native NPY header bound")
    header = ast.literal_eval(raw[10 : 10 + length].decode("ascii"))
    shape, dtype = ((count,), "<f8") if scores else ((count, 10), "<f4")
    if (
        header != {"descr": dtype, "fortran_order": False, "shape": shape}
        or type(header.get("fortran_order")) is not bool
        or len(raw) != 10 + length + math.prod(shape) * np.dtype(dtype).itemsize
    ):
        raise ValueError("native NPY shape/dtype/extent")
    values = np.frombuffer(raw, dtype=dtype, offset=10 + length).reshape(shape).copy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("native array range")
    return values


def storage(out: Path):
    total = 0
    for path in out.rglob("*"):
        if path.is_symlink():
            raise ValueError("native output symlink")
        if path.is_file():
            size = path.stat().st_size
            if size > base.MIB:
                raise ValueError("native file size")
            total += size
    if total > SPEC["output_bytes_max"]:
        raise ValueError("native storage limit")
    return total


def profile(root: Path, protocol: Path, media: Path, out: Path, qa: Path):
    started = base.old.process_started()
    sha = quality(root, protocol, qa)
    base.resources()
    prep, selected = records(root)
    if base.digest(media / "manifest.json") != prep["acquisition_manifest_sha256"]:
        raise ValueError("native acquisition manifest binding")
    base.new_output(out)
    base.old.save(out / "profile-started.json", {"protocol_sha256": sha, "optimizer_updates": 0})
    expires = started + 295
    torch = base.runtime()
    oracles = originals = 0
    for role, rows in selected.items():
        values = []
        for row in rows:
            base.old.check_deadline(expires)
            raw = base.wav_pilot.materialize(media, row["source"], row["recipe"])
            if row["recipe"] is None:
                samples = base.wav_pilot.pcm16_samples(raw)
                with wave.open(io.BytesIO(raw), "rb") as stream:
                    if (
                        stream.getframerate() != 48000
                        or stream.getnchannels() != 1
                        or stream.getsampwidth() != 2
                        or stream.getcomptype() != "NONE"
                    ):
                        raise ValueError("native WAV header")
                source = row["source"]
                if (
                    len(raw) != source["bytes"]
                    or len(samples) != source["frames"]
                    or hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest()
                    != source["decoded_pcm_sha256"]
                ):
                    raise ValueError("native decoded PCM binding")
                originals += 1
            else:
                oracles += 1
            values.append(base.candidate.inputs(raw)[0])
        base.array_save(out / f"{role}.npy", np.asarray(values, dtype=np.float32))
    arrays = {role: read_array(out / f"{role}.npy", len(rows)) for role, rows in selected.items()}
    state = comparison.normalization(arrays["train"])
    base.old.save(out / "normalization.json", state)
    with torch.random.fork_rng():
        torch.manual_seed(2026102206)
        model = base.candidate.network()
    before = comparison.parameter_sha(model)
    x = torch.from_numpy(comparison.transform(arrays["train"], state, "train_zscore")[:64])
    smoke_started = time.monotonic()
    for _ in range(20):
        base.old.check_deadline(expires)
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), torch.arange(64) % 2).backward()
        base.finite_gradients(torch, model)
    smoke = time.monotonic() - smoke_started
    if comparison.parameter_sha(model) != before or oracles != 4800 or originals != 600:
        raise ValueError("native zero-update proof")
    projection = time.monotonic() - started + smoke / 20 * 2280 * 10 + 120
    if projection > 1800:
        raise ValueError("native time projection")
    quality(root, protocol, qa)
    base.old.check_deadline(expires)
    result = {
        "status": "passed_native_zero_update_profile",
        "protocol_sha256": sha,
        "qa_sha256": base.digest(qa),
        "optimizer_updates": 0,
        "originals_verified": originals,
        "payload_oracles_passed": oracles,
        "arrays_sha256": {r: base.digest(out / f"{r}.npy") for r in selected},
        "row_order_sha256": {r: base.row_sha(rows) for r, rows in selected.items()},
        "normalization_sha256": base.digest(out / "normalization.json"),
        "acquisition_manifest_sha256": prep["acquisition_manifest_sha256"],
        "parameters_unchanged": True,
        "backward_batches": 20,
        "profile_wall_seconds": time.monotonic() - started,
        "conservative_total_projection_seconds": projection,
        "resources": base.resources(),
        "output_bytes": storage(out),
        "reserved_media_opened": False,
        "real_fit_authorized": False,
        "checkpoint_written": False,
    }
    base.old.save(out / "profile.json", result)
    return result


def authorize(path: Path, sha: str, profile_sha: str, elapsed: float):
    value, _ = base.old.gate.read_document(path)
    debit = value.get("profile_debit_seconds")
    if (
        value.get("schema_version") != "wav-pair-native-approval-v1"
        or value.get("protocol_sha256") != sha
        or value.get("profile_sha256") != profile_sha
        or value.get("explicit_user_approval") is not True
        or type(value.get("fit_attempts_allowed")) is not int
        or value["fit_attempts_allowed"] != 1
        or not isinstance(value.get("approval_reference"), str)
        or not 1 <= len(value["approval_reference"]) <= 200
        or not base.number(debit, max(elapsed, 0.000001), 300)
    ):
        raise ValueError("native explicit approval/profile debit required")
    return debit


def cell_report(rows, scores, reference):
    if (
        scores.shape != (len(rows),)
        or reference.shape != scores.shape
        or not np.isfinite(scores).all()
        or not np.isfinite(reference).all()
        or np.any((scores < 0) | (scores > 1))
        or np.any((reference < 0) | (reference > 1))
    ):
        raise ValueError("native metric score domain")
    groups = [r["group"] for r in rows[::9]]
    draws = np.vstack([np.ones(len(groups)), metrics.bootstrap_weights(groups)])
    result = []
    for index, (method, rate, plane) in enumerate(base.wav_pilot.RECIPES, 1):
        pairs = np.column_stack((scores[::9], scores[index::9]))
        prior = np.column_stack((reference[::9], reference[index::9]))
        measured, baseline = (
            metrics.measured(pairs, draws, 0.5),
            metrics.measured(prior, draws, 0.5),
        )
        result.append(
            {
                "method": method,
                "rate_percent": rate,
                "plane": plane,
                "pairs": len(groups),
                "speaker_groups": len(set(groups)),
                "threshold": 0.5,
                "metrics": {k: float(v[0]) for k, v in measured.items()},
                "ci95": {
                    k: np.quantile(v[1:], [0.025, 0.975]).tolist() for k, v in measured.items()
                },
                "fixed_global_plane0": {k: float(v[0]) for k, v in baseline.items()},
                "paired_delta_ci95": {
                    k: np.quantile((measured[k] - baseline[k])[1:], [0.025, 0.975]).tolist()
                    for k in measured
                },
            }
        )
    return result


def verify_run(run, seed):
    """Reject partial exposure or missing train/step evidence before scoring."""
    if (
        run.get("domain") != SPEC["domain"]
        or run.get("arm") != "train_zscore"
        or any(
            type(run.get(k)) is not int or run[k] != value
            for k, value in {
                "seed": seed,
                "epochs": 20,
                "optimizer_updates": 1140,
                "row_presentations": 72000,
            }.items()
        )
        or not base.number(run.get("parameter_change_norm"), 1e-30, 1e30)
        or any(
            not comparison.is_sha(run.get(k))
            for k in ("initial_parameter_sha256", "final_parameter_sha256", "order_sha256")
        )
        or run["initial_parameter_sha256"] == run["final_parameter_sha256"]
    ):
        raise ValueError("native full fit evidence")
    rng = np.random.default_rng(seed ^ 0x4E4A5047)
    expected = hashlib.sha256()
    for _ in range(20):
        expected.update(rng.permutation(3600).astype("<i4").tobytes())
    epochs, steps = run.get("epoch_telemetry"), run.get("step_telemetry")
    if (
        run["order_sha256"] != expected.hexdigest()
        or not isinstance(epochs, list)
        or len(epochs) != 20
        or not isinstance(steps, list)
        or len(steps) != 4
    ):
        raise ValueError("native order/telemetry evidence")
    for index, epoch in enumerate(epochs, 1):
        if (
            not comparison.valid_metrics(epoch)
            or any(
                type(epoch.get(k)) is not int or epoch[k] != value
                for k, value in {
                    "epoch": index,
                    "optimizer_updates": 57 * index,
                    "row_presentations": 3600 * index,
                }.items()
            )
            or any(
                not base.number(epoch.get(k), 0, 1e30)
                for k in ("cover_loss", "unweighted_loss", "weighted_objective")
            )
        ):
            raise ValueError("native epoch evidence")
    for point, update in zip(steps, comparison.POINTS, strict=True):
        if (
            type(point.get("update")) is not int
            or point["update"] != update
            or any(
                not base.number(point.get(k), 0, 1e30)
                for k in ("objective", "gradient_norm", "parameter_step_norm")
            )
        ):
            raise ValueError("native gradient evidence")


def replay_auc(scores, record):
    """Independent all-pairs half-tie check against the rank-based scorer."""
    negatives = scores[::9]

    def scalar(positive):
        return sum(
            float(np.count_nonzero(p > negatives)) + 0.5 * float(np.count_nonzero(p == negatives))
            for p in positive
        ) / (len(positive) * len(negatives))

    if (
        not comparison.valid_metrics(record)
        or not math.isclose(
            scalar(np.delete(scores.reshape(-1, 9), 0, axis=1).ravel()),
            record["auc_all"],
            abs_tol=1e-12,
            rel_tol=0,
        )
        or any(
            not math.isclose(scalar(scores[index::9]), cell["roc_auc"], abs_tol=1e-12, rel_tol=0)
            for index, cell in enumerate(record["cells"], 1)
        )
    ):
        raise ValueError("native independent AUC replay")
    return 9


def fit(root: Path, protocol: Path, cache: Path, qa: Path, approval: Path):
    started = base.old.process_started()
    sha = quality(root, protocol, qa)
    receipt, profile_sha = base.old.gate.read_document(cache / "profile.json")
    debit = authorize(approval, sha, profile_sha, receipt.get("profile_wall_seconds", math.inf))
    if (
        receipt.get("status") != "passed_native_zero_update_profile"
        or receipt.get("protocol_sha256") != sha
        or receipt.get("qa_sha256") != base.digest(qa)
        or type(receipt.get("optimizer_updates")) is not int
        or receipt["optimizer_updates"] != 0
        or receipt.get("parameters_unchanged") is not True
        or receipt.get("originals_verified") != 600
        or receipt.get("payload_oracles_passed") != 4800
        or receipt.get("backward_batches") != 20
        or not base.number(receipt.get("conservative_total_projection_seconds"), 0.000001, 1800)
        or base.digest(cache / "normalization.json") != receipt.get("normalization_sha256")
    ):
        raise ValueError("native profile prerequisite")
    _, selected = records(root)
    arrays = {}
    for role, rows in selected.items():
        if (
            base.digest(cache / f"{role}.npy") != receipt["arrays_sha256"][role]
            or base.row_sha(rows) != receipt["row_order_sha256"][role]
        ):
            raise ValueError("native cache/order binding")
        arrays[role] = read_array(cache / f"{role}.npy", len(rows))
    state, _ = base.old.gate.read_document(cache / "normalization.json")
    if base.old.encoded(state) != base.old.encoded(comparison.normalization(arrays["train"])):
        raise ValueError("native train-only normalizer")
    base.resources()
    out = cache / "fit"
    base.new_output(out)  # An existing attempt is never overwritten or retried.
    base.old.save(
        out / "started.json", {"protocol_sha256": sha, "approval_sha256": base.digest(approval)}
    )
    expires = started + 1800 - debit - 5
    torch = base.runtime()
    progress: dict[str, Any] = {"confirmed_updates_lower_bound": 0, "phase": "fit"}
    runs = []
    for seed in SPEC["seeds"]:
        destination = out / str(seed)
        base.new_output(destination)

        def update():
            progress["confirmed_updates_lower_bound"] += 1
            base.old.progress(out, progress)

        def epoch(record, target=destination):
            base.old.save(target / f"epoch-{record['epoch']:02d}.json", record)

        train = comparison.transform(arrays["train"], state, "train_zscore")
        model, run = comparison.fit_one(
            torch, train, selected["train"], seed, "train_zscore", expires, update, epoch
        )
        verify_run(run, seed)
        base.old.save(destination / "fit.json", run)
        for role, values in arrays.items():
            base.old.check_deadline(expires)
            scores = base.predict(torch, model, comparison.transform(values, state, "train_zscore"))
            target = destination / f"{role}-scores.npy"
            base.array_save(target, scores)
            replay = read_array(target, len(scores), scores=True)
            auc, cells = base.old.metrics(selected[role], replay)
            run[role] = {"auc_all": auc, "cells": cells, "scores_sha256": base.digest(target)}
            run[role]["independent_auc_checks"] = replay_auc(replay, run[role])
            if role == "development":
                run[role]["speaker_cluster_metrics"] = cell_report(
                    selected[role], replay, values[:, 0]
                )
        base.old.save(destination / "seed.json", run)
        runs.append(run)
        del model
    quality(root, protocol, qa)
    base.old.check_deadline(expires)
    result = {
        "status": "completed_native_exploratory_pilot",
        "protocol_sha256": sha,
        "profile_sha256": profile_sha,
        "profile_debit_seconds": debit,
        "runs": runs,
        "confirmed_updates": progress["confirmed_updates_lower_bound"],
        "total_wall_seconds": debit + time.monotonic() - started,
        "resources": base.resources(),
        "output_bytes": storage(cache),
        "reserved_media_opened": False,
        "checkpoint_written": False,
        "production_qualified": False,
    }
    if progress["confirmed_updates_lower_bound"] != 2280:
        raise ValueError("native confirmed exposure")
    result["learning_target"] = (
        "met_exploratory_target"
        if all(
            run["train"]["auc_all"] >= 0.60
            and all(
                cell["roc_auc"] >= 0.70
                for cell in run["development"]["cells"]
                if cell["rate_percent"] == 40 and cell["plane"] == 0
            )
            for run in runs
        )
        else "failed_exploratory_target"
    )
    base.old.save(out / "result.json", result)
    base.old.progress(out, {**progress, "phase": "completed"})
    return result
