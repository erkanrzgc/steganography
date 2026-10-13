"""Bounded continuation contracts, not measured real-GPU learning evidence."""

import copy
import hashlib
import json

import numpy as np
import pytest

from core import jpeg_study as study
from core import jpeg_study_segments as segments
from core.jpeg_study_process import WorkerExit, WorkerFailure
from steganography import research_jpeg_study as frontend


def configuration():
    return {"schema_version": "jpeg-study-config-v1", "action": "controls",
            "root": "private-data", "manifest_sha256": "a" * 64,
            "audit": "private-audit", "audit_sha256": "b" * 64,
            "execution_mode": segments.MODE}


def paired_samples():
    return [{"source_group": origin, "role": "fit", "lineage": f"{origin}:{pair}",
             "label": label}
            for origin in study.data.ORIGINS for pair in range(4)
            for label in ("cover", "stego")]


@pytest.mark.parametrize("seed", study.SEEDS)
@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_half_schedule_is_exact_full_prefix(seed, rows):
    samples = paired_samples()
    partial = study.schedule(samples, seed, rows, 16384)
    full = study.schedule(samples, seed, rows)
    for left, right in zip(partial, full, strict=True):
        np.testing.assert_array_equal(left, right[:16384 // rows])
    np.testing.assert_array_equal(study.view_schedule(samples, seed, partial[0]),
                                  study.view_schedule(samples, seed, full[0])[:16384 // rows])


def run_trial(tmp_path, *, rows=4, fault=None, fail_phase=0, budget=None):
    source = {"source.py": "c" * 64}
    arm = "genuine-batch" if rows != 4 else "legacy"
    recipe = study.recipe(arm, study.SEEDS[0], rows)
    calls = []
    out = tmp_path / "trial"

    def launch(payload, phase_out, stop, timeout):
        index = len(calls)
        calls.append((copy.deepcopy(payload), phase_out, stop, timeout))
        assert "execution_mode" not in payload
        assert 0 < timeout <= 1830
        if index == 0:
            assert not payload.get("resume")
        else:
            resume = payload["resume"][0]
            assert resume == {"arm": arm, "seed": study.SEEDS[0],
                              "path": str(out / "segment-0" / "rows-16384.npz"),
                              "sha256": hashlib.sha256(b"generated-16384").hexdigest()}
            # Internal worker configurations accept the exact owned resume;
            # external segmented resumes remain rejected.
            assert frontend.configuration(payload) == payload
        if index == fail_phase and fault in ("timeout", "cpu", "cancel"):
            raise {"timeout": TimeoutError(), "cpu": WorkerFailure(WorkerExit(-24, 0)),
                   "cancel": InterruptedError()}[fault]
        phase_out.mkdir()
        raw = f"generated-{stop}".encode()
        (phase_out / f"rows-{stop}.npz").write_bytes(raw)
        value = {"status": "checkpointed" if index == 0 else "completed",
                 "source_sha256": source, "recipe": copy.deepcopy(recipe),
                 "presented_rows": stop, "resumed_from_rows": index * 16384,
                 "optimizer_updates": 16384 // rows, "total_optimizer_updates": stop // rows,
                 "execution": {"device": "cuda:0"}, "real_gpu_segment": True,
                 "real_gpu_fit": bool(index), "binding": "b" * 64,
                 "initial_learned_sha256": "d" * 64, "schedule_sha256": "e" * 64,
                 "seconds": 100 + index, "mean_cross_entropy": .6 + .2 * index,
                 "checkpoints": [{"path": f"rows-{stop}.npz", "presented_rows": stop,
                                  "sha256": hashlib.sha256(raw).hexdigest()}]}
        if index:
            value.update(scores=[.5] * 8, cells=[])
        if index == fail_phase:
            if callable(fault):
                fault(value)
            elif fault == "symlink":
                path = phase_out / f"rows-{stop}.npz"
                path.unlink()
                path.symlink_to(tmp_path / "outside")
        return value

    result = segments.trial(configuration(), out, arm=arm, seed=study.SEEDS[0], rows=rows,
                            sources=source, recipe=recipe, launch=launch,
                            budget_left=budget or (lambda: 21600))
    return result, calls, out


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_complete_halves_preserve_total_exposure_and_receipts(tmp_path, rows):
    value, calls, out = run_trial(tmp_path, rows=rows)
    assert len(calls) == 2 and value["status"] == "completed" and value["real_gpu_fit"]
    assert value["execution_mode"] == segments.MODE
    assert value["presented_rows"] == 32768 and value["resumed_from_rows"] == 0
    assert value["optimizer_updates"] == value["total_optimizer_updates"] == 32768 // rows
    assert value["seconds"] == 201 and value["mean_cross_entropy"] == pytest.approx(.7)
    assert [c["path"] for c in value["checkpoints"]] == [
        "segment-0/rows-16384.npz", "segment-1/rows-32768.npz"]
    assert json.loads((out / "receipt-16384.json").read_text())["status"] == "checkpointed"
    assert json.loads((out / "receipt-32768.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("fault", ["timeout", "cpu", "cancel"])
@pytest.mark.parametrize("phase", [0, 1])
def test_incomplete_half_never_qualifies_and_retains_receipts(tmp_path, fault, phase):
    value, calls, out = run_trial(tmp_path, fault=fault, fail_phase=phase)
    assert len(calls) == phase + 1 and not value["real_gpu_fit"]
    assert value["status"] == ("cancelled" if fault == "cancel" else "unavailable")
    assert value["reason"] == {"timeout": "worker_timeout", "cpu": "worker_cpu_limit",
                               "cancel": "user_cancelled"}[fault]
    assert (out / f"receipt-{16384 * (phase + 1)}.json").is_file()
    if phase:
        assert (out / "segment-0" / "rows-16384.npz").is_file()


@pytest.mark.parametrize("fault", [
    lambda v: v.update(status="passed"),
    lambda v: v.update(source_sha256={}),
    lambda v: v.pop("source_sha256"),
    lambda v: v.update(presented_rows=4),
    lambda v: v.update(resumed_from_rows=4),
    lambda v: v.update(optimizer_updates=4),
    lambda v: v.update(total_optimizer_updates=4),
    lambda v: v.update(recipe={}),
    lambda v: v.update(execution={"device": "cpu"}),
    lambda v: v.update(real_gpu_segment=False),
    lambda v: v.update(binding="invalid"),
    lambda v: v.update(initial_learned_sha256=None),
    lambda v: v.update(schedule_sha256="invalid"),
    lambda v: v.update(seconds=None),
    lambda v: v.update(mean_cross_entropy=float("nan")),
    lambda v: v.update(checkpoints=[]),
    lambda v: v.update(checkpoints=[None]),
    lambda v: v["checkpoints"][0].update(path="../outside.npz"),
    lambda v: v["checkpoints"][0].update(presented_rows=4),
    lambda v: v["checkpoints"][0].update(sha256="f" * 64),
    lambda v: v.update(real_gpu_fit=True),
    lambda v: v.update(scores=[]),
    lambda v: v.update(cells=[]),
    "symlink",
])
def test_first_half_adversarial_receipts_fail_closed(tmp_path, fault):
    with pytest.raises((ValueError, OSError)):
        run_trial(tmp_path, fault=fault)


@pytest.mark.parametrize("fault", [
    lambda v: v.update(status="checkpointed"),
    lambda v: v.update(binding="a" * 64),
    lambda v: v.update(initial_learned_sha256="a" * 64),
    lambda v: v.update(real_gpu_fit=False),
])
def test_second_half_cannot_change_identity_or_infer_completion(tmp_path, fault):
    with pytest.raises(ValueError):
        run_trial(tmp_path, fault=fault, fail_phase=1)


@pytest.mark.parametrize("phase", [0, 1])
def test_global_deadline_is_not_restarted_for_next_half(tmp_path, phase):
    values = iter([0] if phase == 0 else [21600, 21600, 0])
    result, calls, _ = run_trial(tmp_path, budget=lambda: next(values))
    assert result["reason"] == "controls_budget_exhausted" and len(calls) == phase
    assert not result["real_gpu_fit"]


def test_configuration_versions_and_external_resume_rejected(tmp_path):
    config = configuration()
    assert frontend.configuration(config) == config
    config["execution_mode"] = "unknown"
    with pytest.raises(ValueError, match="mode"):
        frontend.configuration(config)
    config["execution_mode"] = segments.MODE
    config["resume"] = [{}]
    with pytest.raises(ValueError, match="resume"):
        frontend.configuration(config)
    with pytest.raises(ValueError, match="resume"):
        segments.trial(config, tmp_path / "bad", arm="legacy", seed=study.SEEDS[0], rows=4,
                       sources={}, recipe={}, launch=None, budget_left=None)


def test_existing_or_symlink_output_is_not_overwritten(tmp_path):
    target = tmp_path / "trial"
    target.mkdir()
    (target / "keep").write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        run_trial(tmp_path)
    assert (target / "keep").read_bytes() == b"retained"


@pytest.mark.parametrize("raw", [b"", b"different", b"oversized"])
def test_checkpoint_bytes_checked_before_resume(tmp_path, monkeypatch, raw):
    path = tmp_path / "snapshot"
    path.write_bytes(raw)
    if raw == b"oversized":
        monkeypatch.setattr(segments, "MAX_BYTES", 2)
    with pytest.raises(ValueError):
        segments.checkpoint(path, hashlib.sha256(b"expected").hexdigest())
