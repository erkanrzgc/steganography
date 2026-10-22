"""Matched scaling, independent gradients, immutable state and failed-run evidence."""

import copy
import io
import json
import math
import time

import numpy as np
import pytest

from core import wav_pair_comparison as c
from steganography import research_wav_pair_comparison as cli


def protocol(root):
    root.mkdir(exist_ok=True)
    for name in c.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    previous = {"files": [{"sha256": "old", "decoded_pcm_sha256": "old"}]}
    for name in (c.base.old.CORPUS, c.PREVIOUS):
        (root / name).write_text(json.dumps(previous))
    (root / "benchmarks/wav-audiomnist-preparation-20261017.json").write_text(
        json.dumps(
            {
                "files": [
                    {"role": role, "frames": 640}
                    for role, count in (("train", 400), ("development", 200))
                    for _ in range(count)
                ]
            }
        )
    )
    path = root / "protocol.json"
    data = {
        "schema_version": "wav-pair-comparison-protocol-v1",
        "protocol_id": c.PROTOCOL,
        "spec": c.SPEC,
        "real_fit_authorized": False,
        "attempts_allowed": {"profile": 1, "fit": 1},
        "source_sha256": {name: c.base.old.gate.source_hash(root / name) for name in c.SOURCES},
    }
    path.write_text(json.dumps(data))
    return path


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch = pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("comparison-cache")
    path = protocol(root)
    prior = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c.base, "resources", lambda: {"host_peak_bytes": 1024, "swap_bytes": 0})
        patch.setattr(c.base.old, "process_started", time.monotonic)
        receipt = c.profile(root, path, root / "cache")
    torch.set_num_threads(prior[0])
    torch.use_deterministic_algorithms(prior[1])
    return root, path, receipt


def test_profile_has_full_fresh_oracles_train_only_state_and_no_updates(prepared):
    root, path, receipt = prepared
    assert receipt["optimizer_updates"] == 0
    assert [r["arm"] for r in receipt["smoke"]] == list(c.ARMS)
    assert all(r["parameters_unchanged"] for r in receipt["smoke"])
    assert (
        receipt["smoke"][0]["initial_parameter_sha256"]
        == receipt["smoke"][1]["initial_parameter_sha256"]
    )
    cache = root / "cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    assert (
        manifest["generator_seed"] == c.GENERATOR_SEED
        and manifest["payload_oracles_passed"] == 4800
    )
    assert len({r["decoded_pcm_sha256"] for r in manifest["files"]}) == 600
    assert not list(cache.glob("fresh-comparison-*"))
    train = c.base.read_array(cache / "train.npy", 3600)
    state = json.loads((cache / "normalization.json").read_text())
    assert state == c.normalization(train)
    assert c.load(root, path)[1] == receipt["protocol_sha256"]


def test_normalizer_matches_scalar_population_and_fp32_rounding():
    data = np.resize(np.arange(11, dtype=np.float32) / 10, (3600, 10))
    state = c.normalization(data)
    for column in range(10):
        values = [float(v) for v in data[:, column]]
        mean = math.fsum(values) / len(values)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
        assert math.isclose(state["mean"][column], mean, abs_tol=1e-14)
        assert math.isclose(state["std"][column], std, abs_tol=1e-14)
    actual = c.transform(data, state, "train_zscore")
    expected = np.array(
        [
            [(float(data[i, j]) - state["mean"][j]) / state["std"][j] for j in range(10)]
            for i in range(len(data))
        ],
        dtype=np.float32,
    )
    assert np.array_equal(actual, expected)
    raw = c.transform(data, state, "raw")
    assert np.array_equal(raw, data) and not np.shares_memory(raw, data)
    assert np.max(np.abs(actual.mean(axis=0))) < 1e-6
    frozen = copy.deepcopy(state)
    c.transform(np.zeros((1800, 10), dtype=np.float32), state, "train_zscore")
    assert state == frozen  # Development values cannot estimate/update state.
    constant = c.normalization(np.ones((3600, 10), dtype=np.float32))
    assert constant["std"] == [1e-6] * 10
    assert not np.any(c.transform(np.ones((3, 10), dtype=np.float32), constant, "train_zscore"))


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan"])
def test_bad_training_normalizer_domain(kind):
    values = np.zeros((3600, 10), dtype=np.float32)
    if kind == "dtype":
        values = values.astype(np.float64)
    if kind == "shape":
        values = values[:10]
    if kind == "nan":
        values[0, 0] = np.nan
    with pytest.raises(ValueError, match="normalizer training"):
        c.normalization(values)


@pytest.mark.parametrize("kind", ["arm", "mean_shape", "std_zero", "nan", "overflow"])
def test_invalid_transform_is_bounded(kind):
    values = np.zeros((10, 10), dtype=np.float32)
    state = c.normalization(np.zeros((3600, 10), dtype=np.float32))
    arm = "train_zscore"
    if kind == "arm":
        arm = "wrong"
    if kind == "mean_shape":
        state["mean"] = [0]
    if kind == "std_zero":
        state["std"][0] = 0
    if kind == "nan":
        state["mean"][0] = math.nan
    if kind == "overflow":
        state["mean"][0] = 1e300
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        c.transform(values, state, arm)


@pytest.mark.parametrize("kind", ["spec", "permission", "attempt", "sources", "hash"])
def test_protocol_rejects_contract_and_source_tampering(tmp_path, kind):
    path = protocol(tmp_path)
    value = json.loads(path.read_text())
    if kind == "spec":
        value["spec"]["std_floor"] = 0.01
    if kind == "permission":
        value["real_fit_authorized"] = True
    if kind == "attempt":
        value["attempts_allowed"]["fit"] = True
    if kind == "sources":
        value["source_sha256"].pop(next(iter(value["source_sha256"])))
    if kind == "hash":
        value["source_sha256"][next(iter(value["source_sha256"]))] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="comparison"):
        c.load(tmp_path, path)


def fixture_metrics(auc=0.8):
    return {
        "auc_all": auc,
        "scores_sha256": "e" * 64,
        "cells": [
            {
                "method": method,
                "rate_percent": rate,
                "plane": plane,
                "roc_auc": auc,
                "stego_loss": 0.5,
                "balanced_loss": 0.5,
            }
            for method, rate, plane in c.base.wav_pilot.RECIPES
        ],
    }


def fixture_run(seed, arm):
    return {
        "seed": seed,
        "arm": arm,
        "domain": c.DOMAINS[arm],
        "epochs": 20,
        "optimizer_updates": 1140,
        "row_presentations": 72000,
        "initial_parameter_sha256": str(seed % 10) * 64,
        "final_parameter_sha256": "b" * 64,
        "parameter_change_norm": 0.5,
        "order_sha256": "c" * 64,
        "train": fixture_metrics(),
        "development": fixture_metrics(),
        "epoch_telemetry": [
            {
                **fixture_metrics(),
                "epoch": e,
                "optimizer_updates": 57 * e,
                "row_presentations": 3600 * e,
                "cover_loss": 0.5,
                "unweighted_loss": 0.5,
                "weighted_objective": 0.5,
            }
            for e in range(1, 21)
        ],
        "step_telemetry": [
            {"update": u, "objective": 0.5, "gradient_norm": 0.1, "parameter_step_norm": 0.01}
            for u in c.POINTS
        ],
    }


def fixture_result():
    return {
        "status": "completed_generated_comparison",
        "protocol_sha256": "sha",
        "generator_seed": c.GENERATOR_SEED,
        "total_wall_seconds": 50,
        "resources": {"host_peak_bytes": 1000, "swap_bytes": 0},
        "real_media_opened": False,
        "reserved_media_opened": False,
        "checkpoint_written": False,
        "real_fit_authorized": False,
        "production_qualified": False,
        "runs": [fixture_run(seed, arm) for seed in c.SPEC["seeds"] for arm in c.ARMS],
    }


def test_complete_acceptance_is_separate_per_arm_and_seed():
    result = fixture_result()
    assert set(c.decide(result, "sha", 100).values()) == {"passed_generated_only"}
    result["runs"][0]["development"]["cells"][2]["roc_auc"] = 0.69
    assert c.decide(result, "sha") == {
        "raw": "failed_learning_gate",
        "train_zscore": "passed_generated_only",
    }
    result["runs"][3]["train"]["auc_all"] = 0.59
    assert set(c.decide(result, "sha").values()) == {"failed_learning_gate"}


@pytest.mark.parametrize(
    "kind", ["missing", "changed", "count", "time", "hash", "mismatch", "resources"]
)
def test_profile_requires_both_equal_unchanged_initializations(prepared, kind):
    value = copy.deepcopy(prepared[2])
    if kind == "missing":
        value["smoke"] = []
    if kind == "changed":
        value["smoke"][0]["parameters_unchanged"] = False
    if kind == "count":
        value["smoke"][1]["backward_batches"] = 19
    if kind == "time":
        value["smoke"][0]["seconds"] = math.nan
    if kind == "hash":
        value["smoke"][0]["initial_parameter_sha256"] = "x" * 64
    if kind == "mismatch":
        value["smoke"][1]["initial_parameter_sha256"] = "a" * 64
    if kind == "resources":
        value["resources"]["swap_bytes"] = 1
    assert not c.profile_proof(value)


def test_transform_rejects_raw_invalid_inputs():
    with pytest.raises(ValueError, match="raw comparison"):
        c.transform(np.full((4, 10), np.nan, dtype=np.float32), {}, "raw")


@pytest.mark.parametrize(
    "kind",
    [
        "not_dict",
        "status",
        "protocol",
        "seed",
        "time",
        "external",
        "swap",
        "memory",
        "real",
        "missing",
        "arm",
        "domain",
        "updates",
        "unchanged",
        "digest",
        "matched",
        "metrics",
        "metric_type",
        "duplicate",
        "epoch_missing",
        "epoch_count",
        "loss",
        "cell_loss",
        "point_missing",
        "point_update",
        "point_nan",
    ],
)
def test_incomplete_or_corrupt_result_unavailable(kind):
    result = fixture_result()
    run = result["runs"][0]
    external = 100
    if kind == "not_dict":
        result = []
    if kind == "status":
        result["status"] = "running"
    if kind == "protocol":
        result["protocol_sha256"] = "wrong"
    if kind == "seed":
        result["generator_seed"] = True
    if kind == "time":
        result["total_wall_seconds"] = 1801
    if kind == "external":
        external = 1801
    if kind == "swap":
        result["resources"]["swap_bytes"] = 1
    if kind == "memory":
        result["resources"]["host_peak_bytes"] = 3 * 1024**3
    if kind == "real":
        result["real_fit_authorized"] = True
    if kind == "missing":
        result["runs"].pop()
    if kind == "arm":
        run["arm"] = "wrong"
    if kind == "domain":
        run["domain"] = "wrong"
    if kind == "updates":
        run["optimizer_updates"] = True
    if kind == "unchanged":
        run["parameter_change_norm"] = 0
    if kind == "digest":
        run["initial_parameter_sha256"] = "short"
    if kind == "matched":
        run["initial_parameter_sha256"] = "d" * 64
    if kind == "metrics":
        run["train"] = None
    if kind == "metric_type":
        run["development"]["cells"][0]["plane"] = True
    if kind == "duplicate":
        run["development"]["cells"][1] = run["development"]["cells"][0]
    if kind == "epoch_missing":
        run["epoch_telemetry"].pop()
    if kind == "epoch_count":
        run["epoch_telemetry"][0]["row_presentations"] = 3
    if kind == "loss":
        run["epoch_telemetry"][0]["unweighted_loss"] = math.inf
    if kind == "cell_loss":
        run["epoch_telemetry"][0]["cells"][0]["stego_loss"] = math.nan
    if kind == "point_missing":
        run["step_telemetry"].pop()
    if kind == "point_update":
        run["step_telemetry"][0]["update"] = True
    if kind == "point_nan":
        run["step_telemetry"][0]["gradient_norm"] = math.nan
    assert set(c.decide(result, "sha", external).values()) == {"unavailable"}


def cloned_cache(prepared, tmp_path):
    root, path, receipt = prepared
    out = tmp_path / "cache"
    out.mkdir()
    for item in (root / "cache").iterdir():
        (out / item.name).write_bytes(item.read_bytes())
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "protocol_sha256": receipt["protocol_sha256"],
                "profile_sha256": c.base.digest(out / "profile.json"),
                "profile_debit_seconds": 299,
                "fit_attempts_allowed": 1,
            }
        )
    )
    return root, path, out, binding


def test_full_driver_keeps_four_numeric_runs_and_scores_without_real_updates(
    prepared, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    root, path, out, binding = cloned_cache(prepared, tmp_path)
    monkeypatch.setattr(c.base, "resources", lambda: {"host_peak_bytes": 1000, "swap_bytes": 0})
    monkeypatch.setattr(c.base.old, "process_started", time.monotonic)
    seen = []

    def fake(torch, values, rows, seed, arm, expires, on_update, on_epoch):
        seen.append((seed, arm))
        record = fixture_run(seed, arm)
        for epoch in record["epoch_telemetry"]:
            on_epoch(epoch)
        for _ in range(1140):
            on_update()
        return c.base.candidate.network(), record

    monkeypatch.setattr(c, "fit_one", fake)
    result = c.fit(root, path, out, binding)
    assert seen == [(seed, arm) for seed in c.SPEC["seeds"] for arm in c.ARMS]
    assert len(result["runs"]) == 4
    assert (
        json.loads((out / "fit/progress.json").read_text())["confirmed_updates_lower_bound"] == 4560
    )
    for seed, arm in seen:
        dest = out / "fit" / f"{arm}-{seed}"
        assert len(list(dest.glob("epoch-*.json"))) == 20
        assert (dest / "fit.json").is_file() and (dest / "seed.json").is_file()
        for role, count in (("train", 3600), ("development", 1800)):
            a = np.load(io.BytesIO((dest / f"{role}-scores.npy").read_bytes()), allow_pickle=False)
            assert a.dtype == np.float64 and a.shape == (count,)
    with pytest.raises(FileExistsError):
        c.fit(root, path, out, binding)


@pytest.mark.parametrize(
    "kind",
    [
        "binding",
        "profile",
        "manifest",
        "normalizer",
        "cache",
        "order",
        "norm_semantic",
        "provenance",
    ],
)
def test_failures_precede_optimizer_or_fit(prepared, tmp_path, monkeypatch, kind):
    root, path, out, binding = cloned_cache(prepared, tmp_path)
    monkeypatch.setattr(c.base, "resources", lambda: {"host_peak_bytes": 1000, "swap_bytes": 0})
    monkeypatch.setattr(c, "fit_one", lambda *a: pytest.fail("fit reached after bad prerequisite"))
    p = json.loads((out / "profile.json").read_text())
    if kind == "binding":
        binding.write_text("{}")
    if kind == "profile":
        p["optimizer_updates"] = True
    if kind == "manifest":
        (out / "manifest.json").write_text("{}")
    if kind == "normalizer":
        (out / "normalization.json").write_text("{}")
    if kind == "cache":
        (out / "train.npy").write_bytes(b"broken")
    if kind == "order":
        p["row_order_sha256"]["train"] = "wrong"
    if kind == "provenance":
        m = json.loads((out / "manifest.json").read_text())
        m["generator_seed"] = 9
        (out / "manifest.json").write_text(json.dumps(m))
        p["manifest_sha256"] = c.base.digest(out / "manifest.json")
    if kind == "norm_semantic":
        n = json.loads((out / "normalization.json").read_text())
        n["mean"][0] += 0.001
        (out / "normalization.json").write_text(json.dumps(n))
        p["normalization_sha256"] = c.base.digest(out / "normalization.json")
    if kind not in ("binding", "manifest", "normalizer", "cache"):
        (out / "profile.json").write_text(json.dumps(p))
        b = json.loads(binding.read_text())
        b["profile_sha256"] = c.base.digest(out / "profile.json")
        binding.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="comparison|normalizer"):
        c.fit(root, path, out, binding)
    assert not (out / "fit").exists()


def test_first_arm_evidence_survives_later_failure(prepared, tmp_path, monkeypatch):
    root, path, out, binding = cloned_cache(prepared, tmp_path)
    monkeypatch.setattr(c.base, "resources", lambda: {"host_peak_bytes": 1000, "swap_bytes": 0})
    monkeypatch.setattr(c.base.old, "process_started", time.monotonic)

    def fake(torch, values, rows, seed, arm, expires, on_update, on_epoch):
        if arm == "train_zscore":
            raise RuntimeError("later fit failed")
        on_update()
        return c.base.candidate.network(), fixture_run(seed, arm)

    monkeypatch.setattr(c, "fit_one", fake)
    with pytest.raises(RuntimeError, match="later"):
        c.fit(root, path, out, binding)
    assert (out / "fit/raw-20261016/fit.json").is_file()
    assert (out / "fit/raw-20261016/seed.json").is_file()
    with pytest.raises(FileExistsError):
        c.fit(root, path, out, binding)


def test_full_schedule_actual_gradients_and_telemetry_no_optimizer_updates(prepared, monkeypatch):
    torch = pytest.importorskip("torch")
    root, _, _ = prepared
    rows = c.records(root, json.loads((root / "cache/manifest.json").read_text()))["train"]
    values = c.base.read_array(root / "cache/train.npy", 3600)
    calls, saved = [], []

    class NoUpdate:
        def __init__(self, groups, **kwargs):
            assert kwargs == {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
            }
            self.params = [p for g in groups for p in g["params"]]
            self.groups = groups
            self.before = [p.detach().clone() for p in self.params]
            saved.append(self)

        def zero_grad(self, **kwargs):
            for p in self.params:
                p.grad = None

        def step(self):
            assert all(torch.isfinite(p.grad).all() for p in self.params)
            calls.append([p.grad.detach().clone() for p in self.params])

    monkeypatch.setattr(torch.optim, "AdamW", NoUpdate)
    updates, epochs = [], []
    model, run = c.fit_one(
        torch, values, rows, 20261016, "raw", math.inf, lambda: updates.append(1), epochs.append
    )
    assert len(updates) == len(calls) == 1140 and len(epochs) == 20
    assert run["row_presentations"] == 72000 and run["parameter_change_norm"] == 0
    assert [v["update"] for v in run["step_telemetry"]] == list(c.POINTS)
    assert all(v["parameter_step_norm"] == 0 for v in run["step_telemetry"])
    assert all(torch.equal(p, b) for p, b in zip(saved[0].params, saved[0].before, strict=True))
    # Independent per-row scalar gradient sum checks the first and last actual batch.
    labels = [int(r["recipe"] is not None) for r in rows]
    order = np.random.default_rng(20261016 ^ 0x4E4A5047).permutation(3600)
    for offset, call in ((0, calls[0]), (3584, calls[56])):
        batch = order[offset : offset + 64]
        model.zero_grad(set_to_none=True)
        objective = sum(
            torch.nn.functional.cross_entropy(
                model(torch.from_numpy(values[i : i + 1])), torch.tensor([labels[i]])
            )
            * (4.5 if labels[i] == 0 else 0.5625)
            for i in batch
        ) / len(batch)
        objective.backward()
        for parameter, expected in zip(saved[0].params, call, strict=True):
            assert torch.allclose(parameter.grad, expected, rtol=3e-5, atol=2e-7)
    assert model.training is True


def test_telemetry_matches_scalar_losses_and_keeps_parameters(prepared):
    torch = pytest.importorskip("torch")
    root, _, _ = prepared
    rows = c.records(root, json.loads((root / "cache/manifest.json").read_text()))["train"]
    values = torch.from_numpy(c.base.read_array(root / "cache/train.npy", 3600))
    labels = torch.tensor([int(r["recipe"] is not None) for r in rows])
    model = c.base.candidate.network()
    before = c.parameter_sha(model)
    report = c.telemetry(torch, model, values, labels, rows)
    with torch.inference_mode():
        losses = [
            float(torch.nn.functional.cross_entropy(model(values[i : i + 1]), labels[i : i + 1]))
            for i in range(len(values))
        ]
    assert math.isclose(report["unweighted_loss"], math.fsum(losses) / 3600, abs_tol=1e-7)
    assert math.isclose(
        report["weighted_objective"],
        math.fsum(v * (4.5 if i % 9 == 0 else 0.5625) for i, v in enumerate(losses)) / 3600,
        abs_tol=1e-7,
    )
    assert c.parameter_sha(model) == before
    model[0].weight.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        c.telemetry(torch, model, values, labels, rows)


def test_generation_storage_and_deadline_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(c.base, "MIB", 1)
    with pytest.raises(ValueError, match="storage"):
        c.generate(tmp_path, {"train": [640]}, math.inf)
    with pytest.raises(TimeoutError):
        c.generate(tmp_path, {"train": [640]}, -1)


def test_cli_routes_and_bounded_unavailability(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(cli.service, "profile", lambda *a: calls.append(a) or {"status": "passed"})
    args = ["--protocol", str(tmp_path / "p"), "--cache", str(tmp_path / "cache")]
    assert cli.main(["profile", *args]) == 0 and len(calls) == 1
    assert cli.main(["fit", *args]) == 2
    monkeypatch.setattr(
        cli.service,
        "fit",
        lambda *a: {"status": "done", "decisions": {"raw": "failed_learning_gate"}},
    )
    assert cli.main(["fit", *args, "--binding", str(tmp_path / "binding")]) == 0

    def missing(*a):
        raise ImportError("optional Torch missing")

    monkeypatch.setattr(cli.service, "profile", missing)
    assert cli.main(["profile", *args]) == 2 and "unavailable" in capsys.readouterr().out
