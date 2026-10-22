"""Fresh CPU control contracts, independent accounting and failure preservation."""

import builtins
import copy
import io
import json
import math
import time

import numpy as np
import pytest

from core import wav_pair_control as c
from steganography import research_wav_pair_control as cli


def make_root(root):
    root.mkdir(exist_ok=True)
    for name in c.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture-source")
    (root / c.old.CORPUS).write_text(
        json.dumps({"files": [{"sha256": "old", "decoded_pcm_sha256": "old"}]})
    )
    (root / "benchmarks/wav-audiomnist-preparation-20261017.json").write_text(
        json.dumps(
            {
                "files": [
                    {"role": role, "frames": 640}
                    for role, n in (("train", 400), ("development", 200))
                    for _ in range(n)
                ]
            }
        )
    )
    protocol = root / "protocol.json"
    write_protocol(root, protocol)
    return protocol


def write_protocol(root, path):
    value = {
        "schema_version": "wav-pair-control-protocol-v1",
        "protocol_id": c.PROTOCOL,
        "spec": c.SPEC,
        "real_fit_authorized": False,
        "attempts_allowed": {"profile": 1, "fit": 1},
        "source_sha256": {n: c.digest(root / n) for n in c.SOURCES},
    }
    path.write_text(json.dumps(value))


def measured():
    return {"host_peak_bytes": 1024, "swap_bytes": 0, "process_rss_peak_bytes": 1024}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    torch = pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("pair-profile")
    protocol = make_root(root)
    prior = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, "resources", measured)
        patch.setattr(c.old, "process_started", time.monotonic)
        receipt = c.profile(root, protocol, root / "cache")
    torch.set_num_threads(prior[0])
    torch.use_deterministic_algorithms(prior[1])
    return root, protocol, receipt


def test_full_zero_update_profile_builds_disjoint_bound_cache(prepared):
    root, protocol, receipt = prepared
    cache = root / "cache"
    assert receipt["optimizer_updates"] == 0 and receipt["parameters_unchanged"]
    assert receipt["status"] == "passed_zero_update_profile"
    assert c.load(root, protocol)[1] == receipt["protocol_sha256"]
    manifest = json.loads((cache / "manifest.json").read_text())
    assert len(manifest["files"]) == 600 and manifest["payload_oracles_passed"] == 4800
    assert manifest["generator_seed"] == 2026102201
    assert not list(cache.glob("generated-only-*"))
    assert c.read_array(cache / "train.npy", 3600).shape == (3600, 10)
    assert c.read_array(cache / "development.npy", 1800).shape == (1800, 10)


@pytest.mark.parametrize("kind", ["spec", "permission", "sources", "bool_attempt", "source_hash"])
def test_bad_protocols_rejected(tmp_path, kind):
    protocol = make_root(tmp_path)
    value = json.loads(protocol.read_text())
    if kind == "spec":
        value["spec"]["epochs"] = True
    elif kind == "permission":
        value["real_fit_authorized"] = True
    elif kind == "sources":
        value["source_sha256"].pop(next(iter(value["source_sha256"])))
    elif kind == "bool_attempt":
        value["attempts_allowed"]["fit"] = True
    else:
        value["source_sha256"][next(iter(value["source_sha256"]))] = "0" * 64
    protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="contract|binding"):
        c.load(tmp_path, protocol)


def test_output_symlink_overwrite_and_array_bounds(tmp_path):
    out = tmp_path / "out"
    c.new_output(out)
    with pytest.raises(FileExistsError):
        c.new_output(out)
    alias = tmp_path / "link"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        c.new_output(alias / "new")
    with pytest.raises(ValueError, match="bound"):
        c.array_save(out / "big.npy", np.zeros(2 * c.MIB))
    c.array_save(out / "ok.npy", np.zeros((160, 10), dtype=np.float32))
    with pytest.raises(FileExistsError):
        c.array_save(out / "ok.npy", np.zeros((160, 10)))
    alias = out / "alias.npy"
    alias.symlink_to(out / "ok.npy")
    with pytest.raises(ValueError, match="symlink"):
        c.read_array(alias, 160)


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan", "range", "pickle", "corrupt", "npz"])
def test_bad_array_data_rejected(tmp_path, kind):
    values = np.zeros((2, 10), dtype=np.float32)
    if kind == "dtype":
        values = values.astype(np.float64)
    elif kind == "shape":
        values = values[:, :9]
    elif kind == "nan":
        values[0, 0] = np.nan
    elif kind == "range":
        values[0, 0] = 1.1
    elif kind == "pickle":
        values = np.array([{}], dtype=object)
    buf = io.BytesIO()
    if kind == "npz":
        np.savez(buf, values)
    else:
        np.save(buf, values)
    path = tmp_path / "cache.npy"
    path.write_bytes(b"bad" if kind == "corrupt" else buf.getvalue())
    with pytest.raises(ValueError):
        c.read_array(path, 2)


def test_resource_and_runtime_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(c.old, "cgroup", lambda: tmp_path)
    for name, value in {
        "memory.max": str(2 * 1024**3),
        "memory.peak": "42",
        "memory.swap.current": "0",
    }.items():
        (tmp_path / name).write_text(value)
    assert c.resources()["host_peak_bytes"] == 42
    (tmp_path / "memory.max").write_text(str(8 * 1024**3))
    with pytest.raises(ValueError, match="cap"):
        c.resources()
    (tmp_path / "memory.max").write_text(str(2 * 1024**3))
    (tmp_path / "memory.swap.current").write_text("1")
    with pytest.raises(ValueError, match="usage"):
        c.resources()
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "__version__", "wrong")
    with pytest.raises(RuntimeError, match="unavailable"):
        c.runtime()


@pytest.mark.parametrize("kind", ["count", "domain", "group", "pcm", "old"])
def test_fresh_manifest_rejects_overlap_and_wrong_domain(prepared, kind):
    root, _, _ = prepared
    records = json.loads((root / "cache/manifest.json").read_text())["files"]
    previous = {"files": [{"sha256": "old", "decoded_pcm_sha256": "old"}]}
    if kind == "count":
        records.pop()
    elif kind == "domain":
        records[0]["frames"] = True
    elif kind == "group":
        records[400]["speaker_group"] = records[0]["speaker_group"]
    elif kind == "pcm":
        records[1]["decoded_pcm_sha256"] = records[0]["decoded_pcm_sha256"]
    else:
        previous["files"][0]["sha256"] = records[0]["sha256"]
    with pytest.raises(ValueError):
        c.validate_records(records, previous)


def test_generated_storage_duplicate_and_deadline_guards(tmp_path, monkeypatch):
    schedule = {"train": [640, 640], "development": []}
    monkeypatch.setattr(c.old, "speech_like", lambda rng, n: np.zeros(n, dtype="<i2"))
    with pytest.raises(ValueError, match="duplicate"):
        c.generated(tmp_path, schedule, time.monotonic() + 10)
    with pytest.raises(TimeoutError):
        c.generated(tmp_path, schedule, 0)
    monkeypatch.setattr(c, "MIB", 1)
    with pytest.raises(ValueError, match="storage"):
        c.generated(tmp_path, {"train": [640], "development": []}, time.monotonic() + 10)


def test_profile_failure_keeps_seal_and_forbids_overwrite(tmp_path, monkeypatch):
    protocol = make_root(tmp_path)
    monkeypatch.setattr(c, "resources", measured)
    monkeypatch.setattr(c.old, "process_started", lambda: time.monotonic() - 1000)
    monkeypatch.setattr(c, "runtime", lambda: None)
    out = tmp_path / "failed"
    with pytest.raises(TimeoutError):
        c.profile(tmp_path, protocol, out)
    assert (out / "profile-started.json").is_file()
    assert not (out / "profile.json").exists()
    with pytest.raises(FileExistsError):
        c.profile(tmp_path, protocol, out)


def valid_result():
    metrics = {
        "auc_all": 0.60,
        "cells": [
            {"method": m, "rate_percent": r, "plane": p, "roc_auc": 0.70}
            for m, r, p in c.wav_pilot.RECIPES
        ],
    }
    return {
        "status": "completed_generated_pair_control",
        "protocol_sha256": "bound",
        "domain": c.candidate.DOMAIN,
        "generator_seed": c.SEED,
        "generated_only": True,
        "real_fit_authorized": False,
        "production_qualified": False,
        "real_media_opened": False,
        "reserved_media_opened": False,
        "checkpoint_written": False,
        "total_wall_seconds": 1.0,
        "resources": measured(),
        "seeds": [
            {
                "seed": s,
                "epochs": 20,
                "optimizer_updates": 1140,
                "row_presentations": 72000,
                "train": copy.deepcopy(metrics),
                "development": copy.deepcopy(metrics),
            }
            for s in c.SPEC["seeds"]
        ],
    }


def test_acceptance_exact_thresholds_and_complete_failure():
    r = valid_result()
    assert c.decide(r, "bound") == "passed_generated_only"
    assert c.decide(r, "bound", 1800.1) == "unavailable"
    r["seeds"][1]["development"]["cells"][2]["roc_auc"] = 0.699999
    assert c.decide(r, "bound") == "failed_learning_gate"
    assert c.decide(r, "wrong") == c.decide(None, "bound") == "unavailable"


@pytest.mark.parametrize(
    "kind",
    [
        "time",
        "swap",
        "host",
        "missing_seed",
        "bool_seed",
        "seed_object",
        "metric_object",
        "auc",
        "cells",
        "cell",
        "duplicate",
    ],
)
def test_acceptance_incomplete_or_malformed_is_unavailable(kind):
    r = valid_result()
    if kind == "time":
        r["total_wall_seconds"] = float("nan")
    elif kind == "swap":
        r["resources"]["swap_bytes"] = True
    elif kind == "host":
        r["resources"]["host_peak_bytes"] = "bad"
    elif kind == "missing_seed":
        r["seeds"].pop()
    elif kind == "bool_seed":
        r["seeds"][0]["epochs"] = True
    elif kind == "seed_object":
        r["seeds"][0] = None
    elif kind == "metric_object":
        r["seeds"][0]["train"] = None
    elif kind == "auc":
        r["seeds"][0]["train"]["auc_all"] = float("inf")
    elif kind == "cells":
        r["seeds"][0]["train"]["cells"] = []
    elif kind == "cell":
        r["seeds"][0]["train"]["cells"][0] = None
    else:
        r["seeds"][0]["train"]["cells"][0] = r["seeds"][0]["train"]["cells"][1]
    assert c.decide(r, "bound") == "unavailable"


def test_actual_full_schedule_with_fake_nonmutating_optimizer(monkeypatch):
    torch = pytest.importorskip("torch")
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    steps, calls = [], []

    class NoUpdate:
        def __init__(self, groups, lr, **kwargs):
            assert lr == 0.001
            assert kwargs == {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
            }
            assert [g["weight_decay"] for g in groups] == [0.0001, 0]
            self.params = [p for g in groups for p in g["params"]]
            self.before = [p.detach().clone() for p in self.params]

        def zero_grad(self, set_to_none):
            assert set_to_none
            for p in self.params:
                p.grad = None

        def step(self):
            if len(steps) in (0, 56):
                order = np.random.default_rng(20261016 ^ 0x4E4A5047).permutation(3600)
                indices = order[:64] if not steps else order[-16:]
                with torch.random.fork_rng():
                    manual = c.candidate.network()
                manual_params = [manual[0].weight, manual[2].weight, manual[0].bias, manual[2].bias]
                with torch.no_grad():
                    for p, original in zip(manual_params, self.before, strict=True):
                        p.copy_(original)
                for index in indices:
                    label = int(index >= 400)
                    weight = 0.5625 if label else 4.5
                    loss = torch.nn.functional.cross_entropy(
                        manual(torch.zeros((1, 10))), torch.tensor([label])
                    )
                    (loss * weight / len(indices)).backward()
                assert all(
                    torch.allclose(p.grad, expected.grad, rtol=1e-5, atol=1e-7)
                    for p, expected in zip(self.params, manual_params, strict=True)
                )
            steps.append(1)
            assert all(torch.equal(p, old) for p, old in zip(self.params, self.before, strict=True))

    monkeypatch.setattr(torch.optim, "AdamW", NoUpdate)
    x = torch.zeros((3600, 10))
    y = torch.tensor([0] * 400 + [1] * 3200)
    model, receipt = c.fit_seed(
        torch, x, y, 20261016, time.monotonic() + 120, lambda: calls.append(1)
    )
    assert len(steps) == len(calls) == receipt["optimizer_updates"] == 1140
    assert receipt["row_presentations"] == 72000 and receipt["epochs"] == 20
    p = c.predict(torch, model, x.numpy())
    assert p.shape == (3600,) and p.dtype == np.float64 and np.isfinite(p).all()
    with pytest.raises(TimeoutError):
        c.fit_seed(torch, x, y, 20261016, 0, lambda: None)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="scores"):
        c.predict(torch, model, x.numpy())
    with pytest.raises(ValueError, match="gradients"):
        c.finite_gradients(torch, c.candidate.network())
    torch.set_num_threads(prior)


def fitting_copy(prepared, tmp_path, monkeypatch):
    import shutil

    source, _, _ = prepared
    root = tmp_path / "copy"
    shutil.copytree(source, root)
    cache = root / "cache"
    binding = root / "binding.json"
    c.old.save(
        binding,
        {
            "protocol_sha256": c.load(root, root / "protocol.json")[1],
            "profile_sha256": c.digest(cache / "profile.json"),
            "fit_attempts_allowed": 1,
            "profile_debit_seconds": math.ceil(
                json.loads((cache / "profile.json").read_text())["profile_wall_seconds"]
            )
            + 1,
        },
    )
    monkeypatch.setattr(c, "resources", measured)
    monkeypatch.setattr(c.old, "process_started", time.monotonic)
    return root, cache, binding


def test_fit_keeps_seed_one_if_seed_two_fails_and_never_restarts(prepared, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root, cache, binding = fitting_copy(prepared, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "runtime", lambda: torch)

    def fake(torch, x, labels, seed, expires, callback):
        if seed == 20261017:
            raise RuntimeError("second-seed failure")
        callback()
        return c.candidate.network(), {
            "seed": seed,
            "epochs": 20,
            "optimizer_updates": 1140,
            "row_presentations": 72000,
        }

    monkeypatch.setattr(c, "fit_seed", fake)
    with pytest.raises(RuntimeError, match="second-seed"):
        c.fit(root, root / "protocol.json", cache, binding)
    assert (cache / "fit/fit-20261016.json").is_file()
    assert (cache / "fit/seed-20261016.json").is_file()
    assert not (cache / "fit/result.json").exists()
    with pytest.raises(FileExistsError):
        c.fit(root, root / "protocol.json", cache, binding)


@pytest.mark.parametrize("kind", ["binding", "debit", "manifest", "cache", "split", "projection"])
def test_fit_preflight_rejects_wrong_bindings_before_updates(prepared, tmp_path, monkeypatch, kind):
    root, cache, binding = fitting_copy(prepared, tmp_path, monkeypatch)
    if kind == "binding":
        binding.write_text("{}")
    elif kind == "cache":
        (cache / "train.npy").write_bytes(b"corrupt")
    elif kind == "manifest":
        (cache / "manifest.json").write_text("{}")
    else:
        profile = json.loads((cache / "profile.json").read_text())
        if kind == "debit":
            profile["profile_wall_seconds"] = 301
        elif kind == "projection":
            profile["conservative_total_projection_seconds"] = 1801
        else:
            profile["row_order_sha256"]["train"] = "wrong"
        (cache / "profile.json").write_text(json.dumps(profile))
        b = json.loads(binding.read_text())
        b["profile_sha256"] = c.digest(cache / "profile.json")
        binding.write_text(json.dumps(b))
    with pytest.raises(ValueError):
        c.fit(root, root / "protocol.json", cache, binding)
    assert not (cache / "fit").exists()


def test_fit_completion_uses_new_acceptance_not_old_gate(prepared, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    root, cache, binding = fitting_copy(prepared, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "runtime", lambda: torch)

    def fake(torch, x, labels, seed, expires, callback):
        callback()
        return c.candidate.network(), {
            "seed": seed,
            "epochs": 20,
            "optimizer_updates": 1140,
            "row_presentations": 72000,
        }

    monkeypatch.setattr(c, "fit_seed", fake)
    r = c.fit(root, root / "protocol.json", cache, binding)
    assert r["decision"] == c.decide(r, r["protocol_sha256"])
    assert len(r["seeds"]) == 2 and r["real_fit_authorized"] is False
    assert set(r["fixed_global_plane0"]) == {"train", "development"}
    assert json.loads((cache / "fit/progress.json").read_text())["phase"] == "completed"


def test_thin_adapter_routes_and_errors(monkeypatch):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(c, "profile", lambda *args: {"status": "passed_zero_update_profile"})
    assert cli.main(["profile", "--protocol", "p", "--cache", "c"]) == 0
    monkeypatch.setattr(
        c, "fit", lambda *args: {"status": "completed", "decision": "failed_learning_gate"}
    )
    assert cli.main(["fit", "--protocol", "p", "--cache", "c", "--binding", "b"]) == 2
    assert cli.main(["fit", "--protocol", "p", "--cache", "c"]) == 2
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError):
        c.runtime()
