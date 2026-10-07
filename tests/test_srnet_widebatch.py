"""Independent eight-row matching, optimizer accounting and fail-closed replay."""

import copy
import hashlib
import itertools
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet, srnet_positive, srnet_training
from core import srnet_widebatch as wide
from steganography import research_srnet_widebatch as service
from tests.test_srnet_signal import data


def records():
    return [{"epoch": e, "updates": 4, "mean_pair_loss": 0.693} for e in range(20)]


def test_all_epochs_exact_exposures_distinct_context_hashes_and_first_matching():
    _, rows = data()
    original = copy.deepcopy(rows)
    rng = np.random.get_state()
    for e in range(20):
        groups, _ = wide.srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=e)
        actual, report = wide.epoch_batches(rows, seed=20261012, epoch=e)
        assert actual.shape == (4, 8) and not actual.flags.writeable
        assert sorted(map(tuple, actual.reshape(-1, 2))) == sorted(
            map(tuple, groups.reshape(-1, 2))
        )
        identities = []
        for batch in actual:
            assert [rows[i]["label"] for i in batch] == ["cover", "stego"] * 4
            assert len({(rows[i]["source_group"], rows[i]["lineage"]) for i in batch[::2]}) == 4
            sources = [rows[i]["source_group"] for i in batch[::2]]
            assert sources.count("ALASKA2") == sources.count("BOSSbase-1.01") == 2
            identities.append([rows[i]["sha256"] for i in batch])
        assert (
            report["ordered_batch_sha256"]
            == hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
        )
        assert report["rows"] == 32 and report["optimizer_updates"] == 4
        # Independent exhaustive permutation oracle (first lexicographic full pairing).
        valid = []
        for perm in itertools.permutations(range(8)):
            pairs = tuple((perm[i], perm[i + 1]) for i in range(0, 8, 2))
            if any(a > b for a, b in pairs) or tuple(a for a, _ in pairs) != tuple(
                sorted(a for a, _ in pairs)
            ):
                continue
            if all(
                len(
                    {
                        (rows[i]["source_group"], rows[i]["lineage"])
                        for i in np.concatenate((groups[a], groups[b]))[::2]
                    }
                )
                == 4
                for a, b in pairs
            ):
                valid.append(pairs)
        assert tuple(map(tuple, report["base_group_pairing"])) == min(valid)
        assert wide.epoch_batches(rows, seed=20261012, epoch=e)[1] == report
    assert rows == original and np.array_equal(np.random.get_state()[1], rng[1])


def test_cardinality_missing_complete_matching_and_base_group_limit(monkeypatch):
    _, rows = data()
    with pytest.raises(ValueError, match="24"):
        wide.epoch_batches(rows[:-1], seed=1, epoch=0)
    base = wide.srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=0)
    monkeypatch.setattr(
        wide.srnet_multibatch, "epoch_batches", lambda *a, **kw: (base[0][:6], base[1])
    )
    with pytest.raises(ValueError, match="eight"):
        wide.epoch_batches(rows, seed=1, epoch=0)
    fake = np.tile(base[0][0], (8, 1))
    monkeypatch.setattr(wide.srnet_multibatch, "epoch_batches", lambda *a, **kw: (fake, base[1]))
    with pytest.raises(ValueError, match="cannot form"):
        wide.epoch_batches(rows, seed=1, epoch=0)


@pytest.mark.parametrize("bad", [True, float("nan"), -1, "bad"])
def test_objective_invalid_loss_and_count_fail(bad):
    r = records()
    r[-1]["mean_pair_loss"] = bad
    with pytest.raises(ValueError):
        wide.objectives(r, {"balanced_accuracy": 0.5})
    r = records()
    r[0]["updates"] = 8
    with pytest.raises(ValueError):
        wide.objectives(r, {"balanced_accuracy": 0.5})
    with pytest.raises(ValueError):
        wide.objectives(records()[:-1], {"balanced_accuracy": 0.5})


def test_actual_eight_row_forward_backward_mapping_and_exception_restoration(monkeypatch):
    torch = pytest.importorskip("torch")
    pixels, rows = data()
    before, metadata = pixels.copy(), copy.deepcopy(rows)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    observed, nets, initial = [], [], []
    factory = srnet.network

    def network():
        net = factory()
        nets.append(net)
        initial.extend(p.detach().clone() for p in net.parameters())

        def hook(_, args):
            observed.append(args[0].detach().numpy().copy())
            if len(observed) == 2:
                raise RuntimeError("bounded after one verified optimizer step")

        net.register_forward_pre_hook(hook)
        return net

    monkeypatch.setattr(srnet, "network", network)
    paired = [wide.epoch_pairs(rows, seed=20261012, epoch=e)[1] for e in range(20)]
    with pytest.raises(RuntimeError, match="bounded"):
        srnet_training.fit(
            pixels,
            rows,
            np.arange(24),
            seed=20261012,
            schedule=paired,
            config={**srnet_positive.PARAMS, "batch_recipe": wide.srnet_multibatch.RECIPE},
            wide_context=True,
        )
    expected, _ = wide.epoch_batches(rows, seed=20261012, epoch=0)
    for actual, indices in zip(observed, expected[:2], strict=True):
        np.testing.assert_array_equal(actual, pixels[indices])
    assert all(int(v) == 1 for k, v in nets[0].named_buffers() if k.endswith("num_batches_tracked"))
    # The next iteration cleared gradients before its deliberately failing hook.
    assert all(p.grad is None for p in nets[0].parameters())
    assert any(
        not torch.equal(p, before) for p, before in zip(nets[0].parameters(), initial, strict=True)
    )
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    np.testing.assert_array_equal(pixels, before)
    assert rows == metadata
    for flag, recipe, seed, schedule in (
        (1, wide.srnet_multibatch.RECIPE, 20261012, paired),
        (True, None, 20261012, paired),
        (True, wide.srnet_multibatch.RECIPE, 1, paired),
        (True, wide.srnet_multibatch.RECIPE, 20261012, paired[:1]),
    ):
        with pytest.raises(ValueError, match="restricted"):
            srnet_training.fit(
                pixels,
                rows,
                np.arange(24),
                seed=seed,
                schedule=schedule,
                config={} if recipe is None else {"batch_recipe": recipe},
                wide_context=flag,
            )


@pytest.fixture
def fake(monkeypatch):
    import torch

    model = SimpleNamespace(
        named_buffers=lambda: [(f"bn{i}.num_batches_tracked", 80) for i in range(26)],
        state_dict=lambda: {"counter": torch.tensor(80)},
    )

    def fit(pixels, samples, indices, **kwargs):
        assert kwargs["wide_context"] is True and len(kwargs["schedule"]) == 20
        assert kwargs["seed"] == 20261012 and kwargs["config"] == {
            **srnet_positive.PARAMS,
            "batch_recipe": wide.srnet_multibatch.RECIPE,
        }
        assert indices.tolist() == list(range(24)) and not pixels.flags.writeable
        return model, records()

    monkeypatch.setattr(srnet_training, "fit", fit)
    monkeypatch.setattr(
        wide.srnet_signal,
        "evaluate",
        lambda m, p, s: (np.zeros((24, 2)), {"balanced_accuracy": 0.5, "cross_entropy": 0.693}),
    )
    monkeypatch.setattr(wide.srnet_reference, "reference_logits", lambda *a: np.zeros((1, 2)))
    return model


def test_learn_reload_replay_failures_and_numerical_separation(fake, monkeypatch):
    pixels, rows = data()
    model, arm = wide.learn(pixels, rows, factor=32)
    assert model is fake and arm["optimizer_updates"] == 80 and arm["presented_rows"] == 640
    assert not arm["learning_objectives_passed"] and not arm["optimizer_steps_matched"]
    audit = wide.audit(model, pixels, rows, arm)
    assert audit["numerical_gates_passed"] and len(audit["independent_oracles"]) == 9
    monkeypatch.setattr(wide.srnet_reference, "reference_logits", lambda *a: np.ones((1, 2)))
    assert not wide.audit(model, pixels, rows, arm)["numerical_gates_passed"]


@pytest.mark.parametrize(
    "field",
    [
        "derived_tensor_sha256",
        "epoch_pair_schedule",
        "epoch_batch_schedule",
        "optimizer",
        "seed",
        "batch_recipe",
        "optimizer_updates",
        "presented_rows",
        "baseline_optimizer_updates",
        "baseline_presented_rows",
        "optimizer_steps_matched",
        "own_input_singleton_logits",
        "original_input_train_metrics",
        "learning_objectives",
        "learning_objectives_passed",
    ],
)
def test_forged_accounting_and_replay_rejected(fake, field):
    pixels, rows = data()
    _, arm = wide.learn(pixels, rows, factor=1)
    if field.endswith("singleton_logits"):
        arm[field][0][0] = 1
    elif field.endswith("train_metrics"):
        arm[field]["balanced_accuracy"] = 1
    elif field == "learning_objectives":
        arm[field]["final_batch_loss_at_most_0_35"] = True
    else:
        arm[field] = None
    with pytest.raises(ValueError):
        wide.audit(fake, pixels, rows, arm)


def test_counter_and_oracle_coverage_require_complete_models(fake):
    pixels, rows = data()
    _, arm = wide.learn(pixels, rows, factor=1)
    fake.named_buffers = lambda: [("counter.num_batches_tracked", 80)]
    with pytest.raises(ValueError, match="BN"):
        wide.audit(fake, pixels, rows, arm)
    fake.named_buffers = lambda: [(f"bn{i}.num_batches_tracked", 80) for i in range(26)]
    # A valid two-source generated corpus has only six oracle cells, not nine.
    _, generated = srnet_positive.generate()
    _, arm = wide.learn(pixels, generated, factor=1)
    with pytest.raises(ValueError, match="coverage"):
        wide.audit(fake, pixels, generated, arm)


@pytest.fixture
def setup_service(fake, monkeypatch):
    pixels, rows = data()
    root = service.Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "benchmarks/srnet-signal-strength-20261007.json").read_bytes())
    baseline["selected_rows"] = [
        {
            k: s[k]
            for k in ("sha256", "source_group", "lineage", "quality_factor", "label", "method")
        }
        for s in rows
    ]
    baseline["train_data_sha256"] = "d" * 64
    for arm in baseline["arms"]:
        arm["derived_tensor_sha256"] = wide.srnet_signal.prepare(
            pixels, rows, factor=arm["factor"]
        )[1]
    monkeypatch.setattr(
        service,
        "read_document",
        lambda path: (
            (baseline, service.BASELINE_SHA)
            if path.name.startswith("srnet-signal")
            else ({}, service.MANIFEST_SHA)
        ),
    )
    monkeypatch.setattr(
        service, "load_pixels", lambda *a, **kw: (pixels, rows, {"data_sha256": "d" * 64})
    )
    monkeypatch.setattr(service.srnet_sanity, "select", lambda samples: np.arange(24))
    monkeypatch.setattr(service.srnet_model, "save_model", lambda *a: "a" * 64)
    monkeypatch.setattr(service.srnet_model, "load_model", lambda *a, **kw: fake)
    return {
        "manifest": "secret/manifest.json",
        "manifest_sha256": service.MANIFEST_SHA,
        "cache": "secret/cache.json",
        "cache_sha256": service.CACHE_SHA,
    }, baseline


def test_service_complete_fresh_output_no_paths_and_failed_objectives(setup_service, tmp_path):
    config, _ = setup_service
    out = tmp_path / "result"
    report = service.run(config, out)
    assert report["status"] == "completed" and len(report["arms"]) == 2
    assert not report["deployed"] and not report["validation_pixels_loaded"]
    assert report["row_exposure_matched"] and not report["optimizer_steps_matched"]
    assert all(not a["learning_objectives_passed"] for a in report["arms"])
    assert "secret/" not in (out / "control.json").read_text()
    with pytest.raises(FileExistsError):
        service.run(config, out)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        service.run(config, link / "new")


@pytest.mark.parametrize(
    "bad", ["config", "protocol", "baseline", "manifest", "metadata", "tensor", "deadline"]
)
def test_service_rejects_mismatch_and_never_completes(setup_service, monkeypatch, tmp_path, bad):
    config, baseline = setup_service
    if bad == "config":
        config["cache_sha256"] = "0" * 64
    if bad == "protocol":
        monkeypatch.setattr(service, "PROTOCOL_SHA", "0" * 64)
    if bad == "baseline":
        monkeypatch.setattr(service, "read_document", lambda p: (baseline, "0" * 64))
    if bad == "manifest":
        monkeypatch.setattr(
            service,
            "read_document",
            lambda p: (
                (baseline, service.BASELINE_SHA)
                if p.name.startswith("srnet-signal")
                else ({}, "0" * 64)
            ),
        )
    if bad == "metadata":
        baseline["train_data_sha256"] = "0" * 64
    if bad == "tensor":
        baseline["arms"][0]["derived_tensor_sha256"] = ["0" * 64] * 24
    if bad == "deadline":
        ticks = iter((0, 4000))
        monkeypatch.setattr(service.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError):
        service.run(config, tmp_path / "out")
    assert not (tmp_path / "out/control.json").exists()


def test_cli_bounds_completed_fail_success_and_exception_redaction(monkeypatch, capsys):
    import resource

    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["wide", "--config", "config.json", "--out", "out"])
    monkeypatch.setattr(service, "read_document", lambda *a: ({}, ""))
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"learning_objectives_passed": False, "audit": {"numerical_gates_passed": True}}
            ]
        },
    )
    assert service.main() == 2 and len(calls) == 4
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"learning_objectives_passed": True, "audit": {"numerical_gates_passed": True}}
            ]
        },
    )
    assert service.main() == 0

    def fail(*a):
        raise ValueError("private/password=secret")

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 2 and "secret" not in capsys.readouterr().out
