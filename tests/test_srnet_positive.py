"""Generated positive control is reproducible, bounded and never accuracy evidence."""

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_positive as positive
from core import srnet_positive_audit as replay
from steganography import research_srnet_positive as service

ROOT = Path(__file__).resolve().parents[1]


def records():
    return [{"epoch": e, "updates": 8, "mean_pair_loss": 1 - 0.8 * e / 19} for e in range(20)]


def test_generated_scalar_recipe_hashes_sampler_and_rng():
    before = np.random.get_state()
    pixels, rows = positive.generate()
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    repeated, metadata = positive.generate()
    np.testing.assert_array_equal(pixels, repeated)
    assert rows == metadata and pixels.shape == (24, 1, 256, 256)
    assert pixels.dtype == np.dtype("<f4") and not pixels.flags.writeable
    assert len({r["sha256"] for r in rows}) == 24
    rng = np.random.Generator(np.random.PCG64(20261013))
    for group in range(8):
        source, lineage = divmod(group, 4)
        noise = rng.uniform(-1, 1, (256, 256))
        for y, x in ((0, 0), (0, 1), (127, 130), (255, 255)):
            expected = np.float32(96 + 16 * source + 4 * lineage + noise[y, x])
            assert pixels[3 * group, 0, y, x] == expected
            for j, amplitude in ((1, 48), (2, 56)):
                assert pixels[3 * group + j, 0, y, x] == np.float32(
                    expected + amplitude * (2 * ((y + x) % 2) - 1)
                )
    for i, row in enumerate(rows):
        assert hashlib.sha256(pixels[i].tobytes()).hexdigest() == row["sha256"]
    batches, record = positive.srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=0)
    assert batches.shape == (8, 4) and record["optimizer_updates"] == 8
    assert all([rows[i]["label"] for i in b] == ["cover", "stego"] * 2 for b in batches)


def test_objectives_retain_failures_and_reject_incomplete_nonfinite():
    final = {"balanced_accuracy": 0.95}
    assert positive.objectives(records(), final)["relative_loss_reduction"] == pytest.approx(0.8)
    assert all(
        v
        for k, v in positive.objectives(records(), final).items()
        if k != "relative_loss_reduction"
    )
    flat = [{**r, "mean_pair_loss": 0.693} for r in records()]
    gates = positive.objectives(flat, {"balanced_accuracy": 0.5})
    assert not any(v for k, v in gates.items() if k != "relative_loss_reduction")
    zero = [{**r, "mean_pair_loss": 0} for r in records()]
    assert positive.objectives(zero, final)["relative_loss_reduction"] == 0
    for bad in (
        records()[:-1],
        [{**r, "updates": True} for r in records()],
        [{**r, "mean_pair_loss": float("nan")} for r in records()],
    ):
        with pytest.raises(ValueError):
            positive.objectives(bad, final)
    for accuracy in (True, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            positive.objectives(records(), {"balanced_accuracy": accuracy})


@pytest.fixture
def fake(monkeypatch):
    import torch

    model = SimpleNamespace(
        named_buffers=lambda: [("bn.num_batches_tracked", 160)],
        state_dict=lambda: {"bn.num_batches_tracked": torch.tensor(160)},
    )

    def fit(pixels, samples, indices, **kwargs):
        assert kwargs["config"] == {
            **positive.PARAMS,
            "batch_recipe": positive.srnet_multibatch.RECIPE,
        }
        assert len(kwargs["schedule"]) == 20 and kwargs["seed"] == 20261012
        assert indices.tolist() == list(range(24))
        assert len(samples) == len(pixels) == 24
        return model, records()

    monkeypatch.setattr(positive.srnet_training, "fit", fit)
    monkeypatch.setattr(positive.srnet, "float_logits", lambda model, p: np.array([[0.0, 0.0]]))
    return model


def test_learn_failure_preserved_and_thread_restoration(fake):
    import torch

    threads = torch.get_num_threads()
    model, report = positive.learn()
    assert model is fake and torch.get_num_threads() == threads
    assert not report["sanity_objectives_passed"]
    assert report["accuracy_qualification"] == "unavailable"
    assert report["generated_tensors_not_jpeg_or_steganography"]
    assert not report["real_dataset_loaded"] and not report["primary_detection_changed"]


def test_bn_deadline_and_forward_exception_restore(fake, monkeypatch):
    import torch

    fake.named_buffers = lambda: [("bn.num_batches_tracked", 159)]
    with pytest.raises(ValueError, match="BN"):
        positive.learn()
    fake.named_buffers = lambda: [("bn.num_batches_tracked", 160)]
    times = iter((0, 1861))
    monkeypatch.setattr(positive.time, "monotonic", lambda: next(times))
    previous = torch.get_num_threads()
    with pytest.raises(ValueError, match="deadline"):
        positive.learn()
    assert torch.get_num_threads() == previous
    monkeypatch.setattr(positive.time, "monotonic", lambda: 0)

    def fail(*args):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(positive.srnet, "float_logits", fail)
    with pytest.raises(RuntimeError):
        positive.learn()
    assert torch.get_num_threads() == previous


def test_fresh_output_protocol_and_local_only_snapshot(fake, tmp_path, monkeypatch):
    def save(model, path):
        assert model is fake
        path.write_bytes(b"test snapshot")
        return "a" * 64

    monkeypatch.setattr(service.srnet_model, "save_model", save)
    report = service.run(tmp_path / "result")
    assert report["model_sha256"] == "a" * 64
    assert (tmp_path / "result/control.json").is_file()
    with pytest.raises(FileExistsError):
        service.run(tmp_path / "result")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        service.run(tmp_path / "link/other")
    monkeypatch.setattr(positive, "PROTOCOL_SHA", "b" * 64)
    with pytest.raises(ValueError, match="protocol"):
        service.run(tmp_path / "other")


def test_cli_limits_failure_redaction_and_success(tmp_path, monkeypatch, capsys):
    import resource

    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda k, v: limits.append((k, v)))
    monkeypatch.setattr(sys, "argv", ["control", "--out", str(tmp_path / "output")])
    monkeypatch.setattr(service, "run", lambda out: {"sanity_objectives_passed": True})
    assert service.main() == 0 and len(limits) == 4
    assert (resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3)) in limits
    monkeypatch.setattr(service, "run", lambda out: {"sanity_objectives_passed": False})
    assert service.main() == 2

    def fail(out):
        raise ValueError("secret /host/path")

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 2 and "secret" not in capsys.readouterr().out


@pytest.fixture
def audit_fixture(fake, tmp_path, monkeypatch):
    _, report = positive.learn()
    report["model_sha256"] = "a" * 64
    names = (
        "core/srnet_positive.py",
        "core/srnet_training.py",
        "core/srnet.py",
        "core/srnet_multibatch.py",
        "core/srnet_sampling.py",
        "steganography/research_srnet_positive.py",
    )
    report["execution_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
    monkeypatch.setattr(replay.srnet_model, "load_model", lambda *a, **k: fake)
    monkeypatch.setattr(replay.srnet_reference, "reference_logits", lambda *a: np.zeros((1, 2)))
    return report, tmp_path


def test_audit_separates_learning_and_numeric_failures(audit_fixture, monkeypatch):
    report, directory = audit_fixture
    result = replay.audit(report, directory, ROOT)
    assert result["numerical_gates_passed"] and not result["learning_objectives_passed"]
    assert len(result["independent_oracles"]) == 6
    assert result["singleton_rows_replayed"] == 24
    historical = copy.deepcopy(report)
    historical["execution_source_sha256"]["core/srnet_positive.py"] = replay.HISTORICAL_POSITIVE_SHA
    assert replay.audit(historical, directory, ROOT)["numerical_gates_passed"]
    monkeypatch.setattr(replay.srnet_reference, "reference_logits", lambda *a: np.ones((1, 2)))
    assert not replay.audit(report, directory, ROOT)["numerical_gates_passed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("deployed", True),
        ("generated_seed", 1),
        ("optimizer_updates", 159),
        ("selected_rows", []),
        ("epoch_pair_schedule", []),
        ("epoch_batch_schedule", []),
        ("execution_source_sha256", {"../../untrusted": "a" * 64}),
        ("stored_bn_singleton_train_logits", [[1.0, 1.0]]),
        ("stored_bn_singleton_train_metrics", {}),
        ("sanity_objectives", {}),
        ("sanity_objectives_passed", True),
    ],
)
def test_audit_tampering_rejected(audit_fixture, key, value):
    report, directory = audit_fixture
    report = copy.deepcopy(report)
    report[key] = value
    with pytest.raises(ValueError):
        replay.audit(report, directory, ROOT)


def test_audit_protocol_source_bn_and_thread_guards(audit_fixture, monkeypatch, fake):
    import torch

    report, directory = audit_fixture
    monkeypatch.setattr(positive, "PROTOCOL_SHA", "b" * 64)
    report["protocol_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="protocol"):
        replay.audit(report, directory, ROOT)
    monkeypatch.setattr(
        positive, "PROTOCOL_SHA", "5f34d707ec8c575359ff0d5c42d38a5791c583dbb02b27185589a4f1a1109ab5"
    )
    report["protocol_sha256"] = positive.PROTOCOL_SHA
    report["execution_source_sha256"]["core/srnet.py"] = "b" * 64
    with pytest.raises(ValueError, match="source"):
        replay.audit(report, directory, ROOT)
    report["execution_source_sha256"]["core/srnet.py"] = hashlib.sha256(
        (ROOT / "core/srnet.py").read_bytes()
    ).hexdigest()
    fake.state_dict = lambda: {"bn.num_batches_tracked": torch.tensor(159)}
    with pytest.raises(ValueError, match="BN"):
        replay.audit(report, directory, ROOT)
    fake.state_dict = lambda: {"bn.num_batches_tracked": torch.tensor(160)}
    previous = torch.get_num_threads()

    def fail(*args):
        raise RuntimeError("oracle failed")

    monkeypatch.setattr(replay.srnet_reference, "reference_logits", fail)
    with pytest.raises(RuntimeError):
        replay.audit(report, directory, ROOT)
    assert torch.get_num_threads() == previous


def test_audit_cli_preserves_failure_and_redacts_errors(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "positive_audit_script", ROOT / "scripts/audit-srnet-positive-control.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(
        sys, "argv", ["audit", "--result", str(tmp_path), "--out", str(tmp_path / "audit.json")]
    )
    monkeypatch.setattr(script, "read_document", lambda path: ({}, "a" * 64))
    monkeypatch.setattr(script, "audit", lambda *args: {"numerical_gates_passed": True})
    output = []
    monkeypatch.setattr(script, "write_json", lambda path, doc: output.append(doc))
    assert script.main() == 0 and output[0]["control_report_sha256"] == "a" * 64
    monkeypatch.setattr(script, "audit", lambda *args: {"numerical_gates_passed": False})
    assert script.main() == 2 and output[-1]["numerical_gates_passed"] is False

    def fail(*args):
        raise ValueError("secret /host/path")

    monkeypatch.setattr(script, "audit", fail)
    assert script.main() == 2 and "secret" not in capsys.readouterr().out
