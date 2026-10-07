"""Amplification is bounded and artificial; failures never qualify detection."""

import copy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_positive
from core import srnet_signal as signal
from steganography import research_srnet_signal as service


def data():
    pixels, _ = srnet_positive.generate()
    rows = []
    for source, count, qualities in (("ALASKA2", 4, (None,)), ("BOSSbase-1.01", 2, (75, 95))):
        for lineage in range(count):
            for quality in qualities:
                for method in (None, "JUNIWARD", "UERD"):
                    rows.append(
                        {
                            "split": "train",
                            "format": "JPEG",
                            "source_group": source,
                            "lineage": str(lineage),
                            "quality_factor": quality,
                            "method": method,
                            "label": "cover" if method is None else "stego",
                            "sha256": hashlib.sha256(
                                f"{source}/{lineage}/{quality}/{method}".encode()
                            ).hexdigest(),
                            "path": "/host/private/not-reportable.jpg",
                        }
                    )
    return pixels, rows


def records():
    return [{"epoch": e, "updates": 8, "mean_pair_loss": 0.693} for e in range(20)]


def test_scalar_amplification_hashes_no_clipping_and_no_mutation():
    pixels, rows = data()
    metadata = copy.deepcopy(rows)
    before = pixels.copy()
    unchanged, hashes = signal.prepare(pixels, rows, factor=1)
    np.testing.assert_array_equal(unchanged, pixels)
    assert len(hashes) == 24 and not unchanged.flags.writeable
    amplified, hashes = signal.prepare(pixels, rows, factor=32)
    for i, row in enumerate(rows):
        cover = i - i % 3
        for y, x in ((0, 0), (0, 1), (255, 255)):
            c, s = float(pixels[cover, 0, y, x]), float(pixels[i, 0, y, x])
            assert amplified[i, 0, y, x] == np.float32(c + 32 * (s - c))
        assert hashes[i] == hashlib.sha256(amplified[i].tobytes()).hexdigest()
        if row["label"] == "cover":
            np.testing.assert_array_equal(amplified[i], pixels[i])
    assert amplified.min() < 0 and amplified.max() > 255  # Never silently clip.
    np.testing.assert_array_equal(pixels, before)
    assert rows == metadata and not amplified.flags.writeable


@pytest.mark.parametrize("factor", [True, 1.0, 0, -1, 2, 33, None])
def test_invalid_factor_rejected(factor):
    pixels, rows = data()
    with pytest.raises(ValueError):
        signal.prepare(pixels, rows, factor=factor)


def test_invalid_shape_dtype_values_cardinality_pairs_and_derived_limit():
    pixels, rows = data()
    for bad in (
        pixels[:23],
        pixels.astype("f8"),
        np.full_like(pixels, np.nan),
        np.full_like(pixels, 2**37),
    ):
        with pytest.raises(ValueError):
            signal.prepare(bad, rows, factor=32)
    # Complete shorter metadata still cannot accompany 24 input tensors.
    with pytest.raises(ValueError):
        signal.prepare(pixels, rows[:21], factor=32)
    with pytest.raises(ValueError):
        signal.prepare(pixels, rows[:-1], factor=32)
    excessive = pixels.copy()
    excessive[0] = -(2**35)
    excessive[1] = 2**35
    with pytest.raises(ValueError, match="amplified"):
        signal.prepare(excessive, rows, factor=32)


@pytest.fixture
def fake(monkeypatch):
    import torch

    model = SimpleNamespace(
        named_buffers=lambda: [("bn.num_batches_tracked", 160)],
        state_dict=lambda: {"bn.num_batches_tracked": torch.tensor(160)},
    )

    def fit(pixels, rows, indices, **kwargs):
        assert len(rows) == len(pixels) == 24 and indices.tolist() == list(range(24))
        assert kwargs["seed"] == 20261012 and len(kwargs["schedule"]) == 20
        assert kwargs["config"] == {
            **srnet_positive.PARAMS,
            "batch_recipe": signal.srnet_multibatch.RECIPE,
        }
        assert not pixels.flags.writeable
        return model, records()

    monkeypatch.setattr(signal.srnet_training, "fit", fit)
    monkeypatch.setattr(signal.srnet, "float_logits", lambda *args: np.zeros((1, 2)))
    monkeypatch.setattr(signal.srnet_reference, "reference_logits", lambda *args: np.zeros((1, 2)))
    return model


def test_learn_replay_complete_accounting_failures_and_numeric_separation(fake, monkeypatch):
    pixels, rows = data()
    model, report = signal.learn(pixels, rows, factor=32)
    assert model is fake and report["optimizer_updates"] == 160
    assert not report["learning_objectives_passed"]
    audit = signal.audit(model, pixels, rows, report)
    assert audit["all_own_and_original_singletons_replayed"]
    assert len(audit["independent_oracles"]) == 9 and audit["numerical_gates_passed"]
    monkeypatch.setattr(signal.srnet_reference, "reference_logits", lambda *args: np.ones((1, 2)))
    assert not signal.audit(model, pixels, rows, report)["numerical_gates_passed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("derived_tensor_sha256", []),
        ("epoch_pair_schedule", []),
        ("epoch_batch_schedule", []),
        ("optimizer", {}),
        ("seed", 1),
        ("optimizer_updates", 159),
        ("own_input_singleton_logits", [[0.0, 0.0]]),
        ("original_input_singleton_logits", [[0.0, 0.0]]),
        ("own_input_train_metrics", {}),
        ("original_input_train_metrics", {}),
        ("learning_objectives", {}),
        ("learning_objectives_passed", True),
    ],
)
def test_replay_tampering_rejected(fake, key, value):
    pixels, rows = data()
    model, report = signal.learn(pixels, rows, factor=32)
    report[key] = value
    with pytest.raises(ValueError):
        signal.audit(model, pixels, rows, report)


def test_bn_thread_restore_and_oracle_coverage(fake, monkeypatch):
    import torch

    pixels, rows = data()
    model, report = signal.learn(pixels, rows, factor=1)
    fake.named_buffers = lambda: [("bn.num_batches_tracked", 159)]
    with pytest.raises(ValueError, match="BN"):
        signal.learn(pixels, rows, factor=1)
    with pytest.raises(ValueError, match="BN"):
        signal.audit(model, pixels, rows, report)
    fake.named_buffers = lambda: [("bn.num_batches_tracked", 160)]
    # A valid two-source artificial corpus has only six label/method cells,
    # hence cannot silently masquerade as the required nine-oracle real slice.
    generated, metadata = srnet_positive.generate()
    model, short_report = signal.learn(generated, metadata, factor=1)
    with pytest.raises(ValueError, match="coverage"):
        signal.audit(model, generated, metadata, short_report)
    threads = torch.get_num_threads()

    def fail(*args):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(signal.srnet, "float_logits", fail)
    with pytest.raises(RuntimeError):
        signal.evaluate(model, pixels, rows)
    assert torch.get_num_threads() == threads


@pytest.fixture
def service_fake(fake, monkeypatch):
    pixels, rows = data()
    config = {
        "manifest": "manifest",
        "manifest_sha256": service.MANIFEST_SHA,
        "cache": "cache",
        "cache_sha256": service.CACHE_SHA,
    }
    monkeypatch.setattr(service, "read_document", lambda path: ({}, service.MANIFEST_SHA))
    monkeypatch.setattr(
        service, "load_pixels", lambda *args, **kwargs: (pixels, rows, {"data_sha256": "a" * 64})
    )
    monkeypatch.setattr(service.srnet_model, "load_model", lambda *args, **kwargs: fake)

    def save(model, path):
        path.write_bytes(b"test snapshot")
        return "a" * 64

    monkeypatch.setattr(service.srnet_model, "save_model", save)
    return config


def test_service_both_arms_identical_schedule_no_paths_and_fresh_outputs(service_fake, tmp_path):
    report = service.run(service_fake, tmp_path / "result")
    assert report["status"] == "completed" and [a["factor"] for a in report["arms"]] == [1, 32]
    assert report["arms"][0]["epoch_batch_schedule"] == report["arms"][1]["epoch_batch_schedule"]
    assert not report["arms"][0]["learning_objectives_passed"]
    assert not report["validation_pixels_loaded"] and not report["deployed"]
    assert report["accuracy_qualification"] == "unavailable"
    text = (tmp_path / "result/control.json").read_text()
    assert "/host/private" not in text and str(tmp_path) not in text
    assert (tmp_path / "result/factor-1/arm.json").is_file()
    with pytest.raises(FileExistsError):
        service.run(service_fake, tmp_path / "result")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        service.run(service_fake, tmp_path / "link/new")


def test_service_bindings_protocol_and_total_deadlines(service_fake, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        service.run({**service_fake, "validation": "forbidden"}, tmp_path / "out")
    with pytest.raises(ValueError):
        service.run({**service_fake, "cache_sha256": "b" * 64}, tmp_path / "out")
    monkeypatch.setattr(service, "PROTOCOL_SHA", "b" * 64)
    with pytest.raises(ValueError, match="protocol"):
        service.run(service_fake, tmp_path / "out")
    monkeypatch.setattr(
        service, "PROTOCOL_SHA", "c1a067ba32ae56d19dffedfec8f20725b5b60e6726e0fb992264ec2b7327e0ca"
    )
    monkeypatch.setattr(service, "read_document", lambda path: ({}, "b" * 64))
    with pytest.raises(ValueError, match="manifest"):
        service.run(service_fake, tmp_path / "out")
    monkeypatch.setattr(service, "read_document", lambda path: ({}, service.MANIFEST_SHA))
    clock = iter((0, 3721))
    monkeypatch.setattr(service.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="deadline"):
        service.run(service_fake, tmp_path / "first-deadline")
    # Avoid learning's own clock so the next service post-arm check is isolated.
    monkeypatch.setattr(
        service.srnet_signal, "learn", lambda *a, **k: (None, {"learning_objectives_passed": False})
    )
    monkeypatch.setattr(service.srnet_signal, "audit", lambda *a: {"numerical_gates_passed": True})
    clock = iter((0, 0, 3721))
    with pytest.raises(ValueError, match="deadline"):
        service.run(service_fake, tmp_path / "second-deadline")
    assert not (tmp_path / "second-deadline/control.json").exists()


def test_cli_limits_success_failed_objectives_and_redacted_failure(
    service_fake, tmp_path, monkeypatch, capsys
):
    import resource

    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda k, v: limits.append((k, v)))
    monkeypatch.setattr(sys, "argv", ["signal", "--config", "config", "--out", str(tmp_path)])
    monkeypatch.setattr(service, "read_document", lambda path: (service_fake, "a" * 64))
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"learning_objectives_passed": True, "audit": {"numerical_gates_passed": True}}
            ]
            * 2
        },
    )
    assert service.main() == 0 and len(limits) == 4
    assert (resource.RLIMIT_CPU, (7600, 7601)) in limits
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"learning_objectives_passed": False, "audit": {"numerical_gates_passed": True}}
            ]
            * 2
        },
    )
    assert service.main() == 2

    def fail(*args):
        raise ValueError("secret /host/private")

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 2 and "secret" not in capsys.readouterr().out
