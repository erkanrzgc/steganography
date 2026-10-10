"""Generated normalization/math/security controls; never real detection scores."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from core import jpeg_norm_learning as pilot
from core import srnet, srnet_model
from core import srnet_groupnorm as gn
from core import srnet_norm_training as training
from steganography import research_jpeg_norm_learning as frontend


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_full_variant_retains_initial_parameters_and_distinct_numeric_contract(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261010)
        baseline = srnet.network().eval()
        torch.manual_seed(20261010)
        variant = gn.network().eval()
    assert gn.parameter_sha(baseline) == gn.parameter_sha(variant)
    assert variant.architecture == gn.ARCHITECTURE != baseline.architecture
    assert len([m for m in variant.modules() if isinstance(m, torch.nn.GroupNorm)]) == 26
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in variant.modules())
    for m in variant.modules():
        if isinstance(m, torch.nn.GroupNorm):
            assert m.num_groups == 8 and m.eps == 1e-5
    digest = gn.save_model(variant, tmp_path / "gn.npz")
    with np.load(tmp_path / "gn.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(gn.SHAPES)
        for name, value in variant.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), archive[name])
    with pytest.raises(ValueError, match="contract"):
        srnet_model.load_model(tmp_path / "gn.npz", checksum=digest)


def test_groupnorm_matches_independent_numpy_and_needs_no_batch_partner():
    pixels = np.random.default_rng(8).normal(size=(4, 16, 5, 5)).astype("<f4")
    norm = torch.nn.GroupNorm(8, 16, eps=1e-5)
    groups = pixels.astype(np.float64).reshape(4, 8, 2, 5, 5)
    expected = (
        (groups - groups.mean((2, 3, 4), keepdims=True))
        / np.sqrt(groups.var((2, 3, 4), keepdims=True) + 1e-5)
    ).reshape(pixels.shape)
    result = norm(torch.from_numpy(pixels)).detach().numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)
    for i in range(4):
        singleton = norm(torch.from_numpy(pixels[i : i + 1])).detach().numpy()
        np.testing.assert_array_equal(singleton[0], result[i])
    norm.eval()
    np.testing.assert_array_equal(norm(torch.from_numpy(pixels)).detach().numpy(), result)


def test_full_groupnorm_model_has_finite_gradients_and_singleton_parity():
    model = gn.network().train()
    pixels = torch.from_numpy(generated_fetch(np.arange(4)))
    logits = model(pixels)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert training.srnet_training._finite_gradients(model, "cpu")
    model.eval()
    with torch.no_grad():
        together = model(pixels)
        for i in range(4):
            torch.testing.assert_close(
                model(pixels[i : i + 1])[0], together[i], atol=1e-4, rtol=1e-4
            )


@pytest.mark.parametrize("change", ["keys", "dtype", "shape", "nan"])
def test_numeric_validator_rejects_hostile_states(change):
    state = gn.network().state_dict()
    first = next(iter(state))
    if change == "keys":
        state.pop(first)
    elif change == "dtype":
        state[first] = state[first].double()
    elif change == "shape":
        state[first] = state[first].flatten()
    else:
        state[first].flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        gn.validate(state)


@pytest.mark.parametrize("kind", ["channels", "count"])
def test_conversion_requires_exact_architecture(monkeypatch, kind):
    factory = (
        (lambda: torch.nn.Sequential(torch.nn.BatchNorm2d(2)))
        if kind == "channels"
        else (lambda: torch.nn.Sequential(torch.nn.Linear(2, 2)))
    )
    monkeypatch.setattr(gn.srnet, "network", factory)
    with pytest.raises(ValueError):
        gn.network()


def test_snapshot_guards(tmp_path, monkeypatch):
    model = gn.network()
    with pytest.raises(ValueError):
        gn.save_model(model, tmp_path / "train.npz")
    model.eval()
    present = tmp_path / "existing"
    present.touch()
    with pytest.raises(FileExistsError):
        gn.save_model(model, present)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        gn.save_model(model, link)
    monkeypatch.setattr(gn.srnet_model, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        gn.save_model(model, tmp_path / "large.npz")


class Tiny(torch.nn.Module):
    architecture = "mock-research-only"

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.conv = torch.nn.Conv2d(1, 8, 1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, pixels):
        return self.classifier(self.bn(self.conv(self.pool(pixels))).mean((2, 3)))

    def to(self, device):
        return self if device == "cuda:0" else super().to(device)


def generated_fetch(indices):
    rng = np.random.default_rng(11)
    values = rng.normal(size=(4, 1, 256, 256)).astype("<f4")
    return values[indices].copy()


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(srnet, "network", Tiny)
    monkeypatch.setattr(srnet_model, "validate", lambda _: None)
    monkeypatch.setattr(srnet_model, "validate_tensors", lambda _: None)
    monkeypatch.setattr(gn, "network", Tiny)
    monkeypatch.setattr(gn, "validate", lambda _: None)


def arguments():
    return {
        "fetch": generated_fetch,
        "batches": lambda _: np.array([[0, 1, 2, 3]], dtype=np.int64),
        "epochs": 2,
        "seed": 20261010,
        "params": training.srnet_training.settings({}),
        "deadline": lambda: None,
        "device": "cpu",
    }


def test_new_kernel_bn_oracle_matches_legacy_optimizer_exactly(tiny):
    old, old_records = training.learn(mode="bn", **arguments())
    new, new_records = training.bn_kernel_oracle(**arguments())
    assert old_records == new_records
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    assert all(not m.training for m in new.modules())


def test_gn_adapter_updates_real_torch_parameters_with_fixed_accounting(tiny):
    model, records = training.learn(mode="gn", **arguments())
    assert len(records) == 2 and sum(r["updates"] for r in records) == 2
    assert all(np.isfinite(r["mean_pair_loss"]) for r in records)
    assert not model.training


@pytest.mark.parametrize(
    "change",
    ["mode", "device", "epochs", "schedule", "pixels", "logits", "gradients", "loss", "deadline"],
)
def test_training_limit_and_numeric_failures(tiny, monkeypatch, change):
    kwargs = arguments()
    mode = "gn"
    if change == "mode":
        mode = "bad"
    elif change == "device":
        kwargs["device"] = "cuda:1"
    elif change == "epochs":
        kwargs["epochs"] = 9
    elif change == "schedule":
        kwargs["batches"] = lambda _: np.zeros((145, 4), dtype=np.int64)
    elif change == "pixels":
        kwargs["fetch"] = lambda _: generated_fetch(np.arange(4)).astype(np.float64)
    elif change == "logits":
        monkeypatch.setattr(Tiny, "forward", lambda self, pixels: torch.full((4, 2), float("nan")))
    elif change == "gradients":
        monkeypatch.setattr(training.srnet_training, "_finite_gradients", lambda *a: False)
    elif change == "loss":
        monkeypatch.setattr(
            torch.nn.functional, "cross_entropy", lambda *a: torch.tensor(float("nan"))
        )
    else:

        def fail():
            raise ValueError("deadline")

        kwargs["deadline"] = fail
    with pytest.raises(ValueError):
        training.learn(mode=mode, **kwargs)


def real_metadata():
    # Generated original families, independent of private data and other test imports.
    rows = []
    for source, qualities in (("A", (None,)), ("B", (75, 95)), ("C", (75, 95))):
        for original in range(32):
            for quality in qualities:
                for method in (None, "JUNIWARD", "UERD"):
                    rows.append(
                        {
                            "sha256": f"{len(rows):064x}",
                            "format": "JPEG",
                            "source_group": source,
                            "lineage": f"{source}-{original:02}",
                            "quality_factor": quality,
                            "method": method,
                            "split": "train",
                            "label": "cover" if method is None else "stego",
                        }
                    )
    return rows


def test_complete_eight_epoch_mapping_excludes_probe_originals():
    from core.srnet_population_bn import partition

    samples = real_metadata()
    batches, records, probe = pilot.schedules(samples)
    fit, _ = partition(samples)
    assert len(batches) == len(records) == 8
    assert all(a.shape == (144, 4) and not a.flags.writeable for a in batches)
    assert all(set(a.flat) == set(fit.flat) and not set(a.flat) & set(probe.flat) for a in batches)


def test_frontend_sources_and_no_overwrite(tmp_path):
    assert "docs/JPEG_NORM_LEARNING_PROTOCOL.md" in frontend.sources()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        frontend.execute(tmp_path, tmp_path, "gn", existing)


@pytest.mark.parametrize("arm", ["bn", "gn"])
def test_frontend_persistence_and_changed_sources(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(pilot, "fit_arm", lambda *a: (object(), {"status": "mock"}))
    monkeypatch.setattr(srnet_model, "save_model", lambda *a: "bn-mock")
    monkeypatch.setattr(gn, "save_model", lambda *a: "gn-mock")
    out = tmp_path / arm
    report = frontend.execute(tmp_path, tmp_path, arm, out)
    assert report["model_sha256"] == arm + "-mock" and (out / "report.json").is_file()
    counter = 0

    def changing():
        nonlocal counter
        counter += 1
        return {"counter": counter}

    monkeypatch.setattr(frontend, "sources", changing)
    with pytest.raises(ValueError, match="sources changed"):
        frontend.execute(tmp_path, tmp_path, arm, tmp_path / "not-written")


@pytest.mark.parametrize("passed", [True, False])
def test_frontend_main_resource_limits_and_exit_status(tmp_path, monkeypatch, passed):
    calls = []
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *a: calls.append(a))
    monkeypatch.setattr(
        frontend,
        "execute",
        lambda *a: {
            "status": "mock",
            "optimizer_updates": 1152,
            "singleton_batch_parity": {"passed": passed},
        },
    )
    args = [
        item for name in ("root", "audit", "out") for item in ("--" + name, str(tmp_path / name))
    ] + ["--arm", "gn"]
    assert frontend.main(args) == (0 if passed else 2)
    assert len(calls) == 3


@pytest.mark.parametrize("failure", [None, "accounting", "unchanged"])
def test_pilot_mocked_control_flow_is_not_hardware_evidence(tiny, monkeypatch, tmp_path, failure):
    class Reader:
        def __init__(self, *a, **kw):
            self.samples = real_metadata()

        def batch(self, indices):
            return np.broadcast_to(
                indices.astype("<f4")[:, None, None, None], (4, 1, 256, 256)
            ).copy()

        def verify(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(pilot, "TimingReader", Reader)
    monkeypatch.setattr(pilot.srnet_cuda, "inspect", lambda: {"device": "mock-not-hardware"})
    monkeypatch.setattr(pilot.srnet_cuda, "policy", nullcontext)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 0)

    def learned(**kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(pilot.SEED)
            model = Tiny().eval()
        if failure != "unchanged":
            with torch.no_grad():
                model.classifier.weight.add_(1)
        records = [{"epoch": i, "updates": 144, "mean_pair_loss": 0.69} for i in range(8)]
        return model, records if failure != "accounting" else []

    monkeypatch.setattr(pilot.srnet_norm_training, "learn", learned)
    if failure:
        with pytest.raises(ValueError):
            pilot.fit_arm(tmp_path, tmp_path, "gn")
    else:
        model, report = pilot.fit_arm(tmp_path, tmp_path, "gn")
        assert report["execution"]["device"] == "mock-not-hardware"
        assert report["optimizer_updates"] == 1152 and len(report["probe_cells"]) == 10
        assert len(report["probe_logits"]) == 120 and not model.training
        assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]
    with pytest.raises(ValueError):
        pilot.fit_arm(tmp_path, tmp_path, "bad")


def test_pilot_schedule_failure_preserves_bounds(monkeypatch):
    rows = real_metadata()
    monkeypatch.setattr(
        pilot.srnet_diversity_sampling,
        "epoch_batches",
        lambda *a, **kw: (np.zeros((1, 4), dtype=np.int64), {}),
    )
    with pytest.raises(ValueError):
        pilot.schedules(rows)
    monkeypatch.setattr(
        pilot.srnet_diversity_sampling,
        "epoch_batches",
        lambda *a, **kw: (np.zeros((144, 4), dtype=np.int64), {}),
    )
    with pytest.raises(ValueError):
        pilot.schedules(rows)


def test_frontend_oversized_source_guard(monkeypatch):
    import io

    monkeypatch.setattr(frontend, "regular_open", lambda path: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError):
        frontend.sources()
