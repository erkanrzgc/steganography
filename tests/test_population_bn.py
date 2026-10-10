"""Independent central-moment checks; generated controls, not real accuracy."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from core import jpeg_population_control as control
from core import srnet_population_bn as population
from steganography import research_jpeg_population_bn as frontend


def samples():
    rows = []
    for source, qualities in (("ALASKA2", (None,)), ("BOSS", (75, 95)), ("BOWS", (75, 95))):
        for original in range(32):
            for quality in qualities:
                for method in (None, "JUNIWARD", "UERD"):
                    rows.append(
                        {
                            "source_group": source,
                            "lineage": f"{source}-{original:02}",
                            "quality_factor": quality,
                            "method": method,
                            "label": "cover" if method is None else "stego",
                            "split": "train",
                        }
                    )
    return rows


class Tiny(torch.nn.Module):
    architecture = "srnet-gray12-cpu-v1"

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.layers = torch.nn.Sequential(*(torch.nn.BatchNorm2d(1) for _ in range(26)))
        self.classifier = torch.nn.Linear(1, 2, bias=False)

    def forward(self, pixels):
        return self.classifier(self.layers(self.pool(pixels)).flatten(1))

    def to(self, device):
        # Explicit mock of CUDA control flow; no physical GPU evidence.
        return self if device == "cuda:0" else super().to(device)


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(population.srnet_model, "validate_tensors", lambda state: None)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield Tiny().eval()
    torch.set_num_threads(previous)


def fetch(indices):
    return np.broadcast_to(indices.astype("<f4")[:, None, None, None], (4, 1, 256, 256)).copy()


def test_partition_keeps_all_original_derivatives_disjoint():
    rows = samples()
    calibration, probe = population.partition(rows)
    assert calibration.shape == (90, 4) and probe.shape == (30, 4)
    a = {rows[i]["lineage"] for i in calibration.flat}
    b = {rows[i]["lineage"] for i in probe.flat}
    assert len(a) == 72 and len(b) == 24 and not a & b
    assert set(calibration.flat) | set(probe.flat) == set(range(480))


@pytest.mark.parametrize("change", ["size", "split", "groups", "leakage", "counts"])
def test_partition_rejects_invalid_accounting(change):
    rows = samples()
    if change == "size":
        rows.pop()
    elif change == "split":
        rows[0]["split"] = "validation"
    elif change == "groups":
        for row in rows:
            row["source_group"] = "one"
    elif change == "leakage":
        for row in rows:
            row["lineage"] = row["lineage"].split("-")[-1]
    else:
        rows[0]["lineage"] = rows[-1]["lineage"]
    with pytest.raises(ValueError):
        population.partition(rows)


def test_unequal_central_moment_merge_matches_numpy_not_mean_of_variances():
    chunks = [np.array([[1.0, 4.0]]), np.array([[100.0, 9.0], [120.0, 8.0], [200.0, 4.0]])]
    merged = None
    for chunk in chunks:
        merged = population.merge(merged, len(chunk), chunk.mean(0), chunk.var(0) * len(chunk))
    count, mean, m2 = merged
    together = np.concatenate(chunks)
    assert count == 4
    np.testing.assert_allclose(mean, together.mean(0), atol=1e-12)
    np.testing.assert_allclose(m2 / count, together.var(0), atol=1e-12)
    with pytest.raises(ValueError):
        population.merge(None, 0, mean, m2)


def test_layerwise_population_matches_closed_form_and_source_is_immutable(tiny):
    saved = {k: v.clone() for k, v in tiny.state_dict().items()}
    clone, layers = population.refresh(tiny, fetch, np.arange(360).reshape(90, 4), lambda: None)
    assert len(layers) == 26 and layers[0]["activation_count_per_channel"] == 360
    np.testing.assert_allclose(clone.layers[0].running_mean.numpy(), [179.5])
    np.testing.assert_allclose(clone.layers[0].running_var.numpy(), [np.arange(360).var()])
    np.testing.assert_allclose(clone.layers[1].running_mean.numpy(), [0], atol=1e-6)
    np.testing.assert_allclose(clone.layers[1].running_var.numpy(), [1], atol=1e-6)
    assert all(torch.equal(v, saved[k]) for k, v in tiny.state_dict().items())
    assert all(int(m.num_batches_tracked) == 90 for m in clone.layers)
    assert all(not m.training and not m._forward_pre_hooks for m in clone.modules())
    assert torch.equal(clone.classifier.weight, tiny.classifier.weight)


@pytest.mark.parametrize(
    "change",
    [
        "mode",
        "architecture",
        "layers",
        "tracking",
        "batches",
        "pixels",
        "unreached",
        "nonfinite",
        "deadline",
    ],
)
def test_refresh_rejects_bad_state_without_mutating_source(tiny, change):
    batches = np.arange(360).reshape(90, 4)
    reader, deadline = fetch, lambda: None
    if change == "mode":
        tiny.train()
    elif change == "architecture":
        tiny.architecture = "other"
    elif change == "layers":
        tiny.layers = torch.nn.Sequential(torch.nn.BatchNorm2d(1))
    elif change == "tracking":
        tiny.layers[0].track_running_stats = False
    elif change == "batches":
        batches = np.zeros((90, 4), dtype=np.int64)
    elif change == "pixels":

        def reader(indices):
            return fetch(indices).astype(np.float64)
    elif change == "unreached":
        tiny.forward = lambda pixels: pixels
    elif change == "nonfinite":
        tiny.pool.forward = lambda pixels: pixels[:, :, :1, :1] * float("nan")
    else:

        def deadline():
            raise ValueError("deadline")

    saved = {k: v.clone() for k, v in tiny.state_dict().items()}
    with pytest.raises(ValueError):
        population.refresh(tiny, reader, batches, deadline)
    assert all(torch.equal(v, saved[k]) for k, v in tiny.state_dict().items())


def test_all_cells_reported_and_shared_cover_counts_explicit():
    rows = samples()
    _, probe = population.partition(rows)
    logits = {int(i): np.array([0.0, float(rows[i]["label"] == "stego")]) for i in probe.flat}
    cells = control.cells(rows, logits)
    assert len(cells) == 10
    assert all(c["true_positive"] == 8 and c["false_positive"] == 8 for c in cells)
    assert all(c["cover_rows_shared_across_method_cells"] for c in cells)
    logits.pop(int(probe.flat[0]))
    with pytest.raises(ValueError):
        control.cells(rows, logits)
    with pytest.raises(ValueError):
        control.cells([], {})


def test_singleton_scoring_parity_and_eval_guard(tiny):
    class Reader:
        batch = staticmethod(fetch)

    logits, parity = control.score(tiny, Reader(), np.arange(4).reshape(1, 4), lambda: None)
    assert len(logits) == 4 and parity
    tiny.train()
    with pytest.raises(ValueError):
        control.score(tiny, Reader(), np.arange(4).reshape(1, 4), lambda: None)


def test_frontend_sources_and_no_overwrite(tmp_path):
    hashes = frontend.sources()
    assert "docs/JPEG_POPULATION_BN_PROTOCOL.md" in hashes
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        frontend.execute(tmp_path, tmp_path, tmp_path, out)


def test_control_mocked_cuda_flow_has_no_optimizer_or_validation(tiny, monkeypatch, tmp_path):
    import hashlib

    model = tmp_path / "source"
    model.write_bytes(b"mock numeric state")
    monkeypatch.setattr(control, "MODEL_SHA", hashlib.sha256(model.read_bytes()).hexdigest())
    monkeypatch.setattr(control.srnet_model, "load_model", lambda *a, **kw: tiny)
    monkeypatch.setattr(control.srnet_cuda, "inspect", lambda: {"device": "mock-not-hardware"})
    monkeypatch.setattr(control.srnet_cuda, "policy", nullcontext)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 0)

    class Reader:
        def __init__(self, *args, **kwargs):
            self.samples = samples()

        batch = staticmethod(fetch)

        def verify(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(control, "TimingReader", Reader)
    clone, report = control.control(tmp_path, tmp_path, model)
    assert report["optimizer_updates"] == 0 and not report["validation_used"]
    assert report["execution"]["device"] == "mock-not-hardware"
    assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]
    assert report["probe_is_historical_training_data"]
    assert clone is not tiny and report["source_model_unchanged"]


@pytest.mark.parametrize("single", [False, True])
def test_scoring_rejects_nonfinite_batch_or_singleton(tiny, single):
    def bad(pixels):
        if not single or len(pixels) == 1:
            return torch.full((len(pixels), 2), float("nan"))
        return torch.zeros((len(pixels), 2))

    tiny.forward = bad

    class Reader:
        batch = staticmethod(fetch)

    with pytest.raises(ValueError):
        control.score(tiny, Reader(), np.arange(4).reshape(1, 4), lambda: None)


def test_refresh_rejects_changed_learned_parameter(tiny, monkeypatch):
    calls = 0

    def validate(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            state["classifier.weight"].add_(1)

    monkeypatch.setattr(population.srnet_model, "validate_tensors", validate)
    with pytest.raises(ValueError, match="learned weights"):
        population.refresh(tiny, fetch, np.arange(360).reshape(90, 4), lambda: None)


def test_refresh_rejects_empty_moment_accumulation(tiny, monkeypatch):
    monkeypatch.setattr(population, "merge", lambda *args: None)
    with pytest.raises(ValueError, match="moments absent"):
        population.refresh(tiny, fetch, np.arange(360).reshape(90, 4), lambda: None)


def test_frontend_completed_and_changed_sources(tiny, monkeypatch, tmp_path):
    monkeypatch.setattr(control, "control", lambda *args: (tiny, {"status": "completed"}))
    monkeypatch.setattr(frontend.srnet_model, "save_model", lambda *args: "mock-digest")
    report = frontend.execute(tmp_path, tmp_path, tmp_path, tmp_path / "out")
    assert report["refreshed_model_sha256"] == "mock-digest"
    assert (tmp_path / "out/report.json").is_file()
    counter = 0

    def sources():
        nonlocal counter
        counter += 1
        return {"changed": counter}

    monkeypatch.setattr(frontend, "sources", sources)
    with pytest.raises(ValueError, match="sources changed"):
        frontend.execute(tmp_path, tmp_path, tmp_path, tmp_path / "not-written")
    assert not (tmp_path / "not-written").exists()


@pytest.mark.parametrize("passed", [False, True])
def test_frontend_main_limits_and_exit_status(monkeypatch, tmp_path, passed):
    limits = []
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda kind, bound: limits.append(bound))
    monkeypatch.setattr(
        frontend,
        "execute",
        lambda *args: {
            "status": "completed",
            "numerical_gates_passed": passed,
        },
    )
    args = [
        item
        for name in ("root", "audit", "model", "out")
        for item in ("--" + name, str(tmp_path / name))
    ]
    assert frontend.main(args) == (0 if passed else 2)
    assert (3600, 3601) in limits and (32 * 1024**2, 32 * 1024**2) in limits


def test_frontend_oversized_source_rejected(monkeypatch):
    import io

    monkeypatch.setattr(frontend, "regular_open", lambda path: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError, match="exceeds bound"):
        frontend.sources()
