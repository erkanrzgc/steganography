"""Independent pooling/BN equations, generated controls and bounded wiring."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from core import jpeg_learnability as learning
from core import srnet, srnet_groupnorm, srnet_model, srnet_samepool
from steganography import research_jpeg_learnability as frontend


@pytest.mark.parametrize("shape", [(8, 8), (7, 7), (8, 7), (16, 16)])
def test_same_pool_matches_independent_window_mean_and_gradient(shape):
    h, w = shape
    values = np.arange(h * w, dtype=np.float32).reshape(h, w)
    ph, pw = (1 if h % 2 == 0 else 2), (1 if w % 2 == 0 else 2)
    expected = np.array(
        [
            [
                values[
                    max(0, 2 * r - ph // 2) : min(h, 2 * r - ph // 2 + 3),
                    max(0, 2 * c - pw // 2) : min(w, 2 * c - pw // 2 + 3),
                ].mean()
                for c in range((w + 1) // 2)
            ]
            for r in range((h + 1) // 2)
        ]
    )
    pixels = torch.tensor(values[None, None], requires_grad=True)
    result = srnet_samepool.same_average(pixels)
    np.testing.assert_allclose(result.detach().numpy()[0, 0], expected, rtol=0, atol=1e-6)
    result.sum().backward()
    assert torch.isfinite(pixels.grad).all()
    if shape == (8, 8):
        legacy = torch.nn.AvgPool2d(3, 2, 1, count_include_pad=False)(pixels)
        assert result[0, 0, 0, 0] == 9 and legacy[0, 0, 0, 0] == 4.5


def test_pool_rejects_wrong_rank_or_too_small():
    for pixels in (torch.zeros(3, 3), torch.zeros(1, 1, 2, 2)):
        with pytest.raises(ValueError):
            srnet_samepool.same_average(pixels)


def test_reference_changes_only_bound_numerics_not_learned_initialization():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        legacy = srnet.network()
        torch.manual_seed(7)
        reference = srnet_samepool.network()
    assert reference.architecture == srnet_samepool.ARCHITECTURE != legacy.architecture
    assert srnet_groupnorm.parameter_sha(legacy) == srnet_groupnorm.parameter_sha(reference)
    assert set(reference.state_dict()) == set(srnet_model.SHAPES)
    norms = [m for m in reference.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(norms) == 26 and all(m.eps == 1e-3 for m in norms)
    layer = norms[0].eval()
    layer.running_mean.fill_(2)
    layer.running_var.fill_(0.0003)
    values = torch.full((1, 64, 3, 3), 2.002)
    observed = layer(values).detach().numpy()
    expected = (values.numpy().astype(np.float64) - 2) / np.sqrt(0.0003 + 0.001)
    np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=1e-5)


def test_reference_topology_guard(monkeypatch):
    monkeypatch.setattr(srnet, "network", lambda: torch.nn.Linear(1, 2))
    with pytest.raises(ValueError):
        srnet_samepool.network()


def test_complete_reference_network_has_finite_gradients_and_eval_parity():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            model = srnet_samepool.network().train()
            pixels = torch.randn(4, 1, 256, 256)
            loss = torch.nn.functional.cross_entropy(model(pixels), torch.tensor([0, 1, 0, 1]))
            loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
            model.eval()
            with torch.no_grad():
                batched = model(pixels)
                single = torch.cat([model(x[None]) for x in pixels])
            torch.testing.assert_close(single, batched, atol=1e-4, rtol=1e-4)
    finally:
        torch.set_num_threads(previous)


def samples():
    return [
        {
            "source_group": source,
            "lineage": f"{source}-{i:02}",
            "quality_factor": q,
            "method": method,
            "label": "cover" if method is None else "stego",
            "split": "train",
        }
        for source, qualities in (
            ("ALASKA2", (None,)),
            ("BOSSbase-1.01", (75, 95)),
            ("BOWS2", (75, 95)),
        )
        for i in range(32)
        for q in qualities
        for method in (None, "JUNIWARD", "UERD")
    ]


def test_fixed_selection_uses_only_first_two_fit_original_pairs():
    rows = samples()
    indices = learning.select(rows)
    assert len(indices) == 4 and indices.dtype == np.int64
    assert [rows[i]["label"] for i in indices] == ["cover", "stego"] * 2
    assert {rows[i]["lineage"] for i in indices} == {"BOWS2-00", "BOWS2-01"}
    assert {rows[i]["quality_factor"] for i in indices} == {75}


@pytest.mark.parametrize("kind", ["size", "role", "groups", "pair"])
def test_invalid_selection_is_not_success(kind):
    rows = samples()
    if kind == "size":
        rows.pop()
    elif kind == "role":
        rows[0]["split"] = "test"
    elif kind == "groups":
        for r in rows:
            if r["lineage"] == "BOWS2-00":
                r["source_group"] = "other"
    else:
        rows[int(learning.select(rows)[0])]["label"] = "stego"
    with pytest.raises(ValueError):
        learning.select(rows)


def test_scalar_metrics_closed_form_and_failures():
    values = {i: np.array([4.0, 0.0] if i % 2 == 0 else [0.0, 4.0]) for i in range(4)}
    result = learning.metrics(values)
    assert result["correct"] == 4 and result["learnability_criterion_passed"]
    assert result["cross_entropy"] == pytest.approx(np.log1p(np.exp(-4)))
    chance = learning.metrics({i: np.zeros(2) for i in range(4)})
    assert chance["correct"] == 2 and not chance["learnability_criterion_passed"]
    with pytest.raises(ValueError):
        learning.metrics({})
    for invalid in (np.zeros(3), np.array([np.nan, 0]), np.array([2**37, 0])):
        with pytest.raises(ValueError):
            learning.metrics({**values, 0: invalid})


@pytest.mark.parametrize("arm", ["legacy", "reference"])
@pytest.mark.parametrize("failure", [None, "signal", "updates", "unchanged"])
def test_control_flow_is_bounded_train_only_not_hardware_evidence(monkeypatch, arm, failure):
    class Model:
        architecture = "mock-not-hardware"

        def to(self, device):
            return self

        def eval(self):
            return self

    class Reader:
        samples = samples()
        verified = False

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def batch(self, indices):
            assert len(indices) == 4
            pixels = np.zeros((4, 1, 256, 256), dtype="<f4")
            if failure != "signal":
                pixels[1::2] = 1
            return pixels

        def verify(self):
            Reader.verified = True

    seen = []

    def train(**kwargs):
        assert kwargs["device"] == "cuda:0" and kwargs["epochs"] == 8
        schedule = kwargs["batches"](0)
        assert schedule.shape == (128, 4) and np.all(schedule == schedule[0])
        assert len(set(schedule.flat)) == 4
        seen.append(True)
        return Model(), [{"updates": 1 if failure == "updates" else 128} for _ in range(8)]

    def score(model, reader, batches, deadline):
        return {int(i): np.zeros(2) for i in batches.flat}, True

    monkeypatch.setattr(learning, "TimingReader", Reader)
    monkeypatch.setattr(learning.srnet_cuda, "inspect", lambda: {"device": "mock-not-hardware"})
    monkeypatch.setattr(learning.srnet_cuda, "policy", nullcontext)
    monkeypatch.setattr(learning.srnet, "network", Model)
    monkeypatch.setattr(learning.srnet_samepool, "network", Model)
    monkeypatch.setattr(learning.srnet_norm_training, "kernel", train)
    monkeypatch.setattr(learning.jpeg_population_control, "score", score)
    identities = iter(["a", "a" if failure == "unchanged" else "b"])
    monkeypatch.setattr(learning.srnet_groupnorm, "parameter_sha", lambda m: next(identities))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 1)
    if failure:
        with pytest.raises(ValueError):
            learning.fit_arm("root", "audit", arm)
    else:
        report = learning.fit_arm("root", "audit", arm)
        assert seen and Reader.verified and report["optimizer_updates"] == 1024
        assert report["probe_pixels_read"] == 0 and report["in_sample_only"]
        assert not report["deployed"] and report["model_discarded"]
        assert not report["final"]["learnability_criterion_passed"]
    with pytest.raises(ValueError):
        learning.fit_arm("root", "audit", "invalid")


def test_frontend_source_binding_exclusive_output_and_failure(monkeypatch, tmp_path):
    assert "core/srnet_samepool.py" in frontend.sources()
    report = {
        "status": "completed",
        "final": {"learnability_criterion_passed": False},
        "singleton_batch_parity": {"initial": True, "final": True},
    }
    monkeypatch.setattr(frontend.jpeg_learnability, "fit_arm", lambda *args: report.copy())
    out = tmp_path / "out"
    result = frontend.execute(tmp_path, tmp_path, "legacy", out)
    assert (out / "report.json").exists() and result["source_sha256"]
    assert not (out / "model.npz").exists()
    with pytest.raises(FileExistsError):
        frontend.execute(tmp_path, tmp_path, "legacy", out)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(FileExistsError):
        frontend.execute(tmp_path, tmp_path, "legacy", link)
    calls = iter([{}, {"changed": "x"}])
    monkeypatch.setattr(frontend, "sources", lambda: next(calls))
    with pytest.raises(ValueError):
        frontend.execute(tmp_path, tmp_path, "legacy", tmp_path / "new")


@pytest.mark.parametrize("parity", [True, False])
def test_main_keeps_parity_failure_and_learning_failure_distinct(monkeypatch, tmp_path, parity):
    limits = []
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(
        frontend,
        "execute",
        lambda *args: {
            "status": "completed",
            "final": {"learnability_criterion_passed": False},
            "singleton_batch_parity": {"initial": True, "final": parity},
        },
    )
    assert frontend.main(
        [
            "--root",
            str(tmp_path),
            "--audit",
            str(tmp_path),
            "--out",
            str(tmp_path / "new"),
            "--arm",
            "legacy",
        ]
    ) == (0 if parity else 2)
    assert len(limits) == 3
