"""Architecture/limit/readiness contracts, never detection accuracy fixtures."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import cli
from core import srnet
from steganography import research_srnet as service

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_structure_initialization_and_no_early_pool():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = srnet.network().eval()
    finally:
        torch.set_default_dtype(previous)
    assert sum(p.numel() for p in model.parameters()) == 4779616
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    assert len(model.front) == 7 and len(model.middle) == 4
    assert not any(isinstance(m, torch.nn.AvgPool2d) for m in model.front.modules())
    assert model.classifier.bias is None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            assert torch.all(m.bias == 0.2)
    values = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    with torch.no_grad():
        for stage in model.front:
            values = stage(values)
            assert values.shape[-2:] == (128, 128)
        for stage, channels, side in zip(
            model.middle, (16, 64, 128, 256), (64, 32, 16, 8), strict=True
        ):
            values = stage(values)
            assert values.shape == (1, channels, side, side)
    block = model.front[2]
    for p in block.parameters():
        p.data.zero_()
    assert torch.equal(block(-torch.ones(1, 16, 8, 8)), -torch.ones(1, 16, 8, 8))


@pytest.mark.parametrize("side", [128, 256])
def test_bounded_eval_and_immutable_bn(side):
    model = srnet.network().eval()
    raw = np.zeros((1, 1, side, side), dtype=np.uint8)
    before = {k: v.clone() for k, v in model.named_buffers()}
    assert srnet.logits(model, raw).shape == (1, 2)
    assert all(torch.equal(before[k], v) for k, v in model.named_buffers())
    model.front[0][1].train()
    with pytest.raises(ValueError, match="eval"):
        srnet.logits(model, raw)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        np.zeros((1, 1, 128, 128)),
        np.zeros((1, 128, 128), dtype="u1"),
        np.zeros((1, 3, 128, 128), dtype="u1"),
        np.zeros((1, 1, 64, 64), dtype="u1"),
        np.zeros((1, 1, 128, 256), dtype="u1"),
        np.zeros((0, 1, 128, 128), dtype="u1"),
        np.zeros((5, 1, 128, 128), dtype="u1"),
    ],
)
def test_input_limits(raw):
    with pytest.raises(ValueError, match="bounded"):
        srnet.logits(None, raw)


@pytest.mark.parametrize("bad", ["shape", "nan"])
def test_output_guards(bad):
    class Bad(torch.nn.Module):
        def forward(self, values):
            return (
                torch.zeros((len(values), 1))
                if bad == "shape"
                else torch.full((len(values), 2), float("nan"))
            )

    with pytest.raises(ValueError, match="output"):
        srnet.logits(Bad().eval(), np.zeros((1, 1, 128, 128), dtype="u1"))


def test_optional_dependency_absence(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="optional"):
        srnet.network()
    assert service.smoke()["status"] == "unavailable"


def test_learning_smoke_restores_state():
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    report = service.smoke()
    assert (
        report["status"] == "completed"
        and report["finite_gradients"]
        and report["first_layer_updated"]
    )
    assert report["singleton_batch_decisions_equal"]
    assert torch.equal(torch.get_rng_state(), rng) and torch.get_num_threads() == threads


@pytest.mark.parametrize("fault", ["loss", "gradient", "batch"])
def test_learning_failures(monkeypatch, fault):
    if fault == "loss":
        original = torch.nn.functional.cross_entropy
        monkeypatch.setattr(
            torch.nn.functional, "cross_entropy", lambda *a, **k: original(*a, **k) * float("nan")
        )
    elif fault == "gradient":
        factory = srnet.network

        def incomplete():
            m = factory()
            m.register_parameter("unused", torch.nn.Parameter(torch.ones(1)))
            return m

        monkeypatch.setattr(srnet, "network", incomplete)
    else:
        call = 0

        def different(model, raw):
            nonlocal call
            call += 1
            return np.full((len(raw), 2), call, dtype="f4")

        monkeypatch.setattr(srnet, "logits", different)
    assert service.smoke()["status"] == "failed"


@pytest.mark.parametrize(
    "fault", ["ok", "unavailable", "exit", "large", "json", "type", "status", "timeout", "os"]
)
def test_worker_and_cli(monkeypatch, tmp_path, fault, capsys):
    def run(command, **kwargs):
        assert command[-2:] == ["-m", "steganography.research_srnet"]
        assert kwargs["timeout"] == 90 and kwargs["env"]["OMP_NUM_THREADS"] == "2"
        if fault == "timeout":
            raise service.subprocess.TimeoutExpired(command, 90)
        if fault == "os":
            raise OSError("secret-path")
        payload = {
            "ok": b'{"status":"completed"}',
            "unavailable": b'{"status":"unavailable"}',
            "large": b"x" * 65537,
            "json": b"invalid",
            "type": b"[]",
            "status": b'{"status":"bogus"}',
        }
        kwargs["stdout"].write(payload.get(fault, b"{}"))
        return SimpleNamespace(returncode=1 if fault == "exit" else 0)

    monkeypatch.setattr(service.subprocess, "run", run)
    out = tmp_path / "report.json"
    assert cli.main(["research", "srnet-preflight", "--out", str(out)]) == (
        0 if fault == "ok" else 2
    )
    report = json.loads(out.read_text())
    assert report["qualification"] == "unavailable" and report["synthetic_only"]
    assert not report["deployed"] and "secret-path" not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        service.run_preflight(out)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(FileExistsError):
        service.run_preflight(link)


@pytest.mark.parametrize("fault", ["ok", "limits", "smoke"])
def test_worker_main_limits_and_redaction(monkeypatch, fault, capsys):
    import resource

    limits = []

    def setlimit(kind, values):
        if fault == "limits":
            raise OSError("secret-limit-error")
        limits.append((kind, values))

    def smoke():
        if fault == "smoke":
            raise RuntimeError("secret-worker-error")
        return {"status": "completed"}

    monkeypatch.setattr(resource, "setrlimit", setlimit)
    monkeypatch.setattr(service, "smoke", smoke)
    assert service.main() == 0
    printed = capsys.readouterr().out
    report = json.loads(printed)
    assert (
        report["status"] == {"ok": "completed", "limits": "unavailable", "smoke": "failed"}[fault]
    )
    assert "secret" not in printed
    if fault != "limits":
        assert (resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3)) in limits
        assert (resource.RLIMIT_CPU, (60, 61)) in limits
