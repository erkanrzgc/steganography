"""CPU arithmetic and mocked hardware orchestration, not physical GPU evidence."""

import io
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_batch_probe as c
from steganography import research_batch_probe as f


@pytest.mark.parametrize("rows", c.ROWS)
def test_generated_contract(rows):
    values = c.generated(rows)
    assert values.shape == (rows, 1, 256, 256)
    assert values.dtype == np.dtype("<f4") and np.isfinite(values).all()
    np.testing.assert_array_equal(values, c.generated(rows))
    assert not np.array_equal(values[0], values[1])


@pytest.mark.parametrize("rows", [True, 4, 64, 8.0, "8"])
def test_rows_fail_closed(rows):
    with pytest.raises(ValueError):
        c.generated(rows)


def tiny():
    import torch

    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(1, 2)
    )


def test_real_update_and_input_rejections():
    import torch

    model = tiny()
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, foreach=False)
    before = [p.detach().clone() for p in model.parameters()]
    loss = c.update(model, optimizer, c.generated(8), "cpu")
    assert np.isfinite(loss) and loss >= 0
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before, strict=True))
    for values in (
        c.generated(8)[:4], c.generated(8).astype(np.float64),
        np.full((8, 1, 256, 256), np.nan, dtype="<f4"),
        np.full((8, 1, 256, 256), 2**37, dtype="<f4"),
        c.generated(8)[:, :, :128],
    ):
        with pytest.raises(ValueError, match="pixels"):
            c.update(model, optimizer, values, "cpu")


def test_invalid_math_rejected():
    import torch

    for width, bias, message in ((3, 0, "logits"), (2, float("nan"), "logits")):
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(1, width)
        )
        with torch.no_grad():
            model[-1].bias.fill_(bias)
        optimizer = torch.optim.Adamax(model.parameters())
        with pytest.raises(ValueError, match=message):
            c.update(model, optimizer, c.generated(8), "cpu")
    model = tiny()
    optimizer = torch.optim.Adamax(model.parameters())
    handle = model[-1].weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(ValueError, match="gradients"):
        c.update(model, optimizer, c.generated(8), "cpu")
    handle.remove()
    monkey = pytest.MonkeyPatch()
    try:
        def corrupt():
            with torch.no_grad():
                model[-1].weight.fill_(float("nan"))

        monkey.setattr(optimizer, "step", corrupt)
        with pytest.raises(ValueError, match="parameters"):
            c.update(model, optimizer, c.generated(8), "cpu")
    finally:
        monkey.undo()


def test_mocked_probe_restores_threads(monkeypatch):
    import torch

    previous = torch.get_num_threads()
    monkeypatch.setattr(c.srnet_cuda, "policy", lambda: nullcontext({"mock": True}))
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: nullcontext())
    for name in ("manual_seed", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 456)
    model = tiny()
    monkeypatch.setattr(model, "to", lambda _: model)
    monkeypatch.setattr(c.srnet_samepool, "network", lambda: model)
    monkeypatch.setattr(c.srnet_model, "validate_tensors", lambda _: None)
    real_update = c.update
    monkeypatch.setattr(c, "update", lambda m, o, v, _: real_update(m, o, v, "cpu"))
    result = c.probe(8)
    assert result["optimizer_updates"] == 2 and len(result["losses"]) == 2
    assert result["rows"] == 8 and not result["gradient_accumulation"]
    assert result["peak_gpu_reserved_bytes"] == 456
    assert torch.get_num_threads() == previous
    monkeypatch.setattr(c.srnet_cuda, "policy", lambda: nullcontext({}))
    monkeypatch.setattr(c, "update", lambda *_: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError, match="bad"):
        c.probe(8)
    assert torch.get_num_threads() == previous


def setup_front(monkeypatch):
    monkeypatch.setattr(f, "sources", lambda: {"test": "fixed"})


def test_sources_are_bounded_current_closure():
    result = f.sources()
    assert len(result) == 37
    assert all(len(value) == 64 for value in result.values())


def test_sources_reject_oversize(monkeypatch):
    monkeypatch.setattr(f, "previous_sources", lambda: {})
    monkeypatch.setattr(f, "regular_open", lambda _: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError, match="source exceeds bound"):
        f.sources()


def test_nonfinite_loss_rejected(monkeypatch):
    import torch

    model = tiny()
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", lambda *_: torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="loss"):
        c.update(model, torch.optim.Adamax(model.parameters()), c.generated(8), "cpu")


@pytest.mark.parametrize("status", ["completed", "unavailable"])
def test_parent_and_cli(monkeypatch, tmp_path, status):
    setup_front(monkeypatch)

    def child(command, **kwargs):
        assert "--worker" in command and kwargs["timeout"] == 180
        assert kwargs["stdin"] == subprocess.DEVNULL
        kwargs["stdout"].write(json.dumps({
            "status": status, "source_sha256": f.sources(), "rows": int(command[-1]),
        }).encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(f.subprocess, "run", child)
    out = tmp_path / "probe.json"
    report = f.run(8, out)
    assert report["status"] == status and not report["real_data_used"]
    assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]
    assert f.main(["--rows", "16", "--out", str(tmp_path / "other.json")]) == (
        0 if status == "completed" else 2
    )
    with pytest.raises(FileExistsError):
        f.run(8, out)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        f.run(8, link)
    with pytest.raises(SystemExit):
        f.main(["--rows", "8"])


def test_timeout_unavailable_and_changed_sources(monkeypatch, tmp_path):
    setup_front(monkeypatch)
    monkeypatch.setattr(f.subprocess, "run", lambda *_a, **_k: (
        _ for _ in ()
    ).throw(subprocess.TimeoutExpired("fixed", 180)))
    assert f.run(8, tmp_path / "timeout.json")["reason"] == "worker_timeout"
    calls = iter([{"before": "x"}, {"after": "y"}])
    monkeypatch.setattr(f, "sources", lambda: next(calls))
    with pytest.raises(ValueError, match="sources changed"):
        f.run(8, tmp_path / "changed.json")
    assert not (tmp_path / "changed.json").exists()


@pytest.mark.parametrize("raw,code", [
    (b"x" * 65537, 0), (b"", 1), (b"[]", 0),
    (b'{"status":"failed"}', 0),
    (b'{"status":"completed","source_sha256":{}}', 0),
])
def test_worker_response_rejected(monkeypatch, tmp_path, raw, code):
    setup_front(monkeypatch)

    def child(*_, **kwargs):
        kwargs["stdout"].write(raw)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(f.subprocess, "run", child)
    with pytest.raises(ValueError):
        f.run(8, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_worker_bounds_status_and_source_changes(monkeypatch):
    import torch

    setup_front(monkeypatch)
    limits = []
    monkeypatch.setattr(f.resource, "setrlimit", lambda key, value: limits.append((key, value)))
    monkeypatch.setattr(c, "probe", lambda _: {"rows": 8})
    assert f.worker(8)["status"] == "completed"
    assert (f.resource.RLIMIT_CORE, (0, 0)) in limits
    assert (f.resource.RLIMIT_CPU, (160, 165)) in limits
    assert (f.resource.RLIMIT_FSIZE, (65536, 65536)) in limits
    for exception, reason in (
        (f.srnet_cuda.CUDAUnavailable("missing"), "missing"),
        (torch.cuda.OutOfMemoryError("private detail"), "gpu_allocator_out_of_memory"),
        (ImportError("private detail"), "optional_dependency_unavailable"),
    ):
        monkeypatch.setattr(c, "probe", lambda _, exc=exception: (_ for _ in ()).throw(exc))
        assert f.worker(8)["reason"] == reason
        assert f.main(["--worker", "--rows", "8"]) == 0
    monkeypatch.setattr(c, "probe", lambda _: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError, match="bad"):
        f.worker(8)
    monkeypatch.setattr(c, "probe", lambda _: {})
    calls = iter([{"a": "x"}, {"b": "y"}])
    monkeypatch.setattr(f, "sources", lambda: next(calls))
    with pytest.raises(ValueError, match="sources changed"):
        f.worker(8)
