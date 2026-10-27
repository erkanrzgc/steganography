"""Generated CPU numerics and mocked GPU orchestration, never GPU evidence."""

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import colab_artifacts as a
from core import spatial_colab as p

ROOT = Path(__file__).parent.parent
torch = pytest.importorskip("torch")


@pytest.fixture
def protocol(tmp_path):
    path = tmp_path / "protocol.json"
    p.freeze(ROOT, path)
    return path


def test_freeze_and_exact_binding(protocol, tmp_path):
    value = p.verify(ROOT, protocol)
    assert value["optimizer_updates"] == 0 and value["max_seconds"] == 600
    assert value["seeds"] == list(p.SEEDS) and value["batches"] == [2, 4, 8]
    for field in ("source_sha256", "max_seconds", "amp", "environment"):
        broken = copy.deepcopy(value)
        broken[field] = {}
        path = tmp_path / (field + ".json")
        a.put(path, a.encode(broken))
        with pytest.raises(ValueError, match="binding"):
            p.verify(ROOT, path)


@pytest.mark.parametrize("domain", p.DOMAINS)
def test_real_generated_cpu_forward_backward(domain, monkeypatch):
    prior = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    try:
        torch.set_num_threads(2)
        monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args: pytest.fail("optimizer step"))
        result = p.cell(domain, p.SEEDS[0], 2, device="cpu")
        assert result["status"] == "ready" and result["finite_gradients"]
        assert result["parameters_unchanged"] and result["lsb_preserved"]
        assert result["optimizer_updates"] == 0
        assert len(result["forward_backward_seconds"]) == 3
        assert result["peak_vram_allocated_bytes"] is None
        assert torch.equal(torch.get_rng_state(), rng)
        # Every possible uint8 level and its one-LSB neighbour remains distinct.
        raw = np.arange(256, dtype="u1")
        floating = raw.astype("f4")
        np.testing.assert_array_equal(floating.astype("u1"), raw)
        assert np.all(np.abs(floating - (raw ^ 1).astype("f4")) == 1)
    finally:
        torch.set_num_threads(prior)


@pytest.mark.parametrize("fault", ["domain", "seed", "batch", "device"])
def test_invalid_cell(fault):
    kwargs = {"domain": p.DOMAINS[0], "seed": p.SEEDS[0], "batch": 2, "device": "cpu"}
    kwargs[fault] = "bad"
    with pytest.raises(ValueError, match="contract"):
        p.cell(**kwargs)


def test_numerical_and_parameter_failures(monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self, change=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.change = change

        def forward(self, values):
            if self.change:
                with torch.no_grad():
                    self.weight.add_(1)
            score = values.mean((1, 2, 3)) * self.weight
            return torch.stack((score, -score), dim=1)

    monkeypatch.setattr(p.models, "network", lambda domain: Tiny())
    monkeypatch.setattr(
        torch.nn.functional, "cross_entropy", lambda output, target: output.mean() * float("nan")
    )
    with pytest.raises(ValueError, match="gradients"):
        p.cell(p.DOMAINS[0], p.SEEDS[0], 2, device="cpu")
    monkeypatch.undo()
    monkeypatch.setattr(p.models, "network", lambda domain: Tiny(change=True))
    with pytest.raises(ValueError, match="changed parameters"):
        p.cell(p.DOMAINS[0], p.SEEDS[0], 2, device="cpu")


def test_policy_restores_on_error():
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError), p.fp32_policy():
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        raise RuntimeError("generated interruption")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == previous


def test_worker_gpu_missing_mismatch_and_complete(protocol, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert p.worker(ROOT, protocol, tmp_path)["reason"] == "gpu_missing"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(p, "environment", lambda: {**p.ENVIRONMENT, "gpu": "test-double"})
    wrong = {**p.ENVIRONMENT, "torch": "wrong", "gpu": "test-double"}
    with monkeypatch.context() as patch:
        patch.setattr(p, "environment", lambda: wrong)
        assert p.worker(ROOT, protocol, tmp_path)["reason"] == "environment_mismatch"
    seen = []
    monkeypatch.setattr(p, "cell", lambda d, s, b: seen.append((d, s, b)) or {"status": "ready"})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    prior = torch.get_num_threads()
    try:
        assert p.worker(ROOT, protocol, tmp_path)["status"] == "ready"
        assert seen == [(d, s, b) for d in p.DOMAINS for s in p.SEEDS for b in p.BATCHES]
        assert len(list(tmp_path.glob("cell-*.json"))) == 12
    finally:
        torch.set_num_threads(prior)


def test_worker_oom_retains_environment_and_failed_cell(protocol, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    actual = {**p.ENVIRONMENT, "gpu": "test-double"}
    monkeypatch.setattr(p, "environment", lambda: actual)

    def fail(*args):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(p, "cell", fail)
    prior = torch.get_num_threads()
    try:
        report = p.worker(ROOT, protocol, tmp_path)
    finally:
        torch.set_num_threads(prior)
    assert report["status"] == "unavailable" and report["environment"] == actual
    assert report["failed_cell"]["batch"] == 2
    assert a.document(tmp_path / "execution.json")["environment"] == actual


@pytest.mark.parametrize("mode", ["complete", "partial", "timeout", "failed"])
def test_supervisor_bounds_no_retry_and_partial_evidence(protocol, tmp_path, monkeypatch, mode):
    out = tmp_path / "attempt"

    def run(args, **kwargs):
        assert kwargs["timeout"] <= 598 and kwargs["stdin"] == subprocess.DEVNULL
        assert "--protocol" in args and "_worker" in args
        if mode == "timeout":
            a.put(out / "execution.json", a.encode({"environment": {"gpu": "test-double"}}))
            raise subprocess.TimeoutExpired(args, 598)
        for index, (domain, seed, batch) in enumerate(
            (d, s, b) for d in p.DOMAINS for s in p.SEEDS for b in p.BATCHES
        ):
            if mode == "complete" or index == 0:
                a.put(
                    out / f"cell-{index:02d}.json",
                    a.encode(
                        {
                            "domain": domain,
                            "seed": seed,
                            "batch": batch,
                            "forward_backward_seconds": [0.2, 0.3, 0.4],
                        }
                    ),
                )
        a.put(out / "worker.json", a.encode({"status": "ready"}))
        return SimpleNamespace(returncode=1 if mode == "failed" else 0)

    monkeypatch.setattr(p.subprocess, "run", run)
    result = p.supervise(ROOT, protocol, out)
    assert result["status"] == ("ready" if mode == "complete" else "unavailable")
    assert result["optimizer_updates"] == 0 and len(result["cost_table"]) == 12
    assert result["full_training_budget"] == "missing"
    assert not result["real_media_opened"] and not result["retry"]
    assert (out / "measurement.json").exists()
    with pytest.raises(FileExistsError):
        p.supervise(ROOT, protocol, tmp_path / "another")


def test_supervisor_drive_rejected(protocol, tmp_path):
    with pytest.raises(ValueError, match="local"):
        p.supervise(ROOT, protocol, tmp_path / "drive" / "out")


@pytest.mark.parametrize("fault", ["late", "output"])
def test_supervisor_late_or_oversized_receipt_cannot_pass(protocol, tmp_path, monkeypatch, fault):
    out = tmp_path / "attempt"
    clock = [0.0]
    monkeypatch.setattr(p.time, "monotonic", lambda: clock[0])

    def run(args, **kwargs):
        if fault == "late":
            clock[0] = p.MAX_SECONDS + 1
        else:
            kwargs["stdout"].write(bytes(2 * a.MIB))
            kwargs["stdout"].flush()
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(p.subprocess, "run", run)
    if fault == "late":
        assert p.supervise(ROOT, protocol, out)["reason"] == "deadline"
    else:
        with pytest.raises(ValueError, match="output limit"):
            p.supervise(ROOT, protocol, out)


@pytest.mark.parametrize("times", [[0, 1, 2], [1, 2], [1, float("nan"), 2]])
def test_cost_table_does_not_invent_training_time(times):
    with pytest.raises(ValueError, match="times"):
        p.costs(
            [
                {
                    "domain": p.DOMAINS[0],
                    "seed": p.SEEDS[0],
                    "batch": 2,
                    "forward_backward_seconds": times,
                }
            ]
        )
    assert all(r["full_training_seconds"] is None for r in p.costs([]))
