"""Complete validation fixtures and adversarial cache/worker guards."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_block_validation as core
from steganography import research_srnet_epoch_validation as service
from steganography.benchmarking.metrics import classification_metrics
from tests.test_srnet_epochs import config as config
from tests.test_srnet_stream import corpus as corpus


def save(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def ready(config):
    root = Path(config["root"])
    index = json.loads((root / "index.json").read_text())
    for block in index["blocks"]:
        folder = (root / block["path"]).parent / "validation"
        folder.mkdir()
        manifest = json.loads((root / block["path"]).read_text())
        rows = [s for s in manifest["samples"] if s["split"] == "validation"]
        values = np.empty((len(rows), 1, 256, 256), dtype="<f4")
        for row, sample in zip(values, rows, strict=True):
            row.fill(128 if sample["label"] == "cover" else 130)
        (folder / "pixels.f32").write_bytes(values.tobytes())
        data_sha = hashlib.sha256(values.tobytes()).hexdigest()
        cache = {
            "schema_version": "research-float256-cache-v1",
            "feature_version": core.jpeg_float256.FEATURE_VERSION,
            "manifest_sha256": block["manifest_sha256"],
            "split": "validation",
            "dtype": "<f4",
            "shape": [len(rows), 1, 256, 256],
            "decoder": core.jpeg_float256.decoder_contract(),
            "data_sha256": data_sha,
            "rows": [
                {
                    **{k: r[k] for k in ("sha256", "lineage", "label")},
                    "image_size": [512, 512],
                    "region": [128, 128, 256, 256],
                }
                for r in rows
            ],
        }
        cache_sha = save(folder / "cache.json", cache)
        block["caches"]["validation"] = {
            "path": f"{folder.parent.name}/validation/cache.json",
            "sha256": cache_sha,
            "data_sha256": data_sha,
            "rows": len(rows),
        }
    config["index_sha256"] = save(root / "index.json", index)
    audit = json.loads(Path(config["audit"]).read_text())
    audit["index_sha256"] = config["index_sha256"]
    config["audit_sha256"] = save(Path(config["audit"]), audit)
    return config


def reader(config, deadline=lambda: None):
    return core.ValidationBlocks(
        Path(config["root"]),
        index_sha256=config["index_sha256"],
        audit=Path(config["audit"]),
        audit_sha256=config["audit_sha256"],
        deadline=deadline,
    )


@pytest.fixture
def fake_model(monkeypatch):
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(state_dict=lambda: {"value": torch.zeros(1)})

    def forward(model, values):
        margin = values[:, 0, 0, 0] - 129
        return np.stack([np.zeros_like(margin), margin], axis=1)

    monkeypatch.setattr(core.srnet, "float_logits", forward)
    monkeypatch.setattr(
        core.srnet_reference, "reference_logits", lambda arrays, values: forward(model, values)
    )
    return model


def test_complete_bounded_validation_and_six_fixed_cells(ready, fake_model):
    with reader(ready) as opened:
        assert not isinstance(opened, core.TrainBlocks)
        assert len(opened.samples) == 9
        report = core.evaluate(fake_model, opened, lambda: None, metrics=classification_metrics)
        assert report["status"] == "completed" and len(report["cells"]) == 6
        assert len(report["independent_forward_audit"]) == 9
        assert all(
            c["metrics"]["roc_auc"] == 1.0 and c["metrics"]["recommended_threshold"] is None
            for c in report["cells"]
        )
        assert report["threshold"] == 0.5 and report["accuracy_qualification"] == "unavailable"
        assert opened.max_batch_bytes == 4 * 256 * 256 * 4
    with pytest.raises(ValueError):
        opened.batch(np.array([0], dtype="<i8"))


@pytest.mark.parametrize(
    "fault",
    [
        "path",
        "schema",
        "split",
        "shape",
        "rows",
        "identity",
        "size",
        "checksum",
        "nan",
        "symlink",
        "empty",
    ],
)
def test_validation_cache_guards(ready, fault):
    root = Path(ready["root"])
    index = json.loads((root / "index.json").read_text())
    bound = index["blocks"][0]["caches"]["validation"]
    path = root / bound["path"]
    cache = json.loads(path.read_text())
    if fault == "path":
        bound["path"] = "../escape"
    elif fault == "schema":
        cache["schema_version"] = "bad"
    elif fault == "split":
        cache["split"] = "train"
    elif fault == "shape":
        cache["shape"][0] = 40000
    elif fault == "rows":
        cache["rows"] = []
    elif fault == "identity":
        cache["rows"][0]["sha256"] = "0" * 64
    elif fault == "size":
        (path.parent / "pixels.f32").write_bytes(b"short")
    elif fault == "checksum":
        (path.parent / "pixels.f32").write_bytes(b"\0" * (3 * 256 * 256 * 4))
    elif fault == "nan":
        (path.parent / "pixels.f32").write_bytes(
            np.full((3, 1, 256, 256), np.nan, dtype="<f4").tobytes()
        )
    elif fault == "symlink":
        target = path.parent / "pixels.f32"
        target.rename(path.parent / "original.f32")
        target.symlink_to(path.parent / "original.f32")
    else:
        index["splits"]["validation"] = 0
    bound["sha256"] = save(path, cache)
    ready["index_sha256"] = save(root / "index.json", index)
    audit = json.loads(Path(ready["audit"]).read_text())
    audit.update(index_sha256=ready["index_sha256"], splits=index["splits"])
    ready["audit_sha256"] = save(Path(ready["audit"]), audit)
    with pytest.raises((ValueError, KeyError)):
        reader(ready)


def test_deadline_input_and_numerical_failure(ready, fake_model, monkeypatch):
    with pytest.raises(ValueError):
        reader(ready, deadline=None)
    with pytest.raises(ValueError):
        core.evaluate(fake_model, object(), lambda: None, metrics=classification_metrics)
    with reader(ready) as opened:
        monkeypatch.setattr(core.srnet_reference, "compare", lambda *a: {"passed": False})
        assert (
            core.evaluate(fake_model, opened, lambda: None, metrics=classification_metrics)[
                "status"
            ]
            == "failed_numerical_gate"
        )


def test_context_bound_and_model_immutability(ready, fake_model, monkeypatch):
    with reader(ready) as opened:
        original = copy.deepcopy(opened.samples)
        opened.samples = [{**original[0], "source_group": str(i)} for i in range(25)]
        with pytest.raises(ValueError, match="context"):
            core.evaluate(fake_model, opened, lambda: None, metrics=classification_metrics)
        opened.samples = original
        calls = [0]

        def state():
            torch = pytest.importorskip("torch")
            calls[0] += 1
            return {"value": torch.tensor([float(calls[0])])}

        fake_model.state_dict = state
        with pytest.raises(ValueError, match="mutated"):
            core.evaluate(fake_model, opened, lambda: None, metrics=classification_metrics)


def test_service_sources_and_original_plan_gate(ready, tmp_path, monkeypatch):
    assert len(service.sources()) == 20
    plan = tmp_path / "plan.json"
    plan_sha = save(plan, {"wrong": True})
    monkeypatch.setattr(service.epochs, "execute", lambda *a, **k: {})
    with pytest.raises(ValueError, match="original"):
        service.execute(
            ready,
            tmp_path / "out",
            plan_path=plan,
            plan_sha=plan_sha,
            model_dir=tmp_path / "model",
            card_sha="1" * 64,
        )


@pytest.mark.parametrize("fault", [None, "model", "mutation"])
def test_service_publication_binding_and_thread_restore(
    ready, tmp_path, monkeypatch, fake_model, fault
):
    torch = pytest.importorskip("torch")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_sha = save(model_dir / "model.npz", {})
    card_sha = save(model_dir / "epoch.json", {"model_sha256": model_sha})
    plan_sha = save(tmp_path / "plan.json", {"source_sha256": {}})
    monkeypatch.setattr(service.epochs, "execute", lambda *a, **k: {"source_sha256": {}})
    monkeypatch.setattr(
        service.epochs,
        "parent_state",
        lambda *a: {
            "arrays": {"model/value": np.array([1 if fault == "model" else 0], dtype="<f4")}
        },
    )
    monkeypatch.setattr(service.srnet_model, "load_model", lambda *a, **k: fake_model)
    if fault == "mutation":
        calls = iter([service.sources(), {}])
        monkeypatch.setattr(service, "sources", lambda: next(calls))
    previous = torch.get_num_threads()
    arguments = {
        "plan_path": tmp_path / "plan.json",
        "plan_sha": plan_sha,
        "model_dir": model_dir,
        "card_sha": card_sha,
    }
    out = tmp_path / "out.json"
    if fault:
        with pytest.raises(ValueError):
            service.execute(ready, out, **arguments)
        assert not out.exists()
    else:
        report = service.execute(ready, out, **arguments)
        assert report["validation_rows"] == 9 and report["cross_source_heldout"] is False
    assert torch.get_num_threads() == previous


def args(ready, tmp_path):
    config_path = tmp_path / "config.json"
    save(config_path, ready)
    return argparse.Namespace(
        config=config_path,
        out=tmp_path / "out",
        plan=tmp_path / "plan",
        plan_sha="1" * 64,
        model_dir=tmp_path / "model",
        card_sha="2" * 64,
        config_sha256=None,
        worker=False,
    )


@pytest.mark.parametrize("fault", [None, "oversized", "json", "timeout", "status", "sources"])
def test_parent_worker_response_bounds(ready, tmp_path, monkeypatch, fault):
    arguments = args(ready, tmp_path)

    def run(command, **kwargs):
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        record = {"source_sha256": {} if fault == "sources" else service.sources()}
        digest = save(arguments.out, record)
        raw = json.dumps(
            {"status": "failed" if fault == "status" else "completed", "report_sha256": digest}
        ).encode()
        if fault == "oversized":
            raw = b"x" * 65537
        if fault == "json":
            raw = b"bad"
        kwargs["stdout"].write(raw)
        assert kwargs["timeout"] == 210
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(service.subprocess, "run", run)
    if fault:
        with pytest.raises(RuntimeError):
            service.run_job(arguments)
    else:
        assert service.run_job(arguments)["source_sha256"] == service.sources()


def test_main_worker_hard_limits_and_parent_status(ready, tmp_path, monkeypatch, capsys):
    import resource

    arguments = args(ready, tmp_path)
    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(service.epochs.srnet_cuda, "host_bound", lambda: 8 * 1024**3)
    monkeypatch.setattr(service, "execute", lambda *a, **k: save(arguments.out, {}))
    argv = [
        "--config",
        str(arguments.config),
        "--out",
        str(arguments.out),
        "--plan",
        str(arguments.plan),
        "--plan-sha",
        arguments.plan_sha,
        "--model-dir",
        str(arguments.model_dir),
        "--card-sha",
        arguments.card_sha,
    ]
    assert (
        service.main(
            [*argv, "--worker", "--config-sha256", service.epochs.file_sha(arguments.config)]
        )
        == 0
    )
    assert '"completed"' in capsys.readouterr().out
    monkeypatch.setattr(service, "run_job", lambda _: {"status": "completed"})
    assert service.main(argv) == 0
    assert '"completed"' in capsys.readouterr().out
    monkeypatch.setattr(service, "run_job", lambda _: {"status": "failed_numerical_gate"})
    assert service.main(argv) == 2
    monkeypatch.setattr(service, "run_job", lambda _: (_ for _ in ()).throw(ValueError()))
    assert service.main(argv) == 2


def test_real_torch_and_independent_numpy_complete_generated_replay(ready):
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            model = core.srnet.network().eval()
        with reader(ready) as opened:
            report = core.evaluate(model, opened, lambda: None, metrics=classification_metrics)
        assert report["status"] == "completed"
        assert len(report["independent_forward_audit"]) == 9
        assert all(a["passed"] for a in report["independent_forward_audit"])
    finally:
        torch.set_num_threads(previous)
