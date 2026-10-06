"""Hard child limits, bounded output and artifact verification, not accuracy."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from core import jpeg_float256 as fp
from core.srnet_multibatch import RECIPE
from steganography import research_srnet_job as job
from steganography import research_srnet_plan as planner
from tests.test_jrm_reference import sha
from tests.test_srnet_training import config as fit_config  # noqa: F401 — shared pytest fixture
from tests.test_srnet_training import corpus as corpus
from tests.test_srnet_training import plan_config as plan_config

REAL_DECODER = fp.decoder_contract


@pytest.fixture
def config_path(fit_config, tmp_path):  # noqa: F811 — injected fixture
    path = tmp_path / "config.json"
    path.write_text(json.dumps(fit_config))
    return path


@pytest.mark.parametrize("batch_recipe", [None, RECIPE])
def test_actual_hard_limited_worker_fit(config_path, tmp_path, monkeypatch, capsys, batch_recipe):
    pytest.importorskip("jpeglib")
    config = json.loads(config_path.read_bytes())
    monkeypatch.setattr(fp, "decoder_contract", REAL_DECODER)
    cache = Path(config["cache"])
    descriptor = json.loads(cache.read_bytes())
    descriptor["decoder"] = REAL_DECODER()
    cache.write_text(json.dumps(descriptor))
    config["cache_sha256"] = sha(cache)
    extra = {"batch_recipe": batch_recipe} if batch_recipe is not None else {}
    plan = tmp_path / "native-contract-plan.json"
    planner.plan_training(
        {
            **{k: config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")},
            "epochs": 1,
            "seed": 91,
            **extra,
        },
        plan,
    )
    config.update(plan=str(plan), plan_sha256=sha(plan), max_seconds=60)
    config.update(extra)
    config_path.write_text(json.dumps(config))
    out = tmp_path / "isolated"
    assert cli.main(["research", "srnet-fit", "--config", str(config_path), "--out", str(out)]) == 0
    card = json.loads(capsys.readouterr().out)
    assert card["training"] == "completed"
    assert card["epoch_training"][0]["updates"] == (4 if batch_recipe is None else 2)
    assert card["batch_size"] == (2 if batch_recipe is None else 4)
    assert card["model_sha256"] == sha(out / "model.npz")
    assert not card["deployed"] and card["accuracy_metrics"] == "unavailable"
    with pytest.raises(FileExistsError):
        job.run_job(config_path, out)


@pytest.mark.parametrize(
    "fault",
    [
        "timeout",
        "exit",
        "json",
        "oversized",
        "unavailable",
        "missing",
        "hash",
        "oversized-model",
        "symlink",
    ],
)
def test_failed_worker_never_returns_completed_card(config_path, tmp_path, monkeypatch, fault):
    out = tmp_path / "output"

    def run(command, **kwargs):
        assert command[:3] == [job.sys.executable, "-m", "steganography.research_srnet_job"]
        assert kwargs["timeout"] == 1920 and kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        response = {"status": "completed", "card_sha256": "0" * 64, "model_sha256": "0" * 64}
        if fault == "unavailable":
            response = {"status": "unavailable"}
        kwargs["stdout"].write(
            b"x" * (job.OUTPUT_LIMIT + 1)
            if fault == "oversized"
            else b"bad"
            if fault == "json"
            else json.dumps(response).encode()
        )
        if fault in {"hash", "oversized-model", "symlink"}:
            out.mkdir()
            (out / "model-card.json").write_text(json.dumps({"model_sha256": "0" * 64}))
            model = out / "model.npz"
            if fault == "symlink":
                model.symlink_to(config_path)
            else:
                model.write_bytes(bytes(17))
            if fault == "oversized-model":
                monkeypatch.setattr(job, "MAX_BYTES", 16)
        return SimpleNamespace(returncode=1 if fault == "exit" else 0)

    monkeypatch.setattr(job.subprocess, "run", run)
    with pytest.raises(RuntimeError) as exc:
        job.run_job(config_path, out)
    assert str(tmp_path) not in str(exc.value)


def test_worker_limits_config_binding_and_redacted_failures(
    config_path, tmp_path, monkeypatch, capsys
):
    import resource

    seen = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: seen.append(a))
    out = tmp_path / "worker"

    def fit(config, target):
        target.mkdir()
        card = {"model_sha256": "1" * 64}
        (target / "model-card.json").write_text(json.dumps(card))
        return card

    monkeypatch.setattr(job, "train_srnet", fit)
    assert job.main([str(config_path), str(out), sha(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert seen == [
        (resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3)),
        (resource.RLIMIT_CPU, (3660, 3661)),
        (resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2)),
        (resource.RLIMIT_CORE, (0, 0)),
    ]
    for args in ([], [str(config_path), str(out), "0" * 64]):
        assert job.main(args) == 2
        assert json.loads(capsys.readouterr().out) == {"status": "failed"}
    monkeypatch.setattr(resource, "setrlimit", lambda *a: (_ for _ in ()).throw(OSError("secret")))
    assert job.main([str(config_path), str(out), sha(config_path)]) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "unavailable"}
