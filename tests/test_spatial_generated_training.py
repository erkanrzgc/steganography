"""Generated optimizer/resume semantics, never native fitting or a learning gate."""

import copy
import dataclasses
import random

import numpy as np
import pytest

from core import colab_artifacts as a
from core import spatial_colab_state as numeric
from core import spatial_generated_training as t
from core import spatial_training_data as d

torch = pytest.importorskip("torch")


def plan(mode="L", **edits):
    return t.Plan(
        **{
            "spec": d.GeneratedSpec(mode, 2026102801, "train", 1),
            "model_seed": 17,
            "epochs": 2,
            "batch_images": 2,
            "source_sha256": "a" * 64,
            **edits,
        }
    )


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_actual_step_numeric_resume_matches_uninterrupted_and_restores_globals(mode, tmp_path):
    p = plan(mode)
    py, np_state, torch_state, threads = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.get_num_threads(),
    )
    path = tmp_path / "state.numeric"
    with t.session(p) as first:
        result = first.step()
        assert result["optimizer_updates"] == 1 and result["presentations"] == 2
        assert np.isfinite(result["loss"]) and result["generated_only"]
        sha = first.checkpoint(path)
        assert first.output_bytes == path.stat().st_size
        expected_rng = random.random(), np.random.random(), torch.rand(2)  # noqa: S311
        first.step()
        expected_model = {k: v.clone() for k, v in first.model.state_dict().items()}
        expected_optimizer = [
            {k: v.clone() for k, v in row.items()} for row in first.optimizer.state.values()
        ]
        with pytest.raises(StopIteration):
            first.step()
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state) and torch.get_num_threads() == threads
    with t.session(p) as resumed:
        resumed.resume(path, sha)
        assert random.random() == expected_rng[0] and np.random.random() == expected_rng[1]  # noqa: S311
        assert torch.equal(torch.rand(2), expected_rng[2])
        resumed.step()
        assert all(torch.equal(v, expected_model[k]) for k, v in resumed.model.state_dict().items())
        for actual, expected in zip(
            resumed.optimizer.state.values(), expected_optimizer, strict=True
        ):
            assert all(torch.equal(v, expected[k]) for k, v in actual.items())
        saved_model = {k: v.clone() for k, v in resumed.model.state_dict().items()}
        score = resumed.scores(dataclasses.replace(p.spec, split="development"))
        train = resumed.scores(p.spec)
        assert score["scores"].shape == (1, 2) and 0 <= score["auc"] <= 1
        assert train["generated_only"] and resumed.updates == 2 and resumed.model.training
        assert all(torch.equal(v, saved_model[k]) for k, v in resumed.model.state_dict().items())
        with pytest.raises(ValueError, match="fresh"):
            resumed.resume(path, sha)
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), torch_state)


@pytest.fixture
def toy(monkeypatch):
    def network(_):
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(1, 2)
        )

    monkeypatch.setattr(t.models, "network", network)


def test_epoch_tail_pairs_stay_balanced_and_order_is_complete(toy):
    p = plan(spec=d.GeneratedSpec("L", 7, "train", 3), batch_images=4)
    assert np.array_equal(np.sort(t.order(p)), np.arange(6))
    assert np.array_equal(t.order(p), t.order(p))
    with t.session(p) as s:
        assert [s.step()["cursor"] for _ in range(4)] == [2, 3, 5, 6]
        assert s.updates == 4 and s.presentations == 12
    with pytest.raises(ValueError):
        t.order({})


@pytest.mark.parametrize(
    "edit",
    [
        {"spec": None},
        {"spec": d.GeneratedSpec("L", 1, "development", 1)},
        {"epochs": 0},
        {"epochs": 21},
        {"epochs": True},
        {"batch_images": 3},
        {"batch_images": True},
        {"model_seed": -1},
        {"source_sha256": "bad"},
    ],
)
def test_plan_rejects_native_or_invalid_execution(edit):
    with pytest.raises(ValueError):
        plan(**edit)


@pytest.mark.parametrize(
    "options",
    [
        {"max_seconds": 0},
        {"max_seconds": True},
        {"max_seconds": float("nan")},
        {"max_seconds": 1801},
        {"device": "auto"},
        {"device": "cuda:1"},
    ],
)
def test_session_bounds_before_model(options, monkeypatch):
    def forbidden(*_):
        raise AssertionError("model must not be constructed")

    monkeypatch.setattr(t.models, "network", forbidden)
    with pytest.raises(ValueError), t.session(plan(), **options):
        pass
    with pytest.raises(ValueError), t.session(None):
        pass


def test_deadline_and_failed_session_never_retry(toy):
    with t.session(plan()) as s:
        s.started -= 121
        with pytest.raises(ValueError, match="deadline"):
            s.step()
        s.started += 121
        with pytest.raises(ValueError, match="failed"):
            s.step()


@pytest.mark.parametrize("stage", ["loss", "gradient", "model"])
def test_nonfinite_state_or_absent_gradients_fail_closed(toy, monkeypatch, stage):
    with t.session(plan()) as s:
        if stage == "loss":
            monkeypatch.setattr(
                torch.nn.functional,
                "cross_entropy",
                lambda *args: torch.tensor(float("nan"), requires_grad=True),
            )
        elif stage == "gradient":
            s.model.register_forward_hook(lambda _, args, result: result.detach().requires_grad_())
        else:
            step = s.optimizer.step

            def poison():
                step()
                with torch.no_grad():
                    next(s.model.parameters()).fill_(float("nan"))

            monkeypatch.setattr(s.optimizer, "step", poison)
        with pytest.raises(ValueError, match="nonfinite|absent"):
            s.step()
        assert s.failed and s.updates == (1 if stage == "model" else 0)


def test_evaluation_domain_and_mode_restoration(toy):
    with t.session(plan()) as s:
        for spec in (
            None,
            d.GeneratedSpec("RGB", 1, "development", 1),
            d.GeneratedSpec("L", 1, "train", 1),
        ):
            with pytest.raises(ValueError, match="domain|bound"):
                s.scores(spec)
        hook = s.model.register_forward_hook(
            lambda _, args, result: torch.full_like(result, float("nan"))
        )
        with pytest.raises(ValueError, match="nonfinite"):
            s.scores(d.GeneratedSpec("L", 1, "development", 1))
        assert s.model.training and s.updates == 0
        hook.remove()
        s.model.eval()
        s.scores(d.GeneratedSpec("L", 1, "development", 1))
        assert not s.model.training


def test_missing_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(t.srnet_cuda.CUDAUnavailable), t.session(plan(), device="cuda:0"):
        pass


def test_deadline_in_final_evaluation_is_not_reported_ready(toy):
    with t.session(plan()) as s:

        def expire(_, args, result):
            s.started -= 121

        s.model.register_forward_hook(expire)
        with pytest.raises(ValueError, match="deadline"):
            s.scores(d.GeneratedSpec("L", 1, "development", 1))
        assert s.model.training and s.failed


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("spatial-generated-numeric")
    path = root / "state.numeric"
    p = plan(spec=d.GeneratedSpec("L", 9, "train", 3), batch_images=4)
    with t.session(p) as s:
        sha = s.checkpoint(path)
        meta, arrays = numeric.load(
            path,
            checksum=sha,
            environment=t.capacity.environment(),
            source_sha256=s.binding,
            data_sha256=s.data_sha256,
        )
    return p, path, sha, meta, arrays


@pytest.mark.parametrize(
    "fault", ["order", "epoch", "cursor", "presentations", "time", "output", "undersized"]
)
def test_rehashed_checkpoint_cannot_forge_schedule_or_debits(checkpoint, tmp_path, fault):
    p, _, _, metadata, arrays = checkpoint
    meta, changed = copy.deepcopy(metadata), dict(arrays)
    counts = meta["consumption"]
    if fault == "order":
        changed["data/order"] = arrays["data/order"][::-1].copy()
        meta["order_sha256"] = a.digest(changed["data/order"].tobytes())
    elif fault == "epoch":
        counts["epoch"] = 1
    elif fault == "cursor":
        counts["cursor"], counts["presentations"] = 1, 2
    elif fault == "presentations":
        counts["presentations"] = 1
    elif fault == "time":
        counts["elapsed_seconds"] = 121
    elif fault == "output":
        counts["output_bytes"] = t.MAX_OUTPUT_BYTES + 1
    else:
        counts["output_bytes"] = 0
    path = tmp_path / "forged.numeric"
    sha = numeric.save(path, meta, changed)
    with t.session(p) as s:
        before = next(s.model.parameters()).clone()
        with pytest.raises(ValueError, match="schedule/consumption"):
            s.resume(path, sha)
        assert torch.equal(next(s.model.parameters()), before) and s.updates == 0


def test_checkpoint_output_cap_overwrite_symlink_and_bindings(checkpoint, tmp_path, monkeypatch):
    p, original, sha, _, _ = checkpoint
    with t.session(p) as s:
        link = tmp_path / "link"
        link.symlink_to(original)
        with pytest.raises(ValueError, match="symlink"):
            s.resume(link, sha)
        with pytest.raises(ValueError, match="checksum"):
            s.resume(original, "0" * 64)
        path = tmp_path / "numeric"
        first = s.checkpoint(path)
        before = s.output_bytes
        with pytest.raises(FileExistsError):
            s.checkpoint(path)
        assert s.output_bytes == before
        monkeypatch.setattr(t, "MAX_OUTPUT_BYTES", before)
        with pytest.raises(ValueError, match="output limit"):
            s.checkpoint(tmp_path / "over.numeric")
        assert not (tmp_path / "over.numeric").exists() and first
    with (
        t.session(dataclasses.replace(p, model_seed=18)) as s,
        pytest.raises(ValueError, match="incompatible"),
    ):
        s.resume(original, sha)


def test_deadline_in_checkpoint_write_preserves_file_as_unavailable(
    checkpoint, tmp_path, monkeypatch
):
    p, _, _, _, _ = checkpoint
    save = numeric.save
    with t.session(p) as s:

        def expire(*args):
            checksum = save(*args)
            s.started -= 121
            return checksum

        monkeypatch.setattr(numeric, "save", expire)
        path = tmp_path / "late.numeric"
        with pytest.raises(ValueError, match="deadline"):
            s.checkpoint(path)
        assert s.failed and path.is_file() and s.output_bytes == path.stat().st_size
