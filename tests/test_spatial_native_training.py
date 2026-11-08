"""Independent grouped partitions and generated-only native-step/resume QA."""

import copy
import dataclasses
import itertools
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import colab_artifacts as a
from core import spatial_colab_state as numeric
from core import spatial_native_data as d
from core import spatial_native_training as t

torch = pytest.importorskip("torch")


def row(i, mode="RGB", group=None, author=None):
    return d.Source(
        f"fixture-{i}",
        mode,
        group or f"group-{i}",
        author or f"Author {i}",
        f"https://commons.wikimedia.org/wiki/File:Fixture_{i}.png",
        a.digest(f"original-{i}".encode()),
        a.digest(f"crop-{i}".encode()),
        "CC0-1.0",
        d.native.LICENSES["CC0-1.0"],
        "0" * 64,
    )


def catalog(rows):
    return d.Cache(Path("metadata-only"), tuple(rows), "a" * 64)


def prepare(root, mode="RGB", train=3):
    refs = []
    for i in range(train + 2):
        pixels = np.random.default_rng(200 + i).integers(
            0, 256, (256, 256, 3 if mode == "RGB" else 1), dtype="u1"
        )
        path = root / f"fixture-{i}-cover.png"
        Image.fromarray(pixels[:, :, 0] if mode == "L" else pixels).save(path)
        source = dataclasses.replace(row(i, mode), crop_sha256=a.digest(pixels.tobytes()))
        refs.append(d.Input(source, path, a.digest(path.read_bytes())))
    out = root / "cache"
    d.prepare(tuple(refs), out)
    cache = d.load(out, a.digest((out / "cache.json").read_bytes()))
    split = t.Split(
        cache.manifest_sha256,
        mode,
        13,
        tuple(r.source.id for r in refs[:train]),
        (refs[train].source.id,),
        (refs[train + 1].source.id,),
        (),
    )
    return cache, t.Plan(split, 17, 23, 2, 4, "b" * 64)


def test_exact_solver_matches_exhaustive_group_search_and_never_reads_pixels():
    weights = [3, 2, 1, 1]
    rows = []
    for group, weight in enumerate(weights):
        start = len(rows)
        rows.extend(row(start + i, group=f"group-{group}") for i in range(weight))
    cache = catalog(rows)
    possible = set()
    for assignment in itertools.product(range(3), repeat=len(weights)):
        counts = tuple(
            sum(weight for weight, role in zip(weights, assignment, strict=True) if role == r)
            for r in range(3)
        )
        if all(counts):
            possible.add(counts)
    for counts in [(3, 2, 2), (1, 1, 5), (2, 3, 2), (4, 2, 1)]:
        assert counts in possible
        split = t.partition(cache, "RGB", counts, 9)
        split.validate(cache)
        assert (
            tuple(len(getattr(split, r)) for r in ("train", "development", "future_reserved"))
            == counts
        )
        assert split == t.partition(cache, "RGB", counts, 9)
        assert (
            t.split_document(
                dataclasses.asdict(split)
                | {
                    "train": list(split.train),
                    "development": list(split.development),
                    "future_reserved": list(split.future_reserved),
                    "unassigned": [],
                }
            )
            == split
        )
    infeasible = catalog([row(i, group=f"group-{i // 2}") for i in range(6)])
    with pytest.raises(ValueError, match="infeasible"):
        t.partition(infeasible, "RGB", (3, 2, 1), 9)
    with pytest.raises(ValueError, match="inventory"):
        t.partition(cache, "RGB", (1, 1, 1), 9)
    assert all(r.role == "unassigned" for r in rows)


def test_cross_mode_ancestry_stays_unassigned_and_leakage_rejected():
    rows = [
        row(0, author="Shared Author"),
        row(1, "L", author=" shared author "),
        row(2),
        row(3),
        row(4),
    ]
    cache = catalog(rows)
    split = t.partition(cache, "RGB", (1, 1, 1), 0)
    assert set(split.unassigned) == {"fixture-0", "fixture-1"}
    with pytest.raises(ValueError, match="overlap"):
        dataclasses.replace(split, development=split.train)
    with pytest.raises(ValueError, match="binding"):
        split.validate(dataclasses.replace(cache, manifest_sha256="c" * 64))
    with pytest.raises(ValueError, match="complete"):
        dataclasses.replace(split, unassigned=()).validate(cache)
    with pytest.raises(ValueError, match="modes"):
        dataclasses.replace(
            split, train=("fixture-1",), unassigned=("fixture-0", *split.train)
        ).validate(cache)
    connected = catalog([row(0, group="shared"), row(1, group="shared"), row(2)])
    with pytest.raises(ValueError, match="leakage"):
        t.Split("a" * 64, "RGB", 0, ("fixture-0",), ("fixture-1",), ("fixture-2",), ()).validate(
            connected
        )


@pytest.mark.parametrize("counts", [None, (), (1, 1), (0, 1, 2), (True, 1, 1), (128, 1, 1)])
def test_partition_limits(counts):
    with pytest.raises(ValueError):
        t.partition(catalog([row(0)]), "RGB", counts, 0)


def test_split_plan_and_document_limits(tmp_path):
    cache, p = prepare(tmp_path)
    for edits in ({"mode": "RGBA"}, {"train": []}, {"train": ()}, {"train": ("../host",)}):
        with pytest.raises(ValueError):
            dataclasses.replace(p.split, **edits)
    for value in (None, {}, dataclasses.asdict(p.split)):
        with pytest.raises(ValueError):
            t.split_document(value)
    for edits in (
        {"split": None},
        {"epochs": 0},
        {"epochs": 21},
        {"epochs": True},
        {"batch_images": 3},
        {"model_seed": -1},
        {"payload_seed": True},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(p, **edits)
    with pytest.raises(ValueError):
        t.order(None)
    assert np.array_equal(np.sort(t.order(p)), np.arange(6))
    for bad, plan in ((None, p), (cache, None)):
        with pytest.raises(ValueError), t.session(bad, plan):
            pass
    for seconds in (0, 1801, float("nan"), True):
        with pytest.raises(ValueError), t.session(cache, p, max_seconds=seconds):
            pass


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_actual_optimizer_resume_exact_rng_and_reserved_never_opened(tmp_path, monkeypatch, mode):
    torch = pytest.importorskip("torch")
    cache, p = prepare(tmp_path, mode, train=1)
    p = dataclasses.replace(p, batch_images=2)
    read = d.a.read
    opened = []

    def watched(path, *args):
        if path.suffix == ".u8":
            opened.append(path.stem)
            assert path.stem not in p.split.future_reserved
        return read(path, *args)

    monkeypatch.setattr(d.a, "read", watched)
    py, np_state, torch_state, threads = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.get_num_threads(),
    )
    path = tmp_path / "native.numeric"
    with t.session(cache, p) as s:
        assert np.isfinite(s.step()["loss"]) and s.updates == 1
        sha = s.checkpoint(path)
        assert s.output_bytes == path.stat().st_size
        expected_rng = random.random(), np.random.random(), torch.rand(2)  # noqa: S311
        s.step()
        expected = {k: v.clone() for k, v in s.model.state_dict().items()}
        expected_optimizer = [
            {k: v.clone() for k, v in slot.items()} for slot in s.optimizer.state.values()
        ]
        with pytest.raises(StopIteration):
            s.step()
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), torch_state)
    assert (
        np.array_equal(np.random.get_state()[1], np_state[1]) and torch.get_num_threads() == threads
    )
    with t.session(cache, p) as restored:
        restored.resume(path, sha)
        assert random.random() == expected_rng[0] and np.random.random() == expected_rng[1]  # noqa: S311
        assert torch.equal(torch.rand(2), expected_rng[2])
        restored.step()
        assert all(torch.equal(v, expected[k]) for k, v in restored.model.state_dict().items())
        for actual, wanted in zip(
            restored.optimizer.state.values(), expected_optimizer, strict=True
        ):
            assert all(torch.equal(v, wanted[k]) for k, v in actual.items())
        for role in ("train", "development"):
            result = restored.scores(role)
            assert result["scores"].shape == (1, 2) and 0 <= result["auc"] <= 1
            assert "scores" not in restored.scores(role, record_scores=False)
        for role in ("future_reserved", "unassigned", "anything"):
            with pytest.raises(ValueError, match="scope"):
                restored.scores(role)
        assert restored.model.training
        with pytest.raises(ValueError, match="fresh"):
            restored.resume(path, sha)
    assert set(opened) == set(p.split.train + p.split.development)
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), torch_state)


@pytest.fixture
def toy(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        t.models,
        "network",
        lambda _: torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)
        ),
    )


def test_complete_epoch_tails_and_fail_closed_deadline(tmp_path, toy):
    cache, p = prepare(tmp_path)
    with t.session(cache, p) as s:
        assert [s.step()["cursor"] for _ in range(4)] == [2, 3, 5, 6]
        assert s.presentations == 12
        s.started -= 121
        with pytest.raises(ValueError, match="deadline"):
            s.step()
        assert s.failed


@pytest.mark.parametrize("stage", ["loss", "gradient", "model"])
def test_nonfinite_paths_are_unavailable(tmp_path, toy, monkeypatch, stage):

    cache, p = prepare(tmp_path)
    with t.session(cache, p) as s:
        if stage == "loss":
            monkeypatch.setattr(
                torch.nn.functional,
                "cross_entropy",
                lambda *_: torch.tensor(float("nan"), requires_grad=True),
            )
        elif stage == "gradient":
            s.model.register_forward_hook(lambda _, args, result: result.detach().requires_grad_())
        else:
            original = s.optimizer.step

            def poison():
                original()
                with torch.no_grad():
                    next(s.model.parameters()).fill_(float("nan"))

            monkeypatch.setattr(s.optimizer, "step", poison)
        with pytest.raises(ValueError, match="nonfinite|absent"):
            s.step()
        assert s.failed and s.updates == (1 if stage == "model" else 0)


def test_eval_mode_finite_and_final_deadline(tmp_path, toy):

    cache, p = prepare(tmp_path)
    with t.session(cache, p) as s:
        hook = s.model.register_forward_hook(
            lambda _, args, result: torch.full_like(result, float("nan"))
        )
        with pytest.raises(ValueError, match="nonfinite"):
            s.scores("development")
        assert s.model.training
        hook.remove()
        s.model.eval()
        s.scores("train")
        assert not s.model.training

        def expire(_, args, result):
            s.started -= 121

        s.model.register_forward_hook(expire)
        with pytest.raises(ValueError, match="deadline"):
            s.scores("train")


def test_numeric_binding_corruption_and_output_limits(tmp_path, monkeypatch):
    cache, p = prepare(tmp_path, train=1)
    p = dataclasses.replace(p, batch_images=2)
    path = tmp_path / "state.numeric"
    with t.session(cache, p) as s:
        checksum = s.checkpoint(path)
        meta, arrays = numeric.load(
            path,
            checksum=checksum,
            environment=t.capacity.environment(),
            source_sha256=s.binding,
            data_sha256=s.data_sha256,
        )
        with pytest.raises(FileExistsError):
            s.checkpoint(path)
        monkeypatch.setattr(t, "MAX_OUTPUT_BYTES", s.output_bytes)
        with pytest.raises(ValueError, match="output limit"):
            s.checkpoint(tmp_path / "large.numeric")
    monkeypatch.undo()
    for field, value in [
        ("cursor", 1),
        ("epoch", 1),
        ("presentations", 1),
        ("elapsed_seconds", 121),
        ("output_bytes", 0),
    ]:
        changed = copy.deepcopy(meta)
        changed["consumption"][field] = value
        target = tmp_path / (field + ".numeric")
        digest = numeric.save(target, changed, arrays)
        with t.session(cache, p) as s:
            before = next(s.model.parameters()).clone()
            with pytest.raises(ValueError, match="schedule/consumption"):
                s.resume(target, digest)

            assert torch.equal(next(s.model.parameters()), before)
    with (
        t.session(cache, dataclasses.replace(p, payload_seed=24)) as s,
        pytest.raises(ValueError, match="incompatible"),
    ):
        s.resume(path, checksum)


def test_forged_cache_rows_and_missing_cuda_fail_closed(tmp_path, monkeypatch):

    cache, p = prepare(tmp_path)
    forged = dataclasses.replace(cache, rows=cache.rows[::-1])
    with pytest.raises(ValueError, match="source mismatch"), t.session(forged, p):
        pass
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(t.srnet_cuda.CUDAUnavailable), t.session(cache, p, device="cuda:0"):
        pass
