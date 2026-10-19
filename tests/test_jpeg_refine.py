"""Independent schedule, numeric ancestry, bounded controller and arithmetic checks."""

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import jpeg_refine as c
from core import jpeg_refine_metrics as metrics
from core import jpeg_refine_schedule as s
from core import jpeg_refine_state as state
from core import jpeg_refine_training as training
from core import jpeg_study as old
from core import jpeg_study_data as data
from core import jpeg_study_optimizer as rule
from core import srnet_model
from core.jpeg_timing import write_document
from steganography import research_jpeg_refine as frontend

SHA = "a" * 64
SOURCES = {"test-fixture.py": SHA}
EXECUTION = {"device": "cuda:0", "torch_version": "fixture", "cpu_fallback": False}


def rows(role="fit", count=256):
    return [
        {
            "role": role,
            "source_group": origin,
            "label": label,
            "lineage": hashlib.sha256(f"{role}:{origin}:{i}".encode()).hexdigest(),
        }
        for origin in data.ORIGINS
        for i in range(count)
        for label in ("cover", "stego")
    ]


def config(root):
    return {
        "schema_version": "jpeg-refine-config-v1",
        "root": str(root),
        "manifest_sha256": c.MANIFEST_SHA,
        "audit": str(root / "audit.json"),
        "audit_sha256": c.AUDIT_SHA,
        "source_sha256": SOURCES,
        "recipe": c.recipe(),
        "budget_seconds": c.SECONDS,
        "budget_bytes": c.BYTES,
        "expected_gpu": {"name": "NVIDIA GeForce RTX 5060 Laptop GPU", "uuid_sha256": SHA},
    }


@pytest.mark.parametrize("seed", old.SEEDS)
def test_exact_historical_prefix_balance_multisets_determinism_and_zero_partners(seed):
    samples = rows()
    paired = s.schedule(samples, seed, "paired")
    candidate = s.schedule(samples, seed, "unpaired")
    legacy = old.schedule(samples, seed, 16)
    crops = old.view_schedule(samples, seed, legacy[0])
    for value, reference in zip(paired, (*legacy, crops), strict=True):
        np.testing.assert_array_equal(value[:2048], reference)
        assert not value.flags.writeable
    for a, b in zip(candidate, s.schedule(samples, seed, "unpaired"), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(paired, candidate, strict=True):
        np.testing.assert_array_equal(a[:2048], b[:2048])
    assert s.exposure_digest(paired, 1024) == s.exposure_digest(candidate, 1024)
    assert s.digest(paired) != s.digest(candidate)
    for arm, values in (("paired", paired), ("unpaired", candidate)):
        indices = values[0]
        for index, batch in enumerate(indices):
            assert len(set(batch)) == 16
            for origin in data.ORIGINS:
                for label in ("cover", "stego"):
                    assert (
                        sum(
                            samples[i]["source_group"] == origin and samples[i]["label"] == label
                            for i in batch
                        )
                        == 4
                    )
            assert len(set(batch // 2)) == (16 if arm == "unpaired" and index >= 2048 else 8)
        # Independent multiset oracle, including view and D4 tuples per epoch.
        flat = list(zip(*(v.ravel().tolist() for v in values), strict=True))
        if arm == "paired":
            oracle = [sorted(flat[i : i + 1024]) for i in range(0, len(flat), 1024)]
        else:
            assert [sorted(flat[i : i + 1024]) for i in range(0, len(flat), 1024)] == oracle
    tasks = s.segments()
    assert len(tasks) == 56 and len(tasks) * 512 == 28672
    assert tasks[:4] == [
        ("warmup", old.SEEDS[0], 512),
        ("warmup", old.SEEDS[1], 512),
        ("warmup", old.SEEDS[0], 1024),
        ("warmup", old.SEEDS[1], 1024),
    ]
    assert [t[:2] for t in tasks[8:12]] == [
        ("paired", old.SEEDS[0]),
        ("unpaired", old.SEEDS[0]),
        ("paired", old.SEEDS[1]),
        ("unpaired", old.SEEDS[1]),
    ]


@pytest.mark.parametrize("fault", ["arm", "seed", "small", "block", "duplicate", "pair"])
def test_schedule_rejects_ambiguous_inputs(fault):
    samples, seed, arm = rows(count=8), old.SEEDS[0], "paired"
    if fault == "arm":
        arm = "random"
    elif fault == "seed":
        seed = 4
    elif fault == "small":
        samples = rows(count=4)
    elif fault == "block":
        samples = rows(count=10)
    elif fault == "duplicate":
        samples[2]["lineage"] = samples[3]["lineage"] = samples[0]["lineage"]
    else:
        samples[0]["label"] = "stego"
    with pytest.raises(ValueError):
        s.schedule(samples, seed, arm)


@pytest.fixture
def tiny(monkeypatch):
    torch = pytest.importorskip("torch")

    def factory():
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((2, 2)),
            torch.nn.Conv2d(1, 2, 1),
            torch.nn.BatchNorm2d(2, eps=0.001),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 2, bias=False),
        )

    shapes = {k: tuple(v.shape) for k, v in factory().state_dict().items()}
    monkeypatch.setattr(srnet_model, "SHAPES", shapes)
    monkeypatch.setattr(
        state, "PARAMETERS", {k: v for k, v in shapes.items() if k.endswith(("weight", "bias"))}
    )
    return factory


def numeric(
    arm="warmup", seed=20261012, stop=512, bound=SHA, parent=None, warmup=None, schedule_sha=SHA
):
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in state.specs().items()}
    for name, value in arrays.items():
        if name.endswith(("/step", "num_batches_tracked")):
            value[...] = stop
        if name.endswith("running_var"):
            value[...] = 1
    return {
        "metadata": state.metadata(bound, arm, seed, stop, schedule_sha, parent, warmup),
        "arrays": arrays,
    }


@pytest.mark.parametrize(
    "fault",
    [
        "v1",
        "extra",
        "binding",
        "arm",
        "seed",
        "stop",
        "interval",
        "position",
        "parent",
        "parent_binding",
        "warmup",
        "branch",
    ],
)
def test_metadata_rejection(tiny, fault):
    snapshot = numeric()
    meta = snapshot["metadata"]
    if fault == "v1":
        meta["schema_version"] = "jpeg-study-snapshot-v1"
    elif fault == "extra":
        meta["unexpected"] = "no"
    elif fault == "binding":
        meta["binding"] = "x"
    elif fault == "arm":
        meta["arm"] = "unpaired"
    elif fault == "seed":
        meta["seed"] = True
    elif fault == "stop":
        meta["total_updates"] = meta["next_update"] = 513
    elif fault == "interval":
        meta["new_updates"] = True
    elif fault == "position":
        meta["next_update"] = 1024
    elif fault == "parent":
        meta["parent_sha256"] = SHA
    elif fault == "parent_binding":
        meta.update(
            total_updates=1024, next_update=1024, parent_sha256=SHA, parent_binding="b" * 64
        )
    elif fault == "warmup":
        meta["warmup_sha256"] = SHA
    else:
        meta.update(
            arm="unpaired",
            total_updates=2560,
            next_update=2560,
            parent_sha256=SHA,
            parent_binding=SHA,
            warmup_sha256="b" * 64,
        )
    with pytest.raises(ValueError):
        state.validate(snapshot)


@pytest.mark.parametrize(
    "fault",
    ["container", "keys", "shape", "dtype", "nan", "negative_norm", "bn", "step", "variance"],
)
def test_numeric_rejection(tiny, fault):
    snapshot = numeric()
    if fault == "container":
        snapshot = []
    elif fault == "keys":
        snapshot["arrays"].pop(next(iter(snapshot["arrays"])))
    elif fault in ("shape", "dtype", "nan"):
        key = "model/1.weight"
        if fault == "shape":
            snapshot["arrays"][key] = np.zeros(1, dtype="f4")
        elif fault == "dtype":
            snapshot["arrays"][key] = snapshot["arrays"][key].astype("f8")
        else:
            snapshot["arrays"][key].fill(np.nan)
    else:
        key = {
            "negative_norm": "optimizer/1.weight/exp_inf",
            "bn": "model/2.num_batches_tracked",
            "step": "optimizer/1.weight/step",
            "variance": "model/2.running_var",
        }[fault]
        snapshot["arrays"][key][...] = -1
    with pytest.raises(ValueError):
        state.validate(snapshot)


def test_numeric_save_restore_and_overwrite_symlink_checksum_bounds(tiny, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    snapshot = numeric()
    path = tmp_path / "snapshot.npz"
    sha = state.save(snapshot, path)
    loaded = state.load(path, sha, SHA)
    model = tiny()
    optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
    assert (
        state.restore(
            loaded,
            model,
            optimizer,
            SHA,
            arm="warmup",
            seed=20261012,
            start=512,
            parent_sha256=sha,
            schedule_sha256=SHA,
        )
        == 512
    )
    captured = state.capture(model, optimizer, snapshot["metadata"])
    for key, value in snapshot["arrays"].items():
        np.testing.assert_array_equal(captured["arrays"][key], value)
    assert all(v["step"].device == p.device for p, v in optimizer.state.items())
    with pytest.raises(FileExistsError):
        state.save(snapshot, path)
    for checksum, bound in (("b" * 64, SHA), (sha, "b" * 64)):
        with pytest.raises(ValueError):
            state.load(path, checksum, bound)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        state.load(link, sha, SHA)
    with pytest.raises(FileExistsError):
        state.save(snapshot, link)
    monkeypatch.setattr(state, "MAX_BYTES", 1)
    with pytest.raises(ValueError):
        state.load(path, sha, SHA)
    with pytest.raises(ValueError):
        state.save(snapshot, tmp_path / "too-large")
    assert torch.get_num_threads() > 0


@pytest.mark.parametrize(
    "fault", ["members", "compressed", "metadata", "object", "truncated", "expansion", "symlink"]
)
def test_hostile_npz_rejected_before_numeric_loading(tiny, tmp_path, fault):
    snapshot = numeric()
    raw = io.BytesIO()
    arrays = dict(snapshot["arrays"])
    meta = json.dumps(snapshot["metadata"]).encode()
    arrays["__metadata__"] = np.frombuffer(meta, dtype="u1")
    if fault == "members":
        arrays["unexpected"] = np.zeros(1)
    elif fault == "metadata":
        arrays["__metadata__"] = np.zeros(65537, dtype="u1")
    elif fault == "object":
        arrays["model/1.weight"] = np.asarray([object()])
    elif fault == "expansion":
        arrays["model/1.weight"] = np.zeros(96 * 1024**2 + 1, dtype="u1")
    if fault == "compressed":
        np.savez_compressed(raw, **arrays)
    else:
        np.savez(raw, **arrays)
    encoded = raw.getvalue()
    if fault in ("truncated", "symlink"):
        dest = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive, zipfile.ZipFile(dest, "w") as output:
            for index, member in enumerate(archive.infolist()):
                content = archive.read(member)
                if index == 0 and fault == "truncated":
                    content = content[:-1]
                if index == 0 and fault == "symlink":
                    member.external_attr = 0o120777 << 16
                output.writestr(member, content)
        encoded = dest.getvalue()
    path = tmp_path / "hostile.npz"
    path.write_bytes(encoded)
    with pytest.raises(ValueError):
        state.load(path, hashlib.sha256(encoded).hexdigest(), SHA)


@pytest.mark.parametrize("fault", ["binding", "arm", "seed", "start", "schedule", "warmup"])
def test_wrong_branch_restore_rejected_before_model_mutation(tiny, fault):
    model = tiny()
    before = training.model_digest(model)
    snapshot = numeric(arm="paired", stop=3072, parent=SHA, warmup=SHA)
    kwargs = dict(
        arm="paired",
        seed=20261012,
        start=3072,
        parent_sha256=SHA,
        schedule_sha256=SHA,
        warmup_sha256=SHA,
    )
    bound = SHA
    if fault == "binding":
        bound = "b" * 64
    else:
        kwargs[
            {
                "arm": "arm",
                "seed": "seed",
                "start": "start",
                "schedule": "schedule_sha256",
                "warmup": "warmup_sha256",
            }[fault]
        ] = {
            "arm": "unpaired",
            "seed": 20261013,
            "start": 3584,
            "schedule": "b" * 64,
            "warmup": "b" * 64,
        }[fault]
    with pytest.raises(ValueError):
        state.restore(snapshot, model, rule.optimizer(model, reference=True), bound, **kwargs)
    assert training.model_digest(model) == before


@pytest.mark.parametrize("arm", ["paired", "unpaired"])
def test_independent_historical_update_arithmetic_and_resume(tiny, tmp_path, arm):
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(8)
        model, reference = tiny(), tiny()
        reference.load_state_dict(model.state_dict())
        optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
        oracle = rule.optimizer(reference, reference=True, corrected_l2=True)
        pixels = np.random.default_rng(4).normal(size=(16, 1, 8, 8)).astype("f4")
        indices = np.arange(16, dtype=np.int64)
        for _ in range(512):
            actual = training.update(model, optimizer, pixels, indices, "cpu")
            oracle.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                reference(torch.from_numpy(pixels)), torch.tensor([0, 1] * 8)
            )
            objective = loss + rule.convolution_penalty(reference)
            objective.backward()
            oracle.step()
            assert actual == (float(loss.detach()), float(objective.detach()))
        assert training.model_digest(reference) == training.model_digest(model)
        meta = state.metadata(SHA, "warmup", 20261012, 512, SHA)
        path = tmp_path / "checkpoint.npz"
        checksum = state.save(state.capture(model, optimizer, meta), path)
        restored = tiny()
        resumed_optimizer = rule.optimizer(restored, reference=True, corrected_l2=True)
        state.restore(
            state.load(path, checksum, SHA),
            restored,
            resumed_optimizer,
            SHA,
            arm="warmup",
            seed=20261012,
            start=512,
            parent_sha256=checksum,
            schedule_sha256=SHA,
        )
        for _ in range(4):
            if arm == "unpaired":
                indices[1::2] = np.roll(indices[1::2], 4)
            assert training.update(model, optimizer, pixels, indices, "cpu") == training.update(
                restored, resumed_optimizer, pixels, indices, "cpu"
            )
        assert training.model_digest(model) == training.model_digest(restored)
        for (_, original), (_, resumed) in zip(
            optimizer.state.items(), resumed_optimizer.state.items(), strict=True
        ):
            for key in original:
                assert torch.equal(original[key], resumed[key])
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def matrix(tiny, monkeypatch, tmp_path):
    manifest = {"samples": {"fit": rows(), "development": rows("development", 64)}}
    monkeypatch.setattr(c, "prepared", lambda config: manifest)
    monkeypatch.setattr(c, "verify_data", lambda *args: None)
    prepared_schedules = {
        (arm, seed): s.schedule(manifest["samples"]["fit"], seed, arm)
        for arm in ("warmup", *s.ARMS)
        for seed in old.SEEDS
    }
    settings = config(tmp_path / "data")
    calls = []
    fault = {"mode": None}

    def launch(mode, payload, target, timeout, cancelled):
        calls.append((mode, payload, target))
        assert 0 < timeout <= 1830
        assert not cancelled()
        if fault["mode"] == mode:
            raise fault.get("exception", TimeoutError)()
        if mode == "audit":
            return {
                "status": "completed",
                "source_sha256": SOURCES,
                "execution": EXECUTION,
                "manifest_sha256": settings["manifest_sha256"],
                "idct_rows": {r: len(v) for r, v in manifest["samples"].items()},
                "idct_views": {r: 5 * len(v) for r, v in manifest["samples"].items()},
            }
        if mode == "preflight":
            return {
                "status": "completed",
                "source_sha256": SOURCES,
                "execution": EXECUTION,
                "generated_parity": True,
                "save_resume_parity": True,
                "max_update_seconds": fault.get("timing", 0.001),
                "startup_checkpoint_seconds": 0.001,
            }
        if mode == "train":
            arm, seed, stop = payload["task"]
            target.mkdir()
            parent = payload["parent"]
            values = prepared_schedules[(arm, seed)]
            snapshot = numeric(
                arm=arm,
                seed=seed,
                stop=stop,
                bound=payload["binding"],
                parent=parent["sha256"] if parent else None,
                warmup=parent["warmup_sha256"] if parent and arm != "warmup" else None,
                schedule_sha=s.digest(values),
            )
            checksum = state.save(snapshot, target / "snapshot.npz")
            result = {
                "status": "checkpointed",
                "arm": arm,
                "seed": seed,
                "binding": payload["binding"],
                "metadata": snapshot["metadata"],
                "checkpoint_sha256": checksum,
                "initial_sha256": SHA,
                "exposure_sha256": s.exposure_digest(values, 1024),
                "schedule_sha256": s.digest(values),
                "new_updates": 512,
                "total_updates": stop,
                "seconds": 0.1,
                "mean_cross_entropy": 0.7,
                "mean_objective": 0.8,
                "real_gpu_segment": True,
                "source_sha256": SOURCES,
                "execution": EXECUTION,
                "exit_status": 0,
            }
            if fault.get("receipt"):
                result[fault["receipt"]] = fault.get("value", "bad")
            return result
        checkpoint = payload["checkpoint"]
        snapshot = state.load(checkpoint["path"], checkpoint["sha256"], checkpoint["binding"])
        meta = snapshot["metadata"]
        value = {
            "status": "completed",
            "source_sha256": SOURCES,
            "execution": EXECUTION,
            "real_gpu_evaluation": True,
            "checkpoint_sha256": checkpoint["sha256"],
            "metadata": meta,
            "model_sha256_before": SHA,
            "model_sha256_after": SHA,
        }
        for role in ("development",) if meta["arm"] == "warmup" else ("development", "fit"):
            scores = [0.4, 0.4] if meta["arm"] == "warmup" else [0.2, 0.8]
            value[role] = {
                "scores": scores * (len(manifest["samples"][role]) // 2),
                "singleton_batch_parity": True,
            }
        return value

    def execute(out=None, value=None, **kwargs):
        return c.execute(
            value or settings,
            out or tmp_path / "run",
            sources=SOURCES,
            current_sources=lambda: SOURCES,
            inspect=lambda _: EXECUTION,
            launch=launch,
            **kwargs,
        )

    return SimpleNamespace(
        config=settings,
        execute=execute,
        launch=launch,
        fault=fault,
        calls=calls,
        manifest=manifest,
        root=tmp_path,
    )


def test_complete_controller_checkpoints_metrics_selection_and_public_privacy(matrix):
    report = matrix.execute()
    assert report["status"] == "completed"
    assert report["verified_updates"] == report["new_verified_updates"] == 28672
    assert report["learning_gate"] and report["passing_arms"] == ["paired", "unpaired"]
    assert report["selected_arm"] == "paired" and not report["unpaired_superiority"]
    assert len(report["trials"]) == 6 and len(report["receipts"]) == 56
    assert len([v for v in matrix.calls if v[0] == "evaluate"]) == 6
    for trial in report["trials"]:
        for cell in trial["cells"]:
            assert cell["metrics"]["recommended_threshold"] is None
            assert set(cell["metric_ci95"]) == {
                "balanced_accuracy",
                "recall",
                "false_positive_rate",
                "ece",
                "tp",
                "tn",
                "fp",
                "fn",
            }
    assert str(matrix.root) not in json.dumps(report)
    assert not report["deployed"] and report["accuracy_qualification"] == "unavailable"
    assert report["cumulative_output_bytes"] == c.storage(matrix.root / "run")
    before = sorted(
        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in (matrix.root / "run").rglob("*.npz")
    )
    c.verify(
        c.configuration(matrix.config, SOURCES),
        matrix.root / "run",
        EXECUTION,
        report["receipts"],
        report["evaluations"],
    )
    bad = copy.deepcopy(report["receipts"])
    bad[0]["index"] = 1
    with pytest.raises(ValueError, match="order"):
        c.verify(
            c.configuration(matrix.config, SOURCES),
            matrix.root / "run",
            EXECUTION,
            bad,
            report["evaluations"],
        )
    bad[0]["index"] = 0
    bad[0]["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="mismatch"):
        c.verify(
            c.configuration(matrix.config, SOURCES),
            matrix.root / "run",
            EXECUTION,
            bad,
            report["evaluations"],
        )
    bad_evaluation = copy.deepcopy(report["evaluations"])
    bad_evaluation[0]["model_sha256_before"] = "b" * 64
    with pytest.raises(ValueError, match="provenance"):
        c.verify(
            c.configuration(matrix.config, SOURCES),
            matrix.root / "run",
            EXECUTION,
            report["receipts"],
            bad_evaluation,
        )
    assert before == sorted(
        (p.name, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in (matrix.root / "run").rglob("*.npz")
    )


@pytest.mark.parametrize("fault", ["source", "recipe", "budget", "bytes", "gpu", "path", "extra"])
def test_config_frozen_rejections(tmp_path, fault):
    value = config(tmp_path)
    if fault == "source":
        value["source_sha256"] = {}
    elif fault == "recipe":
        value["recipe"]["learning_rate"] = 0.0001
    elif fault == "budget":
        value["budget_seconds"] = True
    elif fault == "bytes":
        value["budget_bytes"] += 1
    elif fault == "gpu":
        value["expected_gpu"]["name"] = "rented"
    elif fault == "path":
        value["root"] = ""
    else:
        value["download"] = True
    with pytest.raises(ValueError):
        c.configuration(value, SOURCES)


@pytest.mark.parametrize(
    "fault",
    [
        "source_sha256",
        "execution",
        "real_gpu_segment",
        "new_updates",
        "mean_objective",
        "initial_sha256",
        "schedule_sha256",
    ],
)
def test_bad_worker_receipt_stops_entire_matrix(matrix, fault):
    matrix.fault["receipt"] = fault
    matrix.fault["value"] = float("nan") if fault == "mean_objective" else None
    report = matrix.execute()
    assert report["status"] == "unavailable" and not report["learning_gate"]
    assert report["verified_updates"] == 0
    assert len([v for v in matrix.calls if v[0] == "train"]) == 1


def test_whole_matrix_estimate_denies_training_and_targets_unchanged(matrix):
    matrix.fault["timing"] = 1.0
    report = matrix.execute()
    assert report["reason"] == "whole_matrix_exceeds_remaining_budget"
    assert not any(v[0] == "train" for v in matrix.calls)
    assert report["recipe"]["total_updates"] == 28672


@pytest.mark.parametrize(
    "exception,status",
    [
        (TimeoutError, "unavailable"),
        (InterruptedError, "cancelled"),
        (KeyboardInterrupt, "cancelled"),
    ],
)
def test_no_retry_on_interruption(matrix, exception, status):
    matrix.fault.update(mode="train", exception=exception)
    report = matrix.execute()
    assert report["status"] == status and report["verified_updates"] == 0
    assert len([v for v in matrix.calls if v[0] == "train"]) == 1


def test_resume_shared_warmup_without_retraining_retains_spent_budget(matrix):
    original = matrix.launch

    def stop(mode, payload, target, timeout, cancelled):
        if mode == "train" and payload["task"][0] != "warmup":
            raise InterruptedError
        return original(mode, payload, target, timeout, cancelled)

    report = c.execute(
        matrix.config,
        matrix.root / "parent",
        sources=SOURCES,
        current_sources=lambda: SOURCES,
        inspect=lambda _: EXECUTION,
        launch=stop,
    )
    assert report["status"] == "cancelled" and report["verified_updates"] == 4096
    parent_path = matrix.root / "parent" / "report.json"
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    value = copy.deepcopy(matrix.config)
    value["resume"] = {
        "parent": str(parent_path.parent),
        "report_sha256": parent_sha,
        "lost_seconds": 3.0,
        "remaining_seconds": 21600 - report["seconds"] - 3,
    }
    matrix.calls.clear()
    resumed = matrix.execute(matrix.root / "resumed", value)
    assert resumed["status"] == "completed"
    assert resumed["new_verified_updates"] == 24576 and resumed["verified_updates"] == 28672
    assert resumed["prior_seconds"] == report["seconds"] + 3
    assert not any(v[0] == "train" and v[1]["task"][0] == "warmup" for v in matrix.calls)
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_sha


def test_incomplete_matrix_never_qualifies(matrix):
    with pytest.raises(ValueError, match="incomplete"):
        c.verify(c.configuration(matrix.config, SOURCES), matrix.root, EXECUTION, [], [])
    with pytest.raises(ValueError, match="complete"):
        metrics.compare(matrix.manifest, [])


def test_storage_bounds_symlink_and_no_overwrite(matrix, monkeypatch):
    target = matrix.root / "artifact"
    target.mkdir()
    (target / "data").write_bytes(b"123")
    assert c.storage(target) == 3
    monkeypatch.setattr(c, "BYTES", 2)
    with pytest.raises(ValueError):
        c.storage(target)
    monkeypatch.setattr(c, "BYTES", 10 * 1024**3)
    (target / "link").symlink_to(target / "data")
    with pytest.raises(ValueError):
        c.storage(target)
    with pytest.raises(FileExistsError):
        matrix.execute(target)
    with pytest.raises(ValueError):
        matrix.execute(Path(matrix.config["root"]) / "child")


def test_cpu_gpu_absence_record_is_unavailable_without_data_or_worker(tmp_path, monkeypatch):
    from core import srnet_cuda

    monkeypatch.setattr(frontend, "sources", lambda: SOURCES)
    monkeypatch.setattr(
        frontend,
        "inspect",
        lambda gpu: (_ for _ in ()).throw(
            srnet_cuda.CUDAUnavailable("cuda_not_visible_or_cpu_torch")
        ),
    )
    path = tmp_path / "config.json"
    write_document(path, config(tmp_path / "data"))
    report = frontend.run(path, tmp_path / "out")
    assert report["status"] == "unavailable"
    assert report["reason"] == "cuda_not_visible_or_cpu_torch"
    assert report["verified_updates"] == 0 and not report["cpu_fallback"]
