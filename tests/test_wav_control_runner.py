"""Generated-only producer and supervisor checks; no CUDA fitting in tests."""

import hashlib
import json
import runpy
import signal
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from core import wav_control_gate as gate
from core import wav_control_runner as runner
from core import wav_pilot, wav_pilot_v3
from steganography import research_wav_control_run as cli


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("generated-control")
    records, manifest = runner.build_corpus(
        root, {"train": [640] * 400, "development": [640] * 200}, time.monotonic() + 300
    )
    return root, records, manifest


def test_corpus_has_native_headers_unique_pcm_exact_groups_and_all_oracles(corpus):
    root, records, manifest = corpus
    assert manifest["evidence"] == {
        "train_files": 400,
        "development_files": 200,
        "train_groups": 20,
        "development_groups": 10,
        "files_per_group": 20,
        "group_overlap": 0,
        "decoded_pcm_overlap": 0,
        "payload_oracles_passed": 4800,
        "length_schedule": "exact_metadata_order",
    }
    assert len(records) == len({r["decoded_pcm_sha256"] for r in records}) == 600
    assert len({r["sha256"] for r in records}) == 600
    assert len(wav_pilot.rows_for_role(records, "train")) == 3600
    assert len(wav_pilot.rows_for_role(records, "development")) == 1800
    row = records[0]
    raw = wav_pilot.materialize(root, row, None)
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    assert len(raw) == 1324 and manifest["generated_bytes"] == 600 * 1324
    assert np.array_equal(wav_pilot_v3.inputs(raw), wav_pilot_v3.inputs(raw))


def test_generated_signal_is_reproducible_without_global_rng_effects():
    np.random.seed(17)
    before = np.random.get_state()
    a = runner.speech_like(np.random.default_rng(12), 400)
    b = runner.speech_like(np.random.default_rng(12), 400)
    assert a.dtype == np.dtype("<i2") and a.shape == (400,) and np.array_equal(a, b)
    assert np.array_equal(before[1], np.random.get_state()[1])
    assert not np.array_equal(a, runner.speech_like(np.random.default_rng(13), 400))


@pytest.mark.parametrize(
    "schedule",
    (
        {},
        {"train": [160] * 399, "development": [160] * 200},
        {"train": [True] * 400, "development": [160] * 200},
    ),
)
def test_invalid_generation_schedule_is_closed(tmp_path, schedule):
    with pytest.raises(ValueError, match="schedule"):
        runner.build_corpus(tmp_path, schedule, time.monotonic() + 300)


def test_generation_limits_duplicates_and_deadline(tmp_path, monkeypatch):
    schedule = {"train": [160] * 400, "development": [160] * 200}
    with pytest.raises(TimeoutError):
        runner.build_corpus(tmp_path, schedule, 0)
    monkeypatch.setattr(runner, "MAX_SCRATCH", 1)
    with pytest.raises(ValueError, match="scratch"):
        runner.build_corpus(tmp_path, schedule, time.monotonic() + 300)
    monkeypatch.setattr(runner, "MAX_SCRATCH", 512 * 1024**2)
    monkeypatch.setattr(runner, "speech_like", lambda *_: np.zeros(160, dtype="<i2"))
    with pytest.raises(ValueError, match="duplicate"):
        runner.build_corpus(tmp_path, schedule, time.monotonic() + 300)


def test_auc_matches_independent_scalar_pair_count_and_keeps_ties():
    negative, positive = np.array([0.1, 0.2, 0.2, 0.8]), np.array([0.1, 0.2, 0.9])
    expected = sum(float(p > n) + 0.5 * float(p == n) for n in negative for p in positive) / 12
    assert runner.auc(negative, positive) == pytest.approx(expected)
    assert runner.auc(np.zeros(2), np.zeros(3)) == 0.5
    assert runner.auc(np.zeros(400), np.r_[np.ones(160), np.zeros(240)]) == 0.70
    assert runner.auc(np.zeros(400), np.r_[np.ones(80), np.zeros(320)]) == 0.60


@pytest.mark.parametrize("scores", ([], [float("nan")], [1.1], [[0.1]], [0.1] * 3201))
def test_auc_rejects_invalid_or_unbounded_scores(scores):
    with pytest.raises(ValueError, match="AUC"):
        runner.auc(np.asarray(scores), np.array([0.5]))


def test_all_recipe_metrics_keep_unrounded_auc(corpus):
    _root, records, _manifest = corpus
    rows = wav_pilot.rows_for_role(records, "development")
    scores = np.asarray([0.1 if r["recipe"] is None else 0.9 for r in rows])
    overall, cells = runner.metrics(rows, scores)
    assert overall == 1 and len(cells) == 8 and all(c["roc_auc"] == 1 for c in cells)


def test_json_progress_is_bounded_exclusive_and_never_follows_link(tmp_path, monkeypatch):
    runner.progress(tmp_path, {"confirmed_updates_lower_bound": 1})
    runner.progress(tmp_path, {"confirmed_updates_lower_bound": 2})
    assert (
        json.loads((tmp_path / "progress.json").read_text())["confirmed_updates_lower_bound"] == 2
    )
    target = tmp_path / "target"
    target.write_text("unchanged")
    (tmp_path / "progress.json").unlink()
    (tmp_path / "progress.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner.progress(tmp_path, {"new": 3})
    assert target.read_text() == "unchanged"
    monkeypatch.setattr(runner, "MAX_JSON", 2)
    with pytest.raises(ValueError, match="size"):
        runner.encoded({"large": 10})


def test_metadata_schedule_never_reads_source_paths(tmp_path):
    path = tmp_path / "benchmarks/wav-audiomnist-preparation-20261017.json"
    path.parent.mkdir()
    rows = [{"role": "train", "frames": 160, "relative_source_path": "/do-not-open"}] * 400
    rows += [{"role": "development", "frames": 170}] * 200
    path.write_text(json.dumps({"files": rows}))
    assert runner.lengths(tmp_path) == {"train": [160] * 400, "development": [170] * 200}
    for changed in (
        {},
        {"files": rows[:-1]},
        {"files": [None] * 600},
        {"files": [{"role": "reserved", "frames": 160}] * 600},
        {"files": [{"role": "train", "frames": 160}] * 600},
    ):
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            runner.lengths(tmp_path)


def test_execution_protocol_binds_parent_and_every_runner_source(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "read_protocol", lambda *_: ({}, "a" * 64))
    hashes = {}
    for name in runner.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        hashes[name] = hashlib.sha256(b"").hexdigest()
    protocol = {
        "schema_version": "wav-control-runner-protocol-v1",
        "protocol_id": runner.PROTOCOL_ID,
        "parent_protocol_sha256": "a" * 64,
        "generator_seed": runner.CORPUS_SEED,
        "reserve_seconds": 10,
        "real_fit_authorized": False,
        "source_closure_sha256": hashes,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    assert runner.load_protocol(tmp_path, path)[2:] == ({}, "a" * 64)
    (tmp_path / runner.SOURCES[0]).write_text("changed")
    with pytest.raises(ValueError, match="source binding"):
        runner.load_protocol(tmp_path, path)
    protocol["real_fit_authorized"] = True
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol binding"):
        runner.load_protocol(tmp_path, path)


def test_kernel_memory_and_swap_limits_are_mandatory(tmp_path):
    proc = tmp_path / "proc"
    folder = tmp_path / "bounded"
    folder.mkdir()
    proc.write_text("0::/bounded\n")
    (folder / "memory.max").write_text(str(8 * 1024**3))
    (folder / "memory.swap.max").write_text("0")
    assert runner.cgroup(tmp_path, proc) == folder
    for value in ("max", str(8 * 1024**3 + 1), "0"):
        (folder / "memory.max").write_text(value)
        with pytest.raises(ValueError, match="bounded memory"):
            runner.cgroup(tmp_path, proc)
    proc.write_text("0::/../escape")
    with pytest.raises(ValueError, match="unified"):
        runner.cgroup(tmp_path, proc)


def test_score_is_finite_deadline_bounded_and_restores_model_mode(corpus):
    torch = pytest.importorskip("torch")
    root, records, _manifest = corpus
    rows = wav_pilot.rows_for_role(records, "development")[:2]
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2)
    ).train()
    values = runner.score(torch, root, rows, model, time.monotonic() + 10, device="cpu")
    assert values.shape == (2,) and np.isfinite(values).all() and model.training
    with pytest.raises(TimeoutError):
        runner.score(torch, root, rows, model, 0, device="cpu")
    with torch.no_grad():
        model[-1].weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        runner.score(torch, root, rows, model, time.monotonic() + 10, device="cpu")
    assert model.training


def test_cuda_absence_and_wrong_runtime_are_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "__version__", "2.14.0+cpu")
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        runner.cuda_runtime()


def test_cuda_runtime_enforces_fp32_determinism_and_allocator(monkeypatch):
    calls = []
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _: "NVIDIA GeForce RTX 5060 Laptop GPU",
        get_device_properties=lambda _: types.SimpleNamespace(total_memory=8 * 1024**3),
        set_per_process_memory_fraction=lambda *a: calls.append(a),
        reset_peak_memory_stats=lambda _: None,
    )
    backend = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace()
    )
    fake = types.SimpleNamespace(
        __version__="2.14.0+cu130",
        cuda=cuda,
        backends=backend,
        set_num_threads=lambda n: calls.append(n),
        use_deterministic_algorithms=lambda v: calls.append(v),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert runner.cuda_runtime() is fake
    assert (0.5, 0) in calls and backend.cudnn.deterministic
    assert not backend.cuda.matmul.allow_tf32 and not backend.cudnn.allow_tf32


def test_stop_process_escalates_and_reaps(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.os, "killpg", lambda *a: calls.append(a))
    process = types.SimpleNamespace(pid=123)
    waits = iter([subprocess.TimeoutExpired("worker", 1), None])

    def wait(**_kwargs):
        result = next(waits)
        if isinstance(result, Exception):
            raise result

    process.wait = wait
    runner.stop_process(process)
    assert calls == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    process.wait = lambda **_kwargs: None
    runner.stop_process(process)
    monkeypatch.setattr(runner.os, "killpg", lambda *_: (_ for _ in ()).throw(ProcessLookupError()))
    runner.stop_process(process)


@pytest.mark.parametrize("fail_second", (False, True))
def test_worker_uses_shared_two_seed_trainer_and_binds_generated_manifest(
    corpus, tmp_path, monkeypatch, fail_second
):
    _root, records, manifest = corpus
    parent = {"source_closure_sha256": {"source": "a" * 64}}
    protocol = {
        "source_closure_sha256": {
            runner.CORPUS: hashlib.sha256(runner.encoded(manifest)).hexdigest()
        }
    }
    monkeypatch.setattr(runner, "load_protocol", lambda *_: (protocol, "b" * 64, parent, "c" * 64))
    monkeypatch.setattr(runner, "cgroup", lambda: None)
    cuda = types.SimpleNamespace(
        get_device_name=lambda _: "NVIDIA GeForce RTX 5060 Laptop GPU",
        max_memory_allocated=lambda _: 1024,
    )
    monkeypatch.setattr(runner, "cuda_runtime", lambda: types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runner, "lengths", lambda *_: {})
    monkeypatch.setattr(runner, "build_corpus", lambda *_: (records, manifest))
    calls = []

    def fit(_data, rows, seed, **kwargs):
        assert len(rows) == 3600 and kwargs["recipe"] is wav_pilot_v3.V3_RECIPE
        calls.append(seed)
        if fail_second and seed == 20261017:
            raise RuntimeError("second seed interrupted")
        for _ in range(1140):
            kwargs["on_update"]()
        return object(), {
            "epochs": 20,
            "optimizer_updates": 1140,
            "row_presentations": 72000,
            "virtual_batch": 64,
            "partial_batch_rows": 16,
            "seconds": 12.5,
            "mean_batch_loss": 0.69,
        }

    monkeypatch.setattr(runner.wav_pilot_training, "fit_seed", fit)
    monkeypatch.setattr(
        runner,
        "score",
        lambda _t, _d, rows, *_: np.asarray(
            [0.1 if row["recipe"] is None else 0.9 for row in rows]
        ),
    )
    if fail_second:
        with pytest.raises(RuntimeError, match="second seed"):
            runner.worker(tmp_path, tmp_path / "protocol", tmp_path, time.monotonic() + 300)
        first = json.loads((tmp_path / "seed-20261016.json").read_text())
        assert first["optimizer_updates"] == 1140 and len(first["development_cells"]) == 8
        assert len(first["train_cells"]) == 8 and first["fit_seconds"] == 12.5
        assert (tmp_path / "fit-20261016.json").exists()
        assert not (tmp_path / "worker.json").exists()
        assert not (tmp_path / "fit-20261017.json").exists()
        assert (
            json.loads((tmp_path / "progress.json").read_text())["confirmed_updates_lower_bound"]
            == 1140
        )
        return
    runner.worker(tmp_path, tmp_path / "protocol", tmp_path, time.monotonic() + 300)
    result = json.loads((tmp_path / "worker.json").read_text())
    assert calls == [20261016, 20261017] and result["seeds"][1]["train_auc_all"] == 1
    assert result["real_optimizer_updates"] == 0 and not result["real_media_opened"]
    decision = gate.evaluate(
        parent,
        "c" * 64,
        {
            **result,
            "wall_seconds": 1,
            "host_peak_bytes": 10000,
            "swap_peak_bytes": 0,
        },
    )
    assert decision["status"] == "passed_generated_only"
    assert (
        json.loads((tmp_path / "progress.json").read_text())["confirmed_updates_lower_bound"]
        == 2280
    )
    protocol["source_closure_sha256"][runner.CORPUS] = "d" * 64
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    with pytest.raises(ValueError, match="corpus binding"):
        runner.worker(tmp_path, tmp_path / "protocol", wrong, time.monotonic() + 300)
    with pytest.raises(ValueError, match="deadline"):
        runner.worker(tmp_path, tmp_path / "protocol", wrong, float("nan"))


def test_supervisor_retains_partial_failure_without_retry(tmp_path, monkeypatch):
    parent = {"source_closure_sha256": {}}
    monkeypatch.setattr(runner, "load_protocol", lambda *_: ({}, "b" * 64, parent, "c" * 64))
    folder = tmp_path / "kernel"
    folder.mkdir()
    (folder / "memory.peak").write_text("10000")
    (folder / "memory.swap.current").write_text("0")
    monkeypatch.setattr(runner, "cgroup", lambda: folder)
    calls = []

    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        runner.progress(tmp_path / "out", {"confirmed_updates_lower_bound": 17})
        return types.SimpleNamespace(poll=lambda: 2, returncode=2)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    result = runner.execute(tmp_path, tmp_path / "protocol", tmp_path / "out")
    assert result["status"] == "unavailable" and len(calls) == 1
    failure = json.loads((tmp_path / "out/result.json").read_text())
    assert failure["partial_progress"]["confirmed_updates_lower_bound"] == 17
    assert calls[0][1]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls[0][1]["start_new_session"] is True
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, tmp_path / "protocol", tmp_path / "out")


@pytest.mark.parametrize("scenario", ("complete", "timeout", "interrupt", "start_error", "symlink"))
def test_supervisor_waits_stops_or_rejects_output_safely(tmp_path, monkeypatch, scenario):
    monkeypatch.setattr(
        runner, "load_protocol", lambda *_: ({}, "b" * 64, {"source_closure_sha256": {}}, "c" * 64)
    )
    monkeypatch.setattr(runner, "process_started", time.monotonic)
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "memory.peak").write_text("1000")
    (kernel / "memory.swap.current").write_text("0")
    monkeypatch.setattr(runner, "cgroup", lambda: kernel)
    stopped, sleeps = [], []
    out = tmp_path / "out"

    class Process:
        returncode = None
        polls = 0

        def __init__(self, *_args, **_kwargs):
            if scenario == "start_error":
                raise OSError("controlled start failure")
            runner.save(out / "worker.json", {"worker_marker": "read_only_fixture"})

        def poll(self):
            self.polls += 1
            if scenario == "complete" and self.polls >= 2:
                self.returncode = 0
            return self.returncode

    def stop(process):
        stopped.append(True)
        process.returncode = -15

    def sleep(seconds):
        sleeps.append(seconds)
        if scenario == "interrupt":
            raise KeyboardInterrupt

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "stop_process", stop)
    monkeypatch.setattr(runner.time, "sleep", sleep)
    if scenario == "timeout":
        monkeypatch.setattr(runner, "SECONDS", 0)
    if scenario == "symlink":
        out.symlink_to(kernel, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            runner.execute(tmp_path, tmp_path / "protocol", out)
    elif scenario == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            runner.execute(tmp_path, tmp_path / "protocol", out)
        assert stopped == [True]
    else:
        decision = runner.execute(tmp_path, tmp_path / "protocol", out)
        assert decision["status"] == "unavailable"  # Fixture is not learning evidence.
        result = json.loads((out / "result.json").read_text())
        if scenario == "complete":
            assert result["worker_marker"] == "read_only_fixture" and sleeps == [0.25]
        if scenario == "timeout":
            assert stopped == [True]


def test_wall_clock_includes_process_startup_imports(tmp_path, monkeypatch):
    proc = tmp_path / "stat"
    proc.write_text("123 (python) " + " ".join(["S"] + ["0"] * 18 + ["9000"]))
    monkeypatch.setattr(runner.os, "sysconf", lambda _: 100)
    monkeypatch.setattr(runner.time, "clock_gettime", lambda _: 100)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 200)
    assert runner.process_started(proc) == 190


def test_cli_is_explicit_and_bounds_core_dumps_and_files(monkeypatch, tmp_path, capsys):
    limits, calls = [], []
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *v: limits.append(v))
    monkeypatch.setattr(runner, "worker", lambda *v: calls.append(v))
    assert (
        cli.main(["--protocol", "bound", "--out", str(tmp_path), "--worker-deadline", "123"]) == 0
    )
    assert calls[0][-1] == 123 and len(limits) == 2
    monkeypatch.setattr(runner, "execute", lambda *_: {"status": "failed_learning_gate"})
    assert cli.main(["--protocol", "bound", "--out", str(tmp_path)]) == 2
    assert "failed_learning_gate" in capsys.readouterr().out
    monkeypatch.setattr(runner, "execute", lambda *_: {"status": "passed_generated_only"})
    assert cli.main(["--protocol", "bound", "--out", str(tmp_path)]) == 0
    monkeypatch.setattr(runner, "execute", lambda *_: (_ for _ in ()).throw(ValueError("invalid")))
    assert cli.main(["--protocol", "bound", "--out", str(tmp_path)]) == 2
    monkeypatch.setattr(
        sys, "argv", ["research_wav_control_run", "--protocol", "bound", "--out", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path("steganography/research_wav_control_run.py", run_name="__main__")
    assert stopped.value.code == 2
