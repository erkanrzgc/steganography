"""Generated software oracles: complete exposure, interrupted ledgers and hard limits.

Fake sessions below exercise orchestration, never claim physical GPU or learning.
The numeric fixture uses the real model/codec with generated optimizer slots.
"""

import contextlib
import copy
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import colab_artifacts as a
from core import spatial_control as c
from core import spatial_control_supervisor as s
from steganography import research_spatial_control as cli

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
PROVE, CHECKPOINT = c.checkpoint_proof, c.checkpoint
NETWORK, ENVIRONMENT = c.capacity.models.network, c.environment
SOURCES = c.sources
CORPUS = a.document(ROOT / c.CORPUS)
ENV = {**c.capacity.environment(), "driver": "generated-software-fixture"}


class FakeSession:
    def __init__(self, plan):
        self.plan, self.model = plan, torch.nn.Linear(1, 1)
        with torch.no_grad():
            for value in self.model.parameters():
                value.zero_()
        self.updates = self.presentations = 0

    def step(self):
        self.updates += 1
        self.presentations += 8
        return {
            "optimizer_updates": self.updates,
            "presentations": self.presentations,
            "loss": 1.0 if self.updates <= 20 else 0.5,
        }

    def scores(self, spec):
        return {"scores": np.tile([0.2, 0.8], (spec.originals, 1))}


def fake_stack(monkeypatch):
    @contextlib.contextmanager
    def session(plan, **kwargs):
        yield FakeSession(plan)

    def saved(experiment, out, phase, cell, *_):
        name = f"{phase}-{cell}.numeric"
        sha = a.put(out / name, b"software-fixture")
        return {"name": name, "sha256": sha, "bytes": 16, "seconds": 0.001}

    def prepare(out, corpus):
        a.directory(out)
        a.put(out / "manifest.json", a.encode({"software_fixture": True}))
        return {}

    def manifest(spec):
        return next(
            row
            for phase in ("profile", "control")
            for row in CORPUS[phase]
            if row["spec"] == asdict(spec)
        )

    monkeypatch.setattr(c, "sources", lambda root: ("source.py", c.CORPUS))
    monkeypatch.setattr(c, "environment", lambda: ENV)
    monkeypatch.setattr(c.engine, "session", session)
    monkeypatch.setattr(c, "checkpoint", saved)
    monkeypatch.setattr(c, "checkpoint_proof", lambda *args: None)
    monkeypatch.setattr(c.data, "manifest", manifest)
    monkeypatch.setattr(c.transport, "prepare", prepare)
    monkeypatch.setattr(c.transport, "batch", lambda *args: None)
    monkeypatch.setattr(c.transport, "verify", lambda *args: {"image_checks": 48})
    monkeypatch.setattr(c.capacity.models, "network", lambda *args: FakeSession(None).model)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 200)


def contract(base):
    root = base / "worktree"
    root.mkdir(parents=True)
    (root / "benchmarks").mkdir()
    a.put(root / "source.py", b"# generated test source\n")
    a.put(root / c.CORPUS, a.encode(CORPUS))
    path = root / "protocol.json"
    c.freeze(root, path, ENV)
    return root, path


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        fake_stack(patch)
        root, protocol = contract(tmp_path_factory.mktemp("spatial-complete"))
        out = root.parent / "output"
        a.directory(out)
        c.worker(root, protocol, out, time.monotonic() + 1700)
        yield (
            root,
            protocol,
            out,
            a.document(out / "worker.json"),
            {
                "wall_seconds": 20.0,
                "output_bytes": s.inventory(out),
                "memory_peak_bytes": 1000,
                "swap_peak_bytes": 0,
                "memory_max_bytes": 8 * 1024**3,
                "memory_swap_max_bytes": 0,
            },
        )


def test_complete_exposure_and_two_independent_auc_oracles(completed):
    root, protocol, out, worker, resources = completed
    result = c.decide(root, protocol, out, worker, resources)
    assert result["status"] == "passed_generated_only"
    assert result["confirmed_updates_lower_bound"] == result["step_intents"] == 1616
    assert result["independent_score_replays"] == 16
    assert result["checkpoints_verified"] == 8
    assert len(list((out / "events").iterdir())) == 3232
    assert all(row["optimizer_updates"] == 400 for row in worker["fits"])


def test_complete_failure_is_preserved_without_early_exit(completed):
    root, protocol, out, worker, resources = completed
    failed = copy.deepcopy(worker)
    failed["fits"][0]["final"]["development"] = [[0.5, 0.5]] * 24
    result = c.decide(root, protocol, out, failed, resources)
    assert result["status"] == "failed_learning_gate"
    assert len(result["cells"]) == 4 and result["confirmed_updates_lower_bound"] == 1616


@pytest.mark.parametrize(
    "change", ["source", "environment", "initial", "scores", "losses", "transport", "auc"]
)
def test_acceptance_rejects_forged_or_changed_evidence(completed, monkeypatch, change):
    root, protocol, out, worker, resources = completed
    altered = copy.deepcopy(worker)
    if change == "source":
        monkeypatch.setattr(c, "checksum", lambda _: "0" * 64)
    elif change == "environment":
        monkeypatch.setattr(c, "environment", lambda: {})
    elif change == "initial":
        altered["fits"][0]["initial_model_sha256"] = "0" * 64
    elif change == "scores":
        altered["fits"][0]["final"]["train"][0][0] = 2
    elif change == "losses":
        altered["fits"][0]["losses"][0] = 2
    elif change == "transport":
        altered["transport_checks"] = {}
    else:
        from core import wav_control_runner

        monkeypatch.setattr(wav_control_runner, "auc", lambda *_: 0)
    with pytest.raises(ValueError):
        c.decide(root, protocol, out, altered, resources)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_seconds", 1801),
        ("wall_seconds", float("nan")),
        ("output_bytes", c.MAX_OUTPUT + 1),
        ("memory_peak_bytes", 9 * 1024**3),
        ("swap_peak_bytes", 1),
        ("memory_max_bytes", 0),
        ("memory_swap_max_bytes", 1),
    ],
)
def test_resource_absence_or_limits_are_unavailable(completed, field, value):
    root, protocol, out, worker, resources = completed
    result = c.decide(root, protocol, out, worker, {**resources, field: value})
    assert result["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("fits", []),
        ("status", "incomplete"),
        ("native_media_opened", True),
        ("protocol_sha256", "0" * 64),
    ],
)
def test_complete_cells_and_generated_scope_required(completed, field, value):
    root, protocol, out, worker, resources = completed
    assert (
        c.decide(root, protocol, out, {**worker, field: value}, resources)["status"]
        == "unavailable"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("allocated_peak_bytes", 0),
        ("reserved_peak_bytes", 5 * 1024**3),
        ("losses", []),
        ("losses", [float("inf")] * 400),
        ("optimizer_updates", 399),
        ("presentations", 3192),
    ],
)
def test_exact_finite_fit_accounting_required(completed, field, value):
    root, protocol, out, worker, resources = completed
    altered = copy.deepcopy(worker)
    altered["fits"][0][field] = value
    assert c.decide(root, protocol, out, altered, resources)["status"] == "unavailable"


def test_profile_gate_forbids_all_learning_updates(completed, tmp_path, monkeypatch):
    root, protocol = contract(tmp_path)
    out = root.parent / "output"
    a.directory(out)
    monkeypatch.setattr(c, "projected_seconds", lambda *_: 1e9)
    c.worker(root, protocol, out, time.monotonic() + 1700)
    assert a.document(out / "worker.json")["reason"] == "profile_budget_gate"
    assert c.ledger(out)["confirmed_updates_lower_bound"] == 16
    assert not list(out.glob("control-*.numeric"))


def test_worker_revalidates_environment_and_generated_manifest(completed, tmp_path, monkeypatch):
    root, protocol = contract(tmp_path)
    out = root.parent / "output"
    a.directory(out)
    monkeypatch.setattr(c, "environment", lambda: {})
    with pytest.raises(ValueError, match="environment"):
        c.worker(root, protocol, out, time.monotonic() + 100)
    monkeypatch.setattr(c, "environment", lambda: ENV)
    monkeypatch.setattr(c.data, "manifest", lambda *_: {})
    with pytest.raises(ValueError, match="regenerated"):
        c.worker(root, protocol, out, time.monotonic() + 100)
    assert c.ledger(out)["confirmed_updates_lower_bound"] == 0


def test_projection_deadline_and_source_closure(tmp_path, monkeypatch):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "x.py").write_text("pass")
    assert "core/x.py" in SOURCES(tmp_path) and c.DOCUMENT in SOURCES(tmp_path)
    row = {
        "step_seconds": [50, 1, 2, 3],
        "transport_seconds": [1, 2, 3, 4],
        "evaluation_seconds": 2,
        "checkpoint_seconds": 3,
        "initialization_seconds": 4,
    }
    profiles = [{**row, "cell": cell} for cell in range(4)]
    assert c.projected_seconds(profiles) == 1.5 * 4 * (400 * 7 + 208 + 3 + 80)
    for bad in ([], [{**profiles[0], "step_seconds": [0] * 4}] + profiles[1:]):
        with pytest.raises(ValueError):
            c.projected_seconds(bad)
    with pytest.raises(TimeoutError):
        c.remaining(time.monotonic() - 1)
    assert 1799 < c.remaining(time.monotonic() + 2000) == 1800
    monkeypatch.setattr(c.subprocess, "run", lambda *args, **kw: SimpleNamespace(stdout="617.42\n"))
    assert ENVIRONMENT()["driver"] == "617.42"


@pytest.mark.parametrize(
    "change",
    ["schema", "setting", "sources", "corpus", "duplicate", "exposure", "missing", "recipe"],
)
def test_protocol_rejects_wrong_hash_exposure_and_ancestry(completed, tmp_path, change):
    root, path = contract(tmp_path)
    value = a.document(path)
    corpus = copy.deepcopy(CORPUS)
    if change == "schema":
        value["schema"] = "other"
    elif change == "setting":
        value["settings"]["epochs"] = 1
    elif change == "sources":
        value["source_sha256"] = {}
    elif change == "corpus":
        value["corpus_sha256"] = "0" * 64
    else:
        if change == "duplicate":
            corpus["control"][0]["sources"][1] = corpus["control"][0]["sources"][0]
        elif change == "exposure":
            corpus["control"][0]["spec"]["originals"] = 79
        elif change == "missing":
            corpus["control"].pop()
        else:
            corpus["recipe"] = "other"
        (root / c.CORPUS).write_bytes(a.encode(corpus))
        value["corpus_sha256"] = value["source_sha256"][c.CORPUS] = c.checksum(root / c.CORPUS)
    path.write_bytes(a.encode(value))
    with pytest.raises(ValueError):
        c.load(root, path)


def test_explicit_environment_and_no_overwrite(completed, tmp_path):
    root, path = contract(tmp_path)
    with pytest.raises(ValueError):
        c.freeze(root, root / "bad.json", {})
    with pytest.raises(FileExistsError):
        c.freeze(root, path, ENV)
    empty = root / "empty.py"
    empty.touch()
    assert c.checksum(empty) == a.digest(b"")


def test_partial_and_corrupt_ledgers_preserve_lower_bound(tmp_path):
    (tmp_path / "events").mkdir()
    state = {"intents": 0, "confirmed_updates": 0, "last_sha256": "0" * 64}
    c.event(tmp_path, state, "profile", 0, 1)
    assert c.ledger(tmp_path)["uncertain_final_step"]
    c.event(
        tmp_path, state, "profile", 0, 1, {"optimizer_updates": 1, "presentations": 8, "loss": 1}
    )
    assert not c.ledger(tmp_path)["uncertain_final_step"]
    c.event(tmp_path, state, "profile", 0, 2)
    (tmp_path / "events" / "000002-2-commit.json").write_text("{cut")
    with pytest.raises(ValueError):
        c.ledger(tmp_path)
    result = c.ledger(tmp_path, strict=False)
    assert result["confirmed_updates_lower_bound"] == 1
    assert result["uncertain_final_step"] and not result["ledger_valid"]


def test_wrong_counters_and_chain_rejected(tmp_path):
    (tmp_path / "events").mkdir()
    state = {"intents": 0, "confirmed_updates": 0, "last_sha256": "0" * 64}
    c.event(tmp_path, state, "profile", 0, 1)
    c.event(
        tmp_path, state, "profile", 0, 1, {"optimizer_updates": 1, "presentations": 7, "loss": 1}
    )
    with pytest.raises(ValueError, match="committed"):
        c.ledger(tmp_path)
    path = tmp_path / "events" / "000001-1-intent.json"
    value = a.document(path)
    value["previous_sha256"] = "a" * 64
    path.write_bytes(a.encode(value))
    with pytest.raises(ValueError, match="ledger"):
        c.ledger(tmp_path)


def test_real_numeric_proof_rejects_recipe_schedule_and_debit(tmp_path, monkeypatch):
    monkeypatch.setattr(c.capacity.models, "network", NETWORK)
    p = c.plan(CORPUS, "profile", 0, "a" * 64)
    binding = a.digest(
        a.encode(
            {
                "plan": asdict(p),
                "device": "cuda:0",
                "recipe": c.data.RECIPE,
                "optimizer": "scalar-adamw-1e-4-v1",
            }
        )
    )
    with torch.random.fork_rng(devices=[]):
        model = c.capacity.models.network(p.spec.domain).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0, foreach=False, fused=False
        )
        for parameter in model.parameters():
            optimizer.state[parameter] = {
                "step": torch.tensor(4.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        meta, arrays = c.numeric.capture(
            model,
            optimizer,
            source_sha256=binding,
            data_sha256=a.digest(a.encode(c.manifest_for(CORPUS, "profile", "L", "train"))),
            order=c.engine.order(p),
            consumption={
                "epoch": 4,
                "cursor": 16,
                "optimizer_updates": 4,
                "presentations": 32,
                "elapsed_seconds": 1,
                "output_bytes": 100 * a.MIB,
            },
        )

        def save(path):
            return c.numeric.save(path, meta, arrays)

        experiment = SimpleNamespace(
            checkpoint=save,
            model=model,
            optimizer=optimizer,
            binding=binding,
            data_sha256=meta["data_sha256"],
        )
        saved = CHECKPOINT(experiment, tmp_path, "profile", 0, CORPUS, "a" * 64)
        row = {"cell": 0, "checkpoint": saved}
        PROVE(tmp_path, row, p, CORPUS, "profile")
        wrong = copy.deepcopy(row)
        wrong["checkpoint"]["name"] = "../escape.numeric"
        with pytest.raises(ValueError, match="name"):
            PROVE(tmp_path, wrong, p, CORPUS, "profile")
        monkeypatch.setattr(c.numeric, "load", lambda *args, **kwargs: (meta, arrays))
        for field, value in [
            ("epoch", 3),
            ("optimizer_updates", 3),
            ("presentations", 31),
            ("cursor", 15),
            ("output_bytes", 1),
            ("elapsed_seconds", 1801),
        ]:
            prior = meta["consumption"][field]
            meta["consumption"][field] = value
            with pytest.raises(ValueError, match="evidence"):
                PROVE(tmp_path, row, p, CORPUS, "profile")
            meta["consumption"][field] = prior
        meta["optimizer"]["lr"] = 0.1
        with pytest.raises(ValueError, match="evidence"):
            PROVE(tmp_path, row, p, CORPUS, "profile")


def test_inventory_symlinks_entries_and_transient_rename(tmp_path, monkeypatch):
    a.put(tmp_path / "file", b"abc")
    (tmp_path / "dir").mkdir()
    assert s.inventory(tmp_path) == 3
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "file")
    with pytest.raises(ValueError, match="symlink"):
        s.inventory(tmp_path)
    link.unlink()
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="type limit"):
        s.inventory(tmp_path)
    (tmp_path / "pipe").unlink()
    original = Path.stat
    vanished = tmp_path / "progress.next.json"
    vanished.touch()

    def stat(path, **kwargs):
        if path == vanished:
            raise FileNotFoundError(2, "owned temporary renamed", str(path))
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert s.inventory(tmp_path) == 3


def test_stop_process_handles_exited_and_hung_children(monkeypatch):
    calls = []

    def kill(pid, sig):
        calls.append(sig)

    process = SimpleNamespace(pid=123, wait=lambda **kw: None)
    monkeypatch.setattr(s.os, "killpg", kill)
    s.stop(process)
    assert calls == [s.signal.SIGTERM]

    def gone(*args):
        raise ProcessLookupError

    monkeypatch.setattr(s.os, "killpg", gone)
    s.stop(process)
    monkeypatch.setattr(s.os, "killpg", kill)

    def wait(timeout):
        if timeout == 1:
            raise subprocess.TimeoutExpired("fixture", timeout)

    process.wait = wait
    s.stop(process)
    assert calls[-1] == s.signal.SIGKILL


@pytest.mark.parametrize(
    "outcome", ["pass", "incomplete", "output", "preflight", "launch", "observer", "observer_open"]
)
def test_independent_parent_limits_and_consumed_attempt(completed, tmp_path, monkeypatch, outcome):
    root, protocol = contract(tmp_path)
    out = root.parent / "output"
    group = tmp_path / "cgroup"
    group.mkdir()
    for name, value in [
        ("memory.peak", 1000),
        ("memory.swap.current", 0),
        ("memory.max", 8 * 1024**3),
        ("memory.swap.max", 0),
    ]:
        (group / name).write_text(str(value))
    monkeypatch.setattr(s, "cgroup", lambda: group)
    monkeypatch.setattr(s, "process_started", time.monotonic)
    monkeypatch.setattr(s.time, "sleep", lambda *_: None)

    class Process:
        pid = 111

        def __init__(self, *args, **kwargs):
            if outcome == "launch":
                raise OSError("fixture")
            self.returncode, self.polls = None, 0
            if outcome == "pass":
                c.worker(root, protocol, out, time.monotonic() + 1700)

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            if outcome in ("observer", "observer_open"):
                return None
            self.polls += 1
            if self.polls > 1:
                self.returncode = 0 if outcome == "pass" else 2
            return self.returncode

    monkeypatch.setattr(s.subprocess, "Popen", Process)
    stopped = []
    monkeypatch.setattr(
        s, "stop", lambda process: (stopped.append(True), setattr(process, "returncode", 2))
    )
    if outcome == "output":
        monkeypatch.setattr(s, "inventory", lambda *_: c.MAX_OUTPUT)
    if outcome == "preflight":
        monkeypatch.setattr(c, "environment", lambda: {})
    if outcome == "observer":

        def broken_inventory(*args):
            raise OSError("observer fixture")

        monkeypatch.setattr(s, "inventory", broken_inventory)
    if outcome == "observer_open":
        original_open = Path.open

        def broken_open(path, *args, **kwargs):
            if path.name == "parent-observations.jsonl":
                raise OSError("observer fixture")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", broken_open)
    result = s.execute(root, protocol, out)
    assert result["status"] == ("passed_generated_only" if outcome == "pass" else "unavailable")
    if outcome in ("output", "observer", "observer_open"):
        assert stopped
    if outcome == "pass":
        lines = (out / "parent-observations.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["worker_progress_sha256"]
    assert (root.parent / "attempt.json").is_file()
    with pytest.raises(FileExistsError):
        s.execute(root, protocol, out)
    with pytest.raises(ValueError, match="dedicated"):
        s.execute(root, protocol, root.parent / "other-output")


def test_worker_entry_requires_live_bound_supervisor(completed, tmp_path, monkeypatch):
    root, protocol = contract(tmp_path)
    out = root.parent / "output"
    binding = c.load(root, protocol)[1]
    attempt = {
        "protocol_sha256": binding,
        "supervisor_pid": os.getppid(),
        "deadline_monotonic": time.monotonic() + 100,
    }
    a.put(root.parent / "attempt.json", a.encode(attempt))
    calls = []
    monkeypatch.setattr(s, "cgroup", lambda: None)
    monkeypatch.setattr(s.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(c, "worker", lambda *args: calls.append(True))
    s.worker(root, protocol, out, time.monotonic() + 50)
    assert calls == [True]
    with pytest.raises(ValueError, match="supervisor"):
        s.worker(root, protocol, out, time.monotonic() + 101)


def test_cli_has_explicit_freeze_worker_and_execution_switches(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *args: None)
    calls = []
    monkeypatch.setattr(c, "freeze", lambda *args: calls.append("freeze"))
    monkeypatch.setattr(s, "worker", lambda *args: calls.append("worker"))
    monkeypatch.setattr(s, "execute", lambda *args: {"status": "passed_generated_only"})
    env = tmp_path / "env.json"
    a.put(env, a.encode({}))
    base = ["--protocol", str(tmp_path / "protocol.json")]
    assert cli.main(base + ["--freeze-environment", str(env)]) == 0
    assert cli.main(base + ["--out", str(tmp_path), "--worker-deadline", "1"]) == 0
    assert cli.main(base + ["--out", str(tmp_path), "--execute"]) == 0
    assert calls == ["freeze", "worker"]
    assert cli.main(base) == 2
    assert cli.main(base + ["--freeze-environment", str(env), "--execute"]) == 2
    assert cli.main(base + ["--worker-deadline", "1"]) == 2
    monkeypatch.setattr(s, "execute", lambda *args: {"status": "failed_learning_gate"})
    assert cli.main(base + ["--out", str(tmp_path), "--execute"]) == 2
    assert "unavailable" in capsys.readouterr().out
