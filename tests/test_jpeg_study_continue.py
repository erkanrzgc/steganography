"""Independent owned-job replay/rejection tests, not GPU learning evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from core import jpeg_study_continue as c
from core import jpeg_study_views as views
from core.jpeg_study_process import WorkerExit, WorkerFailure
from steganography import research_jpeg_continue as cli
from steganography import research_jpeg_study as frontend

SOURCE = {"frozen.py": "a" * 64}
CONTROLLER = {"controller.py": "b" * 64}
TORCH = "test-only-torch"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, allow_nan=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def manifest():
    samples = {}
    for role, count in (("fit", 256), ("development", 64)):
        rows = []
        for source in c.data.ORIGINS:
            for index in range(count):
                lineage = hashlib.sha256(f"{role}:{source}:{index}".encode()).hexdigest()
                for label in ("cover", "stego"):
                    digest = hashlib.sha256(f"{lineage}:{label}".encode()).hexdigest()
                    rows.append({"lineage": lineage, "sha256": digest,
                                 "source_group": source, "role": role,
                                 "label": label, "method": "JUNIWARD" if label == "stego" else None,
                                 "payload_rate": .4 if label == "stego" else None,
                                 "historical_role": "train", "quality_factor": 75,
                                 "path": f"jpeg/{digest}.jpg", "size": 100})
        samples[role] = rows
    from core import jpeg_float256

    return {"schema_version": views.SCHEMA, "samples": samples, "status": "completed",
            "simulation_contract": c.data.SIMULATION,
            "representation": views.contract(), "decoder": jpeg_float256.decoder_contract(),
            "originals_per_source": {"fit": 256, "development": 64}}


def receipt(config, arm, seed, stop, rows=4):
    recipe = c.study.recipe(arm, seed, rows, representation=views.FIVE)
    binding = hashlib.sha256(json.dumps({"recipe": recipe,
                                        "manifest_sha256": config["manifest_sha256"],
                                        "sources": SOURCE, "torch_version": TORCH},
                                       sort_keys=True).encode()).hexdigest()
    raw = f"generated-only:{arm}:{seed}:{stop}".encode()
    phase = stop // 16384 - 1
    value = {"status": "completed" if phase else "checkpointed", "recipe": recipe,
             "binding": binding, "source_sha256": SOURCE,
             "manifest_sha256": config["manifest_sha256"],
             "execution": {"device": "cuda:0", "torch_version": TORCH, "cpu_fallback": False},
             "presented_rows": stop, "resumed_from_rows": phase * 16384,
             "optimizer_updates": 16384 // rows, "total_optimizer_updates": stop // rows,
             "real_gpu_segment": True, "real_gpu_fit": bool(phase),
             "accuracy_qualification": "unavailable", "deployed": False,
             "initial_learned_sha256": "c" * 64, "schedule_sha256": "d" * 64,
             "seconds": 100, "mean_cross_entropy": .7,
             "checkpoints": [{"path": f"rows-{stop}.npz", "presented_rows": stop,
                              "sha256": hashlib.sha256(raw).hexdigest()}]}
    if phase:
        value["scores"] = [.4, .6] * 128
    return value, raw


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root = tmp_path / "parent"
    job = root / "controls-run"
    job.mkdir(parents=True)
    m = manifest()
    config = {"schema_version": "jpeg-study-config-v1", "action": "controls",
              "root": str(root / "data"), "manifest_sha256": write(root / "data/manifest.json", m),
              "audit": str(root / "audit.json"), "audit_sha256": "",
              "execution_mode": c.segments.MODE}
    config["audit_sha256"] = write(root / "audit.json", {
        "schema_version": "jpeg-study-independent-audit-v2", "status": "completed",
        "source_sha256": SOURCE, "manifest_sha256": config["manifest_sha256"],
        "representation": views.contract(),
        "idct_rows": {r: len(v) for r, v in m["samples"].items()},
        "idct_views": {r: 5 * len(v) for r, v in m["samples"].items()},
    })
    config_sha = write(root / "controls.json", config)
    prior = {"schema_version": "jpeg-learning-study-v1", "action": "controls",
             "source_sha256": SOURCE, "config_sha256": config_sha,
             "accuracy_qualification": "unavailable", "deployed": False,
             "status": "cancelled", "reason": "user_cancelled", "seconds": 500,
             "trial_reports": ["trial-legacy-20261012.json"]}
    plan = {"schema_version": c.SCHEMA, "parent": str(job), "config": str(root / "controls.json"),
            "config_sha256": config_sha, "parent_report_sha256": write(job / "report.json", prior),
            "segments": []}
    for arm, stop in (("legacy", 16384), ("legacy", 32768), ("reference-adamax", 16384)):
        seed = c.study.SEEDS[0]
        value, raw = receipt(config, arm, seed, stop)
        name = f"{arm}-{seed}"
        folder = job / name / f"segment-{stop // 16384 - 1}"
        folder.mkdir(parents=True)
        (folder / f"rows-{stop}.npz").write_bytes(raw)
        entry = {"arm": arm, "seed": seed, "stop_rows": stop,
                 "receipt_sha256": write(job / name / f"receipt-{stop}.json", value)}
        if stop == 32768:
            entry["trial_sha256"] = write(job / f"trial-{name}.json", {
                **value, "arm": arm, "seed": seed, "execution_mode": c.segments.MODE})
        plan["segments"].append(entry)

    def generated_load(path, checksum, binding):
        # Bytes are deliberately NOT valid NPZ or training. Real numeric parser
        # is separately integration-tested; orchestration uses this bounded stub.
        c.segments.checkpoint(path, checksum)
        stop = 16384 if "16384" in path.name else 32768
        return {"metadata": {"presented_rows": stop},
                "arrays": {f"optimizer/{n}/step": stop // 4 for n in c.PARAMETERS}}

    monkeypatch.setattr(c.state, "load", generated_load)
    return plan, config, prior


def prepare(plan):
    return c.prepare(plan, SOURCE, torch_version=TORCH)


def execute(tmp_path, owned, *, probe=None, launch=None, clock=lambda: 0,
            current_sources=lambda: SOURCE, current_controller_sources=lambda: CONTROLLER):
    plan, config, _ = owned
    calls = []

    def worker(payload, out, arm, seed, rows, stop, timeout, cancelled):
        assert not cancelled() and 0 < timeout <= 1830
        assert "execution_mode" not in payload
        if stop == 32768:
            expected = out.parent / "segment-0/rows-16384.npz"
            assert payload["resume"][0]["path"] == str(expected)
        calls.append((arm, seed, stop))
        if launch:
            return launch(payload, out, arm, seed, rows, stop, timeout, cancelled)
        value, raw = receipt(config, arm, seed, stop, rows)
        out.mkdir()
        (out / f"rows-{stop}.npz").write_bytes(raw)
        return value

    value = c.execute(plan, tmp_path / "new", sources=SOURCE, torch_version=TORCH,
                      launch=worker, probe=probe or (lambda rows, timeout, cancelled: {
                          "status": "completed" if rows <= 16 else "unavailable", "rows": rows}),
                      current_sources=current_sources, controller_sources=CONTROLLER,
                      current_controller_sources=current_controller_sources, clock=clock)
    return value, calls


def test_exact_prefix_replayed_without_retraining_and_parent_unchanged(tmp_path, owned):
    plan, _, _ = owned
    before = {p: p.read_bytes() for p in Path(plan["parent"]).rglob("*") if p.is_file()}
    value, calls = execute(tmp_path, owned)
    assert value["status"] == "completed" and len(value["trials"]) == 8
    assert len(calls) == 13 and calls[0] == ("reference-adamax", 20261012, 32768)
    assert value["trials"][0]["inherited_segment_indices"] == [0, 1]
    assert value["trials"][1]["inherited_segment_indices"] == [0]
    assert value["prior_controls_seconds"] == value["seconds"] == 500
    assert value["new_controls_seconds"] == 0
    assert value["accuracy_qualification"] == "unavailable" and not value["deployed"]
    assert all(p.read_bytes() == raw for p, raw in before.items())
    half = json.loads((tmp_path / "new/reference-adamax-20261012/receipt-16384.json").read_bytes())
    assert half["inherited_from"]["receipt_sha256"] == plan["segments"][2]["receipt_sha256"]
    assert str(tmp_path) not in json.dumps(value)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version="bad"), lambda p: p.update(extra=True),
    lambda p: p.update(parent=""), lambda p: p.update(parent_report_sha256="f" * 64),
    lambda p: p.update(config_sha256="f" * 64), lambda p: p.update(segments=[]),
    lambda p: p["segments"].reverse(), lambda p: p["segments"][0].update(seed=True),
    lambda p: p["segments"][0].update(stop_rows=True),
    lambda p: p["segments"][0].update(receipt_sha256="f" * 64),
    lambda p: p["segments"][1].update(trial_sha256="f" * 64),
    lambda p: p["segments"][0].update(path="../escape"),
])
def test_bad_plan_prefix_checksum_fields_fail_closed(owned, mutation):
    plan = copy.deepcopy(owned[0])
    mutation(plan)
    with pytest.raises(ValueError):
        prepare(plan)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(status="completed"), lambda p: p.update(reason="timeout"),
    lambda p: p.update(source_sha256={}), lambda p: p.update(config_sha256="f" * 64),
    lambda p: p.update(accuracy_qualification="supported"), lambda p: p.update(deployed=True),
    lambda p: p.update(seconds=True), lambda p: p.update(seconds=-1),
    lambda p: p.update(seconds=21600), lambda p: p.update(seconds=1),
    lambda p: p.update(trial_reports=[]),
])
def test_invalid_parent_budget_ownership_or_source_not_reclassified(owned, mutation):
    plan, _, prior = owned
    mutation(prior)
    plan["parent_report_sha256"] = write(Path(plan["parent"]) / "report.json", prior)
    with pytest.raises(ValueError):
        prepare(plan)


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(binding="f" * 64), lambda v: v.update(recipe={}),
    lambda v: v.update(source_sha256={}), lambda v: v.update(presented_rows=32768),
    lambda v: v.update(resumed_from_rows=16384), lambda v: v.update(optimizer_updates=1),
    lambda v: v.update(real_gpu_fit=True), lambda v: v.update(real_gpu_segment=False),
    lambda v: v.update(scores=[]), lambda v: v.update(seconds=-1),
    lambda v: v.update(mean_cross_entropy=True), lambda v: v.update(checkpoints=[]),
    lambda v: v["checkpoints"][0].update(path="../escape"),
    lambda v: v["checkpoints"][0].update(sha256="f" * 64),
    lambda v: v.update(initial_learned_sha256="bad"),
])
def test_invalid_inherited_receipt_never_launches_gpu(owned, mutation):
    plan = owned[0]
    entry = plan["segments"][0]
    path = Path(plan["parent"]) / "legacy-20261012/receipt-16384.json"
    value = json.loads(path.read_bytes())
    mutation(value)
    entry["receipt_sha256"] = write(path, value)
    with pytest.raises(ValueError):
        prepare(plan)


def test_torch_version_binding_cannot_be_rewritten(owned):
    with pytest.raises(ValueError, match="binding"):
        c.prepare(owned[0], SOURCE, torch_version="changed-torch")


def test_numeric_snapshot_exposure_or_steps_rejected(owned, monkeypatch):
    monkeypatch.setattr(c.state, "load", lambda *a: {
        "metadata": {"presented_rows": 32768}, "arrays": {}})
    with pytest.raises(ValueError, match="exposure"):
        prepare(owned[0])


@pytest.mark.parametrize("what", ["report", "receipt", "snapshot", "parent"])
def test_symlink_evidence_rejected(tmp_path, owned, what):
    parent = Path(owned[0]["parent"])
    path = {"report": parent / "report.json",
            "receipt": parent / "legacy-20261012/receipt-16384.json",
            "snapshot": parent / "legacy-20261012/segment-0/rows-16384.npz",
            "parent": parent}[what]
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=what == "parent")
    with pytest.raises(ValueError, match="symlink"):
        prepare(owned[0])


def test_output_never_overwritten(tmp_path, owned):
    (tmp_path / "new").mkdir()
    with pytest.raises(FileExistsError):
        execute(tmp_path, owned)


def test_budget_includes_discarded_work_and_new_verification(tmp_path, owned):
    ticks = iter([0, 22000, 22000, 22000])
    value, calls = execute(tmp_path, owned, clock=lambda: next(ticks))
    assert not calls and value["status"] == "unavailable"
    assert value["reason"] == "controls_budget_exhausted" and value["seconds"] == 22500


@pytest.mark.parametrize("failure", [TimeoutError(), InterruptedError(),
                                     WorkerFailure(WorkerExit(-24, 0))])
def test_worker_failure_retains_half_without_claiming_completion(tmp_path, owned, failure):
    def fail(*args):
        raise failure

    value, calls = execute(tmp_path, owned, launch=fail)
    assert value["status"] != "completed"
    assert value["trials"][0]["status"] == "completed"
    assert value["trials"][1]["status"] != "completed"
    assert (tmp_path / "new/reference-adamax-20261012/segment-0/rows-16384.npz").is_file()
    assert calls[0][2] == 32768 and not value["learning_gate"]


@pytest.mark.parametrize("controller", [False, True])
def test_source_mutation_fails_closed(tmp_path, owned, controller):
    value, _ = execute(tmp_path, owned, **{
        "current_controller_sources" if controller else "current_sources": lambda: {}})
    assert value["reason"] == "source_mutation" and not value["learning_gate"]


def test_fitting_closure_not_changed_by_separate_orchestrator():
    paths = frontend.sources()
    assert "core/jpeg_study_continue.py" not in paths
    assert "steganography/research_jpeg_continue.py" not in paths
    assert len(cli.sources()) == 3


def test_metadata_digest_bounds(tmp_path):
    path = tmp_path / "document"
    path.write_bytes(b"x")
    assert c.digest(path) == hashlib.sha256(b"x").hexdigest()
    path.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="bounds"):
        c.digest(path)


@pytest.mark.parametrize("fault", ["timeout", "cpu", "source", "rows"])
def test_capacity_adapter_limits_and_source_binding(monkeypatch, fault):
    def child(*args, **kwargs):
        if fault == "timeout":
            raise TimeoutError
        if fault == "cpu":
            raise WorkerFailure(WorkerExit(-24, 0))
        return json.dumps({"status": "completed", "rows": 16 if fault == "rows" else 8,
                           "source_sha256": {}})

    monkeypatch.setattr(frontend, "child", child)
    if fault in ("source", "rows"):
        with pytest.raises(ValueError):
            cli.probe(8, 180, lambda: False)
    else:
        assert cli.probe(8, 180, lambda: False)["status"] == "unavailable"


def test_real_numeric_snapshot_is_not_rebound(owned, monkeypatch):
    import numpy as np

    from core import jpeg_study_state as state

    plan = owned[0]
    plan["segments"] = plan["segments"][:1]
    entry = plan["segments"][0]
    folder = Path(plan["parent"]) / "legacy-20261012"
    path = folder / "receipt-16384.json"
    value = json.loads(path.read_bytes())
    arrays = {}
    for name, (shape, dtype) in state.specs().items():
        array = np.zeros(shape, dtype=dtype)
        if name.endswith("/step") or name.endswith("num_batches_tracked"):
            array[...] = 4096
        elif name.endswith("running_var"):
            array[...] = 1
        arrays[name] = array
    target = folder / "segment-0/real-generated.npz"
    checksum = state.save({"metadata": {"schema_version": "jpeg-study-snapshot-v1",
                                        "presented_rows": 16384, "binding": value["binding"]},
                           "arrays": arrays}, target)
    # Replace this test's stub only; generated zero weights prove parsing, not fitting.
    monkeypatch.undo()
    target.replace(folder / "segment-0/rows-16384.npz")
    value["checkpoints"][0]["sha256"] = checksum
    entry["receipt_sha256"] = write(path, value)
    assert len(prepare(plan)[3]) == 1


@pytest.mark.parametrize("target", ["parent", "data"])
def test_new_output_cannot_be_inside_preserved_evidence(owned, target):
    plan, config, _ = owned
    out = Path(plan["parent"] if target == "parent" else config["root"]) / "new"
    with pytest.raises(ValueError, match="modify"):
        c.execute(plan, out, sources=SOURCE, torch_version=TORCH, launch=None, probe=None,
                  current_sources=lambda: SOURCE, controller_sources=CONTROLLER,
                  current_controller_sources=lambda: CONTROLLER)


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(source_sha256={}), lambda v: v.update(idct_views={}),
    lambda v: v.update(status="unavailable"), lambda v: v.update(representation={}),
])
def test_independent_audit_required_before_new_workers(owned, mutation):
    plan, config, _ = owned
    value = json.loads(Path(config["audit"]).read_bytes())
    mutation(value)
    config["audit_sha256"] = write(Path(config["audit"]), value)
    plan["config_sha256"] = write(Path(plan["config"]), config)
    parent = Path(plan["parent"]) / "report.json"
    prior = json.loads(parent.read_bytes())
    prior["config_sha256"] = plan["config_sha256"]
    plan["parent_report_sha256"] = write(parent, prior)
    with pytest.raises(ValueError, match="audit"):
        prepare(plan)


def test_completed_trial_has_bound_scores(owned):
    plan = owned[0]
    path = Path(plan["parent"]) / "trial-legacy-20261012.json"
    value = json.loads(path.read_bytes())
    value["scores"] = [1]
    plan["segments"][1]["trial_sha256"] = write(path, value)
    with pytest.raises(ValueError, match="parent trial"):
        prepare(plan)


def test_second_half_cannot_change_initialization(owned):
    plan = owned[0]
    path = Path(plan["parent"]) / "legacy-20261012/receipt-32768.json"
    value = json.loads(path.read_bytes())
    value["initial_learned_sha256"] = "f" * 64
    plan["segments"][1]["receipt_sha256"] = write(path, value)
    with pytest.raises(ValueError, match="halves"):
        prepare(plan)


def test_copy_limit_keeps_evidence_unavailable(tmp_path, owned, monkeypatch):
    monkeypatch.setattr(c, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="copy"):
        execute(tmp_path, owned)
    value = json.loads((tmp_path / "new/report.json").read_bytes())
    assert value["reason"] == "continuation_contract_failed"


def test_probe_cancellation_stops_before_gpu(tmp_path, owned):
    def probe(*args):
        raise InterruptedError

    value, calls = execute(tmp_path, owned, probe=probe)
    assert value["status"] == "cancelled" and not calls


def test_capacity_invalid_status_writes_failure_receipt(tmp_path, owned):
    with pytest.raises(ValueError, match="capacity"):
        execute(tmp_path, owned, probe=lambda *args: {"rows": 8, "status": "passed"})
    assert json.loads((tmp_path / "new/report.json").read_bytes())["status"] == "unavailable"


def test_genuine_capacity_absence_cannot_complete(tmp_path, owned):
    value, _ = execute(tmp_path, owned, probe=lambda rows, *args: {
        "rows": rows, "status": "unavailable"})
    assert value["status"] == "unavailable"
    assert all(v["reason"] == "no_genuine_batch_fits" for v in value["trials"]
               if v["arm"] == "genuine-batch")


def test_unmatched_baseline_schedule_rejected(tmp_path, owned):
    def launch(payload, out, arm, seed, rows, stop, *args):
        value, raw = receipt(owned[1], arm, seed, stop, rows)
        out.mkdir()
        (out / f"rows-{stop}.npz").write_bytes(raw)
        value["schedule_sha256"] = "f" * 64
        return value

    with pytest.raises(ValueError, match="unmatched"):
        execute(tmp_path, owned, launch=launch)


@pytest.mark.parametrize("stage", ["probe", "phase", "loop", "symlink"])
def test_user_marker_cancels_or_rejects_symlink(tmp_path, owned, stage):
    calls = []

    def probe(rows, timeout, cancelled):
        calls.append(rows)
        marker = tmp_path / "new/CANCEL"
        if (stage == "probe" and rows == 8) or (stage in ("loop", "symlink") and rows == 32):
            if stage == "symlink":
                marker.symlink_to(tmp_path / "outside")
            else:
                marker.write_bytes(b"")
        return {"rows": rows, "status": "completed"}

    def launch(payload, out, arm, seed, rows, stop, timeout, cancelled):
        (tmp_path / "new/CANCEL").write_bytes(b"")
        value, raw = receipt(owned[1], arm, seed, stop, rows)
        out.mkdir()
        (out / f"rows-{stop}.npz").write_bytes(raw)
        return value

    if stage == "symlink":
        with pytest.raises(ValueError, match="symlink"):
            execute(tmp_path, owned, probe=probe)
    else:
        value, _ = execute(tmp_path, owned, probe=probe,
                           launch=launch if stage == "phase" else None)
        assert value["status"] == "cancelled"


@pytest.mark.parametrize("status", ["completed", "cancelled"])
def test_cli_uses_shared_service_and_returns_correct_exit(tmp_path, monkeypatch, status):
    path = tmp_path / "plan.json"
    write(path, {"test": True})
    monkeypatch.setattr(cli.srnet_cuda, "inspect", lambda: {"torch_version": TORCH})

    def service(plan, out, **kwargs):
        assert plan == {"test": True} and kwargs["torch_version"] == TORCH
        assert kwargs["launch"] is cli.launch and kwargs["probe"] is cli.probe
        return {"status": status, "reason": None}

    monkeypatch.setattr(c, "execute", service)
    assert cli.main(["--plan", str(path), "--out", str(tmp_path / "out")]) == (
        0 if status == "completed" else 2)
    path.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="bounds"):
        cli.main(["--plan", str(path), "--out", str(tmp_path / "out")])


def test_worker_adapter_carries_bounded_owned_resume(monkeypatch, tmp_path):
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs))
        return json.dumps({"status": "completed"})

    monkeypatch.setattr(frontend, "child", child)
    assert cli.launch({"test": True}, tmp_path, "legacy", 20261012, 4, 32768,
                      1800, lambda: False)["status"] == "completed"
    assert "--stop-rows" in calls[0][0] and calls[0][1]["timeout"] == 1800
    assert json.loads(calls[0][1]["payload"]) == {"test": True}


def test_valid_capacity_adapter(monkeypatch):
    monkeypatch.setattr(frontend, "child", lambda *args, **kwargs: json.dumps({
        "status": "completed", "rows": 8, "source_sha256": frontend.capacity_sources()}))
    assert cli.probe(8, 180, lambda: False)["status"] == "completed"
