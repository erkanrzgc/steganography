"""Failure receipts, cumulative-budget resumes, immutable gates and adapter boundaries."""

import copy
import hashlib
import io
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from core import jpeg_refine as c
from core import jpeg_refine_metrics as metrics
from core import jpeg_refine_state as state
from core.jpeg_study_process import WorkerExit, WorkerFailure
from core.jpeg_timing import write_document
from steganography import research_jpeg_refine as frontend
from tests.test_jpeg_refine import EXECUTION, SHA, SOURCES, config, numeric, rows
from tests.test_jpeg_refine import matrix as matrix  # noqa: F401
from tests.test_jpeg_refine import tiny as tiny  # noqa: F401


@pytest.mark.parametrize(
    "fault",
    [
        "audit",
        "probe",
        "probe_binding",
        "timing",
        "evaluation",
        "disk",
        "worker_failure",
        "source",
        "cancel",
        "cancel_symlink",
        "budget",
    ],
)
def test_preflight_and_controller_limits_stop_without_retry(matrix, monkeypatch, fault):
    launch = matrix.launch

    def failed(mode, payload, target, timeout, cancelled):
        if fault == "worker_failure":
            raise WorkerFailure(WorkerExit(-9, 0))
        if fault in ("cancel", "cancel_symlink"):
            marker = matrix.root / "run" / "CANCEL"
            if fault == "cancel":
                marker.touch(exist_ok=True)
            elif not marker.is_symlink():
                marker.symlink_to(matrix.root / "missing")
        value = launch(mode, payload, target, timeout, lambda: False)
        if fault == "audit" and mode == "audit":
            value["idct_rows"] = {}
        if fault == "probe" and mode == "preflight":
            value = {"status": "unavailable", "reason": "generated_unavailable"}
        if fault == "probe_binding" and mode == "preflight":
            value["source_sha256"] = {}
        if fault == "timing" and mode == "preflight":
            value["max_update_seconds"] = float("nan")
        if fault == "evaluation" and mode == "evaluate":
            value["model_sha256_after"] = "b" * 64
        if fault == "disk" and mode == "preflight":
            monkeypatch.setattr(c.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        if fault == "source" and mode == "audit":
            monkeypatch.setattr(c, "digest", lambda _: (_ for _ in ()).throw(ValueError()))
        return value

    settings = dict(
        sources=SOURCES, current_sources=lambda: SOURCES, inspect=lambda _: EXECUTION, launch=failed
    )
    if fault == "source":
        count = 0

        def current_sources():
            nonlocal count
            count += 1
            return SOURCES if count == 1 else {}

        settings["current_sources"] = current_sources
    if fault == "budget":
        ticks = iter([0.0, 21601.0, 21602.0, 21603.0])
        settings["clock"] = lambda: next(ticks)
    report = c.execute(matrix.config, matrix.root / "run", **settings)
    assert report["status"] != "completed" and not report["learning_gate"]
    assert not report["unpaired_superiority"] and report["selected_arm"] is None
    if fault == "worker_failure":
        assert report["reason"] == "worker_signalled" and report["worker_execution"]["signal"] == 9
    if fault == "probe":
        assert report["reason"] == "generated_unavailable"
    if fault == "source":
        assert report["reason"] == "source_mutation"


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "runtime",
        "status",
        "time",
        "bytes",
        "remaining",
        "prefix",
        "entry",
        "lost",
        "parent_symlink",
    ],
)
def test_resume_cannot_rebind_or_reset_budget(matrix, fault):
    parent = matrix.root / "parent"
    parent.mkdir()
    settings = c.configuration(matrix.config, SOURCES)
    prior = {
        "schema_version": "jpeg-refine-report-v1",
        "protocol_sha256": c.digest(c.protocol(settings)),
        "source_sha256": SOURCES,
        "execution": EXECUTION,
        "status": "cancelled",
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "seconds": 10.0,
        "cumulative_output_bytes": 4096,
        "receipts": [],
    }
    if fault == "source":
        prior["source_sha256"] = {}
    elif fault == "runtime":
        prior["execution"] = {}
    elif fault == "status":
        prior["status"] = "completed"
    elif fault == "time":
        prior["seconds"] = -1
    elif fault == "bytes":
        prior["cumulative_output_bytes"] = 0
    elif fault == "prefix":
        prior["receipts"] = [{}] * 57
    elif fault == "entry":
        prior["receipts"] = [{"index": 2}]
    checksum = write_document(parent / "report.json", prior)
    value = copy.deepcopy(matrix.config)
    value["resume"] = {
        "parent": str(parent),
        "report_sha256": checksum,
        "lost_seconds": 0.0,
        "remaining_seconds": 21590.0,
    }
    if fault == "remaining":
        value["resume"]["remaining_seconds"] = 21600.0
    elif fault == "lost":
        value["resume"]["lost_seconds"] = -1
    elif fault == "parent_symlink":
        link = matrix.root / "link"
        link.symlink_to(parent, target_is_directory=True)
        value["resume"]["parent"] = str(link)
    if fault == "lost":
        with pytest.raises(ValueError):
            matrix.execute(matrix.root / "resume", value)
    else:
        report = matrix.execute(matrix.root / "resume", value)
        assert report["status"] == "unavailable" and report["verified_updates"] == 0
    assert not any(v[0] == "train" for v in matrix.calls)


def test_training_failure_and_disk_budget_receipt(matrix, monkeypatch):
    original = matrix.launch

    def launch(mode, payload, target, timeout, cancelled):
        if mode == "train":
            return {"status": "unavailable", "reason": "gpu_allocator_out_of_memory"}
        return original(mode, payload, target, timeout, cancelled)

    report = c.execute(
        matrix.config,
        matrix.root / "failed",
        sources=SOURCES,
        current_sources=lambda: SOURCES,
        inspect=lambda _: EXECUTION,
        launch=launch,
    )
    assert report["reason"] == "gpu_allocator_out_of_memory"
    assert (matrix.root / "failed" / "failure-00.json").is_file()
    monkeypatch.setattr(c, "storage", lambda _: c.BYTES - 1)
    report = matrix.execute(matrix.root / "full")
    assert report["status"] == "unavailable" and report["reason"] == "disk_budget_exhausted"


def test_invalid_numeric_archive_and_metadata_size(tiny, tmp_path):
    path = tmp_path / "notzip"
    path.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="malformed"):
        state.load(path, hashlib.sha256(path.read_bytes()).hexdigest(), SHA)
    path = tmp_path / "big.json"
    path.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="bounds"):
        c.read(path)
    with pytest.raises(ValueError, match="role"):
        metrics.statistics(rows(), [0.5] * 1024, role="test")
    with pytest.raises(ValueError, match="preflight"):
        c.estimate({"status": "completed"}, 28672, 21600)


def test_evaluation_parity_and_metadata_fail_closed(tiny):
    base = {
        "status": "completed",
        "source_sha256": SOURCES,
        "execution": EXECUTION,
        "metadata": numeric(stop=2048, parent=SHA)["metadata"],
        "checkpoint_sha256": SHA,
        "real_gpu_evaluation": True,
        "model_sha256_before": SHA,
        "model_sha256_after": SHA,
        "development": {"singleton_batch_parity": True},
    }
    settings = config(__import__("pathlib").Path("data"))
    assert (
        c.validate_evaluation(base, settings, EXECUTION, {"sha256": SHA}, base["metadata"]) == base
    )
    failed = copy.deepcopy(base)
    failed["development"]["singleton_batch_parity"] = False
    with pytest.raises(ValueError, match="parity"):
        c.validate_evaluation(failed, settings, EXECUTION, {"sha256": SHA}, base["metadata"])


@pytest.mark.parametrize(
    "mode",
    [
        "audit",
        "preflight",
        "evaluate",
        "unknown",
        "mutation",
        "oom",
        "cuda_unavailable",
        "train_parent",
    ],
)
def test_worker_modes_are_thin_adapters(tmp_path, monkeypatch, mode):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(frontend, "sources", lambda: SOURCES)
    monkeypatch.setattr(frontend, "inspect", lambda _: EXECUTION)
    monkeypatch.setattr(frontend.srnet_cuda, "policy", lambda: nullcontext(EXECUTION))
    monkeypatch.setattr(
        frontend.jpeg_study_audit, "audit", lambda *args, **kwargs: {"status": "completed"}
    )
    monkeypatch.setattr(frontend.preflight, "probe", lambda: {"status": "completed"})
    monkeypatch.setattr(
        frontend.training, "evaluate", lambda *args, **kwargs: {"status": "completed"}
    )
    monkeypatch.setattr(
        frontend.training, "train", lambda *args, **kwargs: {"status": "checkpointed"}
    )
    payload = {
        "config": config(tmp_path),
        "checkpoint": {"path": "unused"},
        "parent": {"path": "unused"},
        "task": ["warmup", 20261012, 1024],
        "binding": SHA,
        "execution": EXECUTION,
    }
    chosen = mode
    if mode == "train_parent":
        chosen = "train"
    elif mode == "mutation":
        chosen = "audit"
        values = iter([SOURCES, {}])
        monkeypatch.setattr(frontend, "sources", lambda: next(values))
    elif mode in ("oom", "cuda_unavailable"):
        chosen = "audit"
        exception = (
            torch.cuda.OutOfMemoryError()
            if mode == "oom"
            else (frontend.srnet_cuda.CUDAUnavailable("fixture_unavailable"))
        )
        monkeypatch.setattr(frontend, "inspect", lambda _: (_ for _ in ()).throw(exception))
    result = frontend.worker(chosen, payload, tmp_path)
    assert result["status"] == (
        "unavailable"
        if mode in {"unknown", "mutation", "oom", "cuda_unavailable"}
        else "checkpointed"
        if mode == "train_parent"
        else "completed"
    )


def test_launch_and_worker_main_input_bounds(tmp_path, monkeypatch, capsys):
    captured = {}

    def child(command, **kwargs):
        captured.update(kwargs)
        assert command[1:3] == ["-m", "steganography.research_jpeg_refine"]
        return b'{"status":"completed"}'

    monkeypatch.setattr(frontend, "child", child)
    result = frontend.launch("audit", {"config": {"root": tmp_path}}, tmp_path, 10, lambda: False)
    assert result["exit_status"] == 0 and captured["timeout"] == 10
    monkeypatch.setattr(frontend, "worker", lambda *args: {"status": "completed"})
    monkeypatch.setattr(frontend.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))
    assert frontend.main(["--worker", "audit", "--out", "fresh"]) == 0
    assert "completed" in capsys.readouterr().out
    monkeypatch.setattr(
        frontend.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"x" * (1024**2 + 1)))
    )
    with pytest.raises(SystemExit):
        frontend.main(["--worker", "audit", "--out", "fresh"])


def test_source_file_size_is_bounded(tmp_path, monkeypatch):
    (tmp_path / "oversized").write_bytes(b"x" * (1024**2 + 1))
    monkeypatch.setattr(frontend, "ROOT", tmp_path)
    monkeypatch.setattr(frontend, "NAMES", ("oversized",))
    with pytest.raises(ValueError):
        frontend.sources()


@pytest.mark.parametrize("pattern", ["negative", "unpaired_wins", "one_seed_fails"])
def test_all_cells_required_and_superiority_is_separate(pattern):
    manifest = {"samples": {"development": rows("development", 64), "fit": rows(count=8)}}
    evaluations = []
    for seed in c.study.SEEDS:
        for arm in ("warmup", "paired", "unpaired"):
            stop = 2048 if arm == "warmup" else 8192
            value = {
                "status": "completed",
                "real_gpu_evaluation": True,
                "metadata": {"arm": arm, "seed": seed, "total_updates": stop},
                "model_sha256_before": SHA,
                "model_sha256_after": SHA,
                "checkpoint_sha256": SHA,
            }
            if arm == "warmup":
                scores = [0.5, 0.5] * 128
            elif pattern == "negative":
                scores = [0.8, 0.2] * 128
            elif arm == "paired":
                scores = [0.5, 0.5] * 128
            else:
                scores = [0.2, 0.8] * 128
                if pattern == "one_seed_fails" and seed == c.study.SEEDS[1]:
                    scores[:128] = [0.8, 0.2] * 64
            value["development"] = {"scores": scores}
            if arm != "warmup":
                value["fit"] = {"scores": [0.2, 0.8] * 16}
            evaluations.append(value)
    result = metrics.compare(manifest, evaluations)
    assert result["learning_gate"] is (pattern == "unpaired_wins")
    assert result["unpaired_superiority"] is (pattern == "unpaired_wins")
    assert result["selected_arm"] == ("unpaired" if pattern == "unpaired_wins" else None)


def test_frozen_data_identity_cannot_be_substituted(tmp_path):
    value = config(tmp_path)
    value["manifest_sha256"] = SHA
    with pytest.raises(ValueError, match="frozen data"):
        c.configuration(value, SOURCES)


@pytest.mark.parametrize("fault", ["checkpoint", "copy_limit", "disk_limit"])
def test_inherited_checkpoint_copy_and_budget_are_bounded(matrix, fault):
    parent = matrix.root / "parent"
    parent.mkdir()
    task = ("warmup", c.study.SEEDS[0], 512)
    settings = c.configuration(matrix.config, SOURCES)
    value = matrix.launch(
        "train",
        {"task": task, "parent": None, "binding": c.binding(settings, EXECUTION, task[1])},
        parent / "segment-00",
        1830,
        lambda: False,
    )
    receipt_path = parent / "segment-00" / "receipt.json"
    receipt_sha = write_document(receipt_path, value)
    if fault == "copy_limit":
        with receipt_path.open("ab") as stream:
            stream.write(b" " * 65536)
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    prior = {
        "schema_version": "jpeg-refine-report-v1",
        "protocol_sha256": c.digest(c.protocol(settings)),
        "source_sha256": SOURCES,
        "execution": EXECUTION,
        "status": "cancelled",
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "seconds": 10.0,
        "cumulative_output_bytes": c.BYTES if fault == "disk_limit" else c.storage(parent) + 4096,
        "receipts": [
            {
                "index": 0,
                "arm": task[0],
                "seed": task[1],
                "stop_updates": task[2],
                "receipt_sha256": receipt_sha,
                "checkpoint_sha256": "b" * 64
                if fault == "checkpoint"
                else value["checkpoint_sha256"],
            }
        ],
    }
    checksum = write_document(parent / "report.json", prior)
    config = copy.deepcopy(matrix.config)
    config["resume"] = {
        "parent": str(parent),
        "report_sha256": checksum,
        "lost_seconds": 0.0,
        "remaining_seconds": 21590.0,
    }
    matrix.calls.clear()
    report = matrix.execute(matrix.root / "resume", config)
    assert report["status"] == "unavailable"
    assert not any(v[0] == "train" for v in matrix.calls)


def test_final_data_verifier_rehashes_native_bytes_and_tensors(tmp_path, monkeypatch):
    raw = b"native JPEG fixture bytes"
    path = tmp_path / "fixture.jpg"
    path.write_bytes(raw)
    verified = []

    class Reader:
        def __init__(self, root, checksum, role, *, deadline):
            verified.append(role)

        def verify(self):
            verified.append("hash")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(c.data, "Reader", Reader)
    row = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    manifest = {"samples": {"fit": [row], "development": [row]}}
    config = {"root": tmp_path, "manifest_sha256": SHA}
    c.verify_data(config, manifest)
    assert verified == ["fit", "hash", "development", "hash"]
    path.write_bytes(b"modified")
    with pytest.raises(ValueError):
        c.verify_data(config, manifest)


def test_invalid_worker_shape_is_unavailable_without_training(matrix):
    report = c.execute(
        matrix.config,
        matrix.root / "invalid",
        sources=SOURCES,
        current_sources=lambda: SOURCES,
        inspect=lambda _: EXECUTION,
        launch=lambda *args: [],
    )
    assert report["status"] == "unavailable" and report["verified_updates"] == 0


def test_output_path_cannot_enter_data_through_parent_components(matrix):
    root = matrix.root / "data"
    root.mkdir()
    (matrix.root / "other").mkdir()
    with pytest.raises(ValueError, match="cannot modify"):
        matrix.execute(matrix.root / "other" / ".." / "data" / "new-output")
    assert not (root / "new-output").exists()
