"""Generated reader fixtures and mocked CUDA are never physical timing evidence."""

import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core import jpeg_timing, srnet_cuda, srnet_diversity_sampling
from core import jpeg_timing_probe as probe
from tests.test_jpeg_timing import samples


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "kit"
    (root / "jpeg").mkdir(parents=True)
    rows = samples("ALASKA2", 32) + samples("BOSSbase-1.01", 32) + samples("BOWS2", 32)
    for row in rows:
        raw = row["path"].encode()
        row["path"] = f"jpeg/{row['sha256']}.jpg"
        (root / row["path"]).write_bytes(raw)
    batches, schedule = srnet_diversity_sampling.epoch_batches(rows, seed=jpeg_timing.SEED, epoch=0)
    chunk = bytes(1024**2)
    checksum = hashlib.sha256()
    with (root / "pixels.f32").open("xb") as stream:
        for _ in range(120):
            stream.write(chunk)
            checksum.update(chunk)
    manifest = {
        "schema_version": "jpeg-real-timing-kit-v1",
        "status": "completed",
        "protocol_sha256": "a" * 64,
        "shape": [480, 1, 256, 256],
        "dtype": "<f4",
        "tensor_bytes": 480 * probe.ROW_BYTES,
        "tensor_sha256": checksum.hexdigest(),
        "samples": rows,
        "schedule": schedule,
    }
    proof = {
        "schema_version": "jpeg-real-timing-independent-audit-v1",
        "status": "completed",
        "protocol_sha256": "a" * 64,
        "independent_IDCT_rows": 480,
        "validation_pixels_read": 0,
    }
    audit = tmp_path / "audit.json"

    def save():
        (root / "manifest.json").write_text(json.dumps(manifest))
        digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        monkeypatch.setattr(probe, "MANIFEST_SHA", digest)
        proof["manifest_sha256"] = digest
        audit.write_text(json.dumps(proof))
        monkeypatch.setattr(probe, "AUDIT_SHA", hashlib.sha256(audit.read_bytes()).hexdigest())

    save()
    try:
        yield SimpleNamespace(
            root=root, audit=audit, manifest=manifest, proof=proof, save=save, batches=batches
        )
    finally:
        # Reproducible zero tensors are task-owned fixtures, not user evidence.
        # Do not retain 120 MiB for every test in a RAM-backed temporary folder.
        for name in ("pixels.f32", "original.f32"):
            (root / name).unlink(missing_ok=True)


def test_reader_bounded_batches_integrity_and_closed_state(prepared):
    with probe.TimingReader(prepared.root, prepared.audit, deadline=lambda: None) as reader:
        assert reader.batches.shape == (66, 4) and not reader.batches.flags.writeable
        values = reader.batch(reader.batches[0])
        assert (
            values.shape == (4, 1, 256, 256) and values.nbytes == reader.max_batch_bytes == 1024**2
        )
        assert not np.any(values)
        reader.verify()
        for indices in (
            [0, 1, 2, 3],
            np.array([-1, 1, 2, 3]),
            np.array([480, 1, 2, 3]),
            np.array([0, 1]),
            np.array([0.0, 1.0, 2.0, 3.0]),
        ):
            with pytest.raises(ValueError, match="batch"):
                reader.batch(indices)
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.verify()
    with pytest.raises(ValueError, match="batch"):
        reader.batch(np.arange(4))


@pytest.mark.parametrize(
    "fault",
    [
        "audit",
        "shape",
        "sources",
        "schedule",
        "jpeg-path",
        "jpeg-bytes",
        "tensor-size",
        "tensor-hash",
        "tensor-nan",
        "tensor-link",
        "mutated",
    ],
)
def test_reader_adversarial_inputs_fail_closed(prepared, fault):
    root = prepared.root
    if fault == "audit":
        prepared.proof["independent_IDCT_rows"] = 479
    elif fault == "shape":
        prepared.manifest["shape"][0] = 479
    elif fault == "sources":
        prepared.manifest["samples"][0]["source_group"] = "other"
    elif fault == "schedule":
        prepared.manifest["schedule"] = {}
    elif fault == "jpeg-path":
        prepared.manifest["samples"][0]["path"] = "../escape"
    elif fault == "jpeg-bytes":
        (root / prepared.manifest["samples"][0]["path"]).write_bytes(b"bad")
    elif fault == "tensor-hash":
        prepared.manifest["tensor_sha256"] = "b" * 64
    elif fault == "tensor-size":
        (root / "pixels.f32").write_bytes(b"short")
    elif fault == "tensor-nan":
        with (root / "pixels.f32").open("r+b") as stream:
            stream.write(np.array([np.nan], dtype="<f4").tobytes())
    elif fault == "tensor-link":
        (root / "pixels.f32").rename(root / "original.f32")
        (root / "pixels.f32").symlink_to(root / "original.f32")
    prepared.save()
    if fault == "mutated":
        with probe.TimingReader(root, prepared.audit, deadline=lambda: None) as reader:
            with (root / "pixels.f32").open("r+b") as stream:
                stream.write(b"bad!")
            with pytest.raises(ValueError, match="mutated"):
                reader.verify()
            with pytest.raises(ValueError, match="state"):
                reader.batch(np.arange(4))
    else:
        with pytest.raises(ValueError):
            probe.TimingReader(root, prepared.audit, deadline=lambda: None)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [1] * 46,
        [0] * 47,
        [True] * 47,
        [float("inf")] * 47,
        [float("nan")] * 47,
        [-1] * 47,
        [1e308] * 47,
    ],
)
def test_projection_rejects_partial_nonfinite_and_overflow(value):
    with pytest.raises(ValueError):
        probe.projection(value)


def test_projection_is_fixed_and_no_invented_hourly_price():
    record = probe.projection(list(range(1, 48)))
    assert record["median_update_seconds"] == 24
    assert record["p95_update_seconds"] == pytest.approx(44.7)
    assert record["estimated_optimizer_seconds_per_epoch"] == pytest.approx(44.7 * 4932 * 1.25)
    assert record["estimate_only"] and not record["full_corpus_reader_integrated"]
    assert "disk_charges" in record["excluded"]
    assert "cost" not in record


@pytest.mark.parametrize("count,updates", [(66, 66), (65, 66), (66, 65)])
def test_mocked_profile_control_flow(prepared, monkeypatch, count, updates):
    torch = pytest.importorskip("torch")
    ticks = itertools.count(1, 0.01)
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(srnet_cuda, "inspect", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 2048)

    def learn(**kw):
        assert kw["device"] == "cuda:0" and kw["target_pairs"] == 2 and kw["epochs"] == 1
        assert kw["batches"](0).shape == (66, 4)
        for indices in kw["batches"](0)[:count]:
            assert kw["fetch"](indices).shape == (4, 1, 256, 256)
        return None, [{"updates": updates}]

    monkeypatch.setattr(probe.srnet_training, "_learn", learn)
    if count != updates or count != 66:
        with pytest.raises(ValueError, match="accounting"):
            probe.profile(prepared.root, prepared.audit)
    else:
        report = probe.profile(prepared.root, prepared.audit)
        assert (
            report["real_optimizer_updates"] == 66 and len(report["steady_intervals_seconds"]) == 47
        )
        assert report["real_data_used"] and report["real_model_trained"]
        assert (
            not report["production_model_trained"]
            and report["accuracy_qualification"] == "unavailable"
        )


def test_cuda_absence_precedes_dataset_access(tmp_path, monkeypatch):
    def unavailable():
        raise srnet_cuda.CUDAUnavailable("cuda_not_visible_or_cpu_torch")

    monkeypatch.setattr(srnet_cuda, "inspect", unavailable)
    with pytest.raises(srnet_cuda.CUDAUnavailable):
        probe.profile(tmp_path / "missing", tmp_path / "missing-audit")
