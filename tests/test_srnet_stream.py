"""Independent generated tensor fixtures; never evidence of detector accuracy."""

import copy
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from core import jpeg_float256, srnet_multibatch, srnet_sampling, srnet_training
from core import srnet_scale_sampling as sampling
from core import srnet_stream as stream
from core import srnet_stream_training as training
from steganography import research_srnet_stream as service


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data))
    return digest(path)


@pytest.fixture
def corpus(tmp_path):
    pytest.importorskip("jpeglib")
    root = tmp_path / "blocks"
    root.mkdir()
    blocks, all_rows, tensors = [], [], []
    for source_no, source in enumerate(("A", "B")):
        folder = root / f"block-{source_no:03}"
        folder.mkdir()
        rows = []
        for split in ("train", "validation"):
            lineage = hashlib.sha256(f"{source}-{split}".encode()).hexdigest()
            for quality in (None,) if source == "A" else (75, 95):
                for method in (None, "JUNIWARD", "UERD"):
                    rows.append(
                        {
                            "sha256": hashlib.sha256(
                                f"{source}-{split}-{quality}-{method}".encode()
                            ).hexdigest(),
                            "lineage": lineage,
                            "source_group": source,
                            "split": split,
                            "quality_factor": quality,
                            "format": "JPEG",
                            "label": "cover" if method is None else "stego",
                            "method": method,
                        }
                    )
        manifest_sha = save(folder / "manifest.json", {"schema_version": "1.0", "samples": rows})
        chosen = [r for r in rows if r["split"] == "train"]
        cache_dir = folder / "train"
        cache_dir.mkdir()
        values = np.empty((len(chosen), 1, 256, 256), dtype="<f4")
        for i, row in enumerate(values):
            row.fill(128.125 + i + source_no)
        tensors.extend(values.copy())
        (cache_dir / "pixels.f32").write_bytes(values.tobytes())
        data_sha = digest(cache_dir / "pixels.f32")
        cache_sha = save(
            cache_dir / "cache.json",
            {
                "schema_version": "research-float256-cache-v1",
                "feature_version": jpeg_float256.FEATURE_VERSION,
                "manifest_sha256": manifest_sha,
                "split": "train",
                "dtype": "<f4",
                "shape": [len(chosen), 1, 256, 256],
                "decoder": jpeg_float256.decoder_contract(),
                "data_sha256": data_sha,
                "rows": [
                    {
                        **{k: r[k] for k in ("sha256", "lineage", "label")},
                        "image_size": [512, 512],
                        "region": [128, 128, 256, 256],
                    }
                    for r in chosen
                ],
            },
        )
        blocks.append(
            {
                "path": f"{folder.name}/manifest.json",
                "manifest_sha256": manifest_sha,
                "source_group": source,
                "lineages": sorted({r["lineage"] for r in rows}),
                "caches": {
                    "train": {
                        "path": f"{folder.name}/train/cache.json",
                        "sha256": cache_sha,
                        "data_sha256": data_sha,
                        "rows": len(chosen),
                    }
                },
            }
        )
        all_rows.extend(rows)
    index = {
        "schema_version": "jpeg-scale-blocks-v1",
        "status": "completed",
        "protocol_sha256": "1" * 64,
        "blocks": blocks,
        "jpeg_rows": len(all_rows),
        "original_lineages": 4,
        "splits": dict(Counter(r["split"] for r in all_rows)),
    }
    index_sha = save(root / "index.json", index)
    audit_path = root / "audit.json"
    audit_sha = save(
        audit_path,
        {
            "schema_version": "jpeg-scale-independent-audit-v1",
            "status": "completed",
            "index_sha256": index_sha,
            "protocol_sha256": index["protocol_sha256"],
            "blocks_audited": 2,
            "jpeg_rows": len(all_rows),
            "original_lineages": 4,
            "splits": index["splits"],
            "reserved_identity_overlap": 0,
        },
    )
    config = {
        "root": str(root),
        "index_sha256": index_sha,
        "audit": str(audit_path),
        "audit_sha256": audit_sha,
        "epochs": 1,
        "seed": 91,
        "threads": 1,
        "max_seconds": 180,
    }
    return config, np.asarray(tensors), index


def open_reader(config, deadline=lambda: None):
    return stream.TrainBlocks(
        Path(config["root"]),
        index_sha256=config["index_sha256"],
        audit=Path(config["audit"]),
        audit_sha256=config["audit_sha256"],
        deadline=deadline,
    )


def rebind(config, index):
    config["index_sha256"] = save(Path(config["root"]) / "index.json", index)
    audit_path = Path(config["audit"])
    audit = json.loads(audit_path.read_bytes())
    audit["index_sha256"] = config["index_sha256"]
    config["audit_sha256"] = save(audit_path, audit)


def edit_cache(config, index, edit):
    block = index["blocks"][0]
    path = Path(config["root"]) / block["caches"]["train"]["path"]
    doc = json.loads(path.read_bytes())
    edit(doc, path)
    block["caches"]["train"]["sha256"] = save(path, doc)
    rebind(config, index)


def test_complete_cross_block_exact_pixels_train_only(corpus, monkeypatch):
    config, expected, _ = corpus
    opened = []
    original = stream.regular_open

    def record(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(stream, "regular_open", record)
    with open_reader(config) as reader:
        assert len(reader.samples) == 9 and reader.train_bytes == expected.nbytes
        indices = np.array([8, 0, 3, 4])
        actual = reader.batch(indices)
        np.testing.assert_array_equal(actual, expected[indices])
        actual.fill(0)
        np.testing.assert_array_equal(reader.batch(indices), expected[indices])
        assert reader.max_batch_bytes == 1048576
        handles = [r[0] for r in reader._locations]
    assert all(s.closed for s in handles)
    assert not any("/validation/" in p for p in opened)
    with pytest.raises(ValueError):
        reader.batch(indices)


@pytest.mark.parametrize(
    "indices",
    [
        [],
        [0],
        np.array([], dtype=int),
        np.array([-1]),
        np.array([9]),
        np.zeros(5, dtype=int),
        np.array([True]),
        np.array([0.0]),
        np.array([[0]]),
    ],
)
def test_invalid_batch_limits(corpus, indices):
    with open_reader(corpus[0]) as reader, pytest.raises(ValueError):
        reader.batch(indices)


@pytest.mark.parametrize(
    "fault",
    [
        "index-sha",
        "audit-sha",
        "audit-status",
        "index-status",
        "schema",
        "blocks-empty",
        "blocks-many",
        "overlap",
        "splits",
        "rows",
        "lineages",
        "path",
        "membership",
        "cache-path",
        "block-row-limit",
        "bad-role",
        "bad-source",
        "cross-role",
        "duplicate",
        "row-object",
        "bad-family",
        "manifest-schema",
    ],
)
def test_reject_index_and_lineage_forgery(corpus, fault):
    config, _, index = corpus
    if fault in {"index-sha", "audit-sha"}:
        config["index_sha256" if fault == "index-sha" else "audit_sha256"] = "0" * 64
    elif fault in {"audit-status", "overlap"}:
        path = Path(config["audit"])
        audit = json.loads(path.read_bytes())
        audit["status" if fault == "audit-status" else "reserved_identity_overlap"] = (
            "failed" if fault == "audit-status" else 1
        )
        config["audit_sha256"] = save(path, audit)
    else:
        block = index["blocks"][0]
        if fault == "index-status":
            index["status"] = "partial"
        if fault == "schema":
            index["schema_version"] = "wrong"
        if fault == "blocks-empty":
            index["blocks"] = []
        if fault == "blocks-many":
            index["blocks"] *= 17
        if fault == "splits":
            index["splits"]["train"] -= 1
        if fault == "rows":
            index["jpeg_rows"] -= 1
        if fault == "lineages":
            index["original_lineages"] -= 1
        if fault == "path":
            block["path"] = "../escape.json"
        if fault == "membership":
            block["lineages"] = []
        if fault == "cache-path":
            block["caches"]["train"]["path"] = "block-001/train/cache.json"
        if fault in {
            "block-row-limit",
            "bad-role",
            "bad-source",
            "cross-role",
            "duplicate",
            "row-object",
            "bad-family",
            "manifest-schema",
        }:
            path = Path(config["root"]) / block["path"]
            manifest = json.loads(path.read_bytes())
            rows = manifest["samples"]
            if fault == "block-row-limit":
                manifest["samples"] = rows * 129
            if fault == "bad-role":
                rows[0]["split"] = "test"
            if fault == "bad-source":
                rows[0]["source_group"] = "unknown"
            if fault == "cross-role":
                rows[0]["split"] = "validation"
            if fault == "duplicate":
                rows[1]["sha256"] = rows[0]["sha256"]
            if fault == "row-object":
                rows[0] = None
            if fault == "bad-family":
                rows[1]["method"] = "JMiPOD"
            if fault == "manifest-schema":
                manifest["schema_version"] = "bad"
            block["manifest_sha256"] = save(path, manifest)
        rebind(config, index)
    with pytest.raises(ValueError):
        open_reader(config)


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "split",
        "dtype",
        "decoder",
        "identity",
        "region",
        "size",
        "nan",
        "magnitude",
        "truncated",
        "data-hash",
        "rows",
        "rows-count",
        "binding-count",
    ],
)
def test_reject_cache_contract_and_numeric_forgery(corpus, fault):
    config, _, index = corpus

    def edit(doc, path):
        if fault == "shape":
            doc["shape"][0] -= 1
        if fault == "split":
            doc["split"] = "validation"
        if fault == "dtype":
            doc["dtype"] = "uint8"
        if fault == "decoder":
            doc["decoder"]["rounding"] = "round"
        if fault == "identity":
            doc["rows"][0]["sha256"] = "0" * 64
        if fault == "region":
            doc["rows"][0]["region"][0] += 1
        if fault == "size":
            doc["rows"][0]["image_size"] = [True, 512]
        if fault == "rows":
            doc["rows"][0] = None
        if fault == "rows-count":
            doc["rows"] = []
        if fault == "binding-count":
            index["blocks"][0]["caches"]["train"]["rows"] -= 1
        if fault in {"nan", "magnitude", "truncated", "data-hash"}:
            tensor = path.parent / "pixels.f32"
            raw = bytearray(tensor.read_bytes())
            if fault == "truncated":
                raw.pop()
            elif fault != "data-hash":
                raw[:4] = np.float32(np.nan if fault == "nan" else 2**37).tobytes()
            else:
                raw[:4] = np.float32(12).tobytes()
            tensor.write_bytes(raw)
            if fault != "data-hash":
                doc["data_sha256"] = digest(tensor)
                index["blocks"][0]["caches"]["train"]["data_sha256"] = doc["data_sha256"]

    edit_cache(config, index, edit)
    with pytest.raises(ValueError):
        open_reader(config)


def test_mutation_symlink_fifo_and_failure_closes_handles(corpus, monkeypatch, tmp_path):
    config, _, _ = corpus
    tensor = Path(config["root"]) / "block-000/train/pixels.f32"
    with open_reader(config) as reader:
        with tensor.open("r+b") as f:
            f.write(np.float32(3).tobytes())
        with pytest.raises(ValueError, match="mutated"):
            reader.batch(np.array([0]))
    target = tmp_path / "regular"
    target.write_bytes(b"okay")
    link = tmp_path / "symlink"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        stream.regular_open(link)
    import os

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        stream.regular_open(fifo)
    handles = []
    original = stream.regular_open

    def record(path):
        handle = original(path)
        handles.append(handle)
        return handle

    monkeypatch.setattr(stream, "regular_open", record)
    with pytest.raises(ValueError):
        open_reader(config)
    assert all(h.closed for h in handles)


def test_shared_scaled_schedule_covers_more_than_legacy_cap():
    samples = []
    for source in ("A", "B"):
        for i in range(750):
            for method in (None, "JUNIWARD", "UERD"):
                samples.append(
                    {
                        "source_group": source,
                        "lineage": f"{source}-{i}",
                        "sha256": hashlib.sha256(str(len(samples)).encode()).hexdigest(),
                        "quality_factor": None,
                        "split": "train",
                        "format": "JPEG",
                        "label": "cover" if method is None else "stego",
                        "method": method,
                    }
                )
    batches, record = sampling.epoch_batches(samples, seed=91, epoch=0)
    assert len(samples) == 4500 and batches.shape == (1500, 4)
    assert set(batches.flatten()) == set(range(4500))
    assert record["pair_schedule"]["rows"] == 6000
    with pytest.raises(ValueError, match="row limit"):
        srnet_sampling.epoch_pairs(samples, seed=91, epoch=0)
    with pytest.raises(ValueError, match="row limit"):
        sampling.epoch_batches(samples * 3, seed=91, epoch=0)


def test_actual_shared_optimizer_stream_equals_legacy(corpus):
    torch = pytest.importorskip("torch")
    config, values, _ = corpus
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    with open_reader(config) as reader:
        samples = copy.deepcopy(reader.samples)
        records = [sampling.epoch_batches(samples, seed=91, epoch=0)[1]]
        new, new_records = training.fit(
            reader,
            seed=91,
            schedule=records,
            config={"threads": 1, "max_seconds": 180},
            deadline=lambda: None,
        )
    old, old_records = srnet_training.fit(
        values,
        samples,
        np.arange(len(samples)),
        seed=91,
        schedule=[srnet_sampling.epoch_pairs(samples, seed=91, epoch=0)[1]],
        config={"threads": 1, "max_seconds": 180, "batch_recipe": srnet_multibatch.RECIPE},
    )
    assert new_records == old_records and new_records[0]["updates"] == 4
    for key, value in new.state_dict().items():
        assert torch.equal(value, old.state_dict()[key]), key
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()
    with open_reader(config) as reader:
        with pytest.raises(ValueError, match="schedule"):
            training.fit(reader, seed=91, schedule=[{}], config={}, deadline=lambda: None)
        with pytest.raises(ValueError, match="inputs"):
            training.fit(reader, seed=91, schedule=[], config={}, deadline=lambda: None)

        def expired():
            raise ValueError("expired")

        with pytest.raises(ValueError, match="expired"):
            training.fit(reader, seed=91, schedule=records, config={}, deadline=expired)
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()


@pytest.mark.parametrize("seconds", [0, 1801, True, 1.0])
def test_invalid_deadline(seconds):
    with pytest.raises(ValueError):
        stream.deadline_after(seconds)


def test_propagated_deadline_before_load_during_hash_and_batch(corpus, monkeypatch):
    def expired():
        raise ValueError("expired")

    with pytest.raises(ValueError, match="expired"):
        open_reader(corpus[0], expired)
    ticks = iter([0, 2])
    monkeypatch.setattr(stream.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="deadline"):
        stream.deadline_after(1)()
    checks = 0

    def during():
        nonlocal checks
        checks += 1
        if checks == 4:
            raise ValueError("expired")

    with pytest.raises(ValueError, match="expired"):
        open_reader(corpus[0], during)
    with open_reader(corpus[0]) as reader:
        reader._deadline = expired
        with pytest.raises(ValueError, match="expired"):
            reader.batch(np.array([0]))


def test_service_plan_readiness_and_real_isolated_cli(corpus, tmp_path, capsys):
    config, _, _ = corpus
    plan = service.execute(config, tmp_path / "plan.json", operation="plan")
    assert plan["train_rows"] == 9 and plan["original_train_lineages"] == 2
    report = service.execute(config, tmp_path / "check.json", operation="check")
    assert report["plan"] == plan and report["unique_train_rows_read"] == 9
    assert report["row_presentations"] == 16 and report["planned_optimizer_updates"] == 4
    assert not report["models_trained"] and not report["validation_pixels_opened"]
    assert str(tmp_path) not in json.dumps(report)
    config_path = tmp_path / "config.json"
    save(config_path, config)
    assert (
        service.main(
            [
                "--operation",
                "check",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "isolated.json"),
            ]
        )
        == 0
    )
    assert "completed" in capsys.readouterr().out
    assert json.loads((tmp_path / "isolated.json").read_bytes())["plan"] == plan
    with pytest.raises(FileExistsError):
        service.execute(config, tmp_path / "plan.json", operation="plan")


@pytest.mark.parametrize(
    "edit",
    [
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 51},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"index_sha256": "bad"},
        {"root": ""},
        {"validation_cache": "secret"},
        {"threads": 3},
    ],
)
def test_invalid_configuration(corpus, edit):
    with pytest.raises(ValueError):
        service.configuration({**corpus[0], **edit})


def test_training_service_bound_plan_actual_fit_and_reload(corpus, tmp_path):
    pytest.importorskip("torch")
    config, _, _ = corpus
    path = tmp_path / "plan.json"
    service.execute(config, path, operation="plan")
    config_path = tmp_path / "fit-config.json"
    save(config_path, config)
    card = service.run_job(
        config_path, tmp_path / "fit", operation="fit", plan_path=path, plan_sha256=digest(path)
    )
    assert card["training"] == "completed" and not card["validation_used"] and not card["deployed"]
    assert card["epoch_training"][0]["updates"] == 4
    assert card["accuracy_qualification"] == "unavailable" and str(tmp_path) not in json.dumps(card)
    model = service.srnet_model.load_model(
        tmp_path / "fit/model.npz", checksum=card["model_sha256"]
    )
    assert all(int(v) == 4 for k, v in model.named_buffers() if k.endswith("num_batches_tracked"))
    with pytest.raises(ValueError, match="plan"):
        service.execute(config, tmp_path / "bad-fit", operation="fit")
    fake = json.loads(path.read_bytes())
    fake["seed"] += 1
    save(path, fake)
    with pytest.raises(ValueError, match="plan/provenance"):
        service.execute(
            config,
            tmp_path / "wrong-plan",
            operation="fit",
            plan_path=path,
            plan_sha256=digest(path),
        )


def test_source_snapshot_changes_fail_closed(corpus, tmp_path, monkeypatch):
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        return {"source": calls}

    monkeypatch.setattr(service, "snapshot", changed)
    out = tmp_path / "report.json"
    with pytest.raises(ValueError, match="dependencies changed"):
        service.execute(corpus[0], out, operation="check")
    assert not out.exists()


@pytest.mark.parametrize(
    "response", [b'{"status":"failed"}', b"[]", b"{}", b"not json", b"x" * 65537]
)
def test_worker_failure_redacted(corpus, tmp_path, monkeypatch, response):
    config_path = tmp_path / "config.json"
    save(config_path, corpus[0])

    def failed(*args, **kwargs):
        kwargs["stdout"].write(response)
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(service.subprocess, "run", failed)
    assert (
        service.main(
            [
                "--operation",
                "check",
                "--config",
                str(config_path),
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )


def test_worker_timeout_and_artifact_forgery(corpus, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    save(config_path, corpus[0])

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(service.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="unusable"):
        service.run_job(config_path, tmp_path / "out.json", operation="check")

    def fake(*args, **kwargs):
        kwargs["stdout"].write(
            json.dumps({"status": "completed", "report_sha256": "0" * 64}).encode()
        )
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(service.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="unusable"):
        service.run_job(config_path, tmp_path / "out.json", operation="check")


def test_document_limits_object_and_symlink_output(corpus, tmp_path, monkeypatch):
    path = tmp_path / "doc.json"
    save(path, [])
    with pytest.raises(ValueError, match="object"):
        stream.document(path, digest(path))
    monkeypatch.setattr(stream, "MAX_DOCUMENT", 1)
    with pytest.raises(ValueError, match="size"):
        stream.document(path, digest(path))
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        service.execute(corpus[0], link, operation="plan")
    with pytest.raises(ValueError):
        open_reader(corpus[0], deadline=None)
    with pytest.raises(ValueError):
        service.execute(corpus[0], tmp_path / "bad", operation="other")


def test_worker_limits_redaction(corpus, tmp_path, monkeypatch, capsys):
    import resource

    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda key, value: limits.append((key, value)))
    config_path = tmp_path / "config.json"
    save(config_path, corpus[0])
    args = [
        "--worker",
        "--operation",
        "plan",
        "--config",
        str(config_path),
        "--config-sha256",
        digest(config_path),
        "--out",
        str(tmp_path / "plan.json"),
    ]
    assert service.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3)) in limits
    assert (resource.RLIMIT_CPU, (390, 391)) in limits
    assert (resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2)) in limits
    assert (resource.RLIMIT_CORE, (0, 0)) in limits

    def failed(*a, **kw):
        raise ValueError("host path secret")

    monkeypatch.setattr(service, "execute", failed)
    assert service.main(args) == 2
    assert capsys.readouterr().out == '{"status":"failed"}\n'


def test_dependency_limit_incomplete_readiness_and_bytes(corpus, tmp_path, monkeypatch):
    import io

    monkeypatch.setattr(service, "regular_open", lambda _: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError, match="dependency size"):
        service.snapshot()
    monkeypatch.undo()
    monkeypatch.setattr(stream, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        open_reader(corpus[0])
    monkeypatch.undo()
    monkeypatch.setattr(stream, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="corpus row limit"):
        open_reader(corpus[0])
    monkeypatch.undo()
    original = service.srnet_scale_sampling.epoch_batches

    def omit(*a, **kw):
        batches, record = original(*a, **kw)
        return batches[:1], record

    monkeypatch.setattr(service.srnet_scale_sampling, "epoch_batches", omit)
    with pytest.raises(ValueError, match="omitted"):
        service.execute(corpus[0], tmp_path / "incomplete.json", operation="check")
    assert not (tmp_path / "incomplete.json").exists()


def test_optimizer_outer_deadline_restores_rng_and_threads(corpus):
    torch = pytest.importorskip("torch")
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    calls = 0

    def expired_at_optimizer():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("optimizer job expired")

    with open_reader(corpus[0]) as reader:
        schedule = [sampling.epoch_batches(reader.samples, seed=91, epoch=0)[1]]
        with pytest.raises(ValueError, match="optimizer job expired"):
            training.fit(
                reader,
                seed=91,
                schedule=schedule,
                config={"threads": 1},
                deadline=expired_at_optimizer,
            )
    assert torch.equal(rng, torch.get_rng_state()) and threads == torch.get_num_threads()


def test_direct_fit_publication_and_post_save_source_mutation(corpus, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    config = corpus[0]
    plan_path = tmp_path / "plan.json"
    service.execute(config, plan_path, operation="plan")
    card = service.execute(
        config,
        tmp_path / "direct-fit",
        operation="fit",
        plan_path=plan_path,
        plan_sha256=digest(plan_path),
    )
    assert card["epoch_training"][0]["updates"] == 4
    model = service.srnet_model.load_model(
        tmp_path / "direct-fit/model.npz", checksum=card["model_sha256"]
    )
    # This arm tests fail-closed publication, not another learning claim.
    monkeypatch.setattr(
        service.srnet_stream_training, "fit", lambda *a, **kw: (model, card["epoch_training"])
    )
    original = service.snapshot
    calls = 0

    def changed_after_save():
        nonlocal calls
        calls += 1
        result = original()
        if calls == 3:
            result["core/srnet.py"] = "0" * 64
        return result

    monkeypatch.setattr(service, "snapshot", changed_after_save)
    with pytest.raises(ValueError, match="before publication"):
        service.execute(
            config,
            tmp_path / "partial-fit",
            operation="fit",
            plan_path=plan_path,
            plan_sha256=digest(plan_path),
        )
    assert (tmp_path / "partial-fit/model.npz").exists()
    assert not (tmp_path / "partial-fit/model-card.json").exists()
