"""Independent D4/IDCT, bounded caches and mocked GPU orchestration."""

import hashlib
import io
import json
from argparse import Namespace
from contextlib import nullcontext

import numpy as np
import pytest
from PIL import Image

from core import jpeg_curriculum as c
from core.srnet_stream import ROW_BYTES
from steganography import research_jpeg_curriculum as front


def samples():
    return [
        {
            "source_group": source,
            "lineage": f"{source}-{i:02}",
            "quality_factor": q,
            "method": m,
            "label": "cover" if m is None else "stego",
            "split": "train",
            "payload_rate": None if m is None else 0.2,
            "payload_unit": "bpnzAC",
        }
        for source, qualities in (
            ("ALASKA2", (None,)),
            ("BOSSbase-1.01", (75, 95)),
            ("BOWS2", (75, 95)),
        )
        for i in range(32)
        for q in qualities
        for m in (None, "JUNIWARD", "UERD")
    ]


def test_selection_and_complete_deterministic_exposure():
    rows = samples()
    pairs, probe = c.selection(rows)
    assert pairs.shape == (48, 2) and probe.shape == (30, 4)
    assert set(pairs.flat).isdisjoint(probe.flat)
    assert {rows[i]["source_group"] for i in pairs.flat} == {"BOSSbase-1.01", "BOWS2"}
    for epoch in range(40):
        schedule = c.schedule(pairs, epoch)
        assert schedule.shape == (96, 4)
        np.testing.assert_array_equal(schedule, c.schedule(pairs, epoch))
        indices, counts = np.unique(schedule, return_counts=True)
        np.testing.assert_array_equal(indices, np.sort(pairs.reshape(-1)))
        assert (counts == 4).all()
        assert [rows[i]["label"] for i in schedule[0]] == ["cover", "stego"] * 2
    for p, e in ((pairs[:2], 0), (pairs, 40), (pairs, True), (pairs.astype(np.int32), 0)):
        with pytest.raises(ValueError):
            c.schedule(p, e)
    rows[97]["payload_rate"] = 0.4
    with pytest.raises(ValueError):
        c.selection(rows)
    rows = samples()
    rows[96]["label"] = "stego"
    with pytest.raises(ValueError):
        c.selection(rows)


@pytest.mark.parametrize("orientation", range(8))
def test_d4_independent_coordinate_mapping_keeps_pairs(orientation):
    base = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
    values = np.stack([base + i for i in range(4)])[:, None]
    actual = c.augment(values, orientation)
    expected = Image.fromarray(base)
    if orientation >= 4:
        expected = expected.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    for _ in range(orientation % 4):
        expected = expected.transpose(Image.Transpose.ROTATE_90)
    np.testing.assert_array_equal(actual[0, 0], np.asarray(expected))
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual[1] - actual[0], np.ones((1, 256, 256), np.float32))
    for bad in (-1, 8, True):
        with pytest.raises(ValueError):
            c.augment(values, bad)
    with pytest.raises(ValueError):
        c.augment(values.astype(np.float64), 0)


@pytest.fixture
def jpeg_pair(tmp_path):
    import jpeglib

    cover = tmp_path / "cover.jpg"
    Image.fromarray(np.random.default_rng(7).integers(0, 256, (512, 512), dtype=np.uint8)).save(
        cover, quality=75
    )

    def make(kind="ok"):
        item = jpeglib.read_dct(str(cover))
        item.load()
        item.Y[20, 20, 0 if kind == "dc" else 1, 1 if kind != "dc" else 0] += (
            2 if kind == "large" else 1
        )
        if kind == "table":
            item.qt[0, 1, 1] += 1
        path = tmp_path / f"{kind}.jpg"
        item.write_dct(str(path))
        return cover.read_bytes(), path.read_bytes()

    return make


def test_actual_pair_has_independent_idct_and_coefficient_checks(jpeg_pair):
    values, changes, maximum = c.checked_pixels(*jpeg_pair())
    assert values.shape == (2, 1, 256, 256) and changes == 1 and maximum <= 1e-4


@pytest.mark.parametrize("kind", ["large", "table", "identical", "outside"])
def test_bad_real_pairs_fail_without_filtering(jpeg_pair, kind, tmp_path):
    cover, stego = jpeg_pair(kind if kind in {"dc", "large", "table"} else "ok")
    if kind == "identical":
        stego = cover
    if kind == "outside":
        import jpeglib

        path = tmp_path / "outside.jpg"
        item = jpeglib.read_dct(str(tmp_path / "cover.jpg"))
        item.load()
        item.Y[0, 0, 1, 1] += 1
        item.write_dct(str(path))
        stego = path.read_bytes()
    with pytest.raises(ValueError):
        c.checked_pixels(cover, stego)


def cache(tmp_path, pairs):
    raw = np.zeros((96, 1, 256, 256), dtype="<f4").tobytes()
    (tmp_path / "pixels.f32").write_bytes(raw)
    report = {
        "schema_version": "jpeg-q75-curriculum-cache-v1",
        "status": "completed",
        "manifest_sha256": c.MANIFEST_SHA,
        "audit_sha256": c.AUDIT_SHA,
        "payload_rate": 0.4,
        "payload_unit": "bpnzAC",
        "quality_factor": 75,
        "method": "JUNIWARD",
        "tensor_bytes": 96 * ROW_BYTES,
        "probe_pixels_read": 0,
        "independent_idct_rows": 96,
        "rows": [{"indices": p.tolist()} for p in pairs],
        "tensor_sha256": hashlib.sha256(raw).hexdigest(),
    }
    encoded = json.dumps(report).encode()
    (tmp_path / "manifest.json").write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "failure", [None, "size", "checksum", "nan", "symlink", "metadata", "mutation", "batch"]
)
def test_high_reader_security_and_exclusive_lifecycle(tmp_path, failure):
    pairs, _ = c.selection(samples())
    checksum = cache(tmp_path, pairs)
    path = tmp_path / "pixels.f32"
    if failure == "size":
        path.write_bytes(b"short")
    if failure == "checksum":
        checksum = "0" * 64
    if failure == "nan":
        with path.open("r+b") as stream:
            stream.write(np.array([np.nan], dtype="<f4").tobytes())
    if failure == "symlink":
        path.rename(tmp_path / "old")
        path.symlink_to(tmp_path / "old")
    if failure == "metadata":
        pairs = pairs[::-1].copy()
    if failure in {"size", "checksum", "nan", "symlink", "metadata"}:
        with pytest.raises(ValueError):
            c.HighReader(tmp_path, checksum, pairs, lambda: None)
        return
    reader = c.HighReader(tmp_path, checksum, pairs, lambda: None)
    with reader:
        result = reader.batch(pairs[:2].reshape(4))
        assert result.shape == (4, 1, 256, 256)
        if failure == "mutation":
            with path.open("r+b") as stream:
                stream.write(b"xxxx")
            with pytest.raises(ValueError):
                reader.batch(pairs[:2].reshape(4))
            with pytest.raises(ValueError):
                reader.verify()
        if failure == "batch":
            for bad in (np.array([-1] * 4), pairs[0], np.array([0] * 4, dtype=np.int32)):
                with pytest.raises(ValueError):
                    reader.batch(bad)
        if failure is None:
            reader.verify()
    reader.close()
    with pytest.raises(ValueError):
        reader.verify()


def test_fixed_two_stage_control_keeps_weights_and_reports_all_contexts(monkeypatch, tmp_path):
    pytest.importorskip("torch", reason="optional research dependency unavailable")
    class Model:
        architecture = "mock-not-hardware"

        def to(self, *args):
            return self

        def eval(self):
            return self

    class Reader:
        samples = samples()

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def batch(self, indices):
            return np.zeros((4, 1, 256, 256), dtype="<f4")

        def verify(self):
            pass

    class High(Reader):
        def __init__(self, *args):
            pass

    monkeypatch.setattr(c, "TimingReader", Reader)
    monkeypatch.setattr(c, "HighReader", High)
    monkeypatch.setattr(c.srnet_cuda, "inspect", lambda: {"device": "mock-not-hardware"})
    monkeypatch.setattr(c.srnet_cuda, "policy", nullcontext)
    monkeypatch.setattr(c.srnet_samepool, "network", Model)
    counter = [0]
    monkeypatch.setattr(c.srnet_groupnorm, "parameter_sha", lambda model: str(counter[0]))
    seen = []

    def kernel(**kw):
        model = kw["factory"]()
        seen.append(model)
        if len(seen) == 2:
            assert seen[0] is model
        assert kw["epochs"] == 40 and kw["params"]["learning_rate"] == (
            0.001 if len(seen) == 1 else 0.0001
        )
        for epoch in (0, 39):
            batch = kw["batches"](epoch)
            assert batch.shape == (96, 4)
            assert kw["fetch"](batch[0]).flags.c_contiguous
        counter[0] += 1
        return model, [{"epoch": i, "updates": 96, "mean_pair_loss": 0.69} for i in range(40)]

    monkeypatch.setattr(c.srnet_norm_training, "kernel", kernel)
    monkeypatch.setattr(
        c.jpeg_population_control,
        "score",
        lambda m, r, p, d: ({int(i): np.array([0.0, 0.0]) for i in p.flat}, True),
    )
    report = c.fit(tmp_path, tmp_path, tmp_path, "a" * 64)
    assert report["optimizer_updates"] == 7680 and len(report["probe_cells"]) == 10
    assert all(cell["roc_auc"] == 0.5 for cell in report["probe_cells"])
    assert (
        report["stages"][1]["starting_parameter_sha256"]
        == report["stages"][0]["learned_parameter_sha256"]
    )
    assert report["probe_previously_inspected"] is True and report["validation_test_used"] is False
    assert report["model_discarded"] is True and report["deployed"] is False


def test_frontend_hashes_and_fresh_source_bound_outputs(tmp_path, monkeypatch):
    assert len(front.sources()) == 34
    args = Namespace(
        action="prepare",
        out=tmp_path / "out",
        root=tmp_path,
        audit=tmp_path,
        high=None,
        high_sha256=None,
    )

    def prepare(*args):
        args[2].mkdir()
        return {"status": "completed"}

    monkeypatch.setattr(c, "prepare", prepare)
    report = front.execute(args)
    assert report["source_sha256"] == front.sources()
    assert (args.out / "manifest.json").is_file() and (args.out / "report.json").is_file()
    with pytest.raises(FileExistsError):
        front.execute(args)
    args.action = "train"
    args.out = tmp_path / "new"
    with pytest.raises(ValueError):
        front.execute(args)
    args.high = tmp_path / "out"
    raw = (args.high / "manifest.json").read_bytes()
    args.high_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(c, "fit", lambda *args: {"status": "completed"})
    front.execute(args)
    args.out = tmp_path / "changed"
    monkeypatch.setattr(front, "sources", lambda: {})
    with pytest.raises(ValueError):
        front.execute(args)


@pytest.mark.parametrize("failure", [None, "identity", "output", "preprocessing", "exists"])
def test_preparation_reads_only_fixed_fit_rows_and_retains_failures(tmp_path, monkeypatch, failure):
    rows = samples()
    data = b"bounded-generated-cover"
    digest = hashlib.sha256(data).hexdigest()
    for r in rows:
        r.update(sha256=digest, size=len(data))
    (tmp_path / "jpeg").mkdir()
    (tmp_path / "jpeg" / (digest + ".jpg")).write_bytes(data)
    if failure == "identity":
        (tmp_path / "jpeg" / (digest + ".jpg")).write_bytes(b"wrong")
    seen = []

    class Reader:
        samples = rows

        def __init__(self, *args, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def batch(self, indices):
            seen.extend(indices.tolist())
            return np.full((4, 1, 256, 256), 1 if failure == "preprocessing" else 0, dtype="<f4")

        def verify(self):
            pass

    monkeypatch.setattr(c, "TimingReader", Reader)
    values = np.zeros((2, 1, 256, 256), dtype="<f4")
    values[1] = 1
    monkeypatch.setattr(c, "checked_pixels", lambda *args: (values, 123, 0.0))
    out = tmp_path / "out"
    if failure == "exists":
        out.mkdir()

    def generate(cover, lineage):
        return b"" if failure == "output" else b"bounded-stego"

    if failure is not None:
        with pytest.raises((ValueError, FileExistsError)):
            c.prepare(tmp_path, tmp_path, out, generate)
        return
    report = c.prepare(tmp_path, tmp_path, out, generate)
    assert len(report["rows"]) == 48 and report["independent_idct_rows"] == 96
    assert (out / "pixels.f32").stat().st_size == 96 * ROW_BYTES
    pairs, probe = c.selection(rows)
    assert set(seen) == set(pairs.flat) and set(seen).isdisjoint(probe.flat)
    assert report["probe_pixels_read"] == 0 and report["payload_rate"] == 0.4


def test_actual_bounded_juniward_worker_and_math_audit(jpeg_pair):
    pytest.importorskip("conseal", reason="optional JPEG simulator unavailable")
    cover, _ = jpeg_pair()
    stego = front.generate(cover, "a" * 64)
    values, changes, maximum = c.checked_pixels(cover, stego)
    assert changes > 1 and maximum <= 1e-4 and not np.array_equal(values[0], values[1])


def test_direct_worker_math_and_resource_contract(jpeg_pair, monkeypatch):
    pytest.importorskip("conseal", reason="optional JPEG simulator unavailable")
    cover, _ = jpeg_pair()
    output = io.BytesIO()
    limits = []
    monkeypatch.setattr(front.resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(front.sys, "stdin", Namespace(buffer=io.BytesIO(cover)))
    monkeypatch.setattr(front.sys, "stdout", Namespace(buffer=output))
    front.worker("b" * 64)
    assert output.getvalue().startswith(b"\xff\xd8")
    assert (front.resource.RLIMIT_CPU, (80, 85)) in limits
    assert (front.resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3)) in limits
    monkeypatch.setattr(front.sys, "stdin", Namespace(buffer=io.BytesIO(b"")))
    with pytest.raises(ValueError):
        front.worker("b" * 64)


def test_source_mutation_and_oversized_sources_fail(tmp_path, monkeypatch):
    args = Namespace(action="prepare", out=tmp_path / "out", root=tmp_path, audit=tmp_path)
    calls = [0]

    def sources():
        calls[0] += 1
        return {"revision": str(calls[0])}

    monkeypatch.setattr(front, "sources", sources)
    monkeypatch.setattr(c, "prepare", lambda *args: {"status": "completed"})
    with pytest.raises(ValueError):
        front.execute(args)
    monkeypatch.undo()
    monkeypatch.setattr(front, "regular_open", lambda path: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError):
        front.sources()


def test_high_reader_finite_hash_and_short_read_fail(tmp_path, monkeypatch):
    pairs, _ = c.selection(samples())
    checksum = cache(tmp_path, pairs)
    with c.HighReader(tmp_path, checksum, pairs, lambda: None) as reader:
        monkeypatch.setattr(reader.stream, "read", lambda count: b"")
        with pytest.raises(ValueError):
            reader.batch(pairs[:2].reshape(4))
    monkeypatch.undo()
    raw = json.loads((tmp_path / "manifest.json").read_bytes())
    raw["tensor_sha256"] = "0" * 64
    encoded = json.dumps(raw).encode()
    (tmp_path / "manifest.json").write_bytes(encoded)
    with pytest.raises(ValueError):
        c.HighReader(tmp_path, hashlib.sha256(encoded).hexdigest(), pairs, lambda: None)


@pytest.mark.parametrize("kind", ["version", "exit", "empty", "large"])
def test_worker_wrapper_fails_closed(monkeypatch, kind):
    monkeypatch.setattr(
        front.importlib.metadata, "version", lambda name: "bad" if kind == "version" else "2025.11"
    )

    def run(args, **kw):
        assert args[1:4] == ["-m", "steganography.research_jpeg_curriculum", "worker"]
        assert kw["timeout"] == 90
        kw["stdout"].write(b"x" * (256 * 1024 + 1) if kind == "large" else b"")
        return Namespace(returncode=1 if kind == "exit" else 0)

    monkeypatch.setattr(front.subprocess, "run", run)
    with pytest.raises(ValueError):
        front.generate(b"cover", "a" * 64)


@pytest.mark.parametrize("action", ["prepare", "train", "worker", "bad-worker"])
def test_main_process_limits_and_wiring(monkeypatch, tmp_path, action):
    limits = []
    monkeypatch.setattr(front.resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(front, "worker", lambda lineage: None)
    monkeypatch.setattr(
        front,
        "execute",
        lambda args: {"status": "completed", "singleton_batch_parity": {"passed": False}},
    )
    if action == "worker":
        assert front.main(["worker", "a" * 64]) == 0
    elif action == "bad-worker":
        assert front.main(["worker"]) == 1
    else:
        result = front.main(
            [action, "--root", str(tmp_path), "--audit", str(tmp_path), "--out", str(tmp_path)]
        )
        assert result == (0 if action == "prepare" else 2)
        assert (front.resource.RLIMIT_CPU, (4800, 4801)) in limits
        assert (
            (front.resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3)) in limits
            if action == "prepare"
            else True
        )
