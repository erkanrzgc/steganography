"""Independent generated fixtures and hostile inputs; not accuracy evidence."""

import copy
import hashlib
import io
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import nonjpeg_audit as oracle
from core import nonjpeg_data as service
from steganography import research_nonjpeg as frontend


def save_json(path, value):
    path.write_text(json.dumps(value))
    return service.sha(path.read_bytes())


def source(tmp_path, origin, count=2):
    root = tmp_path / origin
    root.mkdir()
    rows = []
    rng = np.random.default_rng(int(service.sha(origin.encode())[:8], 16))
    roles = ("train", "validation", "test")
    for role in roles:
        for i in range(count):
            if origin in ("DIV2K", "BOWS2"):
                shape = (260, 258, 3) if origin == "DIV2K" else (260, 258)
                image = Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
                output = io.BytesIO()
                image.save(output, format="PNG" if origin == "DIV2K" else "PPM")
                raw = output.getvalue()
                name = f"{role}-{i}.png" if origin == "DIV2K" else f"{role}-{i}.pgm"
            else:
                output = io.BytesIO()
                with wave.open(output, "wb") as audio:
                    audio.setparams(
                        (
                            1,
                            2,
                            8000 if origin == "FSDD-v1.0.10" else 44100,
                            0,
                            "NONE",
                            "not compressed",
                        )
                    )
                    audio.writeframes(rng.integers(0, 65536, size=320, dtype="<u2").tobytes())
                raw = output.getvalue()
                name = f"{role}-{i}.wav"
            (root / name).write_bytes(raw)
            digest = service.sha(raw)
            row = {
                "path": name,
                "sha256": digest,
                "lineage": digest,
                "size": len(raw),
                "source_group": origin,
                "split": role,
                "label": "cover",
                "method": None,
            }
            if origin == "ESC-50":
                row["group_key"] = f"ESC-50:{role}-{i}"
            if origin == "FSDD-v1.0.10":
                row["speaker"] = {"train": "lucas", "validation": "theo", "test": "george"}[role]
                row["split"] = None
            rows.append(row)
    schema = {"BOWS2": "bows2-original-acquisition-v1", "FSDD-v1.0.10": "fsdd-acquisition-v1"}.get(
        origin, "media-diversity-acquisition-v1"
    )
    path = root / "source.json"
    digest = save_json(
        path,
        {
            "schema_version": schema,
            "status": "completed",
            "license": "private-test-license",
            "samples": rows,
        },
    )
    return {"manifest": str(path), "sha256": digest}


@pytest.fixture
def inputs(tmp_path):
    bindings = [source(tmp_path, origin) for origin in ("DIV2K", "BOWS2", "FSDD-v1.0.10", "ESC-50")]
    cfg = {
        "schema_version": "nonjpeg-preparation-config-v1",
        "sources": bindings,
        "counts": {"train": 1, "validation": 1},
    }
    path = tmp_path / "config.json"
    save_json(path, cfg)
    return path, cfg


@pytest.fixture
def corpus(inputs, tmp_path):
    path, cfg = inputs
    out = tmp_path / "data"
    out.mkdir()
    bindings = frontend.fingerprints()
    manifest = service.prepare(
        cfg, out, config_sha256=service.sha(path.read_bytes()), source_sha256=bindings
    )
    return path, cfg, out, manifest, bindings


def test_all_cells_and_independent_audit(corpus, monkeypatch):
    path, _, out, manifest, bindings = corpus
    # Auditor must not use generator, replacement or carrier helpers.
    monkeypatch.setattr(service, "replace", lambda *a, **k: pytest.fail("generator import"))
    monkeypatch.setattr(service, "carrier", lambda *a, **k: pytest.fail("carrier import"))
    audit = oracle.audit(path, out, bindings)
    assert audit["status"] == "completed"
    assert audit["artifact_count"] == 204
    assert audit["independent_scalar_recoveries"] == 192
    assert len(audit["cells"]) == 192
    assert manifest["optimizer_updates"] == 0
    assert audit["test_media_opened"] is False
    assert audit["accuracy_qualification"] == "unavailable"
    assert len(manifest["originals"]) == 8
    assert all(r["split"] != "test" for r in manifest["originals"])
    assert {r["mode"] for r in manifest["originals"]} == {"RGB", "L", "PCM16"}
    assert all(
        r.get("crop") == [1, 2, 256, 256] for r in manifest["originals"] if r["kind"] == "spatial"
    )


def rebind(cfg, index, mutate):
    path = Path(cfg["sources"][index]["manifest"])
    doc = json.loads(path.read_bytes())
    mutate(doc)
    cfg["sources"][index]["sha256"] = save_json(path, doc)


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "schema",
        "license",
        "empty",
        "size",
        "role",
        "lineage",
        "duplicate",
        "path",
        "label",
        "method",
        "origin",
        "row",
        "speaker",
        "recording",
        "decoded-leak",
        "camera-leak",
        "insufficient",
        "count-bound",
    ],
)
def test_selection_fail_closed(inputs, fault):
    _, cfg = inputs

    def mutate(doc):
        rows = doc["samples"]
        if fault == "schema":
            doc["schema_version"] = "unknown"
        elif fault == "license":
            doc["license"] = None
        elif fault == "empty":
            doc["samples"] = []
        elif fault == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif fault == "row":
            rows[0] = None
        elif fault == "insufficient":
            doc["samples"] = [r for r in rows if r["split"] != "validation"]
        elif fault == "count-bound":
            doc["samples"] = rows * 2600
        elif fault.endswith("leak"):
            field = "decoded_sha256" if fault == "decoded-leak" else "camera"
            for row in rows:
                row[field] = "same"
        else:
            field, value = {
                "size": ("size", 0),
                "role": ("split", "invalid"),
                "lineage": ("lineage", "a" * 64),
                "path": ("path", "../escape"),
                "label": ("label", "stego"),
                "method": ("method", "lsb"),
                "origin": ("source_group", "unknown"),
                "speaker": ("speaker", "unknown"),
                "recording": ("group_key", None),
            }[fault]
            rows[0][field] = value

    if fault == "hash":
        cfg["sources"][0]["sha256"] = "0" * 64
    else:
        rebind(cfg, 2 if fault == "speaker" else 3 if fault == "recording" else 0, mutate)
    with pytest.raises(ValueError):
        service.selection(cfg)


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "extra",
        "sources",
        "counts",
        "bool",
        "max",
        "binding",
        "digest",
        "manifest",
        "json-list",
    ],
)
def test_config_fail_closed(inputs, fault):
    path, cfg = inputs
    if fault == "schema":
        cfg["schema_version"] = "unknown"
    elif fault == "extra":
        cfg["adaptive"] = True
    elif fault == "sources":
        cfg["sources"] = []
    elif fault == "counts":
        cfg["counts"] = {"train": 1}
    elif fault == "bool":
        cfg["counts"]["train"] = True
    elif fault == "max":
        cfg["counts"]["validation"] = 9
    elif fault == "binding":
        cfg["sources"][0] = {}
    elif fault == "digest":
        cfg["sources"][0]["sha256"] = "bad"
    elif fault == "manifest":
        cfg["sources"][0]["manifest"] = 3
    else:
        cfg = []
    save_json(path, cfg)
    with pytest.raises(ValueError):
        service.config(path)


@pytest.mark.parametrize("name", ["../x", "/absolute/x", "a/b", "a\\b", ".", "..", "", None])
def test_unsafe_names(name):
    for check in (service.basename, oracle.member):
        with pytest.raises(ValueError):
            check(name)


def test_file_bounds_and_no_follow(tmp_path):
    file = tmp_path / "file"
    file.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(file)
    for reader in (service.regular_read, oracle.read):
        assert reader(file, 3) == b"abc"
        for target, limit in ((file, 2), (link, 3), (tmp_path, 3)):
            with pytest.raises(ValueError):
                reader(target, limit)
    fifo = tmp_path / "fifo"
    import os

    os.mkfifo(fifo)
    for reader in (service.regular_read, oracle.read):
        with pytest.raises(ValueError):
            reader(fifo, 3)
    file.write_bytes(b"")
    for reader in (service.regular_read, oracle.read):
        with pytest.raises(ValueError):
            reader(file, 3)
    with pytest.raises(ValueError):
        service.write(link, b"x")
    with pytest.raises(FileExistsError):
        service.write(file, b"x")


@pytest.mark.parametrize(
    "fault",
    [
        "config",
        "sources",
        "manifest-schema",
        "status",
        "trained",
        "source-hash",
        "license",
        "original-count",
        "carrier-hash",
        "ancestry",
        "crop",
        "audio-meta",
        "missing",
        "extra",
        "cell",
        "file-hash",
        "file-size",
        "payload-hash",
        "payload-size",
        "bits",
        "changed",
        "eligible",
        "count",
        "bytes",
        "denominator",
    ],
)
def test_independent_audit_rejects_corruption(corpus, fault):
    path, cfg, out, manifest, bindings = corpus
    if fault == "config":
        save_json(path, {**cfg, "schema_version": "unknown"})
    elif fault == "sources":
        bindings = {}
    elif fault == "source-hash":
        Path(cfg["sources"][0]["manifest"]).write_text("{}")
    elif fault in (
        "manifest-schema",
        "status",
        "trained",
        "original-count",
        "missing",
        "extra",
        "count",
        "bytes",
        "denominator",
        "license",
    ):
        if fault == "manifest-schema":
            manifest["schema_version"] = "unknown"
        elif fault == "status":
            manifest["status"] = "completed"
        elif fault == "trained":
            manifest["model_trained"] = True
        elif fault == "original-count":
            manifest["originals"].pop()
        elif fault == "missing":
            manifest["samples"].pop()
        elif fault == "extra":
            manifest["samples"].append(manifest["samples"][-1])
        elif fault == "count":
            manifest["artifact_count"] += 1
        elif fault == "bytes":
            manifest["artifact_bytes"] += 1
        elif fault == "denominator":
            manifest["original_counts"] = {}
        else:
            manifest["source_bindings"][0]["license"] = "wrong"
    elif fault in ("carrier-hash", "ancestry", "crop", "audio-meta"):
        original = manifest["originals"][0]
        if fault == "carrier-hash":
            original["carrier_sha256"] = "bad"
        elif fault == "ancestry":
            original["split"] = "test"
        elif fault == "crop":
            original["crop"][0] += 1
        else:
            next(r for r in manifest["originals"] if r["kind"] == "audio")["frames"] += 1
    else:
        sample = next(r for r in manifest["samples"] if r["method"] is not None)
        field, value = {
            "cell": ("plane", 7),
            "file-hash": ("sha256", "0" * 64),
            "file-size": ("size", 1),
            "payload-hash": ("payload_sha256", "0" * 64),
            "payload-size": ("payload_bytes", 1),
            "bits": ("embedded_bits", 1),
            "changed": ("changed_units", 0),
            "eligible": ("eligible_units", 1),
        }[fault]
        sample[field] = value
    save_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError):
        oracle.audit(path, out, bindings)


@pytest.mark.parametrize(
    "fault", ["nonselected", "wrong-payload", "cover", "geometry", "mode", "riff-metadata"]
)
def test_scalar_oracle_rejects_rehashed_media(corpus, fault):
    path, _, out, manifest, bindings = corpus
    sample = next(
        r
        for r in manifest["samples"]
        if r["format"] == ("WAV" if fault == "riff-metadata" else "BMP")
        and r["label"] == ("cover" if fault == "cover" else "stego")
    )
    target = out / sample["path"]
    raw = target.read_bytes()
    if fault == "riff-metadata":
        modified = bytearray(raw)
        modified[28] ^= 1  # Valid PCM decode, but RIFF byte-rate metadata changed.
        raw = bytes(modified)
    else:
        with Image.open(io.BytesIO(raw)) as image:
            pixels = np.array(image)
        if fault == "geometry":
            pixels = pixels[:128]
        elif fault == "mode":
            pixels = np.repeat(pixels[:, :, None], 3, axis=2)
        else:
            pixels.flat[0] ^= 4 if fault == "nonselected" else 1
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="BMP")
        raw = buffer.getvalue()
    target.write_bytes(raw)
    sample.update(size=len(raw), sha256=service.sha(raw))
    save_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError):
        oracle.audit(path, out, bindings)


@pytest.mark.parametrize("fault", ["input-integrity", "decoded-duplicate", "budget", "timeout"])
def test_preparation_retains_failure(inputs, tmp_path, monkeypatch, fault):
    path, cfg = inputs
    chosen, _ = service.selection(cfg)
    if fault == "input-integrity":
        (chosen[0]["root"] / chosen[0]["path"]).write_bytes(b"bad")
    elif fault == "decoded-duplicate":
        real = service.carrier
        first = real(
            (chosen[0]["root"] / chosen[0]["path"]).read_bytes(), chosen[0]["source_group"]
        )
        monkeypatch.setattr(service, "carrier", lambda *a: first)
    elif fault == "budget":
        monkeypatch.setattr(service, "MAX_OUTPUT", 1)
    else:
        values = iter([0, 2000])
        monkeypatch.setattr(service.time, "monotonic", lambda: next(values))
    out = tmp_path / "failed"
    out.mkdir()
    with pytest.raises((ValueError, TimeoutError)):
        service.prepare(cfg, out, config_sha256=service.sha(path.read_bytes()), source_sha256={})
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize(
    "fault", ["method", "rate", "plane", "channel", "dtype", "shape", "small", "empty-payload"]
)
def test_replacement_contract(fault):
    pixels = np.zeros((256, 256), dtype=np.uint8)
    arguments = {
        "lineage": "a" * 64,
        "method": "sequential",
        "rate": 5,
        "channel": "gray",
        "plane": 0,
    }
    if fault in ("method", "rate", "plane", "channel"):
        arguments[fault] = {"method": "unknown", "rate": 2, "plane": 3, "channel": "invalid"}[fault]
    elif fault == "dtype":
        pixels = pixels.astype(np.float32)
    elif fault == "shape":
        arguments["channel"] = "r"
    elif fault == "small":
        pixels = np.zeros(16, dtype="<u2")
        arguments["channel"] = "mono"
    else:
        # 160 samples at 5% is exactly one byte; use zero changes rather than
        # modifying the recipe's minimum. A payload-matching cover must fail.
        pixels = np.zeros(160, dtype="<u2")
        arguments["channel"] = "mono"
        context = "20261012:" + "a" * 64 + ":sequential:5:mono:0"
        pixels[:8] = np.unpackbits(
            np.frombuffer(hashlib.shake_256(context.encode()).digest(1), dtype=np.uint8)
        )
    with pytest.raises(ValueError):
        service.replace(pixels, **arguments)


def test_owned_worker_and_failure_reaping(monkeypatch):
    calls = []

    class Process:
        pid = 4123

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired("redacted", timeout)
            return 0

    monkeypatch.setattr(
        frontend.subprocess, "Popen", lambda *a, **k: (calls.append(k), Process())[1]
    )
    monkeypatch.setattr(frontend.os, "killpg", lambda *a: calls.append(("kill", a)))
    with pytest.raises(subprocess.TimeoutExpired):
        frontend.worker("core.nonjpeg_audit", [])
    assert calls[0]["start_new_session"] is True
    assert calls[-1] == ("wait", None)
    assert calls[-2][0] == "kill"
    with pytest.raises(ValueError):
        frontend.worker("arbitrary.module", [])


def test_incomplete_report_does_not_leak(inputs, tmp_path, monkeypatch):
    path, _ = inputs

    def fail(*a):
        raise ValueError("/home/private password=secret")

    monkeypatch.setattr(frontend, "worker", fail)
    out = tmp_path / "incomplete"
    report = frontend.run(path, out)
    assert report["status"] == "incomplete"
    assert "/home/private" not in (out / "report.json").read_text()
    assert "password" not in (out / "report.json").read_text()
    with pytest.raises(FileExistsError):
        frontend.run(path, out)


def test_real_subprocess_orchestration(inputs, tmp_path):
    path, _ = inputs
    out = tmp_path / "isolated"
    result = frontend.run(path, out)
    assert result["status"] == "completed"
    assert result["gpu_used"] is False
    assert result["independent_scalar_recoveries"] == 192
    assert result["source_sha256"] == frontend.fingerprints()
    assert frontend.main(["--config", str(path), "--out", str(tmp_path / "entry")]) == 0


def test_cli_dispatch(inputs, tmp_path, capsys):
    import argparse

    import cli

    path, _ = inputs
    assert (
        cli.cmd_research_nonjpeg(argparse.Namespace(config=str(path), out=str(tmp_path / "cli")))
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


@pytest.mark.parametrize(
    "fault", ["status", "long-license", "wrong-schema-origin", "group-path", "fsdd-conflict"]
)
def test_source_contract_and_bounded_metadata(inputs, fault):
    _, cfg = inputs

    def mutate(doc):
        if fault == "status":
            doc["status"] = "incomplete"
        elif fault == "long-license":
            doc["license"] = "a" * 257
        elif fault == "wrong-schema-origin":
            doc["schema_version"] = "bows2-original-acquisition-v1"
        elif fault == "group-path":
            doc["samples"][0]["device"] = "/private/host"
        else:
            doc["samples"][0]["split"] = "test"

    rebind(cfg, 2 if fault == "fsdd-conflict" else 0, mutate)
    with pytest.raises(ValueError):
        service.selection(cfg)


@pytest.mark.parametrize(
    "fault",
    [
        "image-mode",
        "image-size",
        "image-limit",
        "riff",
        "rate",
        "channels",
        "width",
        "frames",
        "chunk-bound",
        "many-chunks",
        "two-data",
        "padding",
    ],
)
def test_native_carrier_bounds(fault):
    if fault.startswith("image"):
        mode = "L" if fault == "image-mode" else "RGB"
        size = (
            (128, 256)
            if fault == "image-size"
            else (4001, 2000)
            if fault == "image-limit"
            else (256, 256)
        )
        buffer = io.BytesIO()
        Image.new(mode, size).save(buffer, format="PNG")
        raw, origin = buffer.getvalue(), "DIV2K"
    else:
        buffer = io.BytesIO()
        width, channels, rate, frames = (
            (1 if fault == "width" else 2),
            (2 if fault == "channels" else 1),
            (9000 if fault == "rate" else 8000),
            (159 if fault == "frames" else 320),
        )
        with wave.open(buffer, "wb") as audio:
            audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(frames * width * channels))
        raw, origin = buffer.getvalue(), "FSDD-v1.0.10"
        if fault == "riff":
            raw = b"NOPE" + raw[4:]
        elif fault in ("chunk-bound", "many-chunks", "two-data", "padding"):
            extra = {
                "chunk-bound": b"JUNK" + (100).to_bytes(4, "little"),
                "many-chunks": (b"JUNK" + bytes(4)) * 65,
                "two-data": b"data" + bytes(4),
                "padding": b"x",
            }[fault]
            raw += extra
            raw = raw[:4] + (len(raw) - 8).to_bytes(4, "little") + raw[8:]
    with pytest.raises(ValueError):
        service.carrier(raw, origin)
    if fault in ("image-mode", "image-size", "image-limit", "rate", "channels", "width", "frames"):
        with pytest.raises(ValueError):
            oracle.decode(raw, origin)


def test_test_files_are_never_opened(inputs, tmp_path, monkeypatch):
    path, cfg = inputs
    real = service.regular_read
    seen = []

    def guarded(file, limit):
        assert not file.name.startswith("test-")
        seen.append(file.name)
        return real(file, limit)

    monkeypatch.setattr(service, "regular_read", guarded)
    out = tmp_path / "notest"
    out.mkdir()
    service.prepare(cfg, out, config_sha256=service.sha(path.read_bytes()), source_sha256={})
    assert seen


def test_audit_deadline(corpus, monkeypatch):
    path, _, out, _, bindings = corpus
    ticks = iter([0, 2000])
    monkeypatch.setattr(oracle.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        oracle.audit(path, out, bindings)


def test_worker_failed_exit_and_reaped_group(monkeypatch):
    class Process:
        pid = 4123

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(frontend.subprocess, "Popen", lambda *a, **k: Process())

    def missing(*a):
        raise ProcessLookupError()

    monkeypatch.setattr(frontend.os, "killpg", missing)
    with pytest.raises(ValueError):
        frontend.worker("core.nonjpeg_audit", [])


def test_worker_entrypoints_use_limits(corpus, tmp_path, monkeypatch):
    path, _, out, _, bindings = corpus
    calls = []
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *a: calls.append(a))
    monkeypatch.setattr(frontend, "prepare", lambda *a, **k: calls.append("prepare"))
    assert frontend.main(["--config", str(path), "--out", str(out), "--worker"]) == 0
    assert len(calls) == 5
    assert calls[-1] == "prepare"
    bindings_path = tmp_path / "bindings.json"
    save_json(bindings_path, bindings)
    monkeypatch.setattr(oracle, "audit", lambda *a: {"status": "test-only"})
    monkeypatch.setattr(oracle, "__name__", "unit-test")
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--config", str(path), "--root", str(out), "--bindings", str(bindings_path)],
    )
    oracle.main()
    assert json.loads((out / "independent-audit.json").read_bytes())["status"] == "test-only"


def test_compatible_manifests_and_empty_family(corpus, tmp_path):
    from steganography.research import verify_dataset_manifest

    _, _, out, manifest, _ = corpus
    exports = frontend.research_manifests(out, "a" * 64)
    assert set(exports) == {"spatial-research.json", "audio-research.json"}
    for name, digest in exports.items():
        path = out / name
        assert service.sha(path.read_bytes()) == digest
        result = verify_dataset_manifest(json.loads(path.read_bytes()), source=out)
        assert result["splits"]["test"] == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    save_json(empty / "manifest.json", {**manifest, "samples": []})
    assert frontend.research_manifests(empty, "a" * 64) == {}


def test_cli_parser_and_failure_dispatch(inputs, tmp_path, monkeypatch, capsys):
    import cli

    path, _ = inputs
    args = cli.build_parser().parse_args(
        ["research", "nonjpeg-prepare", "--config", str(path), "--out", str(tmp_path / "unused")]
    )
    monkeypatch.setattr(frontend, "run", lambda *a: {"status": "incomplete"})
    assert args.fn(args) == 2
    assert frontend.main(["--config", str(path), "--out", str(tmp_path / "unused")]) == 2
    assert "incomplete" in capsys.readouterr().out


def test_sigterm_cancellation_restores_handler(inputs, tmp_path, monkeypatch):
    path, _ = inputs
    handlers = []
    monkeypatch.setattr(frontend.signal, "signal", lambda sig, handler: handlers.append(handler))

    def cancel(*args):
        frontend.cancel_requested(None, None)

    monkeypatch.setattr(frontend, "worker", cancel)
    previous = frontend.signal.getsignal(frontend.signal.SIGTERM)
    result = frontend.run(path, tmp_path / "cancelled")
    assert result["status"] == "cancelled"
    assert handlers == [frontend.cancel_requested, previous]


@pytest.mark.parametrize("stage", ["before-audit", "after-audit", "invalid-audit"])
def test_changed_bindings_cannot_report_completion(inputs, tmp_path, monkeypatch, stage):
    path, cfg = inputs
    out = tmp_path / "changed"

    def worker(module, args):
        if module == "steganography.research_nonjpeg":
            if stage == "before-audit":
                save_json(path, {**cfg, "counts": {"train": 2, "validation": 1}})
        else:
            result = {
                "status": "incomplete" if stage == "invalid-audit" else "completed",
                "config_sha256": service.sha(path.read_bytes()),
                "source_sha256": frontend.fingerprints(),
            }
            if stage == "after-audit":
                save_json(path, {**cfg, "counts": {"train": 2, "validation": 1}})
            save_json(out / "independent-audit.json", result)

    monkeypatch.setattr(frontend, "worker", worker)
    result = frontend.run(path, out)
    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("config-count", "audit config bounds"),
        ("empty-source", "audit original metadata"),
        ("source-row-bound", "audit original count bound"),
        ("source-role", "audit source role contract"),
        ("repeated-original", "audit repeated original"),
        ("decoded-leakage", "audit grouping leakage"),
        ("insufficient-selection", "audit fixed selection count"),
        ("original-integrity", "audit original integrity"),
    ],
)
def test_additional_independent_source_guards(corpus, fault, message):
    path, cfg, out, manifest, bindings = corpus
    source_path = Path(cfg["sources"][0]["manifest"])
    doc = json.loads(source_path.read_bytes())
    if fault == "config-count":
        cfg["counts"]["train"] = True
    elif fault == "insufficient-selection":
        cfg["counts"]["train"] = 3
    elif fault == "original-integrity":
        selected = manifest["originals"][0]
        binding = next(
            b for b in cfg["sources"] if b["sha256"] == selected["source_manifest_sha256"]
        )
        original_source = Path(binding["manifest"])
        original_doc = json.loads(original_source.read_bytes())
        row = next(r for r in original_doc["samples"] if r["sha256"] == selected["sha256"])
        (original_source.parent / row["path"]).write_bytes(b"corrupted original")
    else:
        if fault == "empty-source":
            doc["samples"] = []
        elif fault == "source-row-bound":
            doc["samples"] = [doc["samples"][0]] * 15001
        elif fault == "source-role":
            doc["samples"][0]["split"] = "unknown"
        elif fault == "repeated-original":
            doc["samples"][1] = copy.deepcopy(doc["samples"][0])
        elif fault == "decoded-leakage":
            for index in (0, 2):  # Same decoded identity in train and validation.
                doc["samples"][index]["decoded_sha256"] = "a" * 64
        digest = save_json(source_path, doc)
        cfg["sources"][0]["sha256"] = digest
        manifest["source_bindings"][0]["sha256"] = digest
    manifest["config_sha256"] = save_json(path, cfg)
    save_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        oracle.audit(path, out, bindings)


def test_additional_independent_decode_guards():
    buffer = io.BytesIO()
    Image.new("L", (257, 256)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="audit saved image size"):
        oracle.decode(buffer.getvalue(), "BOWS2", "PNG")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(640))
    with pytest.raises(ValueError, match="audit PCM truncation"):
        oracle.decode(buffer.getvalue()[:-2], "FSDD-v1.0.10")
    with pytest.raises(ValueError, match="truncated PCM"):
        raw = buffer.getvalue()[:-2]
        raw = raw[:4] + (len(raw) - 8).to_bytes(4, "little") + raw[8:]
        service.carrier(raw, "FSDD-v1.0.10")


def test_independent_audit_main_rejects_symlink_output(tmp_path, monkeypatch):
    bindings_path = tmp_path / "bindings.json"
    save_json(bindings_path, {})
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("untouched")
    (tmp_path / "independent-audit.json").symlink_to(sentinel)
    monkeypatch.setattr(oracle.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(oracle, "audit", lambda *a: {"status": "test-only"})
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--config", "unused", "--root", str(tmp_path), "--bindings", str(bindings_path)],
    )
    with pytest.raises(ValueError, match="audit symlink output"):
        oracle.main()
    assert sentinel.read_text() == "untouched"
