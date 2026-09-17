import argparse
import asyncio
import base64
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
import wave
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from PIL import Image
from textual.widgets import Input, Select

from api.app import APISettings, create_app
from cli import _parse_size, main
from core.context import AnalysisContext
from core.ctf import (
    CTFLimits,
    CTFService,
    _archive_members,
    _credible,
    _decoded_candidates,
    _decompress,
    _safe_stem,
    _signature_offsets,
    _trailer,
)
from core.ctf_types import Artifact, Coverage, Recommendation, ToolExecution
from core.database import Database
from core.models import ModelRegistry, ModelVerificationError, _download_https
from core.service import StegoService
from core.tools import ToolRunner
from modules.audio_wav import AudioWav
from modules.image_gif import ImageGifAnalyzer, _extension_counts
from modules.text_anomalies import TextAnomalyAnalyzer
from report.v2 import html_v2, sarif_v2, write_evidence_bundle
from steganography.research import (
    ResearchManifestError,
    _apply_temperature,
    _expected_calibration_error,
    _load_predictions,
    _log_loss,
    calibrate_predictions,
    export_onnx,
    train_model,
)
from ui.tui import CTFScreen, SteganographyTUI


class FakeToolRunner:
    def __init__(self, *, create_output: bool = False):
        self.calls = []
        self.create_output = create_output

    def run(self, tool, args, **kwargs):
        self.calls.append((tool, list(args), kwargs))
        inputs = [value for value in args if value.startswith("artifacts/")]
        assert inputs and all((kwargs["cwd"] / value).is_file() for value in inputs)
        if self.create_output and tool == "stegseek":
            (kwargs["cwd"] / "stegseek.bin").write_bytes(b"flag{external}")
        command = tuple(
            "[REDACTED]" if value in kwargs["secret_values"] else value for value in (tool, *args)
        )
        return ToolExecution(tool, "test 1", "completed", 1.0, 0, command, "ok")


def test_analysis_context_reuses_file_image_jpeg_and_wav(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert AnalysisContext(empty).entropy == 0.0
    text = tmp_path / "sample.bin"
    text.write_bytes(b"aabbccdd")
    context = AnalysisContext(text)
    assert context.data is context.data
    assert context.entropy > 0
    assert len(context.entropy_map(block_size=64)) == 1
    with pytest.raises(ValueError, match="block size"):
        context.entropy_map(block_size=8)
    with pytest.raises(ValueError, match="context limit"):
        AnalysisContext(text, max_bytes=1)
    link = tmp_path / "link"
    link.symlink_to(text)
    with pytest.raises(ValueError, match="non-symlink"):
        AnalysisContext(link)

    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(image_path)
    image_context = AnalysisContext(image_path)
    assert image_context.image_rgba.shape == (4, 5, 4)
    assert image_context.image_rgba is image_context.image_rgba
    with pytest.raises(ValueError, match="pixel"):
        _oversized_image = AnalysisContext(image_path, max_pixels=1).image_rgba

    jpeg = tmp_path / "image.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(jpeg)
    segments = AnalysisContext(jpeg).jpeg_segments
    assert segments and segments[0]["marker"] == 0xE0
    assert AnalysisContext(text).jpeg_segments == ()
    broken_jpeg = tmp_path / "broken.jpg"
    broken_jpeg.write_bytes(b"\xff\xd8\xff\xe1\x00\x20short")
    assert AnalysisContext(broken_jpeg).jpeg_segments[-1]["truncated"] is True

    for width, values in (
        (1, bytes([0, 128, 255])),
        (2, b"\x01\x00\xff\xff"),
        (4, b"\x01\x00\x00\x00"),
    ):
        wav_path = tmp_path / f"width-{width}.wav"
        with wave.open(str(wav_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(width)
            output.setframerate(8000)
            output.writeframes(values)
        samples, rate, sample_width = AnalysisContext(wav_path).wav_samples
        assert samples.size and rate == 8000 and sample_width == width
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        output.writeframes(b"\x00" * 8)
    assert AnalysisContext(stereo).wav_samples[0].shape == (2, 2)
    wav_result = AudioWav().analyze(stereo)
    assert {signal.name for signal in wav_result.signals} >= {
        "channel_difference_lsb",
        "wav_lsb_bias",
    }


def test_gif_and_text_native_analysis(tmp_path: Path):
    gif = tmp_path / "animated.gif"
    frames = [Image.fromarray(np.full((8, 8, 3), index, dtype=np.uint8)) for index in (0, 0, 30)]
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        comment=b"candidate comment",
    )
    result = ImageGifAnalyzer().analyze(gif)
    names = {signal.name for signal in result.signals}
    assert "gif_comment_extensions" in names and "gif_frame_differences" in names
    assert _extension_counts(b"x\x21\xfey\x21\xffz") == (1, 1)
    ordinary = tmp_path / "ordinary.bin"
    ordinary.write_bytes(b"not gif")
    assert ImageGifAnalyzer().analyze(ordinary).status == "unsupported"

    text = tmp_path / "unicode.txt"
    text.write_text("Latin аааа e\u0301\x01")
    text_result = TextAnomalyAnalyzer().analyze(text)
    text_names = {signal.name for signal in text_result.signals}
    assert "embedded_control_characters" in text_names
    assert "unicode_normalization_difference" in text_names
    assert "mixed_script_homoglyphs" in text_names


def test_ctf_helpers_cover_decoding_carving_and_archives(tmp_path: Path):
    with pytest.raises(ValueError, match="depth"):
        CTFLimits(max_depth=-1)
    with pytest.raises(ValueError, match="artifact"):
        CTFLimits(max_artifacts=0)
    with pytest.raises(ValueError, match="timeouts"):
        CTFLimits(tool_timeout=0)
    assert _safe_stem("../../odd name!.txt") == "odd-name"
    png = b"\x89PNG\r\n\x1a\nbody"
    assert _credible(png, b"x")
    assert not _credible(b"", b"x") and not _credible(b"same", b"same")
    assert list(_signature_offsets(b"prefix" + png))[0] == (6, "png")
    assert _trailer(b"PNG-IEND\xaeB`\x82tail", ".png") == b"tail"
    assert _trailer(b"plain", ".txt") == b""

    candidates = _decoded_candidates(base64.b64encode(b"flag{base64}"), deep=False)
    assert any(item[0] == b"flag{base64}" for item in candidates)
    assert _decompress(gzip.compress(b"flag{gzip}")) == b"flag{gzip}"
    assert _decompress(zlib.compress(b"flag{zlib}")) == b"flag{zlib}"
    assert _decompress(b"ordinary") is None
    xor = bytes(value ^ 42 for value in b"flag{xor}")
    assert any("XOR" in item[2] for item in _decoded_candidates(xor, deep=True))
    url = _decoded_candidates(b"flag%7Burl%7D", deep=False)
    assert any(item[0] == b"flag{url}" for item in url)

    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("safe/flag.txt", b"flag{zip}")
        archive.writestr("../../escape", b"bad")
    members = _archive_members(zipped.getvalue(), CTFLimits(max_bytes=1000))
    assert members == [(b"flag{zip}", "flag.txt", "ZIP member flag.txt")]

    tarred = io.BytesIO()
    with tarfile.open(fileobj=tarred, mode="w") as archive:
        info = tarfile.TarInfo("inside.txt")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"flag"))
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/escape"  # noqa: S108 - malicious archive fixture
        archive.addfile(link)
    assert _archive_members(tarred.getvalue(), CTFLimits(max_bytes=20_000))[0][0] == b"flag"
    assert _archive_members(b"not an archive", CTFLimits()) == []


def test_ctf_service_cli_reports_limits_and_exact_recovery(tmp_path: Path, capsys):
    source = tmp_path / "challenge.txt"
    source.write_bytes(base64.b64encode(b"flag{candidate}"))
    report = CTFService().solve(source, tmp_path / "quick", mode="quick")
    value = report.to_dict()
    assert report.status == "completed" and report.verdict == "suspicious"
    assert value["schema_revision"] == 1 and value["calibration"]["state"] == "not_calibrated"
    assert all(not item["name"].startswith("/") for item in value["artifacts"])
    assert any("base64" in item["provenance"] for item in value["artifacts"])
    with pytest.raises(FileExistsError):
        CTFService().solve(source, tmp_path / "quick")
    with pytest.raises(ValueError, match="unknown"):
        CTFService().solve(source, tmp_path / "bad-mode", mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wordlist"):
        CTFService().solve(source, tmp_path / "bad-wordlist", wordlist=tmp_path / "missing")
    with pytest.raises(ValueError, match="byte limit"):
        CTFService().solve(source, tmp_path / "small", limits=CTFLimits(max_bytes=1))
    cancelled = CTFService().solve(source, tmp_path / "cancelled", should_cancel=lambda: True)
    assert cancelled.status == "cancelled" and cancelled.verdict == "inconclusive"

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"flag{exact}")
    cover = tmp_path / "cover.png"
    Image.fromarray(np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        cover
    )
    stego = tmp_path / "stego.png"
    StegoService().embed(payload, cover, stego)
    fake = FakeToolRunner()
    deep = CTFService(tool_runner=fake).solve(stego, tmp_path / "deep", mode="deep")
    assert deep.verdict == "confirmed"
    assert any(item.provenance.startswith("successful extraction") for item in deep.artifacts)
    assert any("visualization" in item.provenance for item in deep.artifacts)
    assert {call[0] for call in fake.calls} >= {"zsteg", "pngcheck", "binwalk", "zbarimg"}

    cli_out = tmp_path / "cli"
    assert (
        main(
            [
                "--quiet",
                "ctf",
                str(source),
                "--out",
                str(cli_out),
                "--mode",
                "quick",
                "--report",
                "bundle",
            ]
        )
        == 0
    )
    assert zipfile.is_zipfile(cli_out / "evidence-bundle.zip")
    assert "CTF completed" in capsys.readouterr().out
    assert _parse_size("1.5MiB") == int(1.5 * 1024**2)
    assert _parse_size("2kb") == 2000
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_size("nope")


def test_ctf_external_password_redaction_and_formats(tmp_path: Path):
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("password\n")
    fake = FakeToolRunner(create_output=True)
    for suffix in (".jpg", ".gif", ".wav"):
        source = tmp_path / f"sample{suffix}"
        source.write_bytes(b"ordinary")
        report = CTFService(tool_runner=fake).solve(
            source,
            tmp_path / f"out-{suffix[1:]}",
            mode="deep" if suffix in {".gif", ".wav"} else "balanced",
            wordlist=wordlist if suffix == ".jpg" else None,
            password="top-secret" if suffix == ".jpg" else None,
        )
        serialized = json.dumps(report.to_dict())
        assert "top-secret" not in serialized
    names = {call[0] for call in fake.calls}
    assert {"stegseek", "steghide", "outguess", "gifsicle", "sox", "ffmpeg", "exiftool"} <= names


def test_tool_runner_process_states_and_types(tmp_path: Path, monkeypatch):
    artifact = Artifact("a", "a.bin", "application/octet-stream", 1, "0" * 64, 0, "test")
    assert artifact.to_dict()["id"] == "a"
    assert Coverage("x", "available").to_dict()["status"] == "available"
    assert Recommendation(1, "act", "why").to_dict()["action"] == "act"
    assert ToolExecution("x", None, "unavailable", 0, None, ("x",)).to_dict()["tool"] == "x"
    with pytest.raises(ValueError, match="positive"):
        ToolRunner(timeout=0)

    runner = ToolRunner(timeout=1, max_output=8)
    tool = Path(sys.executable).name
    completed = runner.run(
        tool,
        ["-c", "print('secret-value-long')", "secret"],
        cwd=tmp_path,
        secret_values=("secret",),
    )
    assert completed.status == "completed" and "[REDACTED]" in completed.command
    assert "truncated" in completed.output
    noisy = runner.run(tool, ["-c", "import sys; sys.stdout.write('x' * 2000000)"], cwd=tmp_path)
    assert noisy.status == "completed"
    assert noisy.output == "xxxxxxxx\n[output truncated]"
    link = tmp_path / "linked-workdir"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="working directory"):
        runner.run(tool, [], cwd=link)
    failed = runner.run(tool, ["-c", "raise SystemExit(3)"], cwd=tmp_path)
    assert failed.status == "failed" and failed.exit_code == 3
    timed = runner.run(tool, ["-c", "import time; time.sleep(2)"], cwd=tmp_path, timeout=0.05)
    assert timed.status == "timed_out"
    cancelled = runner.run(tool, ["-c", "print('x')"], cwd=tmp_path, should_cancel=lambda: True)
    assert cancelled.status == "cancelled"
    with pytest.raises(ValueError, match="timeout"):
        runner.run(tool, [], cwd=tmp_path, timeout=0)
    with pytest.raises(ValueError, match="working directory"):
        runner.run(tool, [], cwd=tmp_path / "missing")
    monkeypatch.setattr(shutil, "which", lambda _tool: None)
    assert runner.run("missing-tool", [], cwd=tmp_path).status == "unavailable"
    assert runner.version("another-missing") is None


def test_explicit_signed_model_catalog_install(tmp_path: Path, monkeypatch, capsys):
    model_bytes = b"test onnx bytes"
    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(private.public_key().public_bytes_raw()).decode()
    manifest = {
        "id": "spatial-test",
        "version": "1",
        "domain": "spatial-srnet-v1",
        "file": "model.onnx",
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "license": "test-only",
        "model_card": "fixture",
        "benchmark_summary": {},
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["signature"] = base64.b64encode(private.sign(canonical)).decode()
    catalog = {
        "schema_version": "1.0",
        "models": [
            {
                "id": "spatial-test",
                "version": "1",
                "download_url": "https://models.example/model.onnx",
                "public_key": public,
                "manifest": manifest,
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    registry = ModelRegistry(Database(tmp_path / "models.sqlite3"), tmp_path / "models")
    malformed = tmp_path / "malformed-catalog.json"
    malformed.write_text("{}")
    with pytest.raises(ModelVerificationError, match="malformed"):
        registry.catalog(malformed)
    with pytest.raises(ModelVerificationError, match="cannot read"):
        registry.catalog(tmp_path / "missing-catalog.json")
    with pytest.raises(ModelVerificationError, match="accept-license"):
        registry.install_catalog("spatial-test@1", accept_license=False, catalog_path=catalog_path)
    with pytest.raises(ModelVerificationError, match="not present"):
        registry.install_catalog("missing@1", accept_license=True, catalog_path=catalog_path)

    class Response:
        def __init__(self, value=model_bytes, url="https://models.example/model.onnx"):
            self.value = value
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return self.url

        def read(self, _size):
            value, self.value = self.value, b""
            return value

    monkeypatch.setattr("core.models.urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    installed = registry.install_catalog(
        "spatial-test@1", accept_license=True, catalog_path=catalog_path
    )
    assert installed["state"] == "installed"
    assert (
        main(
            [
                "--quiet",
                "models",
                "--state-dir",
                str(tmp_path / "cli-models"),
                "catalog",
                "--catalog",
                str(catalog_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["models"][0]["id"] == "spatial-test"

    monkeypatch.setattr(
        "core.models.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(url="http://insecure.example/model"),
    )
    with pytest.raises(ModelVerificationError, match="outside HTTPS"):
        _download_https("https://models.example/model", tmp_path / "redirect", maximum=100)
    monkeypatch.setattr(
        "core.models.urllib.request.urlopen", lambda *_args, **_kwargs: Response(value=b"large")
    )
    with pytest.raises(ModelVerificationError, match="byte limit"):
        _download_https("https://models.example/model", tmp_path / "large", maximum=1)
    monkeypatch.setattr(
        "core.models.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(ModelVerificationError, match="download failed"):
        _download_https("https://models.example/model", tmp_path / "offline", maximum=10)


def test_research_calibration_and_opt_in_training(tmp_path: Path):
    source = tmp_path / "dataset"
    source.mkdir()
    samples = []
    predictions = {}
    for name, label, score in (
        ("cover.png", "cover", 0.2),
        ("stego.png", "stego", 0.8),
    ):
        path = source / name
        path.write_bytes(name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        samples.append(
            {
                "path": name,
                "sha256": digest,
                "size": path.stat().st_size,
                "label": label,
                "source_group": "camera-a",
                "lineage": name,
                "split": "validation",
            }
        )
        predictions[digest] = score
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "source": str(source), "samples": samples})
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps({"predictions": predictions}))
    result = calibrate_predictions(manifest_path, predictions_path, tmp_path / "calibration.json")
    assert result["method"] == "scalar_temperature"
    assert result["after"]["ece"] <= result["before"]["ece"]
    assert _apply_temperature(0.5, 2) == 0.5
    assert _log_loss([(True, 0.9), (False, 0.1)], 1) < 0.2
    assert _expected_calibration_error([]) == 0
    with pytest.raises(ValueError, match="split"):
        calibrate_predictions(
            manifest_path,
            predictions_path,
            tmp_path / "bad-calibration.json",
            split="test",
        )
    predictions_path.write_text(json.dumps({"predictions": {}}))
    with pytest.raises(ResearchManifestError, match="missing"):
        calibrate_predictions(
            manifest_path, predictions_path, tmp_path / "missing-calibration.json"
        )
    invalid_predictions = tmp_path / "invalid-predictions.json"
    invalid_predictions.write_text('"bad"')
    with pytest.raises(ResearchManifestError, match="predictions"):
        _load_predictions(invalid_predictions)
    invalid_predictions.write_text(json.dumps(["bad"]))
    with pytest.raises(ResearchManifestError, match="object"):
        _load_predictions(invalid_predictions)
    with pytest.raises(RuntimeError, match="research"):
        train_model(tmp_path / "config.json", tmp_path / "checkpoint.pt")
    with pytest.raises(RuntimeError, match="research"):
        export_onnx(tmp_path / "checkpoint.pt", tmp_path / "model.onnx")


def test_ctf_report_renderers_and_bundle_safety(tmp_path: Path):
    source = tmp_path / "challenge.txt"
    source.write_text("flag{plain}")
    report = CTFService().solve(source, tmp_path / "job", mode="quick")
    value = report.to_dict()
    assert "Artifact graph" in html_v2(value)
    assert sarif_v2(value)["version"] == "2.1.0"
    bundle = tmp_path / "bundle.zip"
    paths = [(item.name, item.path) for item in report.artifacts if item.path]
    write_evidence_bundle(value, paths, bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert {"report.json", "report.html", "report.sarif"} <= set(archive.namelist())
    with pytest.raises(FileExistsError):
        write_evidence_bundle(value, paths, bundle)
    symlink = tmp_path / "artifact-link"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        write_evidence_bundle(value, [("bad", symlink)], tmp_path / "bad.zip")


def test_v2_ctf_job_events_download_and_errors(tmp_path: Path):
    app = create_app(APISettings(state_dir=tmp_path / "state", api_key="secret"))
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer secret"}
        assert client.post("/v2/ctf/jobs", files={"file": ("a", b"x")}).status_code == 401
        invalid = client.post(
            "/v2/ctf/jobs",
            headers=headers,
            files={"file": ("a.txt", b"x")},
            data={"mode": "invalid"},
        )
        assert invalid.status_code == 422
        assert client.get("/v2/models/catalog", headers=headers).json()["models"] == []
        assert (
            client.post("/v2/models", headers=headers, json={"target": "missing@1"}).status_code
            == 422
        )
        assert client.post("/v2/models", headers=headers, json={}).status_code == 422
        created = client.post(
            "/v2/ctf/jobs",
            headers=headers,
            files={"file": ("a.txt", base64.b64encode(b"flag{api}"), "text/plain")},
            data={"mode": "quick"},
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        for _ in range(100):
            job = client.get(f"/v2/ctf/jobs/{job_id}", headers=headers).json()
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert job["status"] == "completed" and job["report"]["report_type"] == "ctf"
        events = client.get(f"/v2/ctf/jobs/{job_id}/events", headers=headers)
        assert "event: terminal" in events.text
        artifact = job["artifacts"][-1]
        assert client.get(artifact["download_url"], headers=headers).content == b"flag{api}"
        assert client.delete(f"/v2/ctf/jobs/{job_id}", headers=headers).status_code == 202
        assert client.get("/v2/ctf/jobs/missing", headers=headers).status_code == 404
        assert client.delete("/v2/ctf/jobs/missing", headers=headers).status_code == 404
        assert client.get("/v2/ctf/jobs/missing/events", headers=headers).status_code == 404
        missing_artifact = f"/v2/ctf/jobs/{job_id}/artifacts/missing"
        assert client.get(missing_artifact, headers=headers).status_code == 404


def test_guided_ctf_tui_runs_and_validates_output(tmp_path: Path):
    source = tmp_path / "tui.txt"
    source.write_bytes(base64.b64encode(b"flag{tui}"))
    output = tmp_path / "tui-output"

    async def scenario():
        app = SteganographyTUI(state_dir=tmp_path / "state")
        async with app.run_test(size=(100, 34)) as pilot:
            app.push_screen(CTFScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CTFScreen)
            screen.query_one("#ctf-input", Input).value = str(source)
            screen.query_one("#ctf-output", Input).value = str(output)
            screen.query_one("#ctf-mode", Select).value = "quick"
            screen.start()
            for _index in range(100):
                await pilot.pause(0.02)
                if (output / "report.json").is_file():
                    break
            assert json.loads((output / "report.json").read_text())["status"] == "completed"
            assert "base64" in str(screen.query_one("#ctf-result").render())
            screen.start()
            screen.cancel()

    asyncio.run(scenario())
