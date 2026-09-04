import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

from cli import main


def test_cli_v2_analysis_scan_and_case_lifecycle(tmp_path: Path, capsys):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("ordinary evidence\n")
    assert main(["--quiet", "analyze", "--in", str(evidence), "--format", "json-v2"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "2.0"

    scan_out = tmp_path / "scan.sarif"
    assert (
        main(
            [
                "--quiet",
                "scan",
                "--dir",
                str(tmp_path),
                "--report",
                "sarif",
                "--out",
                str(scan_out),
                "--jobs",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(scan_out.read_text())["version"] == "2.1.0"
    capsys.readouterr()
    ndjson_out = tmp_path / "scan.ndjson"
    assert main(
        [
            "--quiet",
            "scan",
            "--dir",
            str(tmp_path),
            "--report",
            "ndjson",
            "--out",
            str(ndjson_out),
            "--jobs",
            "1",
        ]
    ) == 0
    assert all(
        json.loads(line)["schema_version"] == "2.0"
        for line in ndjson_out.read_text().splitlines()
    )
    capsys.readouterr()

    state = tmp_path / "state"
    assert (
        main(
            [
                "--quiet",
                "case",
                "--state-dir",
                str(state),
                "create",
                "--name",
                "CLI Case",
            ]
        )
        == 0
    )
    case = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "--quiet",
                "case",
                "--state-dir",
                str(state),
                "add",
                case["id"],
                "--file",
                str(evidence),
                "--password",
                "vault-password",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--quiet",
                "case",
                "--state-dir",
                str(state),
                "scan",
                case["id"],
                "--password",
                "vault-password",
            ]
        )
        == 0
    )
    completed = json.loads(capsys.readouterr().out)
    report = tmp_path / "case.html"
    assert (
        main(
            [
                "--quiet",
                "case",
                "--state-dir",
                str(state),
                "export",
                completed["id"],
                "--format",
                "html",
                "--out",
                str(report),
            ]
        )
        == 0
    )
    assert "steganography DFIR report" in report.read_text()
    capsys.readouterr()
    assert main(["--quiet", "doctor", "--state-dir", str(state)]) == 0
    assert json.loads(capsys.readouterr().out)["network_default"] == "disabled"


def test_cli_payload_v3_models_and_research(tmp_path: Path, capsys):
    cover = tmp_path / "cover.png"
    Image.fromarray(np.random.default_rng(91).integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(
        cover
    )
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"v3 cli payload" * 5)
    stego = tmp_path / "stego.png"
    assert (
        main(
            [
                "--quiet",
                "embed",
                "--in",
                str(payload),
                "--carrier",
                str(cover),
                "--out",
                str(stego),
                "--payload-version",
                "3",
                "--compress",
                "--ecc-symbols",
                "4",
                "--password",
                "payload-password",
            ]
        )
        == 0
    )
    recovered = tmp_path / "recovered.bin"
    assert (
        main(
            [
                "--quiet",
                "extract",
                "--in",
                str(stego),
                "--out",
                str(recovered),
                "--password",
                "payload-password",
            ]
        )
        == 0
    )
    assert recovered.read_bytes() == payload.read_bytes()
    capsys.readouterr()

    state = tmp_path / "models-state"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    manifest = {
        "id": "cli-model",
        "version": "1",
        "domain": "spatial-srnet-v1",
        "file": model.name,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "license": "MIT",
        "model_card": "test",
        "benchmark_summary": {"dataset": "test", "roc_auc": 0.9},
    }
    private = Ed25519PrivateKey.generate()
    unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["signature"] = base64.b64encode(private.sign(unsigned)).decode()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    public = base64.b64encode(private.public_key().public_bytes_raw()).decode()
    assert (
        main(
            [
                "--quiet",
                "models",
                "--state-dir",
                str(state),
                "install",
                "--manifest",
                str(manifest_path),
                "--public-key",
                public,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "--quiet",
                "models",
                "--state-dir",
                str(state),
                "verify",
                "cli-model",
                "1",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "valid"
    assert main(["--quiet", "models", "--state-dir", str(state), "list"]) == 0
    assert len(json.loads(capsys.readouterr().out)["items"]) == 1

    dataset = tmp_path / "dataset" / "cover"
    dataset.mkdir(parents=True)
    (dataset / "one.png").write_bytes(b"one")
    manifest_out = tmp_path / "dataset.json"
    assert (
        main(
            [
                "--quiet",
                "research",
                "import",
                "--source",
                str(dataset.parent),
                "--out",
                str(manifest_out),
            ]
        )
        == 0
    )
    assert json.loads(manifest_out.read_text())["sample_count"] == 1
    capsys.readouterr()

    held_out = tmp_path / "held-out"
    samples = []
    predictions = {}
    for index, (label, group, score) in enumerate(
        (
            ("cover", "camera-a", 0.01),
            ("stego", "camera-a", 0.99),
            ("cover", "camera-b", 0.02),
            ("stego", "camera-b", 0.98),
        )
    ):
        relative = Path(group) / label / f"{index}.png"
        sample_path = held_out / relative
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_bytes(f"held-out-{index}".encode())
        digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        samples.append(
            {
                "path": relative.as_posix(),
                "sha256": digest,
                "size": sample_path.stat().st_size,
                "label": label,
                "source_group": group,
                "split": "test",
            }
        )
        predictions[digest] = score
    held_manifest = tmp_path / "held-manifest.json"
    held_manifest.write_text(
        json.dumps({"schema_version": "1.0", "source": str(held_out), "samples": samples})
    )
    prediction_file = tmp_path / "predictions.json"
    prediction_file.write_text(json.dumps({"predictions": predictions}))
    benchmark_out = tmp_path / "research-report.json"
    assert (
        main(
            [
                "--quiet",
                "research",
                "benchmark",
                "--manifest",
                str(held_manifest),
                "--predictions",
                str(prediction_file),
                "--out",
                str(benchmark_out),
                "--min-samples",
                "4",
                "--min-source-groups",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert json.loads(benchmark_out.read_text())["evaluation"]["metrics"]["roc_auc"] == 1.0
