import base64
import json
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from PIL import Image

from api.app import APISettings, create_app
from core.automation import iter_ndjson, sign_webhook
from core.cases import CaseService
from core.database import Database
from core.ensemble import HybridEnsemble, ModelScore
from core.models import ModelRegistry, ModelVerificationError
from core.pipeline import AnalysisPipeline
from core.result import AnalysisResult, Signal
from core.scans import ScanService
from core.service import AnalysisService, ExtractionError, StegoService
from core.vault import VaultAuthenticationError, VaultIntegrityError, VaultService
from modules.archive_safe import ArchiveSafeAnalyzer
from report.v2 import html_v2, sarif_v2
from steganography.research import (
    ResearchManifestError,
    benchmark_predictions,
    import_dataset,
    verify_dataset_manifest,
    verify_split_isolation,
)


def test_vault_cases_dedup_tamper_and_audit(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite3")
    vault = VaultService(database, tmp_path / "vault")
    vault.initialize("vault password")  # noqa: S106
    cases = CaseService(database, vault)
    first = cases.create_case("Case One", retention_days=30)
    second = cases.create_case("Case Two")
    expiring = cases.create_case("Expired", retention_days=0)
    assert database.purge_expired_cases() == 1
    with pytest.raises(LookupError):
        cases.get_case(expiring["id"])
    source = tmp_path / "evidence.bin"
    source.write_bytes(b"evidence bytes")
    one = cases.add_evidence(first["id"], source)
    two = cases.add_evidence(second["id"], source)
    assert one["sha256"] == two["sha256"]
    with database.connect() as connection:
        obj = connection.execute("SELECT * FROM vault_objects").fetchone()
        assert obj["ref_count"] == 2
        encrypted = tmp_path / "vault" / "objects" / obj["storage_name"]
    assert b"evidence bytes" not in encrypted.read_bytes()
    vault.lock()
    with pytest.raises(VaultAuthenticationError):
        vault.unlock("wrong")
    vault.unlock("vault password")  # noqa: S106
    assert vault.read(one["sha256"]) == b"evidence bytes"
    blob = bytearray(encrypted.read_bytes())
    blob[-1] ^= 1
    encrypted.write_bytes(blob)
    with pytest.raises(VaultIntegrityError):
        vault.read(one["sha256"])
    valid, events, error = database.verify_audit()
    assert valid and events >= 5 and error is None


def test_ensemble_and_offline_automation_helpers():
    results = (
        AnalysisResult(
            "one",
            60,
            (
                Signal("a", 60, "a", category="correlated"),
                Signal("b", 40, "b", category="correlated"),
            ),
            None,
        ),
    )
    model = ModelScore(
        "model",
        "spatial-srnet-v1",
        0.8,
        "ok",
        execution_provider="CPUExecutionProvider",
        calibrated=True,
    )
    score = HybridEnsemble().combine(results, model=model)
    assert score.score == 80 and score.deterministic_score == 60
    assert list(iter_ndjson([{"b": 2, "a": 1}])) == [b'{"a":1,"b":2}\n']
    headers = sign_webhook(b"{}", b"long-enough-webhook-secret", timestamp=1)
    assert headers["X-Steganography-Signature"].startswith("sha256=")


def test_pipeline_scan_reports_and_archive_limits(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    vault = VaultService(database, tmp_path / "vault")
    vault.initialize("password")  # noqa: S106
    cases = CaseService(database, vault)
    case = cases.create_case("Scan")
    evidence = tmp_path / "plain.txt"
    evidence.write_text("ordinary local evidence\n")
    cases.add_evidence(case["id"], evidence)
    scans = ScanService(
        database,
        vault,
        cases,
        lambda profile: AnalysisPipeline(AnalysisService(profile=profile, ai_provider=None)),
    )
    scan = scans.create(case["id"])
    completed = scans.run(scan["id"])
    assert completed["status"] == "completed"
    assert completed["result"]["summary"]["files"] == 1
    assert "<!doctype html>" in html_v2(completed["result"])
    assert sarif_v2(completed["result"])["version"] == "2.1.0"

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../../escape.txt", b"x")
    result = ArchiveSafeAnalyzer().analyze(archive)
    assert any(signal.name == "archive_path_traversal" for signal in result.signals)


def test_payload_v3_argon2_compression_ecc_roundtrip(tmp_path: Path):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"compressible payload " * 30)
    cover = tmp_path / "cover.png"
    array = np.random.default_rng(8).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(array).save(cover)
    stego = tmp_path / "stego.png"
    service = StegoService()
    service.embed(
        secret,
        cover,
        stego,
        password="payload password",  # noqa: S106
        payload_version=3,
        compress=True,
        ecc_symbols=8,
    )
    recovered, _, version = service.extract(stego, password="payload password")  # noqa: S106
    assert recovered == secret.read_bytes()
    assert version == 3
    with pytest.raises(ExtractionError):
        service.extract(stego)
    with pytest.raises(ExtractionError, match="authentication"):
        service.extract(stego, password="wrong")  # noqa: S106 - deliberate test input


def test_signed_model_registry_and_research_manifest(tmp_path: Path, monkeypatch):
    database = Database(tmp_path / "db.sqlite3")
    models = ModelRegistry(database, tmp_path / "models")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake onnx for registry test")
    import hashlib

    manifest = {
        "id": "spatial-test",
        "version": "1.0.0",
        "domain": "spatial-srnet-v1",
        "file": model.name,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "license": "MIT",
        "model_card": "test-only",
        "benchmark_summary": {"dataset": "test", "roc_auc": 0.9},
        "calibrated": True,
    }
    private = Ed25519PrivateKey.generate()
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["signature"] = base64.b64encode(private.sign(canonical)).decode()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    public = private.public_key().public_bytes_raw()
    installed = models.install(manifest_path, public_key=public)
    assert installed["state"] == "installed"
    assert models.verify_installed("spatial-test", "1.0.0")
    missing_score = models.score(
        "missing", "1", np.zeros((1,)), domain="spatial-srnet-v1"
    )
    assert missing_score.status == "unavailable"
    assert models.score(
        "spatial-test", "1.0.0", np.zeros((1,)), domain="jpeg-srnet-v1"
    ).status == "unsupported"
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert models.score(
        "spatial-test", "1.0.0", np.zeros((1,)), domain="spatial-srnet-v1"
    ).status == "unavailable"

    class FakeSession:
        def __init__(self, path, providers):
            assert Path(path).is_file()
            self.providers = providers

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def run(self, _outputs, inputs):
            assert "input" in inputs
            return [np.array([[2.0]])]

        def get_providers(self):
            return self.providers

    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    score = models.score(
        "spatial-test", "1.0.0", np.zeros((1,)), domain="spatial-srnet-v1"
    )
    assert score.status == "ok" and score.calibrated
    assert score.execution_provider == "CPUExecutionProvider"
    manifest["sha256"] = "0" * 64
    with pytest.raises(ModelVerificationError):
        models.verify_manifest(manifest, public_key=public)

    dataset = tmp_path / "dataset"
    (dataset / "cover").mkdir(parents=True)
    (dataset / "stego").mkdir()
    for index in range(10):
        target = dataset / ("cover" if index % 2 == 0 else "stego") / f"{index}.png"
        target.write_bytes(b"png" + bytes([index]))
    research = import_dataset(dataset, tmp_path / "dataset.json", seed=7)
    assert research["sample_count"] == 10
    assert verify_split_isolation(research)


def test_research_held_out_benchmark_gates_and_integrity(tmp_path: Path):
    source = tmp_path / "held-out"
    samples = []
    predictions = {}
    for index, (label, group, score) in enumerate(
        (("cover", "camera-a", 0.01), ("stego", "camera-a", 0.99),
         ("cover", "camera-b", 0.02), ("stego", "camera-b", 0.98))
    ):
        relative = Path(group) / label / f"{index}.png"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sample-{index}".encode())
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        samples.append(
            {
                "path": relative.as_posix(),
                "sha256": digest,
                "size": path.stat().st_size,
                "label": label,
                "source_group": group,
                "split": "test",
            }
        )
        predictions[digest] = score
    manifest = {"schema_version": "1.0", "source": str(source), "samples": samples}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps({"predictions": predictions}))
    with pytest.raises(ValueError, match="between 0 and 1"):
        benchmark_predictions(manifest_path, predictions_path, tmp_path / "bad.json", threshold=2)
    with pytest.raises(ValueError, match="positive"):
        benchmark_predictions(manifest_path, predictions_path, tmp_path / "bad.json", min_samples=0)
    report = benchmark_predictions(
        manifest_path,
        predictions_path,
        tmp_path / "report.json",
        min_samples=4,
        min_source_groups=2,
    )
    assert report["status"] == "passed"
    assert report["evaluation"]["metrics"]["roc_auc"] == 1.0
    assert report["evaluation"]["metrics"]["false_positive_rate"] == 0.0
    assert verify_dataset_manifest(manifest)["split_isolation"] is True

    (source / samples[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ResearchManifestError, match="integrity"):
        benchmark_predictions(
            manifest_path,
            predictions_path,
            tmp_path / "tampered-report.json",
            min_samples=4,
        )

    (source / samples[0]["path"]).write_bytes(b"sample-0")
    predictions_path.write_text(
        json.dumps({"predictions": {samples[0]["sha256"]: 0.99}})
    )
    failed = benchmark_predictions(
        manifest_path,
        predictions_path,
        tmp_path / "failed-report.json",
        min_samples=5,
        min_source_groups=3,
    )
    assert failed["status"] == "failed"
    assert len(failed["gates"]["reasons"]) >= 5


def test_v2_api_auth_vault_case_evidence_scan_and_report(tmp_path: Path):
    app = create_app(APISettings(state_dir=tmp_path / "state", api_key="api-secret"))
    with TestClient(app) as client:
        assert "steganography" in client.get("/").text
        assert client.get("/v2/health").status_code == 200
        assert client.get("/v2/cases").status_code == 401
        assert client.post("/v2/session", json={"api_key": "wrong"}).status_code == 401
        login = client.post("/v2/session", json={"api_key": "api-secret"})
        assert login.status_code == 200
        csrf = login.json()["csrf_token"]
        headers = {"X-CSRF-Token": csrf}
        assert client.get("/v2/vault").json()["initialized"] is False
        assert client.post("/v2/cases", json={"name": "No CSRF"}).status_code == 403
        assert (
            client.post(
                "/v2/vault/initialize", json={"password": "vault-secret"}, headers=headers
            ).status_code
            == 200
        )
        case = client.post("/v2/cases", json={"name": "API Case"}, headers=headers)
        assert case.status_code == 201
        case_id = case.json()["id"]
        evidence = client.post(
            f"/v2/cases/{case_id}/evidence",
            headers=headers,
            files={"file": ("evidence.txt", b"normal evidence\n", "text/plain")},
        )
        assert evidence.status_code == 201
        evidence_id = evidence.json()["id"]
        assert len(client.get(f"/v2/cases/{case_id}/evidence").json()["items"]) == 1
        assert client.get("/v2/cases/missing/evidence").status_code == 404
        assert client.get(f"/v2/evidence/{evidence_id}").status_code == 200
        assert client.get("/v2/evidence/missing").status_code == 404
        scan = client.post(
            f"/v2/cases/{case_id}/scans",
            headers=headers,
            json={"profile": "sensitive"},
        )
        assert scan.status_code == 202
        job_id = scan.json()["id"]
        assert len(client.get(f"/v2/cases/{case_id}/scans").json()["items"]) == 1
        assert client.get("/v2/cases/missing/scans").status_code == 404
        for _ in range(100):
            job = client.get(f"/v2/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed"
        events = client.get(f"/v2/jobs/{job_id}/events")
        assert events.status_code == 200 and "event: progress" in events.text
        assert client.get(f"/v2/reports/{job_id}").status_code == 200
        assert client.get(f"/v2/reports/{job_id}?format=html").status_code == 200
        assert client.get(f"/v2/reports/{job_id}?format=sarif").status_code == 200
        assert client.get(f"/v2/reports/{job_id}?format=bad").status_code == 422
        assert client.get("/v2/jobs/missing").status_code == 404

        carrier = np.random.default_rng(10).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        carrier_path = tmp_path / "api-cover.png"
        Image.fromarray(carrier).save(carrier_path)
        capacity = client.post(
            "/v2/studio/capacity",
            headers=headers,
            files={"carrier": ("cover.png", carrier_path.read_bytes(), "image/png")},
        )
        assert capacity.status_code == 200
        embedded = client.post(
            "/v2/studio/embed",
            headers=headers,
            files={
                "carrier": ("cover.png", carrier_path.read_bytes(), "image/png"),
                "payload": ("secret.bin", b"studio secret", "application/octet-stream"),
            },
        )
        assert embedded.status_code == 201
        artifact = embedded.json()["artifact"]
        download = client.get(artifact["download_url"])
        assert download.status_code == 200
        extracted = client.post(
            "/v2/studio/extract",
            headers=headers,
            files={"carrier": ("stego.png", download.content, "image/png")},
        )
        assert extracted.status_code == 201
        recovered = client.get(extracted.json()["artifact"]["download_url"])
        assert recovered.content == b"studio secret"
        assert client.get("/v2/artifacts/missing/download").status_code == 404

        assert client.get("/v2/models").status_code == 200
        assert (
            client.post(
                "/v2/models",
                headers=headers,
                json={"manifest_path": "/missing", "public_key": "bad"},
            ).status_code
            == 422
        )
        assert client.get("/v2/audit/verify").json()["valid"]
        assert client.post("/v2/vault/lock", headers=headers).status_code == 200
        assert (
            client.post("/v2/vault/unlock", headers=headers, json={"password": "wrong"}).status_code
            == 401
        )
        assert (
            client.post(
                "/v2/vault/unlock", headers=headers, json={"password": "vault-secret"}
            ).status_code
            == 200
        )
        assert client.get(f"/v2/cases/{case_id}").status_code == 200
        assert client.delete(f"/v2/cases/{case_id}", headers=headers).status_code == 204
        assert client.get(f"/v2/cases/{case_id}").status_code == 404
        assert client.delete("/v2/session", headers=headers).status_code == 204
