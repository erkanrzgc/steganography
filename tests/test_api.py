import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import APISettings, _is_loopback, create_app, run_server
from api.jobs import JobStore


def _wait_for_job(client: TestClient, job_id: str, headers: dict) -> dict:
    for _ in range(100):
        job = client.get(f"/v1/scans/{job_id}", headers=headers).json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("scan job did not complete")


def test_api_auth_health_modules_and_single_analysis(tmp_path: Path):
    state = tmp_path / "state"
    app = create_app(APISettings(state_dir=state, api_key="test-key"))
    headers = {"Authorization": "Bearer test-key"}
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 401
        assert client.get("/healthz", headers=headers).json()["status"] == "ok"
        modules = client.get("/v1/modules", headers=headers)
        assert modules.status_code == 200
        assert any(item["id"] == "image_lsb" for item in modules.json()["carriers"])

        response = client.post(
            "/v1/analyze",
            headers=headers,
            files={"file": ("evidence.txt", b"ordinary text\n", "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == "1.0"
        assert body["file"]["name"] == "evidence.txt"
        assert body["file"]["path"] == "evidence.txt"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
    assert not list(state.glob("single-*"))


def test_batch_job_persists_and_uploads_are_removed(tmp_path: Path):
    state = tmp_path / "state"
    settings = APISettings(state_dir=state, retention_days=30)
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/v1/scans?profile=balanced",
            files=[
                ("files", ("a.txt", b"alpha\n", "text/plain")),
                ("files", ("b.txt", b"beta\n", "text/plain")),
            ],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = _wait_for_job(client, job_id, {})
        assert job["status"] == "completed"
        assert job["result"]["summary"]["files"] == 2
        assert [item["file"]["name"] for item in job["result"]["files"]] == [
            "a.txt",
            "b.txt",
        ]
        assert not (state / "uploads" / job_id).exists()
        assert client.get("/v1/scans/missing").status_code == 404
    restarted = JobStore(state / "api.sqlite3")
    assert restarted.get(job_id)["status"] == "completed"
    assert restarted.delete(job_id)
    assert restarted.get(job_id) is None


def test_api_limits_and_delete(tmp_path: Path):
    settings = APISettings(
        state_dir=tmp_path / "state",
        max_file_bytes=3,
        max_batch_files=1,
        max_batch_bytes=3,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        too_large = client.post(
            "/v1/analyze", files={"file": ("x.bin", b"four", "application/octet-stream")}
        )
        assert too_large.status_code == 413
        too_many = client.post(
            "/v1/scans",
            files=[
                ("files", ("a", b"a", "application/octet-stream")),
                ("files", ("b", b"b", "application/octet-stream")),
            ],
        )
        assert too_many.status_code == 413

        response = client.post(
            "/v1/scans", files=[("files", ("a", b"a", "application/octet-stream"))]
        )
        job_id = response.json()["job_id"]
        _wait_for_job(client, job_id, {})
        assert client.delete(f"/v1/scans/{job_id}").status_code == 204
        assert client.delete(f"/v1/scans/{job_id}").status_code == 404

    batch_app = create_app(
        APISettings(
            state_dir=tmp_path / "batch-state",
            max_file_bytes=3,
            max_batch_files=2,
            max_batch_bytes=3,
        )
    )
    with TestClient(batch_app) as client:
        response = client.post(
            "/v1/scans",
            files=[
                ("files", ("a", b"aa", "application/octet-stream")),
                ("files", ("b", b"bb", "application/octet-stream")),
            ],
        )
        assert response.status_code == 413
        assert "batch byte" in response.json()["detail"]


def test_ai_file_upload_requires_server_and_request_opt_in(
    tmp_path: Path, monkeypatch
):
    observed: list[bool] = []

    def fake_make_provider(*, upload_file=True, **kwargs):
        observed.append(upload_file)

        def provider(info, signals):
            return 1, "ok"

        return provider

    import modules.ai_provider_nim

    monkeypatch.setattr(modules.ai_provider_nim, "make_provider", fake_make_provider)
    app = create_app(
        APISettings(state_dir=tmp_path / "off", allow_ai_file_upload=False)
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/analyze?ai=true&allow_ai_file_upload=true",
            files={"file": ("x.txt", b"x", "text/plain")},
        )
        assert response.status_code == 200
    assert observed == [False]

    observed.clear()
    app = create_app(
        APISettings(state_dir=tmp_path / "on", allow_ai_file_upload=True)
    )
    with TestClient(app) as client:
        client.post(
            "/v1/analyze?ai=true&allow_ai_file_upload=true",
            files={"file": ("x.txt", b"x", "text/plain")},
        )
    assert observed == [True]


def test_loopback_and_public_bind_policy(monkeypatch):
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")  # noqa: S104 - policy test only
    assert not _is_loopback("not-an-address")
    monkeypatch.delenv("STEGANO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="STEGANO_API_KEY"):
        run_server(host="0.0.0.0", port=8000)  # noqa: S104 - policy rejects it


def test_job_store_marks_interrupted_jobs_failed_and_expires(tmp_path: Path):
    database = tmp_path / "jobs.sqlite3"
    store = JobStore(database, retention_days=30)
    store.create("running", "sensitive")
    store.set_running("running")
    restarted = JobStore(database, retention_days=30)
    job = restarted.get("running")
    assert job["status"] == "failed"
    assert "restarted" in job["error"]

    expiring = JobStore(tmp_path / "expiring.sqlite3", retention_days=0)
    expiring.create("gone", "strict")
    assert expiring.get("gone") is None

    failed = JobStore(tmp_path / "failed.sqlite3", retention_days=30)
    failed.create("failed", "balanced")
    failed.fail("failed", "analysis failed")
    assert failed.get("failed")["error"] == "analysis failed"
