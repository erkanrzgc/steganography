"""Authenticated local-first v2 API routes."""

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from core.cases import CaseNotFoundError, EvidenceNotFoundError
from core.ctf import CTFLimits, CTFMode, CTFService
from core.database import Database, utc_now
from core.models import ModelVerificationError
from core.scans import ScanNotFoundError
from core.service import ExtractionError
from core.vault import (
    VaultAuthenticationError,
    VaultError,
    VaultLockedError,
)
from core.version import __version__
from core.workspace import create_workspace
from report.v2 import html_v2, sarif_v2

_CHUNK_SIZE = 1024 * 1024


class SessionRequest(BaseModel):
    api_key: str


class VaultPassword(BaseModel):
    password: str = Field(min_length=1)


class CaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10_000)
    retention_days: int | None = Field(default=None, ge=0, le=36_500)


class ScanCreate(BaseModel):
    profile: str = "sensitive"


class ModelInstall(BaseModel):
    manifest_path: str | None = None
    public_key: str | None = None
    target: str | None = None
    catalog_path: str | None = None
    accept_license: bool = False


class V2Security:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.sessions: dict[str, tuple[str, float]] = {}

    def login(self, supplied: str) -> tuple[str, str]:
        if not hmac.compare_digest(supplied, self.api_key):
            raise HTTPException(status_code=401, detail="invalid API key")
        session = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        self.sessions[session] = (csrf, time.monotonic() + 12 * 60 * 60)
        return session, csrf

    async def authenticate(self, request: Request) -> None:
        authorization = request.headers.get("authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(supplied, self.api_key):
            return
        session = request.cookies.get("steg_session", "")
        session_value = self.sessions.get(session)
        if session_value is None:
            raise HTTPException(status_code=401, detail="authentication required")
        csrf, expires_at = session_value
        if time.monotonic() >= expires_at:
            self.sessions.pop(session, None)
            raise HTTPException(status_code=401, detail="session expired")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not hmac.compare_digest(
            request.headers.get("x-csrf-token", ""), csrf
        ):
            raise HTTPException(status_code=403, detail="CSRF token missing or invalid")


def install_v2_routes(
    app: Any,
    *,
    state_dir: Path,
    configured_api_key: str | None,
    executor: ThreadPoolExecutor,
    max_file_bytes: int,
) -> None:
    workspace = create_workspace(state_dir)
    database = workspace.database
    vault = workspace.vault
    cases = workspace.cases
    models = workspace.models
    studio = workspace.studio
    scans = workspace.scans
    api_key = configured_api_key or _local_api_key(state_dir / "v2-api.key")
    security = V2Security(api_key)

    app.state.workspace = workspace
    app.state.database = database
    app.state.vault = vault
    app.state.cases = cases
    app.state.models = models
    app.state.scans_v2 = scans
    app.state.v2_api_key_path = state_dir / "v2-api.key"

    ctf_jobs: dict[str, dict[str, Any]] = {}
    ctf_cancel: dict[str, threading.Event] = {}
    ctf_lock = threading.Lock()
    ctf_root = state_dir / "ctf-jobs"
    ctf_uploads = state_dir / "ctf-uploads"
    ctf_root.mkdir(parents=True, exist_ok=True)
    ctf_uploads.mkdir(parents=True, exist_ok=True)
    app.state.ctf_jobs = ctf_jobs

    router = APIRouter(prefix="/v2")
    auth = Annotated[None, Depends(security.authenticate)]

    @router.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "vault": "unlocked" if vault.unlocked else "locked",
        }

    @router.post("/session")
    async def create_session(body: SessionRequest) -> JSONResponse:
        session, csrf = security.login(body.api_key)
        response = JSONResponse({"status": "ok", "csrf_token": csrf})
        response.set_cookie(
            "steg_session",
            session,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/v2",
        )
        return response

    @router.delete("/session", status_code=204)
    async def delete_session(request: Request, _: auth) -> Response:
        session = request.cookies.get("steg_session", "")
        security.sessions.pop(session, None)
        response = Response(status_code=204)
        response.delete_cookie("steg_session", path="/v2")
        return response

    @router.get("/vault")
    async def vault_status(_: auth) -> dict[str, Any]:
        return {"initialized": vault.initialized, "unlocked": vault.unlocked}

    @router.post("/vault/initialize")
    async def vault_initialize(body: VaultPassword, _: auth) -> dict[str, bool]:
        try:
            vault.initialize(body.password)
        except VaultError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"initialized": True, "unlocked": True}

    @router.post("/vault/unlock")
    async def vault_unlock(body: VaultPassword, _: auth) -> dict[str, bool]:
        try:
            vault.unlock(body.password)
        except VaultAuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except VaultError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"initialized": True, "unlocked": True}

    @router.post("/vault/lock")
    async def vault_lock(_: auth) -> dict[str, bool]:
        vault.lock()
        return {"unlocked": False}

    @router.get("/cases")
    async def list_cases(_: auth) -> dict[str, Any]:
        return {"items": cases.list_cases()}

    @router.post("/cases", status_code=201)
    async def create_case(body: CaseCreate, _: auth) -> dict[str, Any]:
        return cases.create_case(
            body.name,
            description=body.description,
            retention_days=body.retention_days,
        )

    @router.get("/cases/{case_id}")
    async def get_case(case_id: str, _: auth) -> dict[str, Any]:
        try:
            value = cases.get_case(case_id)
            value["evidence"] = cases.list_evidence(case_id)
            value["scans"] = scans.list_for_case(case_id)
            return value
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc

    @router.delete("/cases/{case_id}", status_code=204)
    async def delete_case(case_id: str, _: auth) -> Response:
        try:
            cases.delete_case(case_id)
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc
        return Response(status_code=204)

    @router.post("/cases/{case_id}/evidence", status_code=201)
    async def add_evidence(
        case_id: str,
        _: auth,
        file: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        temporary_dir = Path(tempfile.mkdtemp(prefix="v2-evidence-", dir=state_dir))
        try:
            path, _upload_size = await _save_upload(file, temporary_dir, max_file_bytes)
            return cases.add_evidence(
                case_id,
                path,
                original_name=Path(file.filename or "evidence.bin").name,
                media_type=file.content_type,
            )
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc
        except VaultLockedError as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        finally:
            await file.close()
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @router.get("/cases/{case_id}/evidence")
    async def list_evidence(case_id: str, _: auth) -> dict[str, Any]:
        try:
            return {"items": cases.list_evidence(case_id)}
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc

    @router.get("/evidence/{evidence_id}")
    async def get_evidence(evidence_id: str, _: auth) -> dict[str, Any]:
        try:
            return cases.get_evidence(evidence_id)
        except EvidenceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="evidence not found") from exc

    @router.post("/cases/{case_id}/scans", status_code=202)
    async def create_scan(case_id: str, body: ScanCreate, _: auth) -> dict[str, Any]:
        if body.profile not in {"sensitive", "balanced", "strict"}:
            raise HTTPException(status_code=422, detail="invalid analysis profile")
        if not vault.unlocked:
            raise HTTPException(status_code=423, detail="vault is locked")
        try:
            scan = scans.create(case_id, profile=body.profile)
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc
        executor.submit(scans.run, scan["id"])
        return scan

    @router.get("/cases/{case_id}/scans")
    async def list_scans(case_id: str, _: auth) -> dict[str, Any]:
        try:
            return {"items": scans.list_for_case(case_id)}
        except CaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail="case not found") from exc

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str, _: auth) -> dict[str, Any]:
        try:
            return scans.get(job_id)
        except ScanNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, _: auth) -> StreamingResponse:
        try:
            scans.get(job_id)
        except ScanNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

        async def events():
            previous = ""
            while True:
                value = scans.get(job_id)
                encoded = json.dumps(value, separators=(",", ":"))
                if encoded != previous:
                    yield f"event: progress\ndata: {encoded}\n\n"
                    previous = encoded
                if value["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.get("/reports/{report_id}")
    async def get_report(report_id: str, _: auth, format: str = "json") -> Response:
        try:
            scan = scans.get(report_id)
        except ScanNotFoundError as exc:
            raise HTTPException(status_code=404, detail="report not found") from exc
        if scan["status"] != "completed" or scan["result"] is None:
            raise HTTPException(status_code=409, detail="report is not ready")
        if format == "html":
            return HTMLResponse(html_v2(scan["result"]))
        if format == "sarif":
            return JSONResponse(sarif_v2(scan["result"]), media_type="application/sarif+json")
        if format != "json":
            raise HTTPException(status_code=422, detail="format must be json, html or sarif")
        return JSONResponse(scan["result"])

    def run_ctf_job(
        job_id: str,
        input_path: Path,
        output_path: Path,
        mode: str,
        wordlist_path: Path | None,
        password: str | None,
        limits: CTFLimits,
    ) -> None:
        cancel = ctf_cancel[job_id]

        def event(value: dict[str, Any]) -> None:
            with ctf_lock:
                job = ctf_jobs[job_id]
                job["events"].append(value)
                if value.get("stage"):
                    job["stage"] = value["stage"]

        with ctf_lock:
            ctf_jobs[job_id]["status"] = "running"
        try:
            report = CTFService().solve(
                input_path,
                output_path,
                mode=cast(CTFMode, mode),
                wordlist=wordlist_path,
                password=password,
                limits=limits,
                should_cancel=cancel.is_set,
                on_event=event,
            )
            value = report.to_dict()
            with ctf_lock:
                job = ctf_jobs[job_id]
                job["status"] = report.status
                job["report"] = value
                job["artifacts"] = {
                    artifact.id: artifact.path
                    for artifact in report.artifacts
                    if artifact.path is not None
                }
                job["artifact_metadata"] = {
                    artifact.id: artifact.to_dict() for artifact in report.artifacts
                }
        except Exception as exc:  # worker failures must become observable job state
            with ctf_lock:
                ctf_jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(input_path.parent, ignore_errors=True)

    @router.post("/ctf/jobs", status_code=202)
    async def create_ctf_job(
        _: auth,
        file: Annotated[UploadFile, File()],
        mode: Annotated[str, Form()] = "balanced",
        wordlist: Annotated[UploadFile | None, File()] = None,
        password: Annotated[str | None, Form()] = None,
        max_depth: Annotated[int, Form(ge=0, le=20)] = 3,
        max_artifacts: Annotated[int, Form(ge=1, le=4096)] = 256,
        max_bytes: Annotated[int, Form(ge=1)] = 1024 * 1024 * 1024,
        timeout: Annotated[float, Form(gt=0, le=3600)] = 180.0,
    ) -> dict[str, Any]:
        if mode not in {"quick", "balanced", "deep"}:
            raise HTTPException(status_code=422, detail="invalid CTF mode")
        job_id = uuid.uuid4().hex
        upload_dir = ctf_uploads / job_id
        upload_dir.mkdir()
        try:
            input_path, _input_size = await _save_upload(file, upload_dir, max_file_bytes)
            wordlist_path: Path | None = None
            if wordlist is not None:
                wordlist_path, _wordlist_size = await _save_upload(
                    wordlist, upload_dir, max_file_bytes
                )
            limits = CTFLimits(
                max_depth=max_depth,
                max_artifacts=max_artifacts,
                max_bytes=min(max_bytes, 8 * 1024 * 1024 * 1024),
                tool_timeout=min(30.0, timeout),
                job_timeout=timeout,
            )
            public: dict[str, Any] = {
                "id": job_id,
                "status": "queued",
                "stage": "queued",
                "created_at": utc_now(),
                "events": [],
                "report": None,
                "error": None,
            }
            with ctf_lock:
                ctf_jobs[job_id] = {
                    **public,
                    "artifacts": {},
                    "artifact_metadata": {},
                }
                ctf_cancel[job_id] = threading.Event()
            executor.submit(
                run_ctf_job,
                job_id,
                input_path,
                ctf_root / job_id,
                mode,
                wordlist_path,
                password,
                limits,
            )
            return public
        except Exception:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        finally:
            await file.close()
            if wordlist is not None:
                await wordlist.close()

    @router.get("/ctf/jobs/{job_id}")
    async def get_ctf_job(job_id: str, _: auth) -> dict[str, Any]:
        with ctf_lock:
            job = ctf_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="CTF job not found")
            return _public_ctf_job(job)

    @router.get("/ctf/jobs/{job_id}/events")
    async def ctf_job_events(job_id: str, _: auth) -> StreamingResponse:
        with ctf_lock:
            if job_id not in ctf_jobs:
                raise HTTPException(status_code=404, detail="CTF job not found")

        async def events():
            cursor = 0
            while True:
                with ctf_lock:
                    job = ctf_jobs[job_id]
                    pending = list(job["events"][cursor:])
                    status = job["status"]
                for event_value in pending:
                    name = event_value.get("event", "progress")
                    yield f"event: {name}\ndata: {json.dumps(event_value)}\n\n"
                    cursor += 1
                if status in {"completed", "failed", "cancelled"}:
                    yield f"event: terminal\ndata: {json.dumps({'status': status})}\n\n"
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.delete("/ctf/jobs/{job_id}", status_code=202)
    async def cancel_ctf_job(job_id: str, _: auth) -> dict[str, str]:
        with ctf_lock:
            cancel = ctf_cancel.get(job_id)
            job = ctf_jobs.get(job_id)
            if cancel is None or job is None:
                raise HTTPException(status_code=404, detail="CTF job not found")
            if job["status"] not in {"completed", "failed", "cancelled"}:
                cancel.set()
            return {"id": job_id, "status": "cancellation_requested"}

    @router.get("/ctf/jobs/{job_id}/artifacts/{artifact_id}")
    async def download_ctf_artifact(job_id: str, artifact_id: str, _: auth) -> Response:
        with ctf_lock:
            job = ctf_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="CTF job not found")
            path = job["artifacts"].get(artifact_id)
            metadata = job["artifact_metadata"].get(artifact_id)
        if path is None or metadata is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        path = Path(path)
        allowed = (ctf_root / job_id / "artifacts").resolve()
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(allowed):
            raise HTTPException(status_code=410, detail="artifact is unavailable")
        headers = {"Content-Disposition": f'attachment; filename="{Path(metadata["name"]).name}"'}
        return Response(path.read_bytes(), media_type=metadata["media_type"], headers=headers)

    @router.post("/studio/capacity")
    async def studio_capacity(
        _: auth,
        carrier: Annotated[UploadFile, File()],
        method: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        temporary_dir = Path(tempfile.mkdtemp(prefix="v2-studio-", dir=state_dir))
        try:
            path, _upload_size = await _save_upload(carrier, temporary_dir, max_file_bytes)
            return studio.capacity(path, method=method)
        finally:
            await carrier.close()
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @router.post("/studio/embed", status_code=201)
    async def studio_embed(
        _: auth,
        payload: Annotated[UploadFile, File()],
        carrier: Annotated[UploadFile, File()],
        password: Annotated[str | None, Form()] = None,
        steg_key: Annotated[str | None, Form()] = None,
        method: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        temporary_dir = Path(tempfile.mkdtemp(prefix="v2-studio-", dir=state_dir))
        try:
            payload_path, _payload_size = await _save_upload(payload, temporary_dir, max_file_bytes)
            carrier_path, _carrier_size = await _save_upload(carrier, temporary_dir, max_file_bytes)
            suffix = Path(carrier.filename or "carrier.bin").suffix
            output = temporary_dir / f"stego{suffix}"
            result = studio.embed(
                payload_path,
                carrier_path,
                output,
                password=password,
                steg_key=steg_key,
                method=method,
            )
            artifact = _store_artifact(database, state_dir, output, "studio-output")
            return {"artifact": artifact, "method": result.carrier}
        finally:
            await payload.close()
            await carrier.close()
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @router.post("/studio/extract", status_code=201)
    async def studio_extract(
        _: auth,
        carrier: Annotated[UploadFile, File()],
        password: Annotated[str | None, Form()] = None,
        steg_key: Annotated[str | None, Form()] = None,
        method: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        temporary_dir = Path(tempfile.mkdtemp(prefix="v2-studio-", dir=state_dir))
        try:
            carrier_path, _carrier_size = await _save_upload(carrier, temporary_dir, max_file_bytes)
            try:
                data, used_method, payload_version = studio.extract(
                    carrier_path, password=password, steg_key=steg_key, method=method
                )
            except ExtractionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            output = temporary_dir / "extracted.bin"
            output.write_bytes(data)
            artifact = _store_artifact(database, state_dir, output, "extracted-payload")
            return {
                "artifact": artifact,
                "method": used_method,
                "payload_version": payload_version,
            }
        finally:
            await carrier.close()
            shutil.rmtree(temporary_dir, ignore_errors=True)

    @router.get("/artifacts/{artifact_id}/download")
    async def download_artifact(artifact_id: str, _: auth) -> Response:
        with database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        path = Path(row["path"])
        allowed = (state_dir / "artifacts").resolve()
        if not path.is_file() or not path.resolve().is_relative_to(allowed):
            raise HTTPException(status_code=410, detail="artifact is unavailable")
        headers = {"Content-Disposition": f'attachment; filename="{Path(row["name"]).name}"'}
        return Response(path.read_bytes(), media_type=row["media_type"], headers=headers)

    @router.get("/models")
    async def list_models(_: auth) -> dict[str, Any]:
        return {"items": models.list(), "runtime": models.runtime_status()}

    @router.get("/models/catalog")
    async def model_catalog(_: auth) -> dict[str, Any]:
        return models.catalog()

    @router.post("/models", status_code=201)
    async def install_model(body: ModelInstall, _: auth) -> dict[str, Any]:
        try:
            if body.target:
                return models.install_catalog(
                    body.target,
                    accept_license=body.accept_license,
                    catalog_path=Path(body.catalog_path) if body.catalog_path else None,
                )
            if body.manifest_path and body.public_key:
                return models.install(Path(body.manifest_path), public_key=body.public_key)
            raise ValueError("provide target or manifest_path with public_key")
        except (OSError, ValueError, ModelVerificationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/audit/verify")
    async def verify_audit(_: auth) -> dict[str, Any]:
        valid, events, error = database.verify_audit()
        return {"valid": valid, "events": events, "error": error}

    app.include_router(router)


async def _save_upload(upload: UploadFile, directory: Path, max_bytes: int) -> tuple[Path, int]:
    suffix = Path(upload.filename or "upload.bin").suffix.lower()
    if len(suffix) > 12 or not all(c.isalnum() or c == "." for c in suffix):
        suffix = ".bin"
    path = directory / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with path.open("xb") as stream:
        while chunk := await upload.read(_CHUNK_SIZE):
            size += len(chunk)
            if size > max_bytes:
                stream.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="file byte limit exceeded")
            stream.write(chunk)
    return path, size


def _local_api_key(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
    return value


def _store_artifact(database: Database, state_dir: Path, source: Path, kind: str) -> dict[str, Any]:
    artifact_id = uuid.uuid4().hex
    root = state_dir / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    name = Path(source.name).name
    destination = root / f"{artifact_id}-{name}"
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    size = destination.stat().st_size
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with database.transaction() as connection:
        connection.execute(
            """INSERT INTO artifacts
               (id, kind, name, media_type, path, sha256, size, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, kind, name, media_type, str(destination), digest, size, utc_now()),
        )
        database.audit(
            "artifact.created",
            object_type="artifact",
            object_id=artifact_id,
            details={"kind": kind, "sha256": digest, "size": size},
            connection=connection,
        )
    return {
        "id": artifact_id,
        "name": name,
        "media_type": media_type,
        "sha256": digest,
        "size": size,
        "download_url": f"/v2/artifacts/{artifact_id}/download",
    }


def _public_ctf_job(job: dict[str, Any]) -> dict[str, Any]:
    value = {
        key: job.get(key) for key in ("id", "status", "stage", "created_at", "report", "error")
    }
    value["artifacts"] = [
        {
            **metadata,
            "download_url": f"/v2/ctf/jobs/{job['id']}/artifacts/{artifact_id}",
        }
        for artifact_id, metadata in job.get("artifact_metadata", {}).items()
    ]
    return value
