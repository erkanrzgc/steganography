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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from core.cases import CaseNotFoundError, EvidenceNotFoundError
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
    manifest_path: str
    public_key: str


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

    @router.post("/models", status_code=201)
    async def install_model(body: ModelInstall, _: auth) -> dict[str, Any]:
        try:
            return models.install(Path(body.manifest_path), public_key=body.public_key)
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
