"""FastAPI application exposing analysis-only endpoints."""

import hmac
import ipaddress
import os
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from platformdirs import user_state_path
from starlette.concurrency import run_in_threadpool

from api.jobs import JobStore
from core.result import ScanReport
from core.service import AnalysisService
from core.version import __version__
from registry import Registry

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class APISettings:
    state_dir: Path
    api_key: str | None = None
    allow_ai_file_upload: bool = False
    max_file_bytes: int = 50 * 1024 * 1024
    max_batch_files: int = 20
    max_batch_bytes: int = 200 * 1024 * 1024
    workers: int = 2
    retention_days: int = 30

    @classmethod
    def from_environment(cls, *, workers: int = 2, retention_days: int = 30) -> "APISettings":
        state_value = os.environ.get("STEGANO_STATE_DIR")
        state_dir = (
            Path(state_value)
            if state_value
            else user_state_path("steganography", ensure_exists=True)
        )
        return cls(
            state_dir=state_dir,
            api_key=os.environ.get("STEGANO_API_KEY") or None,
            allow_ai_file_upload=os.environ.get("STEGANO_ALLOW_AI_FILE_UPLOAD") == "1",
            max_file_bytes=int(os.environ.get("STEGANO_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)),
            max_batch_files=int(os.environ.get("STEGANO_MAX_BATCH_FILES", 20)),
            max_batch_bytes=int(os.environ.get("STEGANO_MAX_BATCH_BYTES", 200 * 1024 * 1024)),
            workers=max(1, workers),
            retention_days=max(0, retention_days),
        )


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings.from_environment()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore(settings.state_dir / "api.sqlite3", retention_days=settings.retention_days)
    executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="steg-analysis")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(
        title="steganography analysis API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.executor = executor

    # The v2 platform is additive: existing /v1 behavior and authentication
    # remain stable while v2 always enforces its own API-key/session boundary.
    from api.v2 import install_v2_routes

    install_v2_routes(
        app,
        state_dir=settings.state_dir,
        configured_api_key=settings.api_key,
        executor=executor,
        max_file_bytes=settings.max_file_bytes,
    )

    async def authenticate(request: Request) -> None:
        if settings.api_key is None:
            return
        authorization = request.headers.get("authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, settings.api_key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    auth = Annotated[None, Depends(authenticate)]
    profile_query = Annotated[Literal["sensitive", "balanced", "strict"], Query()]

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.get("/healthz")
    async def health(_: auth) -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/modules")
    async def modules_endpoint(_: auth) -> dict:
        registry = _registry()
        return {
            "carriers": [
                {
                    "id": carrier.identifier,
                    "extensions": carrier.extensions,
                    "can_embed": carrier.can_embed,
                    "can_extract": carrier.can_extract,
                    "experimental": carrier.experimental,
                }
                for carrier in registry.all_carriers()
            ],
            "analyzers": [analyzer.name for analyzer in registry.all_analyzers()],
            "unavailable": [
                {"plugin": error.plugin, "error": error.error} for error in registry.load_errors()
            ],
        }

    @app.post("/v1/analyze")
    async def analyze_endpoint(
        _: auth,
        file: Annotated[UploadFile, File()],
        profile: profile_query = "sensitive",
        ai: bool = False,
        allow_ai_file_upload: bool = False,
    ) -> JSONResponse:
        upload_dir = Path(tempfile.mkdtemp(prefix="single-", dir=settings.state_dir))
        try:
            path, _size, display_name = await _save_upload(
                file, upload_dir, settings.max_file_bytes
            )
            provider = _provider_for_request(
                ai=ai,
                upload_file=allow_ai_file_upload and settings.allow_ai_file_upload,
            )
            service = AnalysisService(_registry(), profile=profile, ai_provider=provider)
            analysis = await run_in_threadpool(_analyze_uploaded, service, path, display_name)
            return JSONResponse(analysis.to_dict(tool_version=__version__))
        finally:
            await file.close()
            shutil.rmtree(upload_dir, ignore_errors=True)

    @app.post("/v1/scans", status_code=202)
    async def create_scan(
        _: auth,
        files: Annotated[list[UploadFile], File()],
        profile: profile_query = "sensitive",
        ai: bool = False,
        allow_ai_file_upload: bool = False,
    ) -> dict:
        if not files or len(files) > settings.max_batch_files:
            raise HTTPException(
                status_code=413,
                detail=f"batch must contain 1-{settings.max_batch_files} files",
            )
        job_id = uuid.uuid4().hex
        upload_dir = settings.state_dir / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        uploads: list[tuple[Path, str]] = []
        total = 0
        try:
            for upload in files:
                path, size, display_name = await _save_upload(
                    upload, upload_dir, settings.max_file_bytes
                )
                uploads.append((path, display_name))
                total += size
                if total > settings.max_batch_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="batch byte limit exceeded",
                    )
        except Exception:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        finally:
            for upload in files:
                await upload.close()

        store.create(job_id, profile)
        executor.submit(
            _run_scan_job,
            store,
            job_id,
            uploads,
            upload_dir,
            profile,
            ai,
            allow_ai_file_upload and settings.allow_ai_file_upload,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/v1/scans/{job_id}")
    async def get_scan(job_id: str, _: auth) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="scan job not found")
        return job

    @app.delete("/v1/scans/{job_id}", status_code=204)
    async def delete_scan(job_id: str, _: auth) -> Response:
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail="scan job not found")
        return Response(status_code=204)

    project_root = Path(__file__).resolve().parents[1]
    web_dist = next(
        (
            candidate
            for candidate in (
                project_root / "steganography" / "web",
                project_root / "web" / "dist",
            )
            if candidate.is_dir()
        ),
        None,
    )
    if web_dist is not None:
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


async def _save_upload(
    upload: UploadFile, directory: Path, max_bytes: int
) -> tuple[Path, int, str]:
    raw_name = (upload.filename or "upload.bin").replace("\\", "/")
    display_name = Path(raw_name).name or "upload.bin"
    suffix = Path(display_name).suffix.lower()
    if len(suffix) > 12 or not all(char.isalnum() or char == "." for char in suffix):
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
    return path, size, display_name


def _run_scan_job(
    store: JobStore,
    job_id: str,
    uploads: list[tuple[Path, str]],
    upload_dir: Path,
    profile: str,
    ai: bool,
    upload_file: bool,
) -> None:
    try:
        store.set_running(job_id)
        provider = _provider_for_request(ai=ai, upload_file=upload_file)
        service = AnalysisService(_registry(), profile=profile, ai_provider=provider)
        analyses = tuple(
            _analyze_uploaded(service, path, display_name) for path, display_name in uploads
        )
        report = ScanReport(analyses, profile=profile)
        # A completed job guarantees that its untrusted uploads are already
        # gone; clients never observe a completed state during cleanup.
        shutil.rmtree(upload_dir, ignore_errors=True)
        store.complete(job_id, report.to_dict(tool_version=__version__))
    except Exception as exc:
        store.fail(job_id, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


def _analyze_uploaded(service: AnalysisService, path: Path, display_name: str):
    analysis = service.analyze_safe(path)
    file_info = replace(analysis.file, path=display_name, name=display_name)
    return replace(analysis, file=file_info)


def _provider_for_request(*, ai: bool, upload_file: bool):
    if not ai:
        return None
    from modules.ai_provider_nim import make_provider

    return make_provider(upload_file=upload_file)


def _registry() -> Registry:
    registry = Registry()
    registry.autodiscover()
    return registry


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def run_server(*, host: str, port: int, workers: int = 2, retention_days: int = 30) -> None:
    settings = APISettings.from_environment(workers=workers, retention_days=retention_days)
    if not _is_loopback(host) and not settings.api_key:
        raise RuntimeError("STEGANO_API_KEY is required when binding the API outside loopback")
    import uvicorn

    uvicorn.run(create_app(settings), host=host, port=port, workers=1)
