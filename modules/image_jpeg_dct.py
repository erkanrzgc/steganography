"""Experimental JPEG DCT carrier isolated behind a worker subprocess."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_WORKER_MODULE = "modules.image_jpeg_dct_worker"
_TIMEOUT_SECONDS = 15


class ImageJpegDct(Carrier):
    name = "image_jpeg_dct"
    method_id = "image_jpeg_dct"
    extensions = (".jpg", ".jpeg")
    priority = 300
    experimental = True
    requires_explicit = True

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("jpeglib") is not None

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.available else "install the 'dct' extra (jpeglib>=0.14)"

    def _require_available(self) -> None:
        if not self.available:
            raise RuntimeError(self.unavailable_reason)

    def capacity(self, src: Path) -> int:
        self._require_available()
        result = _run_worker("capacity", src)
        return int(result["capacity"])

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        raise ValueError("image_jpeg_dct requires embed_with_options and a steg key")

    def embed_with_options(
        self,
        src: Path,
        payload: bytes,
        out: Path,
        *,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> EmbedResult:
        del options
        self._require_available()
        if not steg_key:
            raise ValueError("image_jpeg_dct requires --steg-key or --password")
        capacity = self.capacity(src)
        if len(payload) > capacity:
            raise InsufficientCapacityError(
                f"payload {len(payload)} > JPEG DCT capacity {capacity}"
            )
        with tempfile.NamedTemporaryFile(prefix="steg-dct-", suffix=".bin") as stream:
            stream.write(payload)
            stream.flush()
            _run_worker("embed", src, out, Path(stream.name), key=steg_key)
        return EmbedResult(self.identifier, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        raise ValueError("image_jpeg_dct requires extract_with_options and a steg key")

    def extract_with_options(
        self,
        src: Path,
        *,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> bytes:
        del options
        self._require_available()
        if not steg_key:
            raise ValueError("image_jpeg_dct requires --steg-key or --password")
        with tempfile.NamedTemporaryFile(prefix="steg-dct-out-", suffix=".bin") as stream:
            _run_worker("extract", src, Path(stream.name), key=steg_key)
            return Path(stream.name).read_bytes()

    def analyze(self, src: Path) -> AnalysisResult:
        if not self.available:
            return AnalysisResult(
                self.identifier,
                0,
                (),
                None,
                status="unavailable",
                error=self.unavailable_reason,
            )
        try:
            result = _run_worker("analyze", src)
        except TimeoutError as exc:
            return AnalysisResult(
                self.identifier, 0, (), None, status="error", error=str(exc)
            )
        except RuntimeError as exc:
            return AnalysisResult(
                self.identifier, 0, (), None, status="error", error=str(exc)
            )
        status = str(result.get("status", "ok"))
        signals = tuple(
            Signal(
                str(item["name"]),
                int(item["score"]),
                str(item["detail"]),
                category=str(item.get("category", "jpeg_dct")),
                evidence=str(item.get("evidence", "heuristic")),
            )
            for item in result.get("signals", [])
        )
        return AnalysisResult(
            self.identifier,
            int(result.get("suspicion", 0)),
            signals,
            result.get("explanation"),
            status=status,
            error=result.get("error"),
        )


def _run_worker(
    operation: str,
    *paths: Path,
    key: str | None = None,
) -> dict[str, Any]:
    command = [sys.executable, "-m", _WORKER_MODULE, operation]
    command.extend(str(path) for path in paths)
    try:
        completed = subprocess.run(  # noqa: S603 - fixed interpreter/module and paths
            command,
            input=(key or "") + "\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"JPEG DCT worker exceeded {_TIMEOUT_SECONDS} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-500:] or "unknown worker failure"
        raise RuntimeError(f"JPEG DCT worker failed: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("JPEG DCT worker returned malformed JSON") from exc
