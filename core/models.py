"""Explicit, signed local model installation and runtime discovery."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from core.database import Database, utc_now
from core.ensemble import ModelScore


class ModelError(Exception):
    pass


class ModelVerificationError(ModelError):
    pass


class ModelRegistry:
    """Manage opt-in ONNX files. Installation never performs network access."""

    SUPPORTED_DOMAINS = frozenset({"spatial-srnet-v1", "jpeg-srnet-v1"})

    def __init__(self, database: Database, root: Path) -> None:
        self.database = database
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM models ORDER BY domain, id, version"
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["manifest"] = json.loads(value.pop("manifest_json"))
            values.append(value)
        return values

    def install(
        self,
        manifest_path: Path,
        *,
        public_key: bytes | str | Path,
    ) -> dict[str, Any]:
        """Verify a local manifest and adjacent model before atomically installing."""
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.verify_manifest(manifest, public_key=public_key)
        required = {
            "id",
            "version",
            "domain",
            "file",
            "sha256",
            "license",
            "model_card",
            "benchmark_summary",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise ModelVerificationError(f"manifest missing fields: {', '.join(missing)}")
        if manifest["domain"] not in self.SUPPORTED_DOMAINS:
            raise ModelVerificationError(f"unsupported model domain: {manifest['domain']}")
        filename = Path(str(manifest["file"]))
        if filename.name != str(manifest["file"]):
            raise ModelVerificationError("model file must be adjacent to the manifest")
        source = manifest_path.parent / filename
        if not source.is_file():
            raise ModelVerificationError(f"model file is missing: {filename}")
        digest = _sha256(source)
        if digest != manifest["sha256"]:
            raise ModelVerificationError("model SHA-256 does not match manifest")
        destination_dir = (
            self.root / _safe_component(manifest["id"]) / _safe_component(manifest["version"])
        )
        destination = destination_dir / filename.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filename.name}.", dir=destination_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO models
                   (id, domain, version, state, path, sha256, manifest_json, installed_at)
                   VALUES (?, ?, ?, 'installed', ?, ?, ?, ?)
                   ON CONFLICT(id, version) DO UPDATE SET state='installed', path=excluded.path,
                     sha256=excluded.sha256, manifest_json=excluded.manifest_json,
                     installed_at=excluded.installed_at""",
                (
                    manifest["id"],
                    manifest["domain"],
                    manifest["version"],
                    str(destination),
                    digest,
                    canonical,
                    utc_now(),
                ),
            )
            self.database.audit(
                "model.installed",
                object_type="model",
                object_id=f"{manifest['id']}@{manifest['version']}",
                details={"domain": manifest["domain"], "sha256": digest},
                connection=connection,
            )
        return self.get(manifest["id"], manifest["version"])

    def get(self, model_id: str, version: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE id=? AND version=?", (model_id, version)
            ).fetchone()
        if row is None:
            raise LookupError(f"unknown model {model_id}@{version}")
        value = dict(row)
        value["manifest"] = json.loads(value.pop("manifest_json"))
        return value

    def verify_installed(self, model_id: str, version: str) -> bool:
        model = self.get(model_id, version)
        path = Path(model["path"])
        valid = path.is_file() and _sha256(path) == model["sha256"]
        if not valid:
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE models SET state='invalid' WHERE id=? AND version=?",
                    (model_id, version),
                )
        return valid

    def score(
        self,
        model_id: str,
        version: str,
        tensor: Any,
        *,
        domain: str,
    ) -> ModelScore:
        """Run an installed ONNX model with CPU-default provider selection."""
        try:
            model = self.get(model_id, version)
        except LookupError as exc:
            return ModelScore(model_id, domain, None, "unavailable", error=str(exc))
        if model["domain"] != domain:
            return ModelScore(
                model_id,
                domain,
                None,
                "unsupported",
                error=f"model domain is {model['domain']}",
            )
        if not self.verify_installed(model_id, version):
            return ModelScore(
                model_id, domain, None, "error", error="model integrity check failed"
            )
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            return ModelScore(
                model_id,
                domain,
                None,
                "unavailable",
                error="onnxruntime is not installed",
            )
        available = ort.get_available_providers()
        requested = os.environ.get("STEGANO_ONNX_PROVIDER", "CPU").upper()
        provider = "CPUExecutionProvider"
        if requested == "CUDA" and "CUDAExecutionProvider" in available:
            provider = "CUDAExecutionProvider"
        try:
            session = ort.InferenceSession(model["path"], providers=[provider])
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: tensor})[0]
            raw = float(np.asarray(output).reshape(-1)[0])
            probability = raw if 0 <= raw <= 1 else 1 / (1 + math.exp(-raw))
            actual_provider = session.get_providers()[0]
            return ModelScore(
                model_id,
                domain,
                probability,
                "ok",
                execution_provider=actual_provider,
                calibrated=bool(model["manifest"].get("calibrated", False)),
            )
        except Exception as exc:
            return ModelScore(
                model_id,
                domain,
                None,
                "error",
                execution_provider=provider,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def verify_manifest(manifest: dict[str, Any], *, public_key: bytes | str | Path) -> None:
        signature_value = manifest.get("signature")
        if not isinstance(signature_value, str):
            raise ModelVerificationError("manifest has no Ed25519 signature")
        unsigned = dict(manifest)
        unsigned.pop("signature", None)
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        try:
            signature = base64.b64decode(signature_value, validate=True)
            key_bytes = _key_bytes(public_key)
            Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, canonical)
        except (ValueError, InvalidSignature) as exc:
            raise ModelVerificationError("invalid Ed25519 manifest signature") from exc

    @staticmethod
    def runtime_status() -> dict[str, Any]:
        try:
            import onnxruntime as ort
        except ImportError:
            return {
                "status": "unavailable",
                "reason": "onnxruntime is not installed",
                "available_providers": [],
                "selected_provider": None,
            }
        providers = ort.get_available_providers()
        requested = os.environ.get("STEGANO_ONNX_PROVIDER", "CPU").upper()
        selected = "CPUExecutionProvider"
        if requested == "CUDA" and "CUDAExecutionProvider" in providers:
            selected = "CUDAExecutionProvider"
        return {
            "status": "available",
            "available_providers": providers,
            "selected_provider": selected,
        }


def _key_bytes(value: bytes | str | Path) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        candidate = Path(value)
        try:
            is_file = len(str(value)) < 4096 and candidate.is_file()
        except OSError:
            is_file = False
        raw = candidate.read_bytes() if is_file else str(value).encode()
    raw = raw.strip()
    if len(raw) == 32:
        return raw
    try:
        decoded = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise ModelVerificationError("public key must be raw or base64 Ed25519 bytes") from exc
    if len(decoded) != 32:
        raise ModelVerificationError("Ed25519 public key must be 32 bytes")
    return decoded


def _safe_component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not value or any(character not in allowed for character in value):
        raise ModelVerificationError(f"unsafe model identifier: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
