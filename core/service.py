"""Application services shared by the CLI and REST API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import zlib
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.crypto import (
    KDF_SALT_LEN,
    DecryptionError,
    decrypt,
    decrypt_v3,
    encrypt,
    encrypt_v3,
)
from core.filetype import detect_type, extension_mismatch
from core.payload import VERSION_3, InvalidPayloadError, ParsedPayload, pack, pack_v3, unpack
from core.result import AnalysisResult, EmbedResult, FileAnalysis, FileInfo, Signal
from registry import Registry


class SteganographyError(Exception):
    """Base class for user-facing service failures."""


class ExtractionError(SteganographyError):
    """No unambiguous, valid payload could be extracted."""


class AmbiguousPayloadError(ExtractionError):
    """More than one carrier produced a valid payload."""


class OutputExistsError(SteganographyError):
    """Raised by the optional no-clobber policy."""


_PROFILE_MINIMUMS = {"sensitive": 10, "balanced": 25, "strict": 50}
_USE_REGISTERED_AI = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _severity(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def aggregate_score(results: Iterable[AnalysisResult], profile: str) -> int:
    """Combine independent signal categories without double-counting a detector."""
    if profile not in _PROFILE_MINIMUMS:
        raise ValueError(f"unknown analysis profile: {profile}")
    threshold = _PROFILE_MINIMUMS[profile]
    categories: dict[str, int] = {}
    ai_score = 0
    verified = False
    for result in results:
        if result.status != "ok":
            continue
        if result.analyzer == "ai_triage":
            if not (result.explanation or "").startswith("heuristic fallback:"):
                ai_score = max(ai_score, result.suspicion)
            continue
        for signal in result.signals:
            if signal.evidence == "informational":
                continue
            if signal.evidence == "verified":
                verified = True
            if signal.score < threshold and signal.evidence != "verified":
                continue
            category = signal.category
            if category == "general":
                category = result.analyzer
            categories[category] = max(categories.get(category, 0), signal.score)

    if categories:
        clean_probability = math.prod(1.0 - score / 100.0 for score in categories.values())
        deterministic = round(100 * (1.0 - clean_probability))
    else:
        deterministic = 0
    if verified:
        deterministic = max(95, deterministic)
    return min(100, max(deterministic, ai_score))


class AnalysisService:
    """Run every applicable detector and return a single resilient verdict."""

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        profile: str = "sensitive",
        max_file_size: int | None = None,
        ai_provider: Any = _USE_REGISTERED_AI,
    ) -> None:
        if profile not in _PROFILE_MINIMUMS:
            raise ValueError(f"unknown analysis profile: {profile}")
        self.registry = registry or _discovered_registry()
        self.profile = profile
        self.max_file_size = max_file_size
        self.ai_provider = ai_provider

    def analyze(self, path: Path) -> FileAnalysis:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if self.max_file_size is not None and size > self.max_file_size:
            raise ValueError(f"file is {size} bytes; configured maximum is {self.max_file_size}")

        detected = detect_type(path)
        info = FileInfo(
            path=str(path),
            name=path.name,
            size=size,
            sha256=_sha256(path),
            detected_type=detected.name,
            mime_type=detected.mime_type,
            extension=path.suffix.lower(),
            extension_mismatch=extension_mismatch(path, detected),
        )
        results: list[AnalysisResult] = []
        for carrier in self.registry.select_carriers(path, detected_extension=detected.extension):
            results.append(_safe_analyze(carrier.identifier, carrier.analyze, path))

        ai_triage: Any | None = None
        for analyzer in self.registry.all_analyzers():
            if analyzer.name == "ai_triage":
                ai_triage = analyzer
                continue
            results.append(_safe_analyze(analyzer.name, analyzer.analyze, path))

        if ai_triage is not None:
            prior = tuple(
                signal for result in results if result.status == "ok" for signal in result.signals
            )
            if self.ai_provider is _USE_REGISTERED_AI:
                results.append(_safe_ai_analyze(ai_triage, path, prior))
            elif self.ai_provider is None:
                results.append(ai_triage.score_from_signals(prior))
            else:
                results.append(_safe_ai_provider(self.ai_provider, path, prior))

        score = aggregate_score(results, self.profile)
        return FileAnalysis(
            file=info,
            overall_score=score,
            severity=_severity(score),
            profile=self.profile,
            results=tuple(results),
        )

    def analyze_safe(self, path: Path) -> FileAnalysis:
        """Analyze a path, converting file-level failures into report evidence."""
        path = Path(path)
        try:
            return self.analyze(path)
        except Exception as exc:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            info = FileInfo(
                path=str(path),
                name=path.name,
                size=size,
                sha256="",
                detected_type="unknown",
                mime_type="application/octet-stream",
                extension=path.suffix.lower(),
                extension_mismatch=False,
            )
            error = AnalysisResult(
                analyzer="scan",
                suspicion=0,
                signals=(),
                explanation=None,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            return FileAnalysis(
                file=info,
                overall_score=0,
                severity="low",
                profile=self.profile,
                results=(error,),
            )


def _safe_analyze(name: str, analyze: Any, path: Path) -> AnalysisResult:
    try:
        return analyze(path)
    except Exception as exc:  # plug-ins must not terminate a directory scan
        return AnalysisResult(
            analyzer=name,
            suspicion=0,
            signals=(),
            explanation=None,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _safe_ai_analyze(analyzer: Any, path: Path, prior: tuple[Signal, ...]) -> AnalysisResult:
    try:
        result = analyzer.analyze_with_context(path, prior)
    except Exception as exc:
        return AnalysisResult(
            analyzer="ai_triage",
            suspicion=0,
            signals=(),
            explanation=None,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    if result.explanation and result.explanation.startswith("NIM provider error:"):
        return replace(result, status="error", error=result.explanation)
    return result


def _safe_ai_provider(provider: Any, path: Path, prior: tuple[Signal, ...]) -> AnalysisResult:
    try:
        score, explanation = provider({"path": str(path), "size": path.stat().st_size}, list(prior))
        if explanation.startswith("NIM provider error:"):
            return AnalysisResult(
                "ai_triage",
                0,
                (),
                None,
                status="error",
                error=explanation,
            )
        return AnalysisResult("ai_triage", score, (), explanation)
    except Exception as exc:
        return AnalysisResult(
            "ai_triage",
            0,
            (),
            None,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


class StegoService:
    """High-level embed/extract operations with crypto and payload validation."""

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or _discovered_registry()

    def embed(
        self,
        input_path: Path,
        carrier_path: Path,
        out_path: Path,
        *,
        password: str | None = None,
        method: str | None = None,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
        no_clobber: bool = False,
        payload_version: int = 2,
        compress: bool = False,
        ecc_symbols: int = 0,
    ) -> EmbedResult:
        input_path = Path(input_path)
        carrier_path = Path(carrier_path)
        out_path = Path(out_path)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if not carrier_path.is_file():
            raise FileNotFoundError(carrier_path)
        if no_clobber and out_path.exists():
            raise OutputExistsError(f"output already exists: {out_path}")

        carrier = self.registry.select_carrier_for_embed(carrier_path, method=method)
        raw = input_path.read_bytes()
        if payload_version == VERSION_3:
            payload_metadata: dict[str, object] = {
                "original_name": input_path.name,
                "original_size": len(raw),
            }
            stored = zlib.compress(raw) if compress else raw
            salt = os.urandom(KDF_SALT_LEN) if password else b"\x00" * 16
            if password:
                nonce, stored = encrypt_v3(
                    stored,
                    password,
                    salt,
                    associated_data=_v3_aad(payload_metadata),
                )
            else:
                nonce = b"\x00" * 12
            if ecc_symbols:
                stored = _ecc_encode(stored, ecc_symbols)
            blob = pack_v3(
                payload=stored,
                encrypted=bool(password),
                salt=salt,
                nonce=nonce,
                compressed=compress,
                ecc_symbols=ecc_symbols,
                metadata=payload_metadata,
            )
        elif payload_version != 2:
            raise ValueError("payload_version must be 2 or 3")
        elif password:
            salt = os.urandom(KDF_SALT_LEN)
            nonce, ciphertext = encrypt(raw, password, salt)
            blob = pack(payload=ciphertext, encrypted=True, salt=salt, nonce=nonce)
        else:
            blob = pack(
                payload=raw,
                encrypted=False,
                salt=b"\x00" * 16,
                nonce=b"\x00" * 12,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_sibling(out_path)
        try:
            result = carrier.embed_with_options(
                carrier_path,
                blob,
                temporary,
                steg_key=steg_key or password,
                options=options,
            )
            os.replace(temporary, out_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return EmbedResult(
            carrier=result.carrier,
            out_path=out_path,
            bytes_written=result.bytes_written,
            encrypted=bool(password),
        )

    def extract(
        self,
        input_path: Path,
        *,
        password: str | None = None,
        method: str | None = None,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[bytes, str, int]:
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        detected = detect_type(input_path)
        if method:
            candidates = [self.registry.get_carrier(method)]
        else:
            candidates = self.registry.select_carriers(
                input_path, detected_extension=detected.extension
            )
        candidates = [carrier for carrier in candidates if carrier.can_extract]
        if not candidates:
            raise ExtractionError(f"no extractable carrier for {input_path}")

        successes: list[tuple[str, ParsedPayload]] = []
        failures: list[str] = []
        for carrier in candidates:
            try:
                blob = carrier.extract_with_options(
                    input_path,
                    steg_key=steg_key or password,
                    options=options,
                )
                successes.append((carrier.identifier, unpack(blob)))
            except (Exception, InvalidPayloadError) as exc:
                failures.append(f"{carrier.identifier}: {type(exc).__name__}: {exc}")

        if not successes:
            detail = "; ".join(failures) if failures else "no candidates"
            raise ExtractionError(f"no valid payload found ({detail})")
        if len(successes) > 1 and method is None:
            names = ", ".join(name for name, _ in successes)
            raise AmbiguousPayloadError(
                f"multiple valid payloads found ({names}); select one with --method"
            )
        carrier_name, parsed = successes[0]
        stored = parsed.payload
        if parsed.version == VERSION_3 and parsed.ecc_symbols:
            stored = _ecc_decode(stored, parsed.ecc_symbols)
        if parsed.encrypted:
            if not password:
                raise ExtractionError(
                    "payload encrypted; set --password, --password-file, or STEGANO_PASSWORD"
                )
            try:
                if parsed.version == VERSION_3:
                    data = decrypt_v3(
                        stored,
                        password,
                        parsed.salt,
                        parsed.nonce,
                        associated_data=_v3_aad(parsed.metadata or {}),
                    )
                else:
                    data = decrypt(stored, password, parsed.salt, parsed.nonce)
            except DecryptionError as exc:
                raise ExtractionError("payload authentication failed") from exc
        else:
            data = stored
        if parsed.version == VERSION_3 and parsed.compressed:
            try:
                data = zlib.decompress(data)
            except zlib.error as exc:
                raise ExtractionError("payload decompression failed") from exc
        return data, carrier_name, parsed.version

    def extract_to(
        self,
        input_path: Path,
        out_path: Path,
        **kwargs: Any,
    ) -> tuple[int, str, int]:
        out_path = Path(out_path)
        no_clobber = bool(kwargs.pop("no_clobber", False))
        if no_clobber and out_path.exists():
            raise OutputExistsError(f"output already exists: {out_path}")
        data, carrier, version = self.extract(input_path, **kwargs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_sibling(out_path)
        try:
            temporary.write_bytes(data)
            os.replace(temporary, out_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return len(data), carrier, version


def _temporary_sibling(path: Path) -> Path:
    descriptor, value = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(value)
    temporary.unlink(missing_ok=True)
    return temporary


def _discovered_registry() -> Registry:
    registry = Registry()
    registry.autodiscover()
    return registry


def _ecc_encode(payload: bytes, symbols: int) -> bytes:
    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise SteganographyError(
            "reedsolo is required when Reed-Solomon protection is enabled"
        ) from exc
    return bytes(RSCodec(symbols).encode(payload))


def _v3_aad(metadata: dict[str, object]) -> bytes:
    canonical = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return b"STEG-payload-v3\x00" + canonical


def _ecc_decode(payload: bytes, symbols: int) -> bytes:
    try:
        from reedsolo import ReedSolomonError, RSCodec
    except ImportError as exc:
        raise ExtractionError("reedsolo is required to decode this protected payload") from exc
    try:
        return bytes(RSCodec(symbols).decode(payload)[0])
    except ReedSolomonError as exc:
        raise ExtractionError("Reed-Solomon recovery failed") from exc
