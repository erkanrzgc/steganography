"""Bounded, recursive CTF playbook shared by CLI, API, and TUI."""

from __future__ import annotations

import base64
import binascii
import bz2
import gzip
import hashlib
import io
import lzma
import mimetypes
import re
import shutil
import tarfile
import time
import urllib.parse
import uuid
import zipfile
import zlib
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from core.coverage import assess_coverage
from core.ctf_types import Artifact, Coverage, Recommendation, ToolExecution
from core.pipeline import AnalysisPipeline
from core.service import AnalysisService, StegoService
from core.tools import ToolRunner
from core.version import __version__

_FLAG_PATTERN = re.compile(rb"([a-zA-Z0-9_]{3,24}\{[ -~]{4,120}\})")

CTFMode = Literal["quick", "balanced", "deep"]
EventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class CTFLimits:
    max_depth: int = 3
    max_artifacts: int = 256
    max_bytes: int = 1024 * 1024 * 1024
    tool_timeout: float = 30.0
    job_timeout: float = 180.0

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max depth must not be negative")
        if self.max_artifacts < 1 or self.max_bytes < 1:
            raise ValueError("artifact and byte limits must be positive")
        if self.tool_timeout <= 0 or self.job_timeout <= 0:
            raise ValueError("timeouts must be positive")


@dataclass(frozen=True, slots=True)
class CTFReport:
    id: str
    mode: CTFMode
    status: Literal["completed", "cancelled", "failed"]
    verdict: str
    verdict_reason: str
    duration_ms: float
    input: dict[str, Any]
    analyses: tuple[dict[str, Any], ...]
    artifacts: tuple[Artifact, ...]
    tools: tuple[ToolExecution, ...]
    coverage: tuple[Coverage, ...]
    recommendations: tuple[Recommendation, ...]
    stages: tuple[str, ...]
    limits: CTFLimits
    error: str | None = None
    schema_version: str = "2.0"
    schema_revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        false_positives = (
            "Entropy, image noise, metadata editors, encoding-like prose, and ordinary "
            "application data can trigger heuristic candidates."
        )
        return {
            "schema_version": self.schema_version,
            "schema_revision": self.schema_revision,
            "tool": {"name": "steganography", "version": __version__},
            "report_type": "ctf",
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "threshold": {"likely": 0.70, "suspicious": 0.30},
            "calibration": {"state": "not_calibrated", "scope": "heuristic_triage"},
            "false_positive_conditions": [false_positives],
            "duration_ms": self.duration_ms,
            "input": self.input,
            "analyses": list(self.analyses),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "tool_executions": [item.to_dict() for item in self.tools],
            "coverage": [item.to_dict() for item in self.coverage],
            "recommendations": [item.to_dict() for item in self.recommendations],
            "stages": list(self.stages),
            "limits": {
                "max_depth": self.limits.max_depth,
                "max_artifacts": self.limits.max_artifacts,
                "max_bytes": self.limits.max_bytes,
                "tool_timeout_seconds": self.limits.tool_timeout,
                "job_timeout_seconds": self.limits.job_timeout,
            },
            "error": self.error,
        }


@dataclass(slots=True)
class _JobState:
    output_dir: Path
    limits: CTFLimits
    started: float = field(default_factory=time.monotonic)
    artifacts: list[Artifact] = field(default_factory=list)
    tools: list[ToolExecution] = field(default_factory=list)
    coverage: list[Coverage] = field(default_factory=list)
    analyses: list[dict[str, Any]] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    output_bytes: int = 0
    counter: int = 0
    confirmed: bool = False


class CTFService:
    """Run a deterministic playbook without allowing artifacts to escape a job."""

    def __init__(
        self,
        *,
        analysis: AnalysisPipeline | None = None,
        stego: StegoService | None = None,
        tool_runner: ToolRunner | None = None,
    ) -> None:
        self.analysis = analysis or AnalysisPipeline(
            AnalysisService(profile="balanced", ai_provider=None)
        )
        self.stego = stego or StegoService()
        self.tool_runner = tool_runner

    def solve(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        mode: CTFMode = "balanced",
        wordlist: Path | None = None,
        password: str | None = None,
        limits: CTFLimits | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_event: EventCallback | None = None,
    ) -> CTFReport:
        if mode not in {"quick", "balanced", "deep"}:
            raise ValueError(f"unknown CTF mode: {mode}")
        limits = limits or CTFLimits()
        source = Path(input_path)
        if not source.is_file() or source.is_symlink():
            raise ValueError("CTF input must be a regular, non-symlink file")
        if source.stat().st_size > limits.max_bytes:
            raise ValueError("CTF input exceeds the configured byte limit")
        if wordlist is not None and (not Path(wordlist).is_file() or Path(wordlist).is_symlink()):
            raise ValueError("wordlist must be a regular, non-symlink file")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=False)
        artifacts_dir = target / "artifacts"
        artifacts_dir.mkdir()
        state = _JobState(target, limits)
        job_id = uuid.uuid4().hex
        status: Literal["completed", "cancelled", "failed"] = "completed"
        error: str | None = None
        root: Artifact | None = None
        try:
            self._stage(state, "identify", on_event)
            root = self._copy_input(state, source)
            queue: deque[Artifact] = deque([root])
            seen = {root.sha256}
            while queue:
                self._checkpoint(state, should_cancel)
                current = queue.popleft()
                self._stage(state, "metadata_structure", on_event)
                self._analyze(state, current)
                # Boundary artifacts are analyzed, but cannot create deeper children.
                if current.depth >= limits.max_depth:
                    continue
                self._stage(state, "native_detectors", on_event)
                generated = self._native_candidates(state, current, password=password, mode=mode)
                if current.depth == 0 and mode != "quick":
                    self._stage(state, "external_tools", on_event)
                    generated.extend(
                        self._external_tools(
                            state,
                            current,
                            mode=mode,
                            wordlist=Path(wordlist) if wordlist else None,
                            password=password,
                            should_cancel=should_cancel,
                        )
                    )
                self._stage(state, "carving_decoding_extraction", on_event)
                generated.extend(self._carve_and_decode(state, current, mode=mode))
                for artifact in generated:
                    if artifact.depth <= limits.max_depth and artifact.sha256 not in seen:
                        seen.add(artifact.sha256)
                        queue.append(artifact)
            self._stage(state, "recursive_analysis", on_event)
        except _Cancelled as exc:
            status = "cancelled"
            error = str(exc)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"

        verdict, reason = self._verdict(state, status)
        report = CTFReport(
            id=job_id,
            mode=mode,
            status=status,
            verdict=verdict,
            verdict_reason=reason,
            duration_ms=round((time.monotonic() - state.started) * 1000, 3),
            input={
                "name": source.name,
                "size": source.stat().st_size,
                "sha256": root.sha256 if root else _sha256(source.read_bytes()),
            },
            analyses=tuple(state.analyses),
            artifacts=tuple(state.artifacts),
            tools=tuple(state.tools),
            coverage=tuple(_deduplicate_coverage(state.coverage)),
            recommendations=tuple(self._recommendations(state, mode)),
            stages=tuple(state.stages),
            limits=limits,
            error=error,
        )
        if on_event:
            on_event({"event": "complete", "status": status, "verdict": verdict})
        return report

    def _checkpoint(self, state: _JobState, should_cancel: Callable[[], bool] | None) -> None:
        if should_cancel and should_cancel():
            raise _Cancelled("job cancelled")
        if time.monotonic() - state.started >= state.limits.job_timeout:
            raise _Cancelled("job timeout reached")

    @staticmethod
    def _stage(state: _JobState, name: str, callback: EventCallback | None) -> None:
        if not state.stages or state.stages[-1] != name:
            state.stages.append(name)
            if callback:
                callback({"event": "stage", "stage": name})

    def _copy_input(self, state: _JobState, source: Path) -> Artifact:
        data = source.read_bytes()
        suffix = source.suffix.lower()[:12]
        return self._store(state, data, f"input{suffix}", None, 0, "input_copy", count_bytes=False)

    def _analyze(self, state: _JobState, artifact: Artifact) -> None:
        artifact_path = _artifact_path(artifact)
        pipeline_report = self.analysis.analyze(artifact_path)
        report = pipeline_report.to_dict()
        report = _redact_paths(report, state.output_dir)
        report["file"]["path"] = artifact.name
        report["file"]["name"] = artifact.name
        report["artifact_id"] = artifact.id
        state.analyses.append(report)
        for result in report.get("findings", []):
            analyzer = result.get("analyzer", {}).get("name", "unknown")
            status = result.get("status", "failed")
            mapped = _analyzer_coverage_status(str(status))
            state.coverage.append(Coverage(analyzer, mapped, False, result.get("error")))

        state.coverage.extend(assess_coverage(pipeline_report.analysis).requirements)

    def _native_candidates(
        self,
        state: _JobState,
        artifact: Artifact,
        *,
        password: str | None,
        mode: CTFMode,
    ) -> list[Artifact]:
        artifact_path = _artifact_path(artifact)
        generated: list[Artifact] = []
        try:
            payload, carrier, _version = self.stego.extract(artifact_path, password=password)
        except Exception:  # noqa: BLE001, S110 - absence of a payload is an expected candidate miss
            pass
        else:
            generated.append(
                self._store(
                    state,
                    payload,
                    "recovered.bin",
                    artifact,
                    artifact.depth + 1,
                    f"successful extraction via {carrier}",
                )
            )
            state.confirmed = True
        if not state.confirmed and artifact_path.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                from modules.image_jpeg_dct import extract_jsteg

                jsteg_payload = extract_jsteg(artifact_path)
            except Exception:  # noqa: BLE001, S110
                pass
            else:
                generated.append(
                    self._store(
                        state,
                        jsteg_payload,
                        "recovered.bin",
                        artifact,
                        artifact.depth + 1,
                        "successful extraction via jsteg",
                    )
                )
                state.confirmed = True
        if not state.confirmed and artifact_path.suffix.lower() in {".wav", ".wave"}:
            try:
                from modules.audio_wav import extract_wav_payload

                recovered = extract_wav_payload(artifact_path)
                if recovered is not None:
                    payload, desc = recovered
                    generated.append(
                        self._store(
                            state,
                            payload,
                            "recovered.bin",
                            artifact,
                            artifact.depth + 1,
                            desc,
                        )
                    )
                    state.confirmed = True
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
        if not state.confirmed and artifact_path.suffix.lower() == ".gif":
            try:
                from modules.image_gif import (
                    extract_gif_comments,
                    extract_gif_delays_payload,
                    parse_gif_stream,
                )

                raw_data = artifact_path.read_bytes()
                parsed = parse_gif_stream(raw_data)
                if parsed:
                    comments = extract_gif_comments(raw_data)
                    if comments:
                        is_flag = bool(_FLAG_PATTERN.search(comments))
                        generated.append(
                            self._store(
                                state,
                                comments,
                                "gif-comment.txt",
                                artifact,
                                artifact.depth + 1,
                                "GIF comment payload" + (" (flag confirmed)" if is_flag else ""),
                            )
                        )
                        if is_flag:
                            state.confirmed = True
                    delays = parsed.get("delays", [])
                    delay_res = extract_gif_delays_payload(delays)
                    if delay_res is not None and not state.confirmed:
                        payload, desc = delay_res
                        is_flag = bool(_FLAG_PATTERN.search(payload))
                        generated.append(
                            self._store(
                                state,
                                payload,
                                "gif-delays.bin",
                                artifact,
                                artifact.depth + 1,
                                desc,
                            )
                        )
                        if is_flag:
                            state.confirmed = True
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
        if not state.confirmed and artifact_path.suffix.lower() in {".txt", ".md"}:
            try:
                from modules.text_whitespace import TextWhitespace
                from modules.text_zerowidth import TextZeroWidth

                raw_ws = TextWhitespace().extract(artifact_path)
                if raw_ws:
                    is_flag = bool(_FLAG_PATTERN.search(raw_ws))
                    printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in raw_ws) / len(
                        raw_ws
                    )
                    if is_flag or (printable >= 0.85 and len(raw_ws) >= 4):
                        generated.append(
                            self._store(
                                state,
                                raw_ws,
                                "whitespace-extracted.txt",
                                artifact,
                                artifact.depth + 1,
                                "trailing whitespace payload"
                                + (" (flag confirmed)" if is_flag else ""),
                            )
                        )
                        if is_flag:
                            state.confirmed = True
                if not state.confirmed:
                    raw_zw = TextZeroWidth().extract(artifact_path)
                    if raw_zw:
                        is_flag = bool(_FLAG_PATTERN.search(raw_zw))
                        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in raw_zw) / len(
                            raw_zw
                        )
                        if is_flag or (printable >= 0.85 and len(raw_zw) >= 4):
                            generated.append(
                                self._store(
                                    state,
                                    raw_zw,
                                    "zerowidth-extracted.txt",
                                    artifact,
                                    artifact.depth + 1,
                                    "zero-width unicode payload"
                                    + (" (flag confirmed)" if is_flag else ""),
                                )
                            )
                            if is_flag:
                                state.confirmed = True
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
        if not state.confirmed and artifact_path.suffix.lower() == ".mp3":
            try:
                from modules.audio_mp3 import extract_mp3_payloads

                raw_data = artifact_path.read_bytes()
                for name, payload, desc, is_flag in extract_mp3_payloads(raw_data):
                    generated.append(
                        self._store(
                            state,
                            payload,
                            name,
                            artifact,
                            artifact.depth + 1,
                            desc + (" (flag confirmed)" if is_flag else ""),
                        )
                    )
                    if is_flag:
                        state.confirmed = True
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
        if not state.confirmed and artifact_path.suffix.lower() == ".pdf":
            try:
                from modules.file_pdf import extract_pdf_payloads

                raw_data = artifact_path.read_bytes()
                for name, payload, desc, is_flag in extract_pdf_payloads(
                    raw_data, max_decoded_bytes=max(0, state.limits.max_bytes - state.output_bytes)
                ):
                    generated.append(
                        self._store(
                            state,
                            payload,
                            name,
                            artifact,
                            artifact.depth + 1,
                            desc + (" (flag confirmed)" if is_flag else ""),
                        )
                    )
                    if is_flag:
                        state.confirmed = True
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
        if not state.confirmed and artifact_path.suffix.lower() in {".jpg", ".jpeg", ".tiff"}:
            try:
                import piexif

                exif_dict = piexif.load(str(artifact_path))
                for ifd in ("0th", "Exif", "1st"):
                    for _tag, val in exif_dict.get(ifd, {}).items():
                        if isinstance(val, bytes) and len(val) >= 4:
                            is_flag = bool(_FLAG_PATTERN.search(val))
                            if is_flag:
                                generated.append(
                                    self._store(
                                        state,
                                        val,
                                        "exif-metadata.txt",
                                        artifact,
                                        artifact.depth + 1,
                                        "EXIF metadata payload (flag confirmed)",
                                    )
                                )
                                state.confirmed = True
                                break
                    if state.confirmed:
                        break
            except _Cancelled:
                raise
            except Exception:  # noqa: BLE001, S110
                pass
        if mode == "deep" and artifact_path.suffix.lower() in {".png", ".bmp"}:
            visualization = _bitplane_visualization(artifact_path)
            if visualization:
                generated.append(
                    self._store(
                        state,
                        visualization,
                        "lsb-planes.png",
                        artifact,
                        artifact.depth + 1,
                        "RGB least-significant-bit visualization",
                    )
                )
        return generated

    def _external_tools(
        self,
        state: _JobState,
        artifact: Artifact,
        *,
        mode: CTFMode,
        wordlist: Path | None,
        password: str | None,
        should_cancel: Callable[[], bool] | None,
    ) -> list[Artifact]:
        artifact_path = _artifact_path(artifact)
        runner = self.tool_runner or ToolRunner(timeout=state.limits.tool_timeout)
        suffix = artifact_path.suffix.lower()
        specs: list[tuple[str, list[str], tuple[str, ...]]] = []
        relative = artifact_path.relative_to(state.output_dir).as_posix()
        if suffix in {".png", ".bmp"}:
            specs.extend(
                (
                    ("zsteg", ["--all", relative], ()),
                    (
                        "openstego",
                        ["extract", "-sf", relative, "-xd", ".", "-xf", "openstego.bin"],
                        (),
                    ),
                )
            )
        if suffix == ".png":
            specs.append(("pngcheck", ["-v", relative], ()))
        if suffix in {".jpg", ".jpeg", ".bmp", ".wav", ".au"}:
            if wordlist:
                local_wordlist = state.output_dir / "wordlist.txt"
                if not local_wordlist.exists():
                    shutil.copyfile(wordlist, local_wordlist)
                specs.append(
                    ("stegseek", [relative, local_wordlist.name, "stegseek.bin", "-f"], ())
                )
            else:
                specs.append(("stegseek", ["--seed", relative, "stegseek.bin"], ()))
            steghide_pass = password if password is not None else ""
            specs.append(
                (
                    "steghide",
                    ["extract", "-sf", relative, "-p", steghide_pass, "-xf", "steghide.bin", "-f"],
                    (steghide_pass,) if steghide_pass else (),
                )
            )
            if suffix in {".jpg", ".jpeg"}:
                outguess_args = ["-r", relative, "outguess.bin"]
                outguess_secrets: tuple[str, ...] = ()
                if password:
                    outguess_args[0:0] = ["-k", password]
                    outguess_secrets = (password,)
                specs.append(("outguess", outguess_args, outguess_secrets))
                specs.append(("jsteg", ["reveal", relative, "jsteg.bin"], ()))
        if suffix == ".gif":
            specs.append(("gifsicle", ["--info", relative], ()))
        if suffix in {".wav", ".mp3"}:
            specs.append(("sox", ["--info", relative], ()))
        specs.append(("exiftool", ["-validate", "-warning", "-error", relative], ()))
        if mode == "deep":
            specs.extend(
                (
                    ("binwalk", [relative], ()),
                    ("zbarimg", ["--quiet", relative], ()),
                )
            )
            if suffix in {".gif", ".wav", ".mp3"}:
                specs.append(("ffmpeg", ["-v", "error", "-i", relative, "-f", "null", "-"], ()))

        generated: list[Artifact] = []
        extractors = {"openstego", "stegseek", "steghide", "outguess", "jsteg"}
        for tool, args, secrets in specs:
            self._checkpoint(state, should_cancel)
            output_path = state.output_dir / f"{tool}.bin" if tool in extractors else None
            # Do not let a prior tool supply or overwrite another tool's evidence.
            if output_path is not None and (output_path.exists() or output_path.is_symlink()):
                state.coverage.append(
                    Coverage(tool, "failed", False, "extraction output already exists")
                )
                continue
            remaining = state.limits.job_timeout - (time.monotonic() - state.started)
            execution = runner.run(
                tool,
                args,
                cwd=state.output_dir,
                secret_values=secrets,
                timeout=min(state.limits.tool_timeout, remaining),
                should_cancel=should_cancel,
            )
            # Report commands use sandbox-relative names and never secrets.
            state.tools.append(execution)
            coverage_status = _tool_coverage_status(execution.status)
            state.coverage.append(Coverage(tool, coverage_status, False, execution.error))
            # Logs, crash dumps and partial/failed output are not extraction evidence.
            # Adopt only this invocation's named, nonempty, successful output.
            if (
                output_path is None
                or execution.status != "completed"
                or execution.exit_code != 0
                or output_path.is_symlink()
                or not output_path.is_file()
            ):
                continue
            data = _bounded_read(output_path, state.limits.max_bytes - state.output_bytes)
            if not data:
                continue
            is_flag = bool(_FLAG_PATTERN.search(data))
            generated.append(
                self._store(
                    state,
                    data,
                    output_path.name,
                    artifact,
                    artifact.depth + 1,
                    f"external tool extraction via {tool}"
                    + (" (flag confirmed)" if is_flag else ""),
                )
            )
            output_path.unlink()
            if is_flag:
                state.confirmed = True
        return generated

    def _carve_and_decode(
        self, state: _JobState, artifact: Artifact, *, mode: CTFMode
    ) -> list[Artifact]:
        if artifact.depth >= state.limits.max_depth:
            return []
        artifact_path = _artifact_path(artifact)
        data = artifact_path.read_bytes()
        candidates: list[tuple[bytes, str, str]] = []
        trailer = _trailer(data, artifact_path.suffix.lower())
        if trailer:
            candidates.append((trailer, "trailer.bin", "data appended after structural end"))
        candidates.extend(_archive_members(data, state.limits))
        candidates.extend(_decoded_candidates(data, deep=mode == "deep"))
        for offset, label in _signature_offsets(data):
            end = (
                offset + int.from_bytes(data[offset + 2 : offset + 6], "little")
                if label == "bmp"
                else len(data)
            )
            suffix = {"gzip": "gz", "jpeg": "jpg"}.get(label, label)
            candidates.append(
                (
                    data[offset:end],
                    f"carved-{label}.{suffix}",
                    f"{label} signature at offset {offset}",
                )
            )
        generated: list[Artifact] = []
        for candidate, name, provenance in candidates:
            if not candidate or candidate == data:
                continue
            generated.append(
                self._store(
                    state,
                    candidate,
                    name,
                    artifact,
                    artifact.depth + 1,
                    provenance,
                )
            )
        return generated

    def _store(
        self,
        state: _JobState,
        data: bytes,
        suggested_name: str,
        parent: Artifact | None,
        depth: int,
        provenance: str,
        *,
        count_bytes: bool = True,
    ) -> Artifact:
        if len(state.artifacts) >= state.limits.max_artifacts:
            raise _Cancelled("artifact count limit reached")
        if count_bytes and state.output_bytes + len(data) > state.limits.max_bytes:
            raise _Cancelled("artifact byte limit reached")
        state.counter += 1
        suffix = Path(suggested_name).suffix.lower()[:12]
        name = f"{state.counter:04d}-{_safe_stem(suggested_name)}{suffix}"
        path = state.output_dir / "artifacts" / name
        with path.open("xb") as output:
            output.write(data)
        if path.is_symlink() or path.resolve().parent != (state.output_dir / "artifacts").resolve():
            raise RuntimeError("artifact escaped the job sandbox")
        if count_bytes:
            state.output_bytes += len(data)
        artifact = Artifact(
            id=uuid.uuid4().hex,
            name=name,
            media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
            size=len(data),
            sha256=_sha256(data),
            depth=depth,
            parent_id=parent.id if parent else None,
            provenance=provenance,
            path=path,
        )
        state.artifacts.append(artifact)
        return artifact

    @staticmethod
    def _verdict(state: _JobState, status: str) -> tuple[str, str]:
        if status != "completed":
            return "inconclusive", f"CTF job {status} before all configured stages completed"
        if state.confirmed:
            return "confirmed", "a payload was successfully extracted and validated"
        scores = [float(item.get("confidence", 0.0)) for item in state.analyses]
        maximum = max(scores, default=0.0)
        if maximum >= 0.70:
            return "likely", "native analysis exceeded the likely threshold"
        if maximum >= 0.30 or len(state.artifacts) > 1:
            return "suspicious", "heuristics or bounded decoding produced unverified candidates"
        missing_required = any(
            item.required and item.status != "available" for item in state.coverage
        )
        if missing_required:
            return "inconclusive", "mandatory native detector coverage was unavailable"
        return "no_indicators", "configured mandatory analyzers completed without indicators"

    @staticmethod
    def _recommendations(state: _JobState, mode: CTFMode) -> list[Recommendation]:
        values: list[Recommendation] = []
        unavailable = [item.component for item in state.coverage if item.status == "unavailable"]
        if unavailable:
            values.append(
                Recommendation(
                    1,
                    "Re-run in the full Docker image",
                    "Optional tools were unavailable: " + ", ".join(sorted(set(unavailable))[:6]),
                    "external-tools",
                )
            )
        if mode != "deep":
            values.append(
                Recommendation(
                    2,
                    "Re-run with --mode deep",
                    "Enables additional carving and XOR candidates",
                )
            )
        values.append(
            Recommendation(
                3,
                "Review artifact provenance and validate candidates independently",
                "Decoder and carving outputs are candidates until extraction or a marker "
                "verifies them",
            )
        )
        return sorted(values, key=lambda item: item.priority)[:3]


class _Cancelled(RuntimeError):
    pass


def _safe_stem(value: str) -> str:
    stem = Path(value).stem
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in stem)
    return (cleaned.strip("-_") or "artifact")[:48]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_read(path: Path, limit: int) -> bytes:
    if limit < 0 or path.stat().st_size > limit:
        raise _Cancelled("artifact byte limit reached")
    return path.read_bytes()


def _trailer(data: bytes, suffix: str) -> bytes:
    if suffix in {".wav", ".wave"}:
        if data.startswith(b"RIFF") and len(data) >= 8:
            riff_size = int.from_bytes(data[4:8], "little")
            structural_len = 8 + riff_size
            return data[structural_len:] if len(data) > structural_len else b""
        return b""
    if suffix == ".gif":
        from modules.image_gif import gif_structural_end

        end = gif_structural_end(data)
        if end is not None and len(data) > end:
            return data[end:]
        return b""
    if suffix == ".mp3":
        from modules.audio_mp3 import mp3_structural_end

        end = mp3_structural_end(data)
        if end is not None and len(data) > end:
            return data[end:]
        return b""
    if suffix == ".pdf":
        from modules.file_pdf import pdf_structural_end

        end = pdf_structural_end(data)
        if end is not None and len(data) > end:
            return data[end:]
        return b""
    markers = {
        ".png": b"IEND\xaeB`\x82",
        ".jpg": b"\xff\xd9",
        ".jpeg": b"\xff\xd9",
    }
    marker = markers.get(suffix)
    if not marker:
        return b""
    offset = data.rfind(marker)
    return b"" if offset < 0 else data[offset + len(marker) :]


_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),
    (b"\x1f\x8b\x08", "gzip"),
    (b"BZh", "bz2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"%PDF-", "pdf"),
    (b"\x7fELF", "elf"),
    (b"ID3", "mp3"),
)


def _signature_offsets(data: bytes) -> Iterable[tuple[int, str]]:
    found: set[tuple[int, str]] = set()
    for magic, label in _MAGICS:
        offset = data.find(magic, 1)
        # Two-byte BM occurs naturally in pixel data. Bound the search and require
        # a coherent header before treating it as a file, including after decoys.
        for _ in range(64):
            if offset < 0:
                break
            if label != "bmp" or _valid_bmp_header(data, offset):
                found.add((offset, label))
                break
            offset = data.find(magic, offset + 1)
    return sorted(found)


def _valid_bmp_header(data: bytes, offset: int) -> bool:
    remaining = len(data) - offset
    if remaining < 26:
        return False
    size = int.from_bytes(data[offset + 2 : offset + 6], "little")
    pixels = int.from_bytes(data[offset + 10 : offset + 14], "little")
    dib = int.from_bytes(data[offset + 14 : offset + 18], "little")
    if (
        data[offset + 6 : offset + 10] != b"\0" * 4
        or dib not in {12, 40, 52, 56, 64, 108, 124}
        or not 14 + dib <= pixels < size <= remaining
    ):
        return False
    width_bytes = 2 if dib == 12 else 4
    width = int.from_bytes(
        data[offset + 18 : offset + 18 + width_bytes], "little", signed=dib != 12
    )
    height = int.from_bytes(
        data[offset + 18 + width_bytes : offset + 18 + 2 * width_bytes], "little", signed=dib != 12
    )
    planes_offset = offset + 18 + 2 * width_bytes
    planes = int.from_bytes(data[planes_offset : planes_offset + 2], "little")
    bpp = int.from_bytes(data[planes_offset + 2 : planes_offset + 4], "little")
    return width > 0 and height != 0 and planes == 1 and bpp in {1, 2, 4, 8, 16, 24, 32}


def _decoded_candidates(data: bytes, *, deep: bool) -> list[tuple[bytes, str, str]]:
    if len(data) > 16 * 1024 * 1024:
        return []
    values: list[tuple[bytes, str, str]] = []
    compact = b"".join(data.split())
    decoders = (
        ("base16", lambda: base64.b16decode(compact, casefold=True)),
        ("base32", lambda: base64.b32decode(compact, casefold=True)),
        ("base64", lambda: base64.b64decode(compact, validate=True)),
        ("base85", lambda: base64.b85decode(compact)),
    )
    for name, decode in decoders:
        try:
            decoded = decode()
        except (ValueError, binascii.Error):
            continue
        if _credible(decoded, data):
            values.append((decoded, f"decoded-{name}.bin", f"bounded {name} decoding candidate"))
    if len(compact) >= 8 and len(compact) % 2 == 0 and re.fullmatch(rb"[0-9a-fA-F]+", compact):
        try:
            hex_decoded = bytes.fromhex(compact.decode("ascii"))
            if _credible(hex_decoded, data):
                values.append((hex_decoded, "decoded-hex.bin", "bounded hex decoding candidate"))
        except ValueError:
            pass
    if b"{galf" in data.lower() or b"}galf" in data.lower():
        rev = data[::-1]
        values.append((rev, "decoded-reversed.bin", "byte-reversed candidate (flag confirmed)"))
    if re.search(rb"%[0-9a-fA-F]{2}", data) and re.fullmatch(rb"[\t\n\r\x20-\x7e]+", data):
        # Decode bytes directly: ignoring non-ASCII corrupts binary carriers and
        # fabricates large, apparently credible candidates with preserved magic.
        decoded = urllib.parse.unquote_to_bytes(data)
        if _credible(decoded, data):
            values.append((decoded, "decoded-url.bin", "URL percent-decoding candidate"))
    decompressed = _decompress(data)
    if decompressed is not None and _credible(decompressed, data):
        values.append((decompressed, "decompressed.bin", "bounded decompression"))
    if deep:
        for key in range(1, 256):
            decoded = bytes(value ^ key for value in data)
            has_magic = any(decoded.startswith(magic) for magic, _ in _MAGICS)
            if has_magic or b"flag{" in decoded.lower():
                values.append(
                    (
                        decoded,
                        f"xor-{key:02x}.bin",
                        f"single-byte XOR candidate key=0x{key:02x}",
                    )
                )
                break
    return values


def _credible(candidate: bytes, original: bytes) -> bool:
    if not candidate or candidate == original or len(candidate) > 256 * 1024 * 1024:
        return False
    if any(candidate.startswith(magic) for magic, _ in _MAGICS):
        return True
    printable = sum(byte in b"\t\n\r" or 32 <= byte < 127 for byte in candidate)
    return len(candidate) >= 4 and printable / len(candidate) >= 0.75


def _decompress(data: bytes, limit: int = 256 * 1024 * 1024) -> bytes | None:
    value: bytes | None = None
    try:
        if data.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as source:
                value = source.read(limit + 1)
        elif len(data) >= 2 and data[0] == 0x78:
            z_dec = zlib.decompressobj()
            value = z_dec.decompress(data, limit + 1)
        elif data.startswith(b"BZh"):
            bz2_dec = bz2.BZ2Decompressor()
            value = bz2_dec.decompress(data, max_length=limit + 1)
        elif data.startswith(b"\xfd7zXZ\x00"):
            lzma_dec = lzma.LZMADecompressor()
            value = lzma_dec.decompress(data, max_length=limit + 1)
        else:
            try:
                raw_dec = zlib.decompressobj(-zlib.MAX_WBITS)
                raw_val = raw_dec.decompress(data, limit + 1)
                if len(raw_val) >= 4:
                    value = raw_val
            except Exception:
                return None
    except (OSError, EOFError, zlib.error, lzma.LZMAError):
        return None
    if value is None or len(value) > limit:
        return None
    return value


def _archive_members(data: bytes, limits: CTFLimits) -> list[tuple[bytes, str, str]]:
    values: list[tuple[bytes, str, str]] = []
    try:
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for index, member in enumerate(archive.infolist()):
                    if index >= limits.max_artifacts or member.is_dir():
                        break
                    path = PurePosixPath(member.filename.replace("\\", "/"))
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or member.file_size > limits.max_bytes
                    ):
                        continue
                    if (
                        member.file_size
                        and member.compress_size
                        and member.file_size / member.compress_size > 200
                    ):
                        continue
                    with archive.open(member) as source:
                        content = source.read(min(member.file_size + 1, limits.max_bytes + 1))
                    if len(content) <= limits.max_bytes:
                        values.append(
                            (
                                content,
                                path.name or "archive-member.bin",
                                f"ZIP member {path.name}",
                            )
                        )
            return values
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return values
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for index, tar_member in enumerate(archive):
                if (
                    index >= limits.max_artifacts
                    or not tar_member.isfile()
                    or tar_member.issym()
                    or tar_member.islnk()
                ):
                    continue
                path = PurePosixPath(tar_member.name.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts or tar_member.size > limits.max_bytes:
                    continue
                tar_source = archive.extractfile(tar_member)
                if tar_source is not None:
                    content = tar_source.read(min(tar_member.size + 1, limits.max_bytes + 1))
                    if len(content) <= limits.max_bytes:
                        values.append(
                            (
                                content,
                                path.name or "archive-member.bin",
                                f"TAR member {path.name}",
                            )
                        )
    except (OSError, tarfile.TarError):
        pass
    return values


def _bitplane_visualization(path: Path) -> bytes | None:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        planes = [((rgb[:, :, channel] & 1) * 255).astype(np.uint8) for channel in range(3)]
        montage = np.concatenate(planes, axis=1)
        output = io.BytesIO()
        Image.fromarray(montage, mode="L").save(output, format="PNG")
        return output.getvalue()
    except (OSError, ValueError):
        return None


def _deduplicate_coverage(values: Iterable[Coverage]) -> list[Coverage]:
    chosen: dict[tuple[str, bool], Coverage] = {}
    priority = {"available": 3, "failed": 2, "unavailable": 1, "unsupported": 0}
    required_priority = {"available": 0, "unsupported": 1, "unavailable": 2, "failed": 3}
    for value in values:
        key = (value.component, value.required)
        current = chosen.get(key)
        order = required_priority if value.required else priority
        if current is None or order[value.status] > order[current.status]:
            chosen[key] = value
    return sorted(chosen.values(), key=lambda item: (not item.required, item.component))


def _artifact_path(artifact: Artifact) -> Path:
    if artifact.path is None:
        raise ValueError("artifact has no local job path")
    return artifact.path


def _tool_coverage_status(
    status: str,
) -> Literal["available", "unavailable", "unsupported", "failed"]:
    if status in {"completed", "failed"}:
        return "available"
    if status == "unavailable":
        return "unavailable"
    return "failed"


def _redact_paths(value: Any, root: Path) -> Any:
    """Remove the only absolute path namespace visible inside a CTF job."""
    if isinstance(value, str):
        return value.replace(str(root.resolve()), ".")
    if isinstance(value, list):
        return [_redact_paths(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _redact_paths(item, root) for key, item in value.items()}
    return value


def _analyzer_coverage_status(
    status: str,
) -> Literal["available", "unavailable", "unsupported", "failed"]:
    if status == "ok":
        return "available"
    if status == "unavailable":
        return "unavailable"
    if status == "unsupported":
        return "unsupported"
    return "failed"
