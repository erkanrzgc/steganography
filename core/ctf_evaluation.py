"""Frozen CTF recovery evaluation, separate from detector accuracy or blind qualification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from core.ctf import CTFLimits, CTFService
from core.version import __version__

_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_HASH = re.compile(r"[0-9a-f]{64}")


def _digest(path: Path, limit: int) -> str:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("evaluation files must be regular and non-symlink")
    if path.stat().st_size > limit:
        raise ValueError("evaluation file exceeds byte limit")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            total += len(chunk)
            if total > limit:
                raise ValueError("evaluation file grew beyond byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or len(value) > 240 or "\\" in value:
        raise ValueError("invalid evaluation relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError("evaluation path must stay inside its corpus")
    return root / value


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise ValueError("invalid evaluation identity")
    return value


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("invalid evaluation checksum")
    return value


def load_suite(manifest: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate every input before starting jobs; never infer an expected answer."""
    _hash(expected_sha256)
    if _digest(manifest, 65536) != expected_sha256:
        raise ValueError("frozen manifest checksum mismatch")
    with manifest.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("manifest changed during validation")
    suite = json.loads(raw)
    if not isinstance(suite, dict) or set(suite) != {
        "schema_version", "suite_id", "mode", "limits", "cases"
    } or suite["schema_version"] != "ctf-evaluation-manifest-v1":
        raise ValueError("invalid evaluation manifest schema")
    _identity(suite["suite_id"])
    if suite["mode"] not in ("quick", "balanced", "deep"):
        raise ValueError("invalid evaluation mode")
    caps = dict(max_depth=3, max_artifacts=256, max_bytes=1024**3,
                tool_timeout=30, job_timeout=180)
    limits = suite["limits"]
    if not isinstance(limits, dict) or set(limits) != set(caps):
        raise ValueError("evaluation limits must be explicit")
    for key, cap in caps.items():
        value = limits[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid evaluation limit")
        if key in {"max_depth", "max_artifacts", "max_bytes"} and type(value) is not int:
            raise ValueError("evaluation counts must be integers")
        if not (0 if key == "max_depth" else 1) <= value <= cap:
            raise ValueError("evaluation limit outside default budget")
    cases = suite["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 120:
        raise ValueError("evaluation needs 1 to 120 cases")
    ids: set[str] = set()
    inputs: set[str] = set()
    for case in cases:
        required = {"id", "source_group", "method_family", "input", "input_sha256",
                    "payload_sha256", "required_tools"}
        if not isinstance(case, dict) or not required <= set(case) or set(case) - required not in (
            set(), {"wordlist", "wordlist_sha256"}
        ):
            raise ValueError("invalid evaluation case fields")
        for key in ("id", "source_group", "method_family"):
            _identity(case[key])
        for key in ("input_sha256", "payload_sha256"):
            _hash(case[key])
        if case["id"] == "evaluation.json" or case["id"] in ids or case["input_sha256"] in inputs:
            raise ValueError("duplicate evaluation case or input")
        ids.add(case["id"])
        inputs.add(case["input_sha256"])
        if case["input_sha256"] == case["payload_sha256"]:
            raise ValueError("input cannot be its own expected payload")
        tools = case["required_tools"]
        if (not isinstance(tools, list) or len(tools) > 12
                or len(set(map(str, tools))) != len(tools)):
            raise ValueError("invalid required tools")
        for tool in tools:
            _identity(tool)
        source = _relative(manifest.parent, case["input"])
        if _digest(source, min(10 * 1024**2, limits["max_bytes"])) != case["input_sha256"]:
            raise ValueError("challenge input checksum mismatch")
        if "wordlist" in case:
            wordlist = _relative(manifest.parent, case["wordlist"])
            if _digest(wordlist, 1024**2) != _hash(case["wordlist_sha256"]):
                raise ValueError("wordlist checksum mismatch")
    return suite


def evaluate_suite(
    manifest: Path, expected_sha256: str, output: Path, *, hardware: str,
    service: CTFService | None = None,
) -> dict[str, Any]:
    """Run frozen jobs once; missing tools/failures remain in the denominator."""
    _identity(hardware)
    suite = load_suite(manifest, expected_sha256)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("evaluation output must not use symlinks")
    output.mkdir(parents=True, exist_ok=False)
    solver = service or CTFService()
    limits = CTFLimits(**suite["limits"])
    rows = []
    for case in suite["cases"]:
        row: dict[str, Any] = {key: case[key] for key in (
            "id", "source_group", "method_family", "input_sha256", "payload_sha256"
        )}
        row.update(status="unavailable", exact_payload=False, completed_exact=False,
                   method_top3=False, duration_ms=None)
        missing = [tool for tool in case["required_tools"] if shutil.which(tool) is None]
        row["unavailable_tools"] = missing
        if not missing:
            job = output / case["id"]
            try:
                source = _relative(manifest.parent, case["input"])
                if _digest(source, min(10 * 1024**2, limits.max_bytes)) != case["input_sha256"]:
                    raise ValueError("challenge changed after validation")
                wordlist = None
                if "wordlist" in case:
                    wordlist = _relative(manifest.parent, case["wordlist"])
                    if _digest(wordlist, 1024**2) != case["wordlist_sha256"]:
                        raise ValueError("wordlist changed after validation")
                # Oracle labels/hashes are used only AFTER solve, never passed to it.
                report = solver.solve(source, job, mode=suite["mode"], limits=limits,
                                      wordlist=wordlist)
                if not math.isfinite(report.duration_ms) or report.duration_ms < 0:
                    raise ValueError("invalid job duration")
                row.update(status=report.status, duration_ms=report.duration_ms)
                for artifact in report.artifacts:
                    if artifact.parent_id is None or artifact.path is None:
                        continue
                    artifact.path.resolve().relative_to(job.resolve())
                    actual = _digest(artifact.path, limits.max_bytes)
                    if actual != artifact.sha256:
                        raise ValueError("artifact identity mismatch")
                    row["exact_payload"] |= actual == case["payload_sha256"]
                row["completed_exact"] = row["exact_payload"] and report.status == "completed"
                top3 = sorted(report.recommendations, key=lambda r: r.priority)[:3]
                row["method_top3"] = report.status == "completed" and any(
                    r.method_family == case["method_family"] for r in top3
                )
                row["tools"] = [{"tool": t.tool, "status": t.status, "exit_code": t.exit_code}
                                for t in report.tools]
            except (OSError, ValueError, RuntimeError):
                row.update(status="failed", exact_payload=False, completed_exact=False,
                           method_top3=False, reason="job or integrity verification failed")
        rows.append(row)
    durations = [row["duration_ms"] / 1000 for row in rows if row["duration_ms"] is not None]
    by_method = {}
    for method in sorted({row["method_family"] for row in rows}):
        cell = [row for row in rows if row["method_family"] == method]
        remaining = [row for row in cell if not row["completed_exact"]]
        by_method[method] = {
            "case_count": len(cell),
            "completed_exact_count": sum(row["completed_exact"] for row in cell),
            "completed_exact_rate": sum(row["completed_exact"] for row in cell) / len(cell),
            "unavailable_count": sum(row["status"] == "unavailable" for row in cell),
            "remaining_count": len(remaining),
            "remaining_method_top3_count": sum(row["method_top3"] for row in remaining),
        }
    result = {
        "schema_version": "ctf-evaluation-v1", "suite_id": suite["suite_id"],
        "manifest_sha256": expected_sha256, "tool_version": __version__,
        "source_sha256": {name: _digest(Path(__file__).with_name(name), 1024**2)
                          for name in ("ctf.py", "tools.py", "ctf_evaluation.py")},
        "hardware": hardware, "mode": suite["mode"], "limits": suite["limits"],
        "case_count": len(rows), "source_groups": len({r["source_group"] for r in rows}),
        "completed_exact_rate": sum(r["completed_exact"] for r in rows) / len(rows),
        "recovered_including_incomplete": sum(r["exact_payload"] for r in rows),
        "median_seconds": float(np.median(durations)) if durations else None,
        "p95_seconds": float(np.percentile(durations, 95)) if durations else None,
        "timed_case_count": len(durations), "cases": rows,
        "by_method": by_method,
        "qualification": "development_only", "blind_gate": "unavailable",
        "qualification_reason": "Independent challenge custody and blind audit are not certified.",
    }
    with (output / "evaluation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result
