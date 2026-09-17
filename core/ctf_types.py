"""Shared, serializable types for CTF jobs and evidence reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ExecutionStatus = Literal["completed", "failed", "timed_out", "cancelled", "unavailable"]


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    name: str
    media_type: str
    size: int
    sha256: str
    depth: int
    provenance: str
    parent_id: str | None = None
    path: Path | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "media_type": self.media_type,
            "size": self.size,
            "sha256": self.sha256,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class ToolExecution:
    tool: str
    version: str | None
    status: ExecutionStatus
    duration_ms: float
    exit_code: int | None
    command: tuple[str, ...]
    output: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "version": self.version,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "command": list(self.command),
            "output": self.output,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class Coverage:
    component: str
    status: Literal["available", "unavailable", "unsupported", "failed"]
    required: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "required": self.required,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Recommendation:
    priority: int
    action: str
    rationale: str
    method_family: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "action": self.action,
            "rationale": self.rationale,
            "method_family": self.method_family,
        }
