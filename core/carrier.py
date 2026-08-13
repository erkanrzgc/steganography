"""Carrier ABC: every embed/extract/analyze module implements this."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.result import AnalysisResult, EmbedResult


class CarrierError(Exception):
    """Base for all carrier-level errors."""


class InsufficientCapacityError(CarrierError):
    pass


class Carrier(ABC):
    name: str = ""
    method_id: str = ""
    extensions: tuple[str, ...] = ()
    can_embed: bool = True
    can_extract: bool = True
    priority: int = 100
    experimental: bool = False
    requires_explicit: bool = False

    @property
    def identifier(self) -> str:
        return self.method_id or self.name

    @abstractmethod
    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult: ...

    @abstractmethod
    def extract(self, src: Path) -> bytes: ...

    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...

    @abstractmethod
    def capacity(self, src: Path) -> int: ...

    def supports(self, src: Path, *, detected_extension: str | None = None) -> bool:
        """Return whether this carrier should inspect ``src``.

        Existing third-party carriers inherit this extension-based behavior.
        The registry may additionally supply a content-derived extension.
        """
        suffixes = {src.suffix.lower()}
        if detected_extension:
            suffixes.add(detected_extension.lower())
        return bool(suffixes.intersection(self.extensions))

    def embed_with_options(
        self,
        src: Path,
        payload: bytes,
        out: Path,
        *,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> EmbedResult:
        """Compatibility-preserving extension point for configurable carriers."""
        del steg_key, options
        return self.embed(src, payload, out)

    def extract_with_options(
        self,
        src: Path,
        *,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> bytes:
        del steg_key, options
        return self.extract(src)
