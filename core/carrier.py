"""Carrier ABC: every embed/extract/analyze module implements this."""
from abc import ABC, abstractmethod
from pathlib import Path

from core.result import AnalysisResult, EmbedResult


class CarrierError(Exception):
    """Base for all carrier-level errors."""


class InsufficientCapacityError(CarrierError):
    pass


class Carrier(ABC):
    name: str = ""
    extensions: tuple[str, ...] = ()
    can_embed: bool = True
    can_extract: bool = True

    @abstractmethod
    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult: ...

    @abstractmethod
    def extract(self, src: Path) -> bytes: ...

    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...

    @abstractmethod
    def capacity(self, src: Path) -> int: ...
