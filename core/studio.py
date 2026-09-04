"""High-level, API-safe Studio operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.filetype import detect_type
from core.payload import unpack
from core.service import StegoService


class StudioService:
    """Expose capacity/embed/extract without retaining passwords or steg keys."""

    def __init__(self, stego: StegoService | None = None) -> None:
        self.stego = stego or StegoService()

    def capacity(self, carrier_path: Path, *, method: str | None = None) -> dict[str, Any]:
        carrier_path = Path(carrier_path)
        detected = detect_type(carrier_path)
        if method:
            candidates = [self.stego.registry.get_carrier(method)]
        else:
            candidates = self.stego.registry.select_carriers(
                carrier_path, detected_extension=detected.extension
            )
        values = []
        for carrier in candidates:
            if not carrier.can_embed:
                continue
            try:
                raw_capacity = carrier.capacity(carrier_path)
                status = "available"
                error = None
            except Exception as exc:
                raw_capacity = 0
                status = "unavailable"
                error = f"{type(exc).__name__}: {exc}"
            values.append(
                {
                    "method": carrier.identifier,
                    "carrier_capacity_bytes": max(0, raw_capacity),
                    "payload_v3_capacity_bytes": max(0, raw_capacity - 128),
                    "status": status,
                    "error": error,
                    "detectability": "increases with payload density",
                    "recompression_resilience": (
                        "low" if detected.extension in {".png", ".bmp"} else "format-dependent"
                    ),
                }
            )
        return {"carrier_type": detected.name, "methods": values}

    def embed(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("payload_version", 3)
        return self.stego.embed(*args, **kwargs)

    def preview(
        self,
        carrier_path: Path,
        *,
        method: str | None = None,
        steg_key: str | None = None,
    ) -> dict[str, Any]:
        """Read authenticated payload framing metadata without extracting data.

        Encrypted payload bytes remain encrypted.  The returned metadata is the
        small v3 envelope (original name and size), never payload contents.
        """
        carrier_path = Path(carrier_path)
        detected = detect_type(carrier_path)
        if method:
            candidates = [self.stego.registry.get_carrier(method)]
        else:
            candidates = self.stego.registry.select_carriers(
                carrier_path, detected_extension=detected.extension
            )
        failures: list[str] = []
        for carrier in candidates:
            if not carrier.can_extract:
                continue
            try:
                parsed = unpack(
                    carrier.extract_with_options(
                        carrier_path,
                        steg_key=steg_key,
                        options=None,
                    )
                )
                return {
                    "method": carrier.identifier,
                    "payload_version": parsed.version,
                    "encrypted": parsed.encrypted,
                    "compressed": parsed.compressed,
                    "ecc_symbols": parsed.ecc_symbols,
                    "metadata": parsed.metadata or {},
                }
            except Exception as exc:
                failures.append(f"{carrier.identifier}: {type(exc).__name__}")
        detail = ", ".join(failures) or "no extractable carrier"
        raise ValueError(f"payload metadata unavailable ({detail})")

    def extract(self, *args: Any, **kwargs: Any):
        return self.stego.extract(*args, **kwargs)
