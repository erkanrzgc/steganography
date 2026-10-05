"""Versioned minimum native coverage; completion is not accuracy qualification."""

from __future__ import annotations

from dataclasses import dataclass

from core.ctf_types import Coverage
from core.result import FileAnalysis

_FORMAT_REQUIREMENTS = {
    "png": ("image_lsb", "image_lsb_scatter", "image_bitplane"),
    "bmp": ("image_lsb", "image_lsb_scatter", "image_bitplane"),
    "jpeg": ("image_jpeg", "image_jpeg_dct"),
    "gif": ("image_gif",),
    "wav": ("audio_wav",),
    "mp3": ("audio_mp3",),
    "pdf": ("file_pdf",),
    "text": ("text_whitespace", "text_zerowidth", "text_anomalies"),
}
POLICY = "native-format-minimum-v1"


@dataclass(frozen=True, slots=True)
class CoverageAssessment:
    detected_type: str
    requirements: tuple[Coverage, ...]

    @property
    def complete(self) -> bool:
        return all(item.status == "available" for item in self.requirements)

    def to_dict(self):
        return {
            "policy": POLICY,
            "detected_type": self.detected_type,
            "complete": self.complete,
            "scope": "minimum native execution, not method support or a clean-file certificate",
            "required": [item.to_dict() for item in self.requirements],
        }


def assess_coverage(analysis: FileAnalysis) -> CoverageAssessment:
    """Content-detected format, not filename; AI/tools/models cannot fill native gaps."""
    fmt = analysis.file.detected_type
    required = ("file_structure", "signatures", *_FORMAT_REQUIREMENTS.get(fmt, ()))
    requirements = []
    for name in required:
        results = [r for r in analysis.results if r.analyzer == name]
        if not results:
            requirements.append(Coverage(name, "unavailable", True, "Required analyzer absent"))
        elif len(results) != 1:
            requirements.append(Coverage(name, "failed", True, "Ambiguous duplicate results"))
        elif results[0].status == "ok":
            requirements.append(Coverage(name, "available", True))
        else:
            status = results[0].status
            requirements.append(
                Coverage(
                    name,
                    "unsupported"
                    if status == "unsupported"
                    else "unavailable"
                    if status == "unavailable"
                    else "failed",
                    True,
                    "Required analyzer did not complete",
                )
            )
    if fmt not in _FORMAT_REQUIREMENTS:
        requirements.append(
            Coverage(
                "format_specific_analysis",
                "unsupported",
                True,
                "No native format-specific coverage policy; generic inspection only",
            )
        )
    return CoverageAssessment(fmt, tuple(requirements))
