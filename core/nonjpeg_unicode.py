"""Pinned Unicode conformance controls; not a natural-language benchmark."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from core.nonjpeg_data import regular_read
from core.research_acquisition import UNICODE_FILES, digest, write_json


def controls(root: Path, acquisition: dict, out: Path) -> dict:
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("fresh nonsymlink Unicode output required")
    rows = []
    counts: Counter[str] = Counter()
    bindings = {}
    for name in UNICODE_FILES:
        binding = acquisition.get("unicode", {}).get(name, {})
        if binding.get("status") != "ready" or binding.get("version") != "17.0.0":
            continue
        raw = regular_read(root / name, 16 * 1024**2)
        if digest(raw) != binding.get("sha256"):
            raise ValueError("Unicode source checksum mismatch")
        bindings[name] = digest(raw)
        for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            content = line.split("#", 1)[0].strip()
            if not content or content.startswith("@"):
                continue
            fields = content.split(";")
            tokens = fields[0].replace("÷", " ").replace("×", " ").split()
            points = [int(token, 16) for token in tokens]
            if (
                not points
                or len(points) > 128
                or any(p > 0x10FFFF or 0xD800 <= p <= 0xDFFF for p in points)
            ):
                continue
            categories = []
            if 0x200D in points:
                categories.append("zwj")
            if 0x200C in points:
                categories.append("zwnj")
            if any(p in (9, 10, 13, 32, 0xA0, 0x2000, 0x2028, 0x2029) for p in points):
                categories.append("whitespace")
            if name == "NormalizationTest.txt":
                categories.append("normalization")
            if name == "BidiCharacterTest.txt":
                categories.append("bidi")
            if name == "emoji-test.txt":
                categories.append("emoji")
            categories = [c for c in categories if counts[c] < 32]
            if not categories:
                continue
            row = {
                "source": name,
                "line": number,
                "source_sha256": digest(raw),
                "codepoints": points,
                "text": "".join(map(chr, points)),
                "categories": categories,
                "role": "algorithm_conformance_negative",
            }
            if name == "NormalizationTest.txt":
                row["normalization_columns"] = [
                    [int(token, 16) for token in field.split()] for field in fields[:5]
                ]
            rows.append(row)
            counts.update(categories)
    required = {"zwj", "zwnj", "whitespace", "normalization", "bidi", "emoji"}
    result = {
        "schema_version": "unicode-normal-controls-v1",
        "version": "17.0.0",
        "status": "ready" if required <= counts.keys() else "incomplete",
        "source_sha256": bindings,
        "counts": dict(counts),
        "missing": sorted(required - counts.keys()),
        "rows": rows,
        "natural_document_coverage": "unavailable",
        "detector_scored": False,
    }
    write_json(out, result)
    return result
