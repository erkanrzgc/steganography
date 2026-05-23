"""JSON and HTML report writers for batch scans."""
import json
from pathlib import Path

from core.result import AnalysisResult


def _serialize(item: tuple[Path, AnalysisResult]) -> dict:
    path, r = item
    return {
        "path": str(path),
        "analyzer": r.analyzer,
        "suspicion": r.suspicion,
        "signals": [{"name": s.name, "score": s.score, "detail": s.detail} for s in r.signals],
        "explanation": r.explanation,
    }


def write_json(results: list[tuple[Path, AnalysisResult]], out: Path) -> None:
    out.write_text(
        json.dumps({"files": [_serialize(r) for r in results]}, indent=2),
        encoding="utf-8",
    )


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>steganography report</title>
<style>
 body{{font-family:system-ui,monospace;background:#111;color:#eee;padding:24px}}
 h1{{font-weight:300}}
 table{{width:100%;border-collapse:collapse}}
 th,td{{padding:8px;border-bottom:1px solid #333;text-align:left;font-size:14px}}
 .high{{color:#f55}} .mid{{color:#fb3}} .low{{color:#6f6}}
 details{{margin:4px 0}}
</style></head><body>
<h1>steganography — scan report</h1>
<table><thead><tr><th>file</th><th>analyzer</th><th>suspicion</th><th>signals</th></tr></thead><tbody>
{rows}
</tbody></table></body></html>"""


def _row(item: tuple[Path, AnalysisResult]) -> str:
    path, r = item
    cls = "high" if r.suspicion >= 70 else "mid" if r.suspicion >= 30 else "low"
    sigs = "<br>".join(f"{s.name}({s.score}): {s.detail}" for s in r.signals) or "—"
    return (
        f"<tr><td>{path}</td><td>{r.analyzer}</td>"
        f"<td class='{cls}'>{r.suspicion}</td><td>{sigs}</td></tr>"
    )


def write_html(results: list[tuple[Path, AnalysisResult]], out: Path) -> None:
    rows = "\n".join(_row(r) for r in results)
    out.write_text(_HTML_TEMPLATE.format(rows=rows), encoding="utf-8")
