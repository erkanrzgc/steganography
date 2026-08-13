"""Legacy and schema-v1 JSON/HTML report writers."""
from __future__ import annotations

import html
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from core.result import AnalysisResult, FileAnalysis, ScanReport
from core.version import __version__


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
    _atomic_write_text(
        out,
        json.dumps({"files": [_serialize(result) for result in results]}, indent=2),
    )


def write_json_v1(report: ScanReport, out: Path) -> None:
    _atomic_write_text(
        out,
        json.dumps(
            report.to_dict(tool_version=__version__),
            indent=2,
            ensure_ascii=False,
        ),
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
    sigs = "<br>".join(
        f"{html.escape(s.name)}({s.score}): {html.escape(s.detail)}" for s in r.signals
    ) or "—"
    return (
        f"<tr><td>{html.escape(str(path))}</td>"
        f"<td>{html.escape(r.analyzer)}</td>"
        f"<td class='{cls}'>{r.suspicion}</td><td>{sigs}</td></tr>"
    )


def write_html(
    results: list[tuple[Path, AnalysisResult]] | ScanReport, out: Path
) -> None:
    if isinstance(results, ScanReport):
        write_html_v1(results, out)
        return
    rows = "\n".join(_row(result) for result in results)
    _atomic_write_text(out, _HTML_TEMPLATE.format(rows=rows))


_HTML_V1_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>steganography report</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}}
main{{max-width:1200px;margin:auto}} h1{{font-weight:500}} .summary{{display:flex;gap:12px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 18px}}
input{{width:100%;box-sizing:border-box;margin:18px 0;padding:10px;background:#161b22;
color:#e6edf3;border:1px solid #30363d;border-radius:6px}}
details{{background:#161b22;border:1px solid #30363d;border-radius:8px;margin:10px 0;padding:12px}}
summary{{cursor:pointer}} table{{width:100%;border-collapse:collapse;margin-top:10px}}
th,td{{padding:7px;border-bottom:1px solid #30363d;text-align:left;vertical-align:top}}
.high{{color:#ff7b72}} .medium{{color:#d29922}} .low{{color:#3fb950}}
.error{{color:#ff7b72}} code{{overflow-wrap:anywhere}}
</style></head><body><main><h1>steganography scan report</h1>
<p>Generated {generated}; profile <b>{profile}</b>; schema {schema}</p>
<div class="summary">{summary}</div>
<input id="filter" placeholder="Filter files, hashes, analyzers or signals…">
<section id="files">{files}</section></main>
<script>
const q=document.getElementById('filter');q.addEventListener('input',()=>{{
 const needle=q.value.toLowerCase();document.querySelectorAll('details.file').forEach(el=>{{
 el.hidden=!el.textContent.toLowerCase().includes(needle);}});}});
</script></body></html>"""


def write_html_v1(report: ScanReport, out: Path) -> None:
    summary = "".join(
        f'<div class="card"><b>{html.escape(label)}</b><br>{value}</div>'
        for label, value in (
            ("Files", len(report.files)),
            ("High", sum(item.severity == "high" for item in report.files)),
            ("Medium", sum(item.severity == "medium" for item in report.files)),
            ("Errors", sum(_has_error(item) for item in report.files)),
        )
    )
    files = "\n".join(_file_block(item) for item in report.files)
    document = _HTML_V1_TEMPLATE.format(
        generated=html.escape(report.generated_at),
        profile=html.escape(report.profile),
        schema=html.escape(report.schema_version),
        summary=summary,
        files=files or "<p>No files analyzed.</p>",
    )
    _atomic_write_text(out, document)


def _file_block(item: FileAnalysis) -> str:
    file_info = item.file
    rows = []
    for result in item.results:
        signals = "<br>".join(
            f"<b>{html.escape(signal.name)}</b> ({signal.score}): "
            f"{html.escape(signal.detail)}"
            for signal in result.signals
        ) or "—"
        status = html.escape(result.status)
        if result.error:
            status += f': <span class="error">{html.escape(result.error)}</span>'
        rows.append(
            f"<tr><td>{html.escape(result.analyzer)}</td><td>{result.suspicion}</td>"
            f"<td>{status}</td><td>{signals}</td></tr>"
        )
    return (
        f'<details class="file"><summary><span class="{item.severity}">'
        f"[{item.overall_score:03d} {html.escape(item.severity.upper())}]</span> "
        f"{html.escape(file_info.path)}</summary>"
        f"<p>SHA-256: <code>{html.escape(file_info.sha256 or 'unavailable')}</code><br>"
        f"Type: {html.escape(file_info.detected_type)}; size: {file_info.size} bytes</p>"
        "<table><thead><tr><th>Analyzer</th><th>Score</th><th>Status</th>"
        f"<th>Signals</th></tr></thead><tbody>{''.join(rows)}</tbody></table></details>"
    )


def _has_error(item: FileAnalysis) -> bool:
    return any(result.status == "error" for result in item.results)


def _atomic_write_text(out: Path, content: str) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=out.suffix, dir=out.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, out)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
