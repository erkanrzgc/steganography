"""Portable JSON v2, HTML and SARIF renderers."""

from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path
from typing import Any


def json_v2(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)


def html_v2(report: dict[str, Any]) -> str:
    files = report.get("files") or report.get("analyses") or [report]
    rows: list[str] = []
    for item in files:
        file_info = item.get("file", {})
        findings = item.get("findings", [])
        policy = item.get("coverage_policy")
        coverage_html = ""
        if isinstance(policy, dict):
            requirements = "".join(
                "<li>{}: {}</li>".format(
                    html.escape(str(entry.get("component", "unknown"))),
                    html.escape(str(entry.get("status", "unavailable"))),
                )
                for entry in policy.get("required", [])
            )
            coverage_html = "<p>Required native coverage: {}. {}</p><ul>{}</ul>".format(
                "complete" if policy.get("complete") else "incomplete",
                html.escape(str(policy.get("scope", ""))),
                requirements,
            )
        finding_html = (
            "".join(
                "<li><code>{}</code> — {}: {}</li>".format(
                    html.escape(str(finding.get("rule_id", "unknown"))),
                    html.escape(str(finding.get("verdict", "inconclusive"))),
                    html.escape(str(finding.get("detail", ""))),
                )
                for finding in findings
            )
            or "<li>No findings.</li>"
        )
        rows.append(
            "<article><h2>{}</h2><p class='{}'>{} ({:.0%})</p><ul>{}</ul>{}</article>".format(
                html.escape(str(file_info.get("name", "evidence"))),
                html.escape(str(item.get("verdict", "inconclusive"))),
                html.escape(str(item.get("verdict", "inconclusive"))),
                float(item.get("confidence", 0)),
                finding_html,
                coverage_html,
            )
        )
    artifact_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(item.get("name", "artifact"))),
            html.escape(str(item.get("sha256", ""))),
            html.escape(str(item.get("parent_id") or "root")),
            html.escape(str(item.get("provenance", ""))),
        )
        for item in report.get("artifacts", [])
    )
    artifact_section = (
        "<h2>Artifact graph</h2><table><thead><tr><th>Name</th><th>SHA-256</th>"
        "<th>Parent</th><th>Provenance</th></tr></thead><tbody>"
        + artifact_rows
        + "</tbody></table>"
        if artifact_rows
        else ""
    )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>steganography report</title>
<style>body{font:15px system-ui;background:#0a1020;color:#e7edf8;max-width:1100px;
margin:auto;padding:2rem}article{border:1px solid #34415a;border-radius:10px;padding:1rem;
margin:1rem 0}table{border-collapse:collapse;width:100%}td,th{border:1px solid #34415a;
padding:.5rem;text-align:left;overflow-wrap:anywhere}code{color:#8bd5ff}.confirmed,.likely{color:#ff7b72}.suspicious{color:#e5c07b}
.no_indicators{color:#7ee787}</style></head><body>
<h1>steganography DFIR report</h1>"""
        + "".join(rows)
        + artifact_section
        + "</body></html>"
    )


def sarif_v2(report: dict[str, Any]) -> dict[str, Any]:
    files = report.get("files") or report.get("analyses") or [report]
    rules: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for item in files:
        uri = item.get("file", {}).get("path") or item.get("file", {}).get("name", "evidence")
        for finding in item.get("findings", []):
            if finding.get("verdict") in {"no_indicators", "inconclusive"}:
                continue
            rule_id = str(finding.get("rule_id", "steganography.unknown"))
            rules.setdefault(
                rule_id,
                {
                    "id": rule_id,
                    "shortDescription": {"text": rule_id},
                    "help": {"text": "Review the evidence and analyzer detail."},
                },
            )
            level = "error" if finding.get("verdict") == "confirmed" else "warning"
            results.append(
                {
                    "ruleId": rule_id,
                    "level": level,
                    "message": {"text": str(finding.get("detail", rule_id))},
                    "locations": [{"physicalLocation": {"artifactLocation": {"uri": str(uri)}}}],
                    "properties": {
                        "verdict": finding.get("verdict"),
                        "confidence": finding.get("confidence"),
                        "evidenceStrength": finding.get("evidence_strength"),
                    },
                }
            )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "steganography",
                        "informationUri": "https://github.com/erkanrzgc/steganography",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
                "properties": {
                    "analyses": [
                        {
                            "file": item.get("file", {}).get("name", "evidence"),
                            "verdict": item.get("verdict", "inconclusive"),
                            "coverage_policy": item.get("coverage_policy"),
                        }
                        for item in files
                    ]
                },
            }
        ],
    }


def write_evidence_bundle(
    report: dict[str, Any],
    artifacts: list[tuple[str, Path]],
    output: Path,
) -> None:
    """Write a portable, no-clobber ZIP containing all normalized report views."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("report.json", json_v2(report))
        bundle.writestr("report.html", html_v2(report))
        bundle.writestr("report.sarif", json.dumps(sarif_v2(report), indent=2))
        for name, path in artifacts:
            source = Path(path)
            if source.is_symlink() or not source.is_file():
                raise ValueError("bundle artifacts must be regular, non-symlink files")
            safe_name = Path(name).name
            bundle.write(source, f"artifacts/{safe_name}")
