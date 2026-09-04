"""Portable JSON v2, HTML and SARIF renderers."""

from __future__ import annotations

import html
import json
from typing import Any


def json_v2(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)


def html_v2(report: dict[str, Any]) -> str:
    files = report.get("files") or [report]
    rows: list[str] = []
    for item in files:
        file_info = item.get("file", {})
        findings = item.get("findings", [])
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
            "<article><h2>{}</h2><p class='{}'>{} ({:.0%})</p><ul>{}</ul></article>".format(
                html.escape(str(file_info.get("name", "evidence"))),
                html.escape(str(item.get("verdict", "inconclusive"))),
                html.escape(str(item.get("verdict", "inconclusive"))),
                float(item.get("confidence", 0)),
                finding_html,
            )
        )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>steganography report</title>
<style>body{font:15px system-ui;background:#0a1020;color:#e7edf8;max-width:1100px;
margin:auto;padding:2rem}article{border:1px solid #34415a;border-radius:10px;padding:1rem;
margin:1rem 0}code{color:#8bd5ff}.confirmed,.likely{color:#ff7b72}.suspicious{color:#e5c07b}
.no_indicators{color:#7ee787}</style></head><body>
<h1>steganography DFIR report</h1>"""
        + "".join(rows)
        + "</body></html>"
    )


def sarif_v2(report: dict[str, Any]) -> dict[str, Any]:
    files = report.get("files") or [report]
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
            }
        ],
    }
