"""Run the analysis engine against a labeled corpus and enforce quality gates."""
from __future__ import annotations

import concurrent.futures
import html
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.service import AnalysisService, aggregate_score
from core.version import __version__
from steganography.benchmarking.corpus import load_manifest
from steganography.benchmarking.metrics import classification_metrics

BENCHMARK_SCHEMA_VERSION = "1.0"
VALID_PROFILES = ("sensitive", "balanced", "strict")


def run_benchmark(
    corpus: Path,
    *,
    profiles: tuple[str, ...] = VALID_PROFILES,
    threshold: int = 70,
    jobs: int = 1,
    min_recall: float = 0.90,
    max_false_positive_rate: float = 0.10,
    baseline: Path | None = None,
    max_recall_drop: float = 0.02,
    max_fpr_increase: float = 0.02,
) -> dict[str, Any]:
    """Analyze a corpus and return a deterministic, machine-readable report."""
    corpus = Path(corpus).resolve()
    manifest = load_manifest(corpus, verify_files=True)
    _validate_configuration(
        profiles=profiles,
        threshold=threshold,
        jobs=jobs,
        min_recall=min_recall,
        max_false_positive_rate=max_false_positive_rate,
        max_recall_drop=max_recall_drop,
        max_fpr_increase=max_fpr_increase,
    )
    samples = list(manifest["samples"])
    if jobs == 1:
        analyzed = [_analyze_sample(corpus, sample, profiles) for sample in samples]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            analyzed = list(
                executor.map(
                    lambda sample: _analyze_sample(corpus, sample, profiles), samples
                )
            )
    analyzed.sort(key=lambda item: item["id"])

    profile_reports: dict[str, Any] = {}
    failures: list[str] = []
    error_samples = [item["id"] for item in analyzed if item["errors"]]
    if error_samples:
        failures.append(f"analysis errors in {len(error_samples)} sample(s)")
    for profile in profiles:
        metrics = _group_metrics(analyzed, profile, threshold)
        profile_reports[profile] = metrics
        overall = metrics["overall"]
        if overall["recall"] < min_recall:
            failures.append(
                f"{profile}: recall {overall['recall']:.4f} < {min_recall:.4f}"
            )
        if overall["false_positive_rate"] > max_false_positive_rate:
            failures.append(
                f"{profile}: FPR {overall['false_positive_rate']:.4f} "
                f"> {max_false_positive_rate:.4f}"
            )

    comparison = None
    if baseline is not None:
        comparison, regression_failures = _compare_baseline(
            profile_reports,
            Path(baseline),
            profiles=profiles,
            corpus_digest=manifest["corpus_digest"],
            max_recall_drop=max_recall_drop,
            max_fpr_increase=max_fpr_increase,
        )
        failures.extend(regression_failures)

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "tool_version": __version__,
        "corpus": {
            "schema_version": manifest["schema_version"],
            "seed": manifest["seed"],
            "digest": manifest["corpus_digest"],
            "sample_count": manifest["sample_count"],
            "skipped": manifest.get("skipped", []),
        },
        "configuration": {
            "profiles": list(profiles),
            "threshold": threshold,
            "min_recall": min_recall,
            "max_false_positive_rate": max_false_positive_rate,
            "max_recall_drop": max_recall_drop,
            "max_fpr_increase": max_fpr_increase,
        },
        "profiles": profile_reports,
        "samples": analyzed,
        "baseline_comparison": comparison,
        "gates": {"passed": not failures, "failures": failures},
    }


def write_benchmark_json(report: dict[str, Any], output: Path) -> None:
    _atomic_write(
        Path(output),
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_benchmark_html(report: dict[str, Any], output: Path) -> None:
    profile_rows = []
    for profile, values in report["profiles"].items():
        metrics = values["overall"]
        confusion = metrics["confusion"]
        profile_rows.append(
            "<tr>"
            f"<td>{html.escape(profile)}</td>"
            f"<td>{metrics['precision']:.3f}</td>"
            f"<td>{metrics['recall']:.3f}</td>"
            f"<td>{metrics['f1']:.3f}</td>"
            f"<td>{metrics['false_positive_rate']:.3f}</td>"
            f"<td>{_format_optional(metrics['roc_auc'])}</td>"
            f"<td>{confusion['tp']}/{confusion['tn']}/"
            f"{confusion['fp']}/{confusion['fn']}</td>"
            "</tr>"
        )
    method_sections = []
    for profile, values in report["profiles"].items():
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{metrics['recall']:.3f}</td>"
            f"<td>{metrics['false_positive_rate']:.3f}</td>"
            f"<td>{metrics['f1']:.3f}</td>"
            "</tr>"
            for method, metrics in values["by_method"].items()
        )
        method_sections.append(
            f"<h2>{html.escape(profile)} by method</h2>"
            "<table><thead><tr><th>Method</th><th>Recall</th><th>FPR</th>"
            f"<th>F1</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    failures = report["gates"]["failures"]
    gate_class = "pass" if report["gates"]["passed"] else "fail"
    failure_html = "".join(f"<li>{html.escape(item)}</li>" for item in failures)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>steganography benchmark</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;
background:#0d1117;color:#e6edf3}} table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}
.pass{{color:#3fb950}} .fail{{color:#ff7b72}} code{{overflow-wrap:anywhere}}
</style></head><body><h1>steganography benchmark</h1>
<p>Corpus <code>{html.escape(report['corpus']['digest'])}</code> ·
{report['corpus']['sample_count']} samples</p>
<h2 class="{gate_class}">Gates: {'PASS' if report['gates']['passed'] else 'FAIL'}</h2>
<ul>{failure_html or '<li>All quality gates passed.</li>'}</ul>
<table><thead><tr><th>Profile</th><th>Precision</th><th>Recall</th><th>F1</th>
<th>FPR</th><th>ROC-AUC</th><th>TP/TN/FP/FN</th></tr></thead>
<tbody>{''.join(profile_rows)}</tbody></table>{''.join(method_sections)}
</body></html>"""
    _atomic_write(Path(output), document)


def _analyze_sample(
    corpus: Path, sample: dict[str, Any], profiles: tuple[str, ...]
) -> dict[str, Any]:
    service = AnalysisService(profile="sensitive", ai_provider=None)
    analysis = service.analyze_safe(corpus / sample["path"])
    errors = [
        {
            "analyzer": result.analyzer,
            "status": result.status,
            "error": result.error,
        }
        for result in analysis.results
        if result.status == "error"
    ]
    scores = {
        profile: aggregate_score(analysis.results, profile) for profile in profiles
    }
    return {
        "id": sample["id"],
        "label": sample["label"],
        "method": sample["method"],
        "format": sample["format"],
        "density": sample["density"],
        "scores": scores,
        "errors": errors,
    }


def _group_metrics(
    samples: list[dict[str, Any]], profile: str, threshold: int
) -> dict[str, Any]:
    grouped_method: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_density: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_format: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped_method[sample["method"]].append(sample)
        grouped_density[sample["density"]].append(sample)
        grouped_format[sample["format"]].append(sample)
    return {
        "overall": _metrics_for(samples, profile, threshold),
        "by_method": {
            key: _metrics_for(value, profile, threshold)
            for key, value in sorted(grouped_method.items())
        },
        "by_density": {
            key: _metrics_for(value, profile, threshold)
            for key, value in sorted(grouped_density.items())
        },
        "by_format": {
            key: _metrics_for(value, profile, threshold)
            for key, value in sorted(grouped_format.items())
        },
    }


def _metrics_for(
    samples: list[dict[str, Any]], profile: str, threshold: int
) -> dict[str, Any]:
    observations = [
        (sample["label"] == "stego", int(sample["scores"][profile]))
        for sample in samples
    ]
    return classification_metrics(observations, threshold=threshold)


def _compare_baseline(
    current: dict[str, Any],
    baseline_path: Path,
    *,
    profiles: tuple[str, ...],
    corpus_digest: str,
    max_recall_drop: float,
    max_fpr_increase: float,
) -> tuple[dict[str, Any], list[str]]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("baseline benchmark schema is incompatible")
    if baseline.get("corpus", {}).get("digest") != corpus_digest:
        raise ValueError("baseline was produced from a different corpus digest")
    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for profile in profiles:
        try:
            before = baseline["profiles"][profile]["overall"]
        except KeyError as exc:
            raise ValueError(f"baseline is missing profile {profile!r}") from exc
        after = current[profile]["overall"]
        recall_delta = after["recall"] - before["recall"]
        fpr_delta = after["false_positive_rate"] - before["false_positive_rate"]
        comparisons[profile] = {
            "recall_delta": round(recall_delta, 6),
            "false_positive_rate_delta": round(fpr_delta, 6),
            "roc_auc_delta": _delta(after.get("roc_auc"), before.get("roc_auc")),
        }
        if recall_delta < -max_recall_drop:
            failures.append(
                f"{profile}: recall regression {recall_delta:.4f} "
                f"< -{max_recall_drop:.4f}"
            )
        if fpr_delta > max_fpr_increase:
            failures.append(
                f"{profile}: FPR regression +{fpr_delta:.4f} "
                f"> +{max_fpr_increase:.4f}"
            )
    return {
        "baseline": str(baseline_path),
        "profiles": comparisons,
    }, failures


def _validate_configuration(
    *,
    profiles: tuple[str, ...],
    threshold: int,
    jobs: int,
    min_recall: float,
    max_false_positive_rate: float,
    max_recall_drop: float,
    max_fpr_increase: float,
) -> None:
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be non-empty and unique")
    unknown = set(profiles) - set(VALID_PROFILES)
    if unknown:
        raise ValueError(f"unknown analysis profiles: {sorted(unknown)}")
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    for name, value in (
        ("min_recall", min_recall),
        ("max_false_positive_rate", max_false_positive_rate),
        ("max_recall_drop", max_recall_drop),
        ("max_fpr_increase", max_fpr_increase),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return round(current - baseline, 6)


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _atomic_write(output: Path, content: str) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=output.suffix, dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
