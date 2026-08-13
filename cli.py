"""steganography CLI: embed, extract, analyze, scan, serve and list-modules."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from core.result import AnalysisResult, ScanReport
from core.service import AnalysisService, StegoService
from core.version import __version__
from registry import Registry
from report.report import write_html, write_json, write_json_v1
from ui.banner import print_gradient_banner


def _build_registry() -> Registry:
    registry = Registry()
    registry.autodiscover()
    return registry


def _password_from_args(args: argparse.Namespace) -> str | None:
    direct = getattr(args, "password", None)
    if direct is not None:
        return direct
    password_file = getattr(args, "password_file", None)
    if password_file:
        return Path(password_file).read_text(encoding="utf-8").rstrip("\r\n")
    if getattr(args, "password_stdin", False):
        return sys.stdin.readline().rstrip("\r\n")
    return os.environ.get("STEGANO_PASSWORD") or None


def _enable_ai_if_requested(args: argparse.Namespace) -> str | None:
    if not getattr(args, "ai", False):
        from modules.ai_triage import set_provider

        set_provider(None)
        return None
    from modules.ai_provider_nim import make_provider
    from modules.ai_triage import set_provider

    upload_file = bool(getattr(args, "allow_ai_file_upload", False))
    provider = make_provider(upload_file=upload_file)
    if provider is None:
        set_provider(None)
        return "warning: --ai requested but NVIDIA_NIM_API_KEY is not set; using heuristics"
    set_provider(provider)
    privacy = "signals + file" if upload_file else "signals only"
    return f"ai: NVIDIA NIM provider registered ({privacy})"


def cmd_embed(args: argparse.Namespace) -> int:
    password = _password_from_args(args)
    result = StegoService().embed(
        Path(args.input),
        Path(args.carrier),
        Path(args.out),
        password=password,
        method=args.method,
        steg_key=args.steg_key,
        options={"channels": args.channels},
        no_clobber=args.no_clobber,
    )
    print(
        f"embedded {result.bytes_written} bytes via {result.carrier} "
        f"→ {result.out_path}"
    )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    password = _password_from_args(args)
    size, carrier, version = StegoService().extract_to(
        Path(args.input),
        Path(args.out),
        password=password,
        method=args.method,
        steg_key=args.steg_key,
        no_clobber=args.no_clobber,
    )
    print(
        f"extracted {size} bytes via {carrier} (payload v{version}) → {args.out}"
    )
    return 0


def _analyze_file(registry: Registry, path: Path) -> list[AnalysisResult]:
    """Backward-compatible internal helper returning per-module results."""
    return list(AnalysisService(registry).analyze(path).results)


def _legacy_result(result: AnalysisResult) -> dict:
    return {
        "analyzer": result.analyzer,
        "suspicion": result.suspicion,
        "signals": [
            {"name": signal.name, "score": signal.score, "detail": signal.detail}
            for signal in result.signals
        ],
        "explanation": result.explanation,
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    registry = _build_registry()
    status = _enable_ai_if_requested(args)
    machine_output = args.json or args.output_format == "json-v1"
    if status and not machine_output:
        print(status, file=sys.stderr)
    maximum = args.max_file_size * 1024 * 1024 if args.max_file_size else None
    analysis = AnalysisService(
        registry,
        profile=args.profile,
        max_file_size=maximum,
    ).analyze(Path(args.input))
    if args.json:
        print(json.dumps([_legacy_result(result) for result in analysis.results], indent=2))
    elif args.output_format == "json-v1":
        print(
            json.dumps(
                analysis.to_dict(tool_version=__version__),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(
            f"  overall [{analysis.overall_score:3d}] "
            f"{analysis.severity} ({analysis.profile})"
        )
        print(
            f"  sha256 {analysis.file.sha256}  "
            f"type={analysis.file.detected_type} size={analysis.file.size}"
        )
        for result in analysis.results:
            suffix = f" error={result.error}" if result.error else ""
            print(f"  [{result.suspicion:3d}] {result.analyzer} [{result.status}]{suffix}")
            for signal in result.signals:
                print(
                    f"        - {signal.name}({signal.score}, {signal.evidence}): "
                    f"{signal.detail}"
                )
    return _failure_exit((analysis.overall_score,), args.fail_on)


def cmd_scan(args: argparse.Namespace) -> int:
    registry = _build_registry()
    status = _enable_ai_if_requested(args)
    if status:
        print(status, file=sys.stderr)
    root = Path(args.dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    out = Path(args.out)
    out_resolved = out.resolve(strict=False)
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (args.follow_symlinks or not path.is_symlink())
        and path.resolve(strict=False) != out_resolved
    ]
    paths.sort(key=lambda path: str(path))
    maximum = args.max_file_size * 1024 * 1024 if args.max_file_size else None
    service = AnalysisService(
        registry,
        profile=args.profile,
        max_file_size=maximum,
    )
    if args.jobs == 1:
        analyses = [service.analyze_safe(path) for path in paths]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            analyses = list(executor.map(service.analyze_safe, paths))
    scan_report = ScanReport(tuple(analyses), profile=args.profile)

    if args.report == "json-v1":
        write_json_v1(scan_report, out)
        result_count = len(analyses)
    elif args.report == "json":
        rows = [
            (Path(item.file.path), result)
            for item in analyses
            for result in item.results
        ]
        write_json(rows, out)
        result_count = len(rows)
    else:
        write_html(scan_report, out)
        result_count = len(analyses)
    print(f"scanned {len(paths)} files / {result_count} results → {out}")
    return _failure_exit(
        (analysis.overall_score for analysis in analyses), args.fail_on
    )


def cmd_list_modules(_: argparse.Namespace) -> int:
    registry = _build_registry()
    print("carriers:")
    for carrier in registry.all_carriers():
        flags = []
        if carrier.requires_explicit:
            flags.append("explicit")
        if carrier.experimental:
            flags.append("experimental")
        unavailable_reason = getattr(carrier, "unavailable_reason", None)
        if unavailable_reason:
            flags.append(f"unavailable: {unavailable_reason}")
        flag_text = f"  [{' '.join(flags)}]" if flags else ""
        print(f"  {carrier.identifier}  {carrier.extensions}{flag_text}")
    print("analyzers:")
    for analyzer in registry.all_analyzers():
        print(f"  {analyzer.name}")
    if registry.load_errors():
        print("unavailable:")
        for error in registry.load_errors():
            print(f"  {error.plugin}: {error.error}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from api.app import run_server
    except ImportError as exc:
        raise RuntimeError(
            "REST API dependencies are missing; install with 'pip install .[api]'"
        ) from exc
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        retention_days=args.retention_days,
    )
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    from steganography.benchmarking.corpus import generate_corpus

    manifest = generate_corpus(
        Path(args.out),
        seed=args.seed,
        force=args.force,
        include_dct=not args.exclude_dct,
        methods=set(args.method) if args.method else None,
    )
    print(
        f"generated {manifest['sample_count']} samples → {args.out} "
        f"(digest {manifest['corpus_digest']})"
    )
    for skipped in manifest["skipped"]:
        print(
            f"warning: skipped {skipped['recipe']}: {skipped['reason']}",
            file=sys.stderr,
        )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from steganography.benchmarking.runner import (
        run_benchmark,
        write_benchmark_html,
        write_benchmark_json,
    )

    report = run_benchmark(
        Path(args.corpus),
        profiles=tuple(args.profile or ("sensitive", "balanced", "strict")),
        threshold=args.threshold,
        jobs=args.jobs,
        min_recall=args.min_recall,
        max_false_positive_rate=args.max_fpr,
        baseline=Path(args.baseline) if args.baseline else None,
        max_recall_drop=args.max_recall_drop,
        max_fpr_increase=args.max_fpr_increase,
    )
    write_benchmark_json(report, Path(args.out))
    if args.html:
        write_benchmark_html(report, Path(args.html))
    for profile, values in report["profiles"].items():
        metrics = values["overall"]
        print(
            f"{profile}: precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
            f"fpr={metrics['false_positive_rate']:.3f}"
        )
    status = "passed" if report["gates"]["passed"] else "failed"
    print(f"benchmark gates {status} → {args.out}")
    for failure in report["gates"]["failures"]:
        print(f"  - {failure}", file=sys.stderr)
    return 0 if report["gates"]["passed"] else 1


def _failure_exit(scores: Iterable[int], fail_on: str | None) -> int:
    if fail_on is None:
        return 0
    threshold = 70 if fail_on == "high" else 30
    return 1 if any(score >= threshold for score in scores) else 0


def _add_password_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--password")
    group.add_argument("--password-file")
    group.add_argument("--password-stdin", action="store_true")


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=("sensitive", "balanced", "strict"),
        default="sensitive",
    )
    parser.add_argument("--ai", action="store_true", help="enable signal-only NIM triage")
    parser.add_argument(
        "--allow-ai-file-upload",
        action="store_true",
        help="explicitly allow the analyzed image to be sent to the AI provider",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=0,
        metavar="MIB",
        help="skip/fail files larger than this many MiB (0 = unlimited)",
    )
    parser.add_argument("--fail-on", choices=("medium", "high"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steganography")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--quiet", action="store_true", help="suppress banner")
    commands = parser.add_subparsers(dest="cmd", required=True)

    embed = commands.add_parser("embed")
    embed.add_argument("--in", dest="input", required=True)
    embed.add_argument("--carrier", required=True)
    embed.add_argument("--out", required=True)
    embed.add_argument("--method")
    embed.add_argument("--steg-key")
    embed.add_argument("--channels", default="rgb")
    embed.add_argument("--no-clobber", action="store_true")
    _add_password_options(embed)
    embed.set_defaults(fn=cmd_embed)

    extract = commands.add_parser("extract")
    extract.add_argument("--in", dest="input", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--method")
    extract.add_argument("--steg-key")
    extract.add_argument("--no-clobber", action="store_true")
    _add_password_options(extract)
    extract.set_defaults(fn=cmd_extract)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--in", dest="input", required=True)
    analyze.add_argument("--json", action="store_true", help="legacy JSON output")
    analyze.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json-v1"),
        default="text",
    )
    _add_analysis_options(analyze)
    analyze.set_defaults(fn=cmd_analyze)

    scan = commands.add_parser("scan")
    scan.add_argument("--dir", required=True)
    scan.add_argument("--report", choices=("json", "html", "json-v1"), default="html")
    scan.add_argument("--out", required=True)
    scan.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    scan.add_argument("--follow-symlinks", action="store_true")
    _add_analysis_options(scan)
    scan.set_defaults(fn=cmd_scan)

    list_modules = commands.add_parser("list-modules")
    list_modules.set_defaults(fn=cmd_list_modules)

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=2)
    serve.add_argument("--retention-days", type=int, default=30)
    serve.set_defaults(fn=cmd_serve)

    corpus = commands.add_parser("corpus", help="generate a labeled benchmark corpus")
    corpus.add_argument("--out", required=True)
    corpus.add_argument("--seed", type=int, default=20260813)
    corpus.add_argument("--force", action="store_true")
    corpus.add_argument("--exclude-dct", action="store_true")
    corpus.add_argument(
        "--method",
        action="append",
        help="include only this carrier method or recipe id (repeatable)",
    )
    corpus.set_defaults(fn=cmd_corpus)

    benchmark = commands.add_parser(
        "benchmark", help="measure analysis quality against a labeled corpus"
    )
    benchmark.add_argument("--corpus", required=True)
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--html")
    benchmark.add_argument(
        "--profile",
        action="append",
        choices=("sensitive", "balanced", "strict"),
        default=None,
    )
    benchmark.add_argument("--threshold", type=int, default=70)
    benchmark.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    benchmark.add_argument("--min-recall", type=float, default=0.90)
    benchmark.add_argument("--max-fpr", type=float, default=0.10)
    benchmark.add_argument("--baseline")
    benchmark.add_argument("--max-recall-drop", type=float, default=0.02)
    benchmark.add_argument("--max-fpr-increase", type=float, default=0.02)
    benchmark.set_defaults(fn=cmd_benchmark)
    return parser


_MACHINE_READABLE_CMDS = frozenset(
    {"list-modules", "extract", "serve", "corpus", "benchmark"}
)


def _should_show_banner(args: argparse.Namespace) -> bool:
    if args.quiet or args.cmd in _MACHINE_READABLE_CMDS:
        return False
    if args.cmd == "analyze":
        return not (args.json or args.output_format == "json-v1")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _should_show_banner(args):
        print_gradient_banner()
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
