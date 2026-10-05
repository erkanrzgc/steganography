"""steganography CLI: embed, extract, analyze, scan, serve and list-modules."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

from core.automation import iter_ndjson
from core.ctf import CTFLimits, CTFService
from core.pipeline import AnalysisPipeline
from core.result import AnalysisResult, ScanReport
from core.service import AnalysisService, StegoService
from core.vault import VaultService
from core.version import __version__
from core.workspace import Workspace, create_workspace
from registry import Registry
from report.report import write_html, write_json, write_json_v1
from report.v2 import html_v2, json_v2, sarif_v2, write_evidence_bundle
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
        payload_version=args.payload_version,
        compress=args.compress,
        ecc_symbols=args.ecc_symbols,
    )
    print(f"embedded {result.bytes_written} bytes via {result.carrier} → {result.out_path}")
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
    print(f"extracted {size} bytes via {carrier} (payload v{version}) → {args.out}")
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
    machine_output = args.json or args.output_format in {"json-v1", "json-v2"}
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
    elif args.output_format == "json-v2":
        report = AnalysisPipeline(
            AnalysisService(registry, profile=args.profile, max_file_size=maximum)
        ).analyze(Path(args.input))
        print(json_v2(report.to_dict()))
    else:
        print(f"  overall [{analysis.overall_score:3d}] {analysis.severity} ({analysis.profile})")
        print(
            f"  sha256 {analysis.file.sha256}  "
            f"type={analysis.file.detected_type} size={analysis.file.size}"
        )
        for result in analysis.results:
            suffix = f" error={result.error}" if result.error else ""
            print(f"  [{result.suspicion:3d}] {result.analyzer} [{result.status}]{suffix}")
            for signal in result.signals:
                print(
                    f"        - {signal.name}({signal.score}, {signal.evidence}): {signal.detail}"
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
        rows = [(Path(item.file.path), result) for item in analyses for result in item.results]
        write_json(rows, out)
        result_count = len(rows)
    elif args.report in {"json-v2", "sarif", "ndjson"}:
        pipeline = AnalysisPipeline(service)
        v2_files = [pipeline.analyze(path).to_dict() for path in paths]
        v2_report = {
            "schema_version": "2.0",
            "profile": args.profile,
            "files": v2_files,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.report == "ndjson":
            out.write_bytes(b"".join(iter_ndjson(v2_files)))
        else:
            content = (
                json.dumps(sarif_v2(v2_report), indent=2)
                if args.report == "sarif"
                else json_v2(v2_report)
            )
            out.write_text(content, encoding="utf-8")
        result_count = len(v2_files)
    else:
        write_html(scan_report, out)
        result_count = len(analyses)
    print(f"scanned {len(paths)} files / {result_count} results → {out}")
    return _failure_exit((analysis.overall_score for analysis in analyses), args.fail_on)


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
    from steganography.benchmarking.corpus import EXTENDED_RECIPES, generate_corpus

    recipes = EXTENDED_RECIPES if getattr(args, "extended", False) else None
    manifest = generate_corpus(
        Path(args.out),
        seed=args.seed,
        force=args.force,
        include_dct=not args.exclude_dct,
        methods=set(args.method) if args.method else None,
        recipes=recipes,
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


def _parse_size(value: str) -> int:
    normalized = value.strip().lower().replace(" ", "")
    suffixes = {
        "kib": 1024,
        "kb": 1000,
        "mib": 1024**2,
        "mb": 1000**2,
        "gib": 1024**3,
        "gb": 1000**3,
        "b": 1,
    }
    for suffix in sorted(suffixes, key=len, reverse=True):
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            break
    else:
        number = normalized
        suffix = "b"
    try:
        result = int(float(number) * suffixes[suffix])
    except (ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value}") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("byte size must be positive")
    return result


def cmd_ctf(args: argparse.Namespace) -> int:
    password = _password_from_args(args)
    limits = CTFLimits(
        max_depth=args.max_depth,
        max_artifacts=args.max_artifacts,
        max_bytes=args.max_bytes,
        tool_timeout=min(30.0, args.timeout),
        job_timeout=args.timeout,
    )
    report = CTFService().solve(
        Path(args.input),
        Path(args.out),
        mode=args.mode,
        wordlist=Path(args.wordlist) if args.wordlist else None,
        password=password,
        limits=limits,
    )
    value = report.to_dict()
    output_dir = Path(args.out)
    if args.report == "bundle":
        report_path = output_dir / "evidence-bundle.zip"
        write_evidence_bundle(
            value,
            [(item.name, item.path) for item in report.artifacts if item.path is not None],
            report_path,
        )
    else:
        report_path = output_dir / f"report.{args.report}"
        if args.report == "html":
            content = html_v2(value)
        elif args.report == "sarif":
            content = json.dumps(sarif_v2(value), indent=2)
        else:
            content = json_v2(value)
        with report_path.open("x", encoding="utf-8") as output:
            output.write(content)
    print(f"CTF {report.status}: {report.verdict} → {report_path}")
    return 0 if report.status == "completed" else 1


def _workspace(args: argparse.Namespace) -> Workspace:
    return create_workspace(args.state_dir)


def _unlock_for_command(vault: VaultService, args: argparse.Namespace) -> None:
    password = _password_from_args(args)
    if not password:
        raise ValueError("vault password is required")
    if vault.initialized:
        vault.unlock(password)
    else:
        vault.initialize(password)


def cmd_case_create(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    value = workspace.cases.create_case(
        args.name,
        description=args.description,
        retention_days=args.retention_days,
    )
    print(json.dumps(value, indent=2))
    return 0


def cmd_case_add(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    try:
        _unlock_for_command(workspace.vault, args)
        value = workspace.cases.add_evidence(args.case_id, Path(args.file))
    finally:
        workspace.lock()
    print(json.dumps(value, indent=2))
    return 0


def cmd_case_scan(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    try:
        _unlock_for_command(workspace.vault, args)
        scan = workspace.scans.create(args.case_id, profile=args.profile)
        result = workspace.scans.run(scan["id"])
    finally:
        workspace.lock()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


def cmd_case_export(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    scan = workspace.scans.get(args.scan_id)
    if scan["result"] is None:
        raise ValueError("scan report is not ready")
    if args.format == "html":
        content = html_v2(scan["result"])
    elif args.format == "sarif":
        content = json.dumps(sarif_v2(scan["result"]), indent=2)
    else:
        content = json_v2(scan["result"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"exported {args.format} report → {out}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    models = _workspace(args).models
    if args.models_cmd == "list":
        print(json.dumps({"items": models.list(), "runtime": models.runtime_status()}, indent=2))
        return 0
    if args.models_cmd == "catalog":
        print(
            json.dumps(
                models.catalog(Path(args.catalog) if args.catalog else None),
                indent=2,
            )
        )
        return 0
    if args.models_cmd == "install":
        if args.manifest:
            if not args.public_key:
                raise ValueError("--public-key is required with --manifest")
            value = models.install(Path(args.manifest), public_key=args.public_key)
        elif args.target:
            value = models.install_catalog(
                args.target,
                accept_license=args.accept_license,
                catalog_path=Path(args.catalog) if args.catalog else None,
            )
        else:
            raise ValueError("provide MODEL@VERSION or --manifest")
        print(json.dumps(value, indent=2))
        return 0
    valid = models.verify_installed(args.model_id, args.model_version)
    print("valid" if valid else "invalid")
    return 0 if valid else 1


def cmd_research_import(args: argparse.Namespace) -> int:
    from steganography.research import import_dataset

    manifest = import_dataset(
        Path(args.source),
        Path(args.out),
        seed=args.seed,
        license_name=args.license,
        source_url=args.source_url,
    )
    print(f"imported {manifest['sample_count']} samples → {args.out}")
    return 0


def cmd_research_partition(args: argparse.Namespace) -> int:
    from steganography.research import partition_dataset

    manifest = partition_dataset(
        Path(args.manifest),
        Path(args.out),
        test_sources=args.test_source,
        reserved_manifests=[Path(path) for path in args.reserved_manifest],
        source=Path(args.source) if args.source else None,
        seed=args.seed,
    )
    print(json.dumps(manifest["partition"], indent=2))
    return 0


def cmd_research_benchmark(args: argparse.Namespace) -> int:
    from steganography.research import benchmark_predictions

    report = benchmark_predictions(
        Path(args.manifest),
        Path(args.predictions),
        Path(args.out),
        source=Path(args.source) if args.source else None,
        threshold=args.threshold,
        min_samples=args.min_samples,
        min_source_groups=args.min_source_groups,
        min_roc_auc=args.min_roc_auc,
        min_balanced_accuracy=args.min_balanced_accuracy,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
        max_ece=args.max_ece,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


def cmd_research_features(args: argparse.Namespace) -> int:
    from steganography.research_features import extract_features

    result = extract_features(
        Path(args.manifest),
        Path(args.out),
        split=args.split,
        source=Path(args.source) if args.source else None,
        feature_version=args.feature_version,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_train(args: argparse.Namespace) -> int:
    from steganography.research import train_model

    result = train_model(Path(args.config), Path(args.out))
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_calibrate(args: argparse.Namespace) -> int:
    from steganography.research import calibrate_predictions

    result = calibrate_predictions(
        Path(args.manifest),
        Path(args.predictions),
        Path(args.out),
        source=Path(args.source) if args.source else None,
        split=args.split,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_export(args: argparse.Namespace) -> int:
    from steganography.research import export_onnx

    result = export_onnx(Path(args.checkpoint), Path(args.out))
    print(json.dumps(result, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    audit_valid, audit_events, audit_error = workspace.database.verify_audit()
    tools = {name: shutil.which(name) for name in ("zsteg", "stegseek", "exiftool")}
    value = {
        "version": __version__,
        "state_dir": str(workspace.state_dir),
        "vault": {
            "initialized": workspace.vault.initialized,
            "unlocked": workspace.vault.unlocked,
        },
        "audit": {"valid": audit_valid, "events": audit_events, "error": audit_error},
        "external_tools": {
            name: {"status": "available" if path else "unavailable", "path": path}
            for name, path in tools.items()
        },
        "models": {
            "installed": len(workspace.models.list()),
            "runtime": workspace.models.runtime_status(),
        },
        "network_default": "disabled",
    }
    print(json.dumps(value, indent=2))
    return 0 if audit_valid else 1


def cmd_ui(args: argparse.Namespace) -> int:
    state_dir = _workspace(args).state_dir
    os.environ["STEGANO_STATE_DIR"] = str(state_dir)
    token_path = state_dir / "v2-api.key"
    from api.v2 import _local_api_key

    _local_api_key(token_path)
    print(f"local UI/API: http://{args.host}:{args.port}")
    print(f"v2 API token file: {token_path}", file=sys.stderr)
    return cmd_serve(args)


def cmd_tui(args: argparse.Namespace) -> int:
    """Launch the direct, local terminal workspace."""
    from ui.tui import run_tui

    run_tui(args.state_dir)
    return 0


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
    commands = parser.add_subparsers(dest="cmd")

    embed = commands.add_parser("embed")
    embed.add_argument("--in", dest="input", required=True)
    embed.add_argument("--carrier", required=True)
    embed.add_argument("--out", required=True)
    embed.add_argument("--method")
    embed.add_argument("--steg-key")
    embed.add_argument("--channels", default="rgb")
    embed.add_argument("--payload-version", type=int, choices=(2, 3), default=2)
    embed.add_argument("--compress", action="store_true")
    embed.add_argument("--ecc-symbols", type=int, choices=range(0, 256), default=0)
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
        choices=("text", "json-v1", "json-v2"),
        default="text",
    )
    _add_analysis_options(analyze)
    analyze.set_defaults(fn=cmd_analyze)

    scan = commands.add_parser("scan")
    scan.add_argument("--dir", required=True)
    scan.add_argument(
        "--report",
        choices=("json", "html", "json-v1", "json-v2", "ndjson", "sarif"),
        default="html",
    )
    scan.add_argument("--out", required=True)
    scan.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    scan.add_argument("--follow-symlinks", action="store_true")
    _add_analysis_options(scan)
    scan.set_defaults(fn=cmd_scan)

    ctf = commands.add_parser("ctf", help="run the bounded CTF recovery playbook")
    ctf.add_argument("input")
    ctf.add_argument("--out", required=True)
    ctf.add_argument("--mode", choices=("quick", "balanced", "deep"), default="balanced")
    ctf.add_argument("--wordlist")
    ctf.add_argument("--max-depth", type=int, default=3)
    ctf.add_argument("--max-artifacts", type=int, default=256)
    ctf.add_argument("--max-bytes", type=_parse_size, default=1024**3, metavar="SIZE")
    ctf.add_argument("--timeout", type=float, default=180.0, metavar="SECONDS")
    ctf.add_argument("--report", choices=("json", "html", "sarif", "bundle"), default="json")
    _add_password_options(ctf)
    ctf.set_defaults(fn=cmd_ctf)

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
    corpus.add_argument(
        "--extended",
        action="store_true",
        help="include extended realistic gradient/spatial recipes",
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

    case = commands.add_parser("case", help="manage local DFIR cases")
    case.add_argument("--state-dir")
    case_commands = case.add_subparsers(dest="case_cmd", required=True)
    case_create = case_commands.add_parser("create")
    case_create.add_argument("--name", required=True)
    case_create.add_argument("--description", default="")
    case_create.add_argument("--retention-days", type=int)
    case_create.set_defaults(fn=cmd_case_create)
    case_add = case_commands.add_parser("add")
    case_add.add_argument("case_id")
    case_add.add_argument("--file", required=True)
    _add_password_options(case_add)
    case_add.set_defaults(fn=cmd_case_add)
    case_scan = case_commands.add_parser("scan")
    case_scan.add_argument("case_id")
    case_scan.add_argument(
        "--profile", choices=("sensitive", "balanced", "strict"), default="sensitive"
    )
    _add_password_options(case_scan)
    case_scan.set_defaults(fn=cmd_case_scan)
    case_export = case_commands.add_parser("export")
    case_export.add_argument("scan_id")
    case_export.add_argument("--format", choices=("json", "html", "sarif"), default="html")
    case_export.add_argument("--out", required=True)
    case_export.set_defaults(fn=cmd_case_export)

    models = commands.add_parser("models", help="manage signed local ONNX models")
    models.add_argument("--state-dir")
    model_commands = models.add_subparsers(dest="models_cmd", required=True)
    model_list = model_commands.add_parser("list")
    model_list.set_defaults(fn=cmd_models)
    model_catalog = model_commands.add_parser("catalog")
    model_catalog.add_argument("--catalog")
    model_catalog.set_defaults(fn=cmd_models)
    model_install = model_commands.add_parser("install")
    model_install.add_argument("target", nargs="?", metavar="MODEL@VERSION")
    model_install.add_argument("--manifest")
    model_install.add_argument("--public-key")
    model_install.add_argument("--catalog")
    model_install.add_argument("--accept-license", action="store_true")
    model_install.set_defaults(fn=cmd_models)
    model_verify = model_commands.add_parser("verify")
    model_verify.add_argument("model_id")
    model_verify.add_argument("model_version")
    model_verify.set_defaults(fn=cmd_models)

    research = commands.add_parser("research", help="reproducible research workflows")
    research_commands = research.add_subparsers(dest="research_cmd", required=True)
    research_import = research_commands.add_parser("import")
    research_import.add_argument("--source", required=True)
    research_import.add_argument("--out", required=True)
    research_import.add_argument("--seed", type=int, default=20260813)
    research_import.add_argument("--license", default="user-supplied")
    research_import.add_argument("--source-url")
    research_import.set_defaults(fn=cmd_research_import)
    research_partition = research_commands.add_parser("partition")
    research_partition.add_argument("--manifest", required=True)
    research_partition.add_argument("--out", required=True)
    research_partition.add_argument("--source")
    research_partition.add_argument("--seed", type=int, default=20260918)
    research_partition.add_argument("--test-source", action="append", required=True)
    research_partition.add_argument("--reserved-manifest", action="append", required=True)
    research_partition.set_defaults(fn=cmd_research_partition)
    research_benchmark = research_commands.add_parser("benchmark", aliases=["benchmark-suite"])
    research_benchmark.add_argument("--manifest", required=True)
    research_benchmark.add_argument("--predictions", required=True)
    research_benchmark.add_argument("--out", required=True)
    research_benchmark.add_argument("--source")
    research_benchmark.add_argument("--threshold", type=float, default=0.9)
    research_benchmark.add_argument("--min-samples", type=int, default=2_000)
    research_benchmark.add_argument("--min-source-groups", type=int, default=2)
    research_benchmark.add_argument("--min-roc-auc", type=float, default=0.90)
    research_benchmark.add_argument("--min-balanced-accuracy", type=float, default=0.85)
    research_benchmark.add_argument("--min-recall", type=float, default=0.80)
    research_benchmark.add_argument("--max-fpr", type=float, default=0.03)
    research_benchmark.add_argument("--max-ece", type=float, default=0.05)
    research_benchmark.add_argument("--bootstrap-samples", type=int, default=200)
    research_benchmark.set_defaults(fn=cmd_research_benchmark)
    research_train = research_commands.add_parser("train")
    research_features = research_commands.add_parser("features")
    research_features.add_argument("--manifest", required=True)
    research_features.add_argument("--out", required=True)
    research_features.add_argument("--source")
    research_features.add_argument("--split", choices=("train", "validation"), default="train")
    research_features.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    research_features.add_argument(
        "--feature-version",
        choices=("spatial-summary-v1", "jpeg-dct-summary-v1", "spatial-cooccurrence-v1"),
        default="spatial-summary-v1",
    )
    research_features.set_defaults(fn=cmd_research_features)
    research_train.add_argument("--config", required=True)
    research_train.add_argument("--out", required=True)
    research_train.set_defaults(fn=cmd_research_train)
    research_calibrate = research_commands.add_parser("calibrate")
    research_calibrate.add_argument("--manifest", required=True)
    research_calibrate.add_argument("--predictions", required=True)
    research_calibrate.add_argument("--out", required=True)
    research_calibrate.add_argument("--source")
    research_calibrate.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    research_calibrate.set_defaults(fn=cmd_research_calibrate)
    research_export = research_commands.add_parser("export")
    research_export.add_argument("--checkpoint", required=True)
    research_export.add_argument("--out", required=True)
    research_export.set_defaults(fn=cmd_research_export)
    research_export_compat = research_commands.add_parser("export-onnx")
    research_export_compat.add_argument("--checkpoint", required=True)
    research_export_compat.add_argument("--out", required=True)
    research_export_compat.set_defaults(fn=cmd_research_export)

    doctor = commands.add_parser("doctor", help="verify local platform dependencies")
    doctor.add_argument("--state-dir")
    doctor.set_defaults(fn=cmd_doctor)

    ui = commands.add_parser("ui", help="start the local web panel and API")
    ui.add_argument("--state-dir")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--workers", type=int, default=2)
    ui.add_argument("--retention-days", type=int, default=30)
    ui.set_defaults(fn=cmd_ui)

    tui = commands.add_parser("tui", help="open the guided terminal workbench")
    tui.add_argument("--state-dir")
    tui.set_defaults(fn=cmd_tui)
    return parser


_MACHINE_READABLE_CMDS = frozenset(
    {
        "list-modules",
        "extract",
        "serve",
        "corpus",
        "benchmark",
        "ctf",
        "case",
        "models",
        "research",
        "doctor",
        "ui",
        "tui",
    }
)


def _should_show_banner(args: argparse.Namespace) -> bool:
    if args.quiet or args.cmd in _MACHINE_READABLE_CMDS:
        return False
    if args.cmd == "analyze":
        return not (args.json or args.output_format in {"json-v1", "json-v2"})
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        source = sys.argv[1:] if argv is None else argv
        interactive = not source and sys.stdin.isatty() and sys.stdout.isatty()
        if interactive:
            from ui.tui import run_tui

            run_tui()
        else:
            parser.print_help()
        return 0
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
