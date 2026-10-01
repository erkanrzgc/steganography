"""Independent embedding/recovery pilot, deliberately separate from support claims."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import platform
import random
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.ctf import CTFLimits, CTFService
from core.service import AnalysisService
from core.version import __version__
from steganography.benchmarking.metrics import classification_metrics

PASSWORD = "public-pilot-fixture"  # noqa: S105 - benchmark credential, never a user secret
SEED = 20260918


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def checked_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe sample path")
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("sample is missing or escapes its root")
    return path


def embed_external(tool: str, cover: Path, payload: Path, output: Path) -> None:
    executable = shutil.which(tool)
    if executable is None:
        raise RuntimeError(f"unavailable: {tool}")
    if output.exists():
        raise FileExistsError(output)
    if tool == "steghide":
        args = [
            "embed",
            "-cf",
            str(cover),
            "-ef",
            str(payload),
            "-sf",
            str(output),
            "-p",
            PASSWORD,
            "-Z",
        ]
    elif tool == "openstego":
        args = [
            "embed",
            "-a",
            "randomlsb",
            "-cf",
            str(cover),
            "-mf",
            str(payload),
            "-sf",
            str(output),
            "-C",
            "-E",
        ]
    else:
        raise ValueError("unsupported generator")
    subprocess.run(  # noqa: S603 - explicit trusted generator, fixed arguments
        [executable, *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    if not output.is_file() or output.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("invalid generator output")


def generate(
    covers: Path, out: Path, *, pairs: int = 1000, workers: int = 4, both_methods: bool = False
) -> dict[str, Any]:
    if not 1 <= pairs <= 1000 or not 1 <= workers <= 8:
        raise ValueError("pairs must be 1..1000 and workers 1..8")
    source = json.loads((covers / "source.json").read_text())
    records = source["samples"][:pairs]
    if len(records) < pairs:
        raise ValueError("insufficient independent covers")
    for tool in ("steghide", "openstego"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"unavailable: {tool}")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("samples", "expected", "challenges"):
        (out / name).mkdir()

    def pair(item: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        index, record = item
        original = checked_path(covers, record["path"])
        data = original.read_bytes()
        if digest(data) != record["sha256"]:
            raise ValueError("source checksum mismatch")
        payload = out / "expected" / f"{index:04d}.bin"
        payload_size = (256, 1024, 4096)[index % 3]
        payload.write_bytes(random.Random(SEED + index).randbytes(payload_size))  # noqa: S311
        methods = (("steghide", "bmp"), ("openstego", "png"))
        selected = methods if both_methods else (methods[index % 2],)
        rows: list[dict[str, Any]] = []
        for tool, extension in selected:
            cover = out / "samples" / f"{index:04d}-cover.{extension}"
            stego = out / "samples" / f"{index:04d}-stego.{extension}"
            with Image.open(original) as image:
                transformation = f"{image.mode} ({image.format}) to RGB, no resize"
                image.convert("RGB").save(cover)
                pixels = image.width * image.height
            embed_external(tool, cover, payload, stego)
            rows.extend(
                {
                    "path": p.relative_to(out).as_posix(),
                    "sha256": digest(p.read_bytes()),
                    "label": label,
                    "source_group": source["source_group"],
                    "lineage": record["sha256"],
                    "split": "test",
                    "method": tool,
                    "format": extension,
                    "payload_bytes": payload_size if label == "stego" else 0,
                    "requested_bits_per_pixel": payload_size * 8 / pixels,
                    "transformation": transformation,
                    "camera": None,
                    "device": None,
                }
                for p, label in ((cover, "cover"), (stego, "stego"))
            )
        return rows

    samples = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, values in enumerate(executor.map(pair, enumerate(records))):
            samples.extend(values)
            if index % 50 == 0:
                print(f"generated {index + 1}/{pairs} pairs", flush=True)
    if len({s["lineage"] for s in samples}) != pairs:
        raise ValueError("duplicate cover lineage")
    challenges = make_challenges(out, samples)
    manifest = {
        "schema_version": "pilot-1",
        "seed": SEED,
        "tool_version": __version__,
        "source": source,
        "samples": samples,
        "challenges": challenges,
        "claim": "exploratory single-source baseline; not blind or cross-source",
        "threshold": 70,
        "profile": "balanced",
        "training_performed": False,
        "both_methods_per_cover": both_methods,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def make_challenges(root: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    methods = ("steghide", "openstego", "base64", "gzip", "zip", "base64-zip")
    for index in range(30):
        method = methods[index % len(methods)]
        payload = f"flag{{pilot_{digest(str(SEED + index).encode())[:24]}}}".encode()
        name = digest(f"challenge-{SEED}-{index}".encode())[:16]
        if method in {"steghide", "openstego"}:
            options = [s for s in samples if s["method"] == method and s["label"] == "cover"]
            if not options:
                records.append({"id": name, "method": method, "status": "unavailable"})
                continue
            cover = checked_path(root, options[(index // 6) % len(options)]["path"])
            expected = root / "expected" / f"challenge-{name}.bin"
            expected.write_bytes(payload)
            path = root / "challenges" / f"{name}{cover.suffix}"
            embed_external(method, cover, expected, path)
        else:
            path = root / "challenges" / f"{name}.bin"
            if method == "base64":
                encoded = base64.b64encode(payload)
            elif method == "gzip":
                encoded = gzip.compress(payload, mtime=0)
            else:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(zipfile.ZipInfo("message.txt", (2020, 1, 1, 0, 0, 0)), payload)
                encoded = buffer.getvalue()
                if method == "base64-zip":
                    encoded = base64.b64encode(encoded)
            path.write_bytes(encoded)
        records.append(
            {
                "id": name,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest(path.read_bytes()),
                "expected_sha256": digest(payload),
                "method": method,
                "status": "ready",
                "password_provided": method == "steghide",
            }
        )
    return records


def score_sample(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    root, sample = item
    path = checked_path(Path(root), sample["path"])
    if digest(path.read_bytes()) != sample["sha256"]:
        raise ValueError("sample checksum mismatch")
    started = time.monotonic()
    report = AnalysisService(profile="balanced", ai_provider=None).analyze(path)
    return {
        "sha256": sample["sha256"],
        "label": sample["label"],
        "method": sample["method"],
        "lineage": sample["lineage"],
        "score": report.overall_score,
        "seconds": time.monotonic() - started,
        "coverage": {r.analyzer: r.status for r in report.results},
        "signals": [
            {"analyzer": r.analyzer, "code": s.name, "score": s.score}
            for r in report.results
            for s in r.signals
            if s.score >= 30
        ],
    }


def evaluate(root: Path, out: Path, *, workers: int = 4, ctf: bool = False) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    manifest = json.loads((root / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if ctf:
        rows = []
        for case in manifest["challenges"]:
            if case["status"] != "ready":
                rows.append({"id": case["id"], "status": "unavailable", "exact_recovery": False})
                continue
            path = checked_path(root, case["path"])
            if digest(path.read_bytes()) != case["sha256"]:
                raise ValueError("challenge checksum mismatch")
            report = CTFService().solve(
                path,
                out / case["id"],
                mode="balanced",
                password=PASSWORD if case["password_provided"] else None,
                limits=CTFLimits(
                    tool_timeout=5, job_timeout=60, max_artifacts=64, max_bytes=16 * 1024 * 1024
                ),
            )
            exact = any(a.sha256 == case["expected_sha256"] for a in report.artifacts)
            rows.append(
                {
                    "id": case["id"],
                    "method": case["method"],
                    "status": report.status,
                    "exact_recovery": exact,
                    "seconds": report.duration_ms / 1000,
                    "tools": [{"name": t.tool, "status": t.status} for t in report.tools],
                }
            )
            print(f"CTF {len(rows)}/30 {case['method']}: recovered={exact}", flush=True)
        result = {
            "kind": "ctf",
            "cases": rows,
            "exact_recovery_rate": sum(r["exact_recovery"] for r in rows) / len(rows),
            "blind": False,
            "password_policy": "known password supplied for steghide",
        }
        result["completed_exact_recovery_rate"] = sum(
            r["exact_recovery"] and r["status"] == "completed" for r in rows
        ) / len(rows)
        timings = [r["seconds"] for r in rows if "seconds" in r]
        result["latency"] = {
            "median_seconds": float(np.median(timings)) if timings else None,
            "p95_seconds": float(np.percentile(timings, 95)) if timings else None,
        }
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, row in enumerate(
                executor.map(
                    score_sample, ((str(root), s) for s in manifest["samples"]), chunksize=4
                )
            ):
                rows.append(row)
                if index % 100 == 0:
                    print(f"analyzed {index + 1}/{len(manifest['samples'])}", flush=True)
        result = {
            "kind": "detection",
            "threshold": 70,
            "profile": "balanced",
            "samples": rows,
            "metrics": summarize(rows),
            "by_method": {
                m: summarize([r for r in rows if r["method"] == m])
                for m in sorted({r["method"] for r in rows})
            },
        }
    result.update(
        {
            "schema_version": "pilot-1",
            "tool_version": __version__,
            "manifest_sha256": digest((root / "manifest.json").read_bytes()),
            "runner_sha256": digest(Path(__file__).read_bytes()),
            "support_status": "experimental",
            "cross_source_gate": "unavailable",
            "elapsed_seconds": time.monotonic() - started,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "workers": workers,
                "logical_cpus": os.cpu_count(),
            },
        }
    )
    write_json(out / "report.json", result)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [(r["label"] == "stego", r["score"]) for r in rows]
    result = classification_metrics(observations, threshold=70)
    result.pop("recommended_threshold", None)  # Never recommend tuning on test outcomes.
    result["analysis_errors"] = sum("error" in r["coverage"].values() for r in rows)
    result["median_seconds"] = float(np.median([r["seconds"] for r in rows]))
    result["p95_seconds"] = float(np.percentile([r["seconds"] for r in rows], 95))
    # Resample entire cover lineages, keeping the paired observations together.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["lineage"], []).append(row)
    grouped = list(groups.values())
    rng = random.Random(SEED)  # noqa: S311 - reproducible statistical resampling
    draws: dict[str, list[float]] = {
        k: [] for k in ("roc_auc", "balanced_accuracy", "recall", "false_positive_rate")
    }
    for _ in range(200):
        bootstrap = [r for group in rng.choices(grouped, k=len(grouped)) for r in group]
        metrics = classification_metrics(
            [(r["label"] == "stego", r["score"]) for r in bootstrap], threshold=70
        )
        for key, values in draws.items():
            if metrics[key] is not None:
                values.append(metrics[key])
    result["lineage_bootstrap_95_ci"] = {
        key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for key, values in draws.items()
        if values
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "detect", "ctf"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--both-methods", action="store_true")
    args = parser.parse_args()
    if args.action == "generate":
        generate(
            args.source,
            args.out,
            pairs=args.pairs,
            workers=args.workers,
            both_methods=args.both_methods,
        )
    else:
        evaluate(args.source, args.out, workers=args.workers, ctf=args.action == "ctf")


if __name__ == "__main__":
    main()
