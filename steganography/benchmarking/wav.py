"""Bounded, preregistered FSDD LSB replacement baseline using shared analysis."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import random
import re
import shutil
import struct
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from core.service import AnalysisService
from steganography.benchmarking.alaska2 import (
    bounded_read,
    detector_fingerprint,
    digest,
    no_symlinks,
    write_json,
)
from steganography.benchmarking.metrics import classification_metrics

SPEAKERS = {
    "george": "test",
    "jackson": "test",
    "lucas": "train",
    "nicolas": "train",
    "theo": "validation",
    "yweweler": "validation",
}
METHODS = ("sequential", "scattered")
RATES = (5, 20, 40)  # percentage of available samples, rounded down to whole bytes
REQUIRED = ("audio_wav", "filestruct_appended", "file_structure", "signatures")
MAX_FILE = 1024 * 1024
MAX_DOCUMENT = 8 * 1024 * 1024
MAX_OUTPUT = 512 * 1024 * 1024
SEED = 20261005


def pcm_region(data: bytes) -> tuple[int, int]:
    """Validate bounded mono PCM16/8k WAV; retain original RIFF metadata bytes."""
    if len(data) > MAX_FILE or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("invalid bounded WAV")
    if len(data) < 12 or int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError("invalid RIFF extent")
    with wave.open(io.BytesIO(data)) as audio:
        frames = audio.getnframes()
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getcomptype(),
        ) != (1, 2, 8000, "NONE") or not 160 <= frames <= 80000:
            raise ValueError("unexpected bounded PCM contract")
        if len(audio.readframes(frames)) != frames * 2:
            raise ValueError("truncated PCM samples")
    offset = 12
    chunks: list[tuple[bytes, int, int]] = []
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        end = offset + 8 + size + (size & 1)
        if end > len(data) or len(chunks) >= 64:
            raise ValueError("RIFF chunk outside limits")
        chunks.append((data[offset : offset + 4], offset + 8, size))
        offset = end
    regions = [(start, size) for name, start, size in chunks if name == b"data"]
    if offset != len(data) or len(regions) != 1 or regions[0][1] != frames * 2:
        raise ValueError("ambiguous PCM region")
    return regions[0][0], frames


def positions(frames: int, count: int, method: str, seed: str) -> list[int]:
    if method not in METHODS or not 0 < count <= frames:
        raise ValueError("invalid embedding method or length")
    if method == "sequential":
        return list(range(count))
    rng = random.Random(seed)  # noqa: S311 - public reproducible research recipe
    return rng.sample(range(frames), count)


def make_pair(data: bytes, *, method: str, rate: int) -> tuple[bytes, dict[str, Any]]:
    """Independent marker-free research generator and scalar extraction oracle."""
    if rate not in RATES:
        raise ValueError("unsupported payload rate")
    offset, frames = pcm_region(data)
    size = (frames * rate // 100) // 8
    seed = f"{SEED}:{digest(data)}:{method}:{rate}"
    payload = hashlib.shake_256(seed.encode()).digest(size)
    order = positions(frames, size * 8, method, seed)
    samples = np.frombuffer(data, dtype="<i2", count=frames, offset=offset).copy()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    samples[order] = (samples[order] & np.int16(-2)) | bits.astype(np.int16)
    encoded = data[:offset] + samples.astype("<i2").tobytes() + data[offset + frames * 2 :]
    # Different extraction implementation: scalar signed PCM and byte assembly.
    recovered = bytearray(size)
    changed = 0
    for bit_index, sample_index in enumerate(order):
        sample = struct.unpack_from("<h", encoded, offset + 2 * sample_index)[0]
        recovered[bit_index // 8] |= (sample & 1) << (7 - bit_index % 8)
        original = struct.unpack_from("<h", data, offset + 2 * sample_index)[0]
        if abs(sample - original) > 1:
            raise ValueError("sample distortion outside LSB contract")
        changed += sample != original
    if bytes(recovered) != payload or not changed or pcm_region(encoded) != (offset, frames):
        raise ValueError("independent payload verification failed")
    return encoded, {
        "payload_sha256": digest(payload),
        "payload_bytes": size,
        "requested_bits_per_sample": rate / 100,
        "actual_bits_per_sample": size * 8 / frames,
        "changed_samples": changed,
        "extraction_verified": True,
        "verification": "independent scalar oracle; known public ordering; not native recovery",
    }


def prepare(source: Path, out: Path, *, source_sha256: str, count: int = 1000) -> dict[str, Any]:
    if not 1 <= count <= 1000:
        raise ValueError("count must be 1..1000")
    raw = bounded_read(source / "source.json", MAX_DOCUMENT)
    document = json.loads(raw)
    if (
        digest(raw) != source_sha256
        or document.get("schema_version") != "fsdd-acquisition-v1"
        or document.get("license") != "CC-BY-SA-4.0"
    ):
        raise ValueError("source integrity or schema mismatch")
    rows = document.get("samples", [])
    if not 6 <= len(rows) <= 3000:
        raise ValueError("source count outside limits")
    identities, partition = set(), []
    for row in rows:
        name = row.get("path", "")
        match = re.fullmatch(r"[0-9]_([a-z]+)_[0-9]+\.wav", name) if isinstance(name, str) else None
        if (
            not match
            or match[1] not in SPEAKERS
            or row.get("speaker") != match[1]
            or row.get("label") != "cover"
            or row.get("lineage") != row.get("sha256")
        ):
            raise ValueError("invalid source identity")
        data = bounded_read(source / name, MAX_FILE)
        if (
            len(data) != row.get("size")
            or digest(data) != row.get("sha256")
            or digest(data) in identities
        ):
            raise ValueError("source checksum or duplicate identity")
        pcm_region(data)
        identities.add(digest(data))
        partition.append(
            {
                "path": name,
                "sha256": digest(data),
                "lineage": digest(data),
                "speaker": match[1],
                "split": SPEAKERS[match[1]],
            }
        )
    if {r["speaker"] for r in partition} != set(SPEAKERS):
        raise ValueError("missing speaker group")
    selected = sorted((r for r in partition if r["split"] == "test"), key=lambda r: r["path"])[
        :count
    ]
    if len(selected) != count:
        raise ValueError("insufficient reserved covers")
    no_symlinks(out)
    out.mkdir(parents=True, exist_ok=False)
    samples, total = [], 0
    for row in selected:
        data = bounded_read(source / row["path"], MAX_FILE)
        if digest(data) != row["sha256"]:
            raise ValueError("source changed during preparation")
        variants: list[tuple[str, bytes, str | None, int | None, dict[str, Any]]] = [
            ("cover", data, None, None, {})
        ]
        for method in METHODS:
            for rate in RATES:
                encoded, verification = make_pair(data, method=method, rate=rate)
                variants.append((f"{method}-{rate}", encoded, method, rate, verification))
        for folder, encoded, variant_method, variant_rate, verification in variants:
            total += len(encoded)
            if total > MAX_OUTPUT:
                raise ValueError("corpus output budget exceeded")
            target = out / folder / row["path"]
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as stream:
                stream.write(encoded)
            samples.append(
                {
                    **row,
                    "path": target.relative_to(out).as_posix(),
                    "sha256": digest(encoded),
                    "size": len(encoded),
                    "label": "cover" if variant_method is None else "stego",
                    "method": variant_method,
                    "rate_percent": variant_rate,
                    **verification,
                }
            )
    result = {
        "schema_version": "fsdd-lsb-pilot-v1",
        "source_sha256": source_sha256,
        "license": document["license"],
        "source_url": document["source_url"],
        "source_commit": document["source_commit"],
        "seed": SEED,
        "speaker_partition": SPEAKERS,
        "reserved_originals": partition,
        "lineages": count,
        "total_bytes": total,
        "samples": samples,
    }
    write_json(out / "manifest.json", result)
    return result


def score_sample(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    root, sample = item
    row = {
        k: sample[k]
        for k in ("path", "sha256", "lineage", "label", "method", "rate_percent", "speaker")
    }
    row.update(score=None, status="failed", coverage={})
    started = time.monotonic()
    try:
        if not re.fullmatch(
            r"(cover|sequential-(5|20|40)|scattered-(5|20|40))/[0-9]_(george|jackson)_[0-9]+\.wav",
            row["path"],
        ):
            raise ValueError("unsafe evaluation path")
        path = Path(root) / row["path"]
        data = bounded_read(path, MAX_FILE)
        if digest(data) != row["sha256"] or len(data) != sample["size"]:
            raise ValueError("evaluation input integrity failed")
        pcm_region(data)
        report = AnalysisService(
            profile="balanced", ai_provider=None, max_file_size=MAX_FILE
        ).analyze(path)
        if (
            report.file.sha256 != row["sha256"]
            or digest(bounded_read(path, MAX_FILE)) != row["sha256"]
        ):
            raise ValueError("evaluation input changed")
        row.update(
            score=report.overall_score,
            status="completed",
            coverage={r.analyzer: r.status for r in report.results if r.analyzer != "ai_triage"},
        )
    except Exception as exc:
        row["failure"] = type(exc).__name__
    row["seconds"] = time.monotonic() - started
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = sum(
        r["status"] != "completed"
        or "error" in r["coverage"].values()
        or any(r["coverage"].get(n) != "ok" for n in REQUIRED)
        for r in rows
    )
    result: dict[str, Any] = {
        "status": "unavailable",
        "samples": len(rows),
        "invalid_samples": failed,
        "metrics": None,
    }
    if not rows or failed:
        return result

    def metrics(values):
        output = classification_metrics(
            [(r["label"] == "stego", r["score"]) for r in values],
            threshold=70,
            recommend_threshold=False,
        )
        for key in ("recommended_threshold", "average_precision"):
            output.pop(key)
        return output

    result.update(status="completed", metrics=metrics(rows))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["lineage"], []).append(row)
    draws: dict[str, list[float]] = {
        k: [] for k in ("roc_auc", "balanced_accuracy", "recall", "false_positive_rate")
    }
    rng = random.Random(SEED)  # noqa: S311 - paired statistical bootstrap
    grouped = list(groups.values())
    for _ in range(200):
        measured = metrics([r for g in rng.choices(grouped, k=len(grouped)) for r in g])
        for key in draws:
            draws[key].append(measured[key])
    result["lineage_bootstrap_95_ci"] = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()
    }
    return result


def evaluate(
    source: Path,
    out: Path,
    *,
    source_sha256: str,
    protocol: Path,
    count: int = 1000,
    workers: int = 4,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    protocol_hash = digest(bounded_read(protocol, MAX_DOCUMENT))
    started = time.monotonic()
    manifest = prepare(source, out, source_sha256=source_sha256, count=count)
    provenance = {
        "schema_version": "fsdd-lsb-baseline-v1",
        "source_sha256": source_sha256,
        "manifest_sha256": digest(bounded_read(out / "manifest.json", MAX_DOCUMENT)),
        "protocol_sha256": protocol_hash,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "detector_code_sha256": detector_fingerprint(),
        "threshold": 70,
        "profile": "balanced",
        "ai_enabled": False,
        "training_performed": False,
        "threshold_search_performed": False,
        "cross_source_gate": "unavailable",
        "support_status": "experimental",
        "calibrated": False,
        "ece": {"status": "unavailable", "reason": "uncalibrated heuristic scores"},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "workers": workers,
            "tools_available": {
                n: bool(shutil.which(n)) for n in ("exiftool", "stegseek", "zsteg")
            },
        },
    }
    write_json(out / "run.json", provenance)
    rows = []
    with (
        (out / "scores.jsonl").open("x") as stream,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        for row in executor.map(score_sample, ((str(out), r) for r in manifest["samples"])):
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 500 == 0:
                print(f"analyzed {len(rows)}/{len(manifest['samples'])}", flush=True)
    cells = {
        f"{method}-{rate}": summarize(
            [
                r
                for r in rows
                if r["method"] is None or (r["method"] == method and r["rate_percent"] == rate)
            ]
        )
        for method in METHODS
        for rate in RATES
    }
    result = {
        **provenance,
        "samples": len(rows),
        "lineages": count,
        "by_method_rate": cells,
        "status": "completed"
        if all(c["status"] == "completed" for c in cells.values())
        else "unavailable",
        "total_bytes": manifest["total_bytes"],
        "elapsed_seconds": time.monotonic() - started,
        "scores_sha256": digest(bounded_read(out / "scores.jsonl", 32 * 1024 * 1024)),
        "latency": {
            "median_seconds": float(np.median([r["seconds"] for r in rows])),
            "p95_seconds": float(np.percentile([r["seconds"] for r in rows], 95)),
        },
    }
    write_json(out / "report.json", result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            args.source,
            args.out,
            source_sha256=args.source_sha256,
            protocol=args.protocol,
            count=args.count,
            workers=args.workers,
        )
    except Exception as exc:
        parser.exit(2, f"WAV baseline failed ({type(exc).__name__}); partial output retained\n")
    print(f"WAV baseline {report['status']}: {report['samples']} files")
    if report["status"] != "completed":
        parser.exit(1)


if __name__ == "__main__":
    main()
