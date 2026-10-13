"""Continue an explicitly bound cancelled own-GPU JPEG study, never an old model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core import jpeg_study_continue as service
from core import srnet_cuda
from core.jpeg_study_process import WorkerFailure
from core.srnet_stream import regular_open
from steganography import research_jpeg_study as study


def sources():
    names = ("core/jpeg_study_continue.py", "steganography/research_jpeg_continue.py",
             "docs/JPEG_OWNED_CONTINUATION.md")
    return {name: service.digest(study.ROOT / name) for name in names}


def launch(payload, out, arm, seed, rows, stop, timeout, cancelled):
    return json.loads(study.child([
        sys.executable, "-m", "steganography.research_jpeg_study", "--worker", "fit",
        "--out", str(out), "--arm", arm, "--seed", str(seed), "--rows", str(rows),
        "--stop-rows", str(stop),
    ], timeout=timeout, cancelled=cancelled, payload=json.dumps(payload).encode()))


def probe(rows, timeout, cancelled):
    try:
        value = json.loads(study.child([
            sys.executable, "-m", "steganography.research_batch_probe", "--worker",
            "--rows", str(rows),
        ], timeout=timeout, cancelled=cancelled))
    except WorkerFailure as exc:
        value = exc.unavailable()
    except TimeoutError:
        value = {"status": "unavailable", "reason": "worker_timeout"}
    if value.get("status") == "completed" and (
        value.get("rows") != rows or value.get("source_sha256") != study.capacity_sources()
    ):
        raise ValueError("continuation capacity source mismatch")
    return {**value, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with regular_open(args.plan) as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("continuation plan exceeds bounds")
    execution = srnet_cuda.inspect()
    plan = json.loads(raw)
    report = service.execute(plan, args.out, sources=study.sources(),
                             torch_version=execution["torch_version"], launch=launch,
                             probe=probe, current_sources=study.sources,
                             controller_sources=sources(), current_controller_sources=sources)
    print(json.dumps({"status": report["status"], "reason": report["reason"]}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
