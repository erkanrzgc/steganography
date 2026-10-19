"""Explicit read-only WAV diagnosis; no fitting, downloads or deployment."""

import argparse
import json
import resource
from pathlib import Path

from core import nonjpeg_comparison as documents
from core import wav_diagnostics as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 301))
    resource.setrlimit(resource.RLIMIT_FSIZE, (service.OUTPUT_BYTES, service.OUTPUT_BYTES))
    config, _ = documents.read(args.config)
    report = service.execute(config, args.out)
    print(json.dumps({"status": report["status"], "reason": report.get("reason")}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
