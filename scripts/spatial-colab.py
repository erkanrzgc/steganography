"""Thin explicit adapter; all computation belongs to core services."""

import argparse
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import colab_artifacts as artifacts  # noqa: E402
from core import spatial_colab as service  # noqa: E402
from core import spatial_colab_bundle as bundle  # noqa: E402
from core import spatial_colab_data as data  # noqa: E402
from core import spatial_colab_state as state  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze",
            "bundle",
            "verify",
            "profile",
            "_worker",
            "validate-data",
            "acquire",
            "prepare",
            "state-probe",
        ),
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--originals", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--prior-ledger", type=Path)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    file_limit = artifacts.MIB if args.phase in {"profile", "_worker"} else state.MAX_BYTES
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    try:
        phase = args.phase
        if phase in {"freeze", "verify", "profile", "_worker"} and args.protocol is None:
            raise ValueError("protocol required")
        if phase not in {"freeze", "verify", "validate-data"} and args.out is None:
            raise ValueError("output required")
        if phase == "freeze":
            service.freeze(ROOT, args.protocol)
            result = {"status": "ready"}
        elif phase == "verify":
            service.verify(ROOT, args.protocol)
            result = {"status": "ready"}
        elif phase == "bundle":
            result = bundle.build(ROOT, args.out)
        elif phase == "profile":
            result = service.supervise(ROOT, args.protocol, args.out)
        elif phase == "_worker":
            try:
                result = service.worker(ROOT, args.protocol, args.out)
            except (ValueError, RuntimeError, ImportError):
                result = {"status": "unavailable", "reason": "gpu_oom_numerics_or_dependency"}
            artifacts.put(args.out / "worker.json", artifacts.encode(result))
        elif phase == "state-probe":
            result = state.generated_probe(args.out)
        else:
            if args.manifest is None:
                raise ValueError("manifest required")
            value = artifacts.document(args.manifest)
            data.manifest(value)
            if phase == "validate-data":
                result = {"status": "ready", "media_opened": False}
            elif phase == "acquire":
                if args.prior_ledger is None:
                    raise ValueError("prior ledger required")
                ledger_raw = artifacts.read(args.prior_ledger, artifacts.MIB)
                ledger = artifacts.document(args.prior_ledger)
                result = data.acquire(
                    value,
                    args.out,
                    allow_download=args.allow_download,
                    prior_ledger={
                        "sha256": artifacts.digest(ledger_raw),
                        "historical_bytes": ledger["historical_bytes"],
                    },
                )
            else:
                if args.originals is None:
                    raise ValueError("originals required")
                result = data.prepare(value, args.originals, args.out)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError):
        print("unavailable: protocol, artifact or environment gate failed")
        return 2
    print(artifacts.encode(result).decode(), end="")
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
