"""Explicit generated learning control; no dataset access or detector change."""

import argparse
import hashlib
from pathlib import Path

from core import srnet_model, srnet_positive
from steganography.research_jpeg import write_json


def run(out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("positive control output must be fresh and non-symlink")
    root = Path(__file__).resolve().parents[1]
    protocol = root / "docs/SRNET_POSITIVE_CONTROL_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != srnet_positive.PROTOCOL_SHA:
        raise ValueError("frozen positive control protocol changed")
    sources = (
        "core/srnet_positive.py",
        "core/srnet_training.py",
        "core/srnet.py",
        "core/srnet_multibatch.py",
        "core/srnet_sampling.py",
        "steganography/research_srnet_positive.py",
    )
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    model, report = srnet_positive.learn()
    out.mkdir(parents=True, exist_ok=False)
    report["model_sha256"] = srnet_model.save_model(model, out / "model.npz")
    report["execution_source_sha256"] = hashes
    write_json(out / "control.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (3660, 3661))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        report = run(args.out)
    except Exception:
        print("failed/unavailable: no complete usable generated-control result")
        return 2
    print("completed: generated learning control; detector qualification unavailable")
    return 0 if report["sanity_objectives_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
