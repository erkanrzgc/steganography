"""Hardened full-image development recovery matrix; no blind/accuracy qualification."""

import hashlib
import json
import tempfile
from pathlib import Path

from core.ctf_evaluation import evaluate_suite
from steganography.benchmarking.ctf_matrix import generate_matrix


def main() -> None:
    reports = {}
    with tempfile.TemporaryDirectory(prefix="ctf-matrix-") as temporary:
        root = Path(temporary)
        for name, external in (("native", False), ("external", True)):
            corpus = root / name
            generate_matrix(corpus, external=external)
            manifest = corpus / "manifest.json"
            checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
            report = evaluate_suite(manifest, checksum, root / (name + "-jobs"),
                                    hardware="full-image-reference", service=None)
            reports[name] = report
            if report["completed_exact_rate"] != 1:
                print(json.dumps(reports))
                raise RuntimeError(f"{name} development fixture recovery failed")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
