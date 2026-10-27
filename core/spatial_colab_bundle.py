"""Deterministic portable source overlay and Colab notebook, without publishing."""

from __future__ import annotations

from pathlib import Path

from core import colab_artifacts as artifacts
from core import spatial_colab as service


def build(root: Path, out: Path) -> dict:
    artifacts.directory(out)
    protocol = service.freeze(root, out / "protocol.json")
    files = {}
    for name in service.SOURCES:
        path = root / name
        files[name] = (
            "" if path.stat().st_size == 0 else artifacts.read(path, artifacts.MIB).decode()
        )
    package = {"schema": "spatial-colab-source-package-v1", "protocol": protocol, "files": files}
    checksum = artifacts.put(out / "sources.json", artifacts.encode(package), 4 * artifacts.MIB)
    expected_files = "\n".join(f"    {name!r}," for name in service.SOURCES)
    cells = []

    def markdown(text):
        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
        )

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": text.splitlines(keepends=True),
                "outputs": [],
                "execution_count": None,
            }
        )

    markdown("""# PNG/BMP SRNet — Colab Pro preparation
Select **Runtime → Change runtime type → Runtime Version 2026.07 → GPU**.
The version and GPU are checked by the service; unsupported versions stop.
No real media download, training, optimizer update or Drive mount runs by default.
Use the accompanying `sources.json` (same delivery folder). The notebook checks
its SHA-256 before applying the source overlay to the fixed base Git commit.
This avoids depending on an unpublished branch. Frozen JPEG files are checked,
never overwritten. Keep all receipts; a failed attempt has no automatic retry.

[Versioned runtimes](https://research.google.com/colaboratory/runtime-version-faq.html)
· [Drive I/O guidance](https://research.google.com/colaboratory/faq.html)
""")
    code(f"""import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

from google.colab import files


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_command(arguments, **options):
    # Call sites use literal commands, validated paths and the verified source package.
    return subprocess.run(arguments, **options)  # noqa: S603


BASE_COMMIT = {service.BASE_COMMIT!r}
SOURCE_SHA256 = {checksum!r}
uploaded = files.upload()  # Select only the accompanying sources.json.
require(set(uploaded) == {{"sources.json"}}, "Select sources.json only")
raw = uploaded["sources.json"]
require(len(raw) <= 4 * 1024**2, "Source package byte limit")
require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, "Source package checksum")
package = json.loads(raw)
require(package["schema"] == "spatial-colab-source-package-v1", "Source package schema")
expected = [
{expected_files}
]
require(set(package["files"]) == set(expected), "Unexpected source files")
require(set(package["protocol"]["source_sha256"]) == set(expected), "Source binding fields")
repo = pathlib.Path("/content/steganography-colab")
require(not repo.exists(), "Existing session: preserve receipts; do not overwrite or retry")
git = shutil.which("git")
require(git is not None, "Git unavailable")
run_command([git, "clone", "--no-checkout",
             "https://github.com/erkanrzgc/steganography.git", str(repo)], check=True)
run_command([git, "checkout", "--detach", BASE_COMMIT], cwd=repo, check=True)
actual_commit = run_command([git, "rev-parse", "HEAD"], cwd=repo,
                            text=True, capture_output=True, check=True).stdout
require(actual_commit.strip() == BASE_COMMIT, "Base commit mismatch")
for name in expected:
    data = package["files"][name].encode()
    require(hashlib.sha256(data).hexdigest() == package["protocol"]["source_sha256"][name],
            "Source checksum mismatch")
    target = repo / name
    require(not any(p.is_symlink() for p in (target, *target.parents)), "Source symlink")
    if target.exists():
        require(target.read_bytes() == data, "Frozen base source differs")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
protocol = repo / "colab-protocol.json"
with protocol.open("x") as stream:
    json.dump(package["protocol"], stream)
command = [sys.executable, str(repo / "scripts/spatial-colab.py")]
run_command(command + ["verify", "--protocol", str(protocol)], cwd=repo, check=True)
""")
    markdown("""## Environment / GPU check
Use Colab's preinstalled GPU stack. `requirements-colab-gpu.txt` records the
exact expected versions; the CPU CI lock is not installed. A missing GPU or
version mismatch is unavailable. Software installation time is outside the
measurement; the measurement command has its own 600-second total limit.
""")
    code("""sys.path.insert(0, str(repo))
from core import spatial_colab as profile

versions = profile.environment()
print(versions)
require(versions["gpu"] is not None, "unavailable: GPU missing")
matches = all(versions[k] == v for k, v in profile.ENVIRONMENT.items() if k != "runtime")
require(matches, "unavailable: pinned environment mismatch")
""")
    markdown("""## Data validation (metadata only)
New sources must have per-file Commons attribution, CC0/CC BY license URL,
original SHA-1, byte count and reviewed whole-source ancestry. No reserved
split is opened. Put an approved manifest on the local disk and enable this
cell explicitly. The delivered candidate list is metadata only; native modes,
near duplicates and natural-image suitability remain unverified until admission.
""")
    code("""VALIDATE_DATA = False
if VALIDATE_DATA:
    run_command(command + ["validate-data", "--manifest", "/content/commons-manifest.json"],
                cwd=repo, check=True)
# Acquisition requires a separate explicit command with --allow-download
# and --prior-ledger. No media is acquired by this notebook.
""")
    markdown("""## Capacity measurement — one explicit run
Synthetic 256×256 FP32, gray/RGB × two seeds × batch 2/4/8. One warm-up and
three timed forward/backward passes per cell; no optimizer is constructed.
AMP/TF32 are disabled. OOM, GPU absence or timeout stays unavailable.
Set RUN_MEASUREMENT only when ready; rerunning cannot overwrite an attempt.
""")
    code("""RUN_MEASUREMENT = False
measurement_dir = pathlib.Path("/content/spatial-capacity")
if RUN_MEASUREMENT:
    args = ["profile", "--protocol", str(protocol), "--out", str(measurement_dir)]
    completed = run_command(command + args, cwd=repo, check=False)
    print("ready" if completed.returncode == 0 else "unavailable; preserve receipts, no retry")
""")
    markdown("""## Report / interruption preparation
The receipt contains GPU/software, every measured interval, VRAM/RAM and
output bytes, with a cost table for both models and both seeds. Optimizer,
evaluation and data I/O costs are unmeasured; no full-training budget follows.
Numeric model/AdamW/RNG/order/consumption checkpoints are tested with generated
states only. The state-probe command writes only to the local working disk.
""")
    code("""if (measurement_dir / "measurement.json").exists():
    receipt = json.loads((measurement_dir / "measurement.json").read_text())
    print(receipt["status"])
    for row in receipt["cost_table"]:
        print(row)
RUN_GENERATED_STATE_PROBE = False
if RUN_GENERATED_STATE_PROBE:
    run_command(command + ["state-probe", "--out", "/content/generated-state-probe"],
                cwd=repo, check=True)
""")
    markdown("""## Export verified receipts
Use the explicit export command below. It rechecks all package hashes before
import. Computation stays on `/content`; Drive is optional storage for this
single verified package only. Do not move loose training files to Drive.
No download, actual fit, role assignment or historical transfer/JPEG-budget
reset is authorized. Real training requires a later frozen protocol and budget.
""")
    code("""if (measurement_dir / "measurement.json").exists():
    from core.spatial_colab_bundle import export_receipts
    receipt_package = pathlib.Path("/content/capacity-receipts.json")
    sha256 = export_receipts(measurement_dir, receipt_package)
    print("Verified receipt package SHA-256:", sha256)
    files.download(str(receipt_package))  # Save this package; Drive upload is manual.
""")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "png_bmp_capacity.ipynb"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    for index, item in enumerate(cells):
        item["id"] = f"spatial-{index:02d}"
    artifacts.put(out / "png_bmp_capacity.ipynb", artifacts.encode(notebook), 4 * artifacts.MIB)
    return {
        "status": "ready",
        "source_package_sha256": checksum,
        "base_commit": service.BASE_COMMIT,
    }


def export_receipts(folder: Path, target: Path) -> str:
    files = {}
    for path in sorted(folder.iterdir()):
        if (
            path.name == "measurement.json"
            or path.name == "worker.json"
            or path.name == "execution.json"
            or path.name in {f"cell-{i:02d}.json" for i in range(12)}
        ):
            raw = artifacts.read(path, artifacts.MIB)
            files[path.name] = {"sha256": artifacts.digest(raw), "text": raw.decode()}
    if "measurement.json" not in files:
        raise ValueError("measurement receipt required")
    result = {"schema": "spatial-colab-receipts-v1", "files": files}
    raw = artifacts.encode(result)
    verify_receipts(result)
    return artifacts.put(target, raw, 4 * artifacts.MIB)


def verify_receipts(package: dict) -> None:
    if (
        set(package) != {"schema", "files"}
        or package["schema"] != "spatial-colab-receipts-v1"
        or not isinstance(package["files"], dict)
        or not 1 <= len(package["files"]) <= 15
    ):
        raise ValueError("receipt package schema")
    allowed = {
        "measurement.json",
        "worker.json",
        "execution.json",
        *(f"cell-{i:02d}.json" for i in range(12)),
    }
    if "measurement.json" not in package["files"] or not set(package["files"]) <= allowed:
        raise ValueError("receipt package members")
    for record in package["files"].values():
        if (
            set(record) != {"sha256", "text"}
            or not isinstance(record["text"], str)
            or len(record["text"].encode()) > artifacts.MIB
            or artifacts.digest(record["text"].encode()) != artifacts.identity(record["sha256"])
        ):
            raise ValueError("receipt package checksum/size")
