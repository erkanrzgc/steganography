"""Reproduce reviewable native-profile sources/protocol; creates no permission."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import colab_artifacts as a  # noqa: E402
from core import colab_execution as process  # noqa: E402
from core import spatial_colab_handoff as handoff  # noqa: E402
from core import spatial_colab_profile as profile  # noqa: E402

EXPECTED = {"python": "3.12.13", "numpy": "2.0.2", "pillow": "11.3.0",
            "torch": "2.11.0+cu128", "cuda": "12.8", "gpu": "Tesla T4", "driver": "580.82.07"}
TARGET = {"python": "3.12.13", "implementation": "CPython", "platform": "linux"}


def build(root, out):
    a.directory(out)
    source_sha = handoff.source_package(root, out / "sources.json")
    split = a.document(root / "benchmarks/png-bmp-rgb-pilot-proposal-20261028.json")["split"]
    payload = {"schema": profile.PROFILE_SCHEMA, "split": split, "recipe": profile.RECIPE,
               "expected_environment": EXPECTED}
    limits = process.Limits(240, 0, 0, 256 * a.MIB, 128 * a.MIB,
                            closing_seconds=10)
    process.freeze(root, out / "prospective-profile.json", "native_profile", payload, limits,
                   target_environment=TARGET)
    binding = a.digest(a.read(out / "prospective-profile.json", 4 * a.MIB))
    payload_raw = json.dumps(payload)
    payload_chunks = "\n".join(
        f"    {payload_raw[n:n + 88]!r}" for n in range(0, len(payload_raw), 88)
    )
    bootstrap = f'''import hashlib
import json
import os
import re
import sys
from pathlib import Path

from google.colab import files

SOURCE_SHA256 = {source_sha!r}
if any(n == "core" or n.startswith("core.") for n in sys.modules):
    raise RuntimeError("Use a fresh kernel; preserve every consumed attempt")
uploaded = files.upload()  # Select the accompanying sources.json only.
if set(uploaded) != {{"sources.json"}}:
    raise ValueError("Select sources.json only")
raw = uploaded["sources.json"]
if len(raw) > 4 * 1024**2 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
    raise ValueError("Reviewed source packet hash/extent mismatch")
packet = json.loads(raw)
if (set(packet) != {{"schema", "base_commit", "files"}}
        or packet["schema"] != "colab-native-source-package-v1"
        or packet["base_commit"] != {handoff.BASE_COMMIT!r}
        or not 3 <= len(packet["files"]) <= 513 or "LICENSE" not in packet["files"]):
    raise ValueError("Reviewed source packet schema mismatch")
prepared = {{}}
for name, row in packet["files"].items():
    allowed = re.fullmatch(
        r"(?:core|steganography)/(?:[a-zA-Z0-9_]+/){{0,3}}[a-zA-Z0-9_]+\\.py", name
    )
    if name != "LICENSE" and allowed is None:
        raise ValueError("Reviewed source path mismatch")
    if set(row) != {{"utf8", "sha256"}} or not isinstance(row["utf8"], str):
        raise ValueError("Reviewed source entry mismatch")
    content = row["utf8"].encode()
    if len(content) > 1024**2 or hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("Reviewed source file hash/extent mismatch")
    prepared[name] = content
source_root = Path("/content/spatial-colab-native-profile-source-20261106")
if any(p.is_symlink() for p in (source_root, *source_root.parents)):
    raise ValueError("Source directory symlink")
source_root.mkdir(mode=0o700)
for name, content in prepared.items():
    target = source_root / name
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
sys.path.insert(0, str(source_root))
print("Reviewed source package installed:", SOURCE_SHA256)
'''
    freeze = f'''from core import colab_artifacts as artifacts
from core import colab_execution as controller
from core import spatial_colab_handoff as handoff
from core import spatial_colab_profile as profile

PAYLOAD = json.loads(
{payload_chunks}
)
TARGET = {TARGET!r}
EXPECTED_CONTRACT = {binding!r}
protocol = source_root / "profile-contract.json"
limits = controller.Limits(240, 0, 0,
                           256 * artifacts.MIB, 128 * artifacts.MIB, closing_seconds=10)
controller.freeze(source_root, protocol, "native_profile", PAYLOAD, limits,
                  target_environment=TARGET)
if artifacts.digest(artifacts.read(protocol, 4 * artifacts.MIB)) != EXPECTED_CONTRACT:
    raise ValueError("Prospective source/environment contract differs")
if "torch" in sys.modules:
    raise RuntimeError("Source-only setup unexpectedly imported Torch")
profile_permission_verified = False
print("Contract:", EXPECTED_CONTRACT, "— permission missing; no profile executed")
'''
    permission = '''VERIFY_APPROVED_PERMISSION = False
if VERIFY_APPROVED_PERMISSION:
    selected = files.upload()  # Only the later, explicitly approved profile-permission.json.
    if set(selected) != {"profile-permission.json"}:
        raise ValueError("Select the approved permission file only")
    permission_path = source_root / "profile-permission.json"
    artifacts.put(permission_path, selected["profile-permission.json"])
    job = artifacts.document(protocol, 4 * artifacts.MIB)
    controller.permission(job, EXPECTED_CONTRACT, permission_path)
    profile_permission_verified = True
'''
    execute = '''RUN_APPROVED_PROFILE = False
CACHE_PACKAGE_SHA256 = ""  # Fill only from the separately reviewed train-only packet.
if RUN_APPROVED_PROFILE:
    if not profile_permission_verified:
        raise ValueError("Bound permission must be verified before any media upload")
    artifacts.identity(CACHE_PACKAGE_SHA256)
    selected = files.upload()  # Select train-cache.json; no reserved/development pixels.
    if set(selected) != {"train-cache.json"}:
        raise ValueError("Select train-cache.json only")
    cache_packet = source_root / "train-cache.json"
    artifacts.put(cache_packet, selected["train-cache.json"], handoff.CACHE_LIMIT)
    split = profile.engine.split_document(PAYLOAD["split"])
    cache = handoff.install_cache(cache_packet, CACHE_PACKAGE_SHA256, split,
                                  source_root / "cache")
    output = source_root / "profile-output"
    result = controller.run(source_root, protocol, output,
                            permission_path=permission_path, cache=cache.root)
    controller.audit(source_root, protocol, output)
    print(json.dumps(result, sort_keys=True, indent=2))
'''
    export = '''DOWNLOAD_VERIFIED_EVIDENCE = False
if DOWNLOAD_VERIFIED_EVIDENCE:
    checked = controller.audit(source_root, protocol, output)
    if checked["status"] != "completed":
        raise ValueError("Unavailable evidence cannot support fitting approval")
    archive = output / "verified-profile.zip"
    checksum = checked["output_files"][archive.name]["sha256"]
    handoff.verify_evidence(archive, checksum, EXPECTED_CONTRACT)
    files.download(str(archive))
    files.download(str(output / "closed.json"))
'''
    description = (
        "# Colab RGB SRNet — source-bound cost-profile preparation\n\n"
        "This notebook defaults to **CPU / Runtime Version 2026.07** for source/metadata setup.\n"
        "Setup installs reviewed source only and freezes the prospective interpreter.\n"
        "No pip installation or Drive mount runs.\n"
        "All GPU/media/permission/export switches are **False**. No permission file is generated.\n"
        "An explicitly approved fresh profile later needs T4 /2026.07 and the same contract hash.\n"
        "The old Colab and WSL attempts are consumed; never rerun them.\n\n"
        "Proposed command: 240 s including closing, provider-managed RAM, 4-GiB Torch allocator,\n"
        "256-MiB total output including the verified archive; zero native /six generated updates.\n"
        "**Hard cgroup RAM/zero-swap enforcement is unavailable.**\n"
        "RAM monitoring leaves 10% observed capacity free; no fixed 8-GiB ceiling is applied.\n"
        "Only the exact 64-source train overlay may be transported. Development/reserved pixels\n"
        "and all 13 L sources stay closed; actual metadata roles remain unassigned.\n"
        "Two generated numeric states and all six commits are audited before packing.\n"
        "This notebook has no fitting command; measured cost cannot approve a fit automatically.\n"
    )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": description.splitlines(True)}]
    for index, code in enumerate((bootstrap, freeze, permission, execute, export), 1):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": code.splitlines(True), "id": f"native-{index}"})
    cells[0]["id"] = "native-0"
    name = "png_bmp_colab_native_profile_20261106.ipynb"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"colab": {"name": name},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    a.put(out / name, a.encode(notebook), a.MIB)
    return {"source_package_sha256": source_sha, "prospective_contract_sha256": binding,
            "permission_created": False, "data_or_gpu_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(a.encode(build(Path(__file__).parent.parent, args.out)).decode())
