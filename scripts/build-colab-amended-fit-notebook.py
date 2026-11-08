"""Deliver a disabled fit-only notebook reusing the consumed, immutable T4 profile."""

import argparse
import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import colab_artifacts as a  # noqa: E402
from core import spatial_colab_budget as budget  # noqa: E402
from core import spatial_colab_fit as training  # noqa: E402
from core import spatial_colab_handoff as handoff  # noqa: E402

SOURCE_NAME = "png_bmp_colab_amended_fit_sources_20261107.json"
NOTEBOOK_NAME = "png_bmp_colab_amended_fit_20261107.ipynb"
METADATA = "benchmarks/png-bmp-colab-consumed-profile-metadata-20261107.json"


def build(root, out):
    a.directory(out)
    source_sha = handoff.source_package(root, out / SOURCE_NAME)
    metadata = a.read(root / METADATA, budget.PACKET_LIMIT)
    prior = out / "accepted-profile"
    a.directory(prior)
    a.put(prior / "profile-metadata.json", metadata, budget.PACKET_LIMIT)
    debit = {"schema": "colab-fit-external-debit-v1", "complete": True,
             "seconds": budget.MIN_EXTERNAL_SECONDS, "output_bytes": budget.MIN_EXTERNAL_BYTES}
    a.put(out / "external-debit.json", a.encode(debit))
    job = training.freeze(root, prior, debit, out / "fit-contract.json", amended=True)
    binding = a.digest(a.read(out / "fit-contract.json", 4 * a.MIB))
    original = a.document(root / "notebooks/png_bmp_colab_native_training_20261107.ipynb")
    bootstrap = "".join(original["cells"][1]["source"])
    bootstrap = re.sub(r"(?m)^SOURCE_SHA256 = .+$", f"SOURCE_SHA256 = {source_sha!r}", bootstrap)
    bootstrap = bootstrap.replace(
        "png_bmp_colab_native_training_sources_20261107.json", SOURCE_NAME
    )
    bootstrap = bootstrap.replace("spatial-colab-native-training-source-20261107",
                                  "spatial-colab-amended-fit-source-20261107")
    bootstrap = bootstrap.replace("import sys\n", "import sys\nimport time\n", 1)
    bootstrap = bootstrap.replace(
        "SOURCE_SHA256 = ", "setup_started = time.monotonic()\n"
        "ALLOCATION_REQUEST_AGE_AT_BOOTSTRAP_SECONDS = None\n\nSOURCE_SHA256 = ", 1
    )
    encoded = base64.b64encode(metadata).decode()
    chunks = "\n".join(f"    {encoded[n:n + 88]!r}" for n in range(0, len(encoded), 88))
    prepare = f'''import base64

from core import colab_artifacts as artifacts
from core import colab_execution as controller
from core import spatial_colab_budget as budget
from core import spatial_colab_fit as training
from core import spatial_colab_handoff as handoff
from core import spatial_colab_profile as profile

metadata_raw = base64.b64decode(
{chunks}
)
METADATA_SHA256 = {a.digest(metadata)!r}
if artifacts.digest(metadata_raw) != METADATA_SHA256:
    raise ValueError("Immutable profile metadata changed")
profile_output = source_root / "accepted-profile"
artifacts.directory(profile_output)
artifacts.put(profile_output / "profile-metadata.json", metadata_raw, budget.PACKET_LIMIT)
fit_protocol = source_root / "fit-contract.json"
debit = {{"schema": "colab-fit-external-debit-v1", "complete": True,
         "seconds": {debit['seconds']}, "output_bytes": {debit['output_bytes']}}}
training.freeze(source_root, profile_output, debit, fit_protocol, amended=True)
fit_job, fit_binding, fit_limits = controller.source_contract(source_root, fit_protocol)
if fit_binding != {binding!r}:
    raise ValueError("Reviewable fit contract changed")
fit_permission_verified = False
print("Fit proposal:", fit_binding)
print("Cumulative seconds:", budget.SECONDS, "remaining:", fit_limits.seconds)
print("No profile rerun; separate fitting permission missing")
'''
    permit = '''VERIFY_APPROVED_FIT = False
if VERIFY_APPROVED_FIT:
    selected = files.upload()  # Later approval of this exact v2 contract.
    if set(selected) != {"fit-permission.json"}:
        raise ValueError("Select fit-permission.json only")
    fit_permission = source_root / "fit-permission.json"
    artifacts.put(fit_permission, selected["fit-permission.json"])
    controller.permission(fit_job, fit_binding, fit_permission)
    fit_permission_verified = True
'''
    run = '''RUN_APPROVED_FIT = False
TRAIN_PACKAGE_SHA256 = "1773c2991478e40718ac6de37e6705141e7a795d02a8fb80b029dc8173d58327"
DEVELOPMENT_PACKAGE_SHA256 = ""  # Prepare only after later fitting approval.
if RUN_APPROVED_FIT:
    if not fit_permission_verified:
        raise ValueError("Separate fitting permission required before media upload")
    budget.setup_age(ALLOCATION_REQUEST_AGE_AT_BOOTSTRAP_SECONDS,
                     time.monotonic() - setup_started)
    artifacts.identity(DEVELOPMENT_PACKAGE_SHA256)
    selected = files.upload()
    if set(selected) != {"train-cache.json", "development-cache.json"}:
        raise ValueError("Select only the approved train/development packets")
    split = profile.engine.split_document(fit_job["payload"]["split"])
    train_packet = source_root / "train-cache.json"
    development_packet = source_root / "development-cache.json"
    artifacts.put(train_packet, selected["train-cache.json"], handoff.CACHE_LIMIT)
    artifacts.put(development_packet, selected["development-cache.json"], handoff.CACHE_LIMIT)
    cache = handoff.install_cache(train_packet, TRAIN_PACKAGE_SHA256, split, source_root / "cache")
    handoff.install_development(cache, split, development_packet, DEVELOPMENT_PACKAGE_SHA256)
    budget.setup_age(ALLOCATION_REQUEST_AGE_AT_BOOTSTRAP_SECONDS,
                     time.monotonic() - setup_started)
    fit_output = source_root / "fit-output"
    fit_result = controller.run(source_root, fit_protocol, fit_output, cache=cache.root,
                                permission_path=fit_permission, profile_out=profile_output)
    controller.audit(source_root, fit_protocol, fit_output)
    print("Execution:", fit_result["status"])
'''
    export = "".join(original["cells"][-1]["source"])
    description = (
        "# Colab RGB SRNet — prospective 45-minute cumulative amendment\n\n"
        "CPU-default metadata preparation; all fitting/export switches are False.\n"
        "The consumed T4 profile is reused through pinned receipt/audit/source metadata.\n"
        "This notebook has no profile execution cell and generates no fitting permission.\n"
        "All previous session charges remain; 300 seconds of future setup/export are reserved.\n"
        "At most 150 seconds precede launch; another 150 remain for export/runtime deletion.\n"
        "Before a later approved GPU launch, the operator must supply the observed allocation\n"
        "age in the first cell; a missing age or exhausted reservation prevents fitting.\n"
        "Current fixed recipe: two seeds /20 epochs /640 updates, 64 train /20 development.\n"
        "Reserved 21 RGB and all 13 L pixels remain closed; actual roles stay unassigned.\n"
        "No hard cgroup RAM/zero-swap or whole-driver VRAM guarantee is claimed.\n"
        "The new allowance and exact fit contract require later explicit user approval.\n"
        "On completion or interruption, preserve evidence and delete the GPU runtime.\n"
    )
    cells = [{"cell_type": "markdown", "id": "amended-0", "metadata": {},
              "source": description.splitlines(True)}]
    for n, source in enumerate((bootstrap, prepare, permit, run, export), 1):
        compile(source, f"amended-{n}", "exec")
        cells.append({"cell_type": "code", "id": f"amended-{n}", "metadata": {},
                      "execution_count": None, "outputs": [], "source": source.splitlines(True)})
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"colab": {"name": NOTEBOOK_NAME},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    a.put(out / NOTEBOOK_NAME, a.encode(notebook), a.MIB)
    return {"source_packet_sha256": source_sha, "fit_contract_sha256": binding,
            "profile_metadata_sha256": a.digest(metadata), "fit_limits": job["limits"],
            "permission_created": False, "data_or_gpu_execution": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(a.encode(build(Path(__file__).parent.parent, args.out)).decode())
