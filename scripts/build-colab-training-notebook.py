"""Reproduce the complete Colab profile/proposal/approved-fit workflow, disabled by default."""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import colab_artifacts as a  # noqa: E402

SOURCE_NAME = "png_bmp_colab_native_training_sources_20261107.json"
NOTEBOOK_NAME = "png_bmp_colab_native_training_20261107.ipynb"


def build(root, out):
    spec = importlib.util.spec_from_file_location(
        "profile_builder", root / "scripts/build-colab-native-notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = module.build(root, out)
    notebook = a.document(out / "png_bmp_colab_native_profile_20261106.ipynb")
    for cell in notebook["cells"]:
        cell["source"] = [
            line.replace(
                "spatial-colab-native-profile-source-20261106",
                "spatial-colab-native-training-source-20261107",
            ).replace('"sources.json"', repr(SOURCE_NAME))
            for line in cell["source"]
        ]
    notebook["cells"][0]["source"] += [
        "\n## Later native training\n",
        "The remaining cells prepare and execute a separately approved RGB pilot.\n",
        "Two seeds, twenty epochs, batch eight: 640 updates /5,120 presentations.\n",
        "Loss and development AUC are shown each epoch; checkpoints at 5/10/15/20.\n",
        "Freeze-fit requires this exact accepted physical profile and measured external debits.\n",
        "The cumulative command allowance is 1,800 seconds /768 MiB, including the profile.\n",
        "A failed estimate, deadline, OOM or unavailable profile prevents fitting.\n",
        "Only a later fit permission allows the twenty development crops to be uploaded.\n",
        "All twenty-one reserved RGB and thirteen L pixels stay closed.\n",
        "All switches below are False; no permission file is created by this notebook.\n",
        "Training evidence contains metadata and eight individual numeric states.\n",
        "Each state is verified separately; the report archive excludes numeric files.\n",
    ]
    notebook["cells"][0]["source"] = [
        s.replace(
            "This notebook has no fitting command; "
            "measured cost cannot approve a fit automatically.",
            "The later fitting cells require separate explicit profile-bound approval.",
        ).replace("source-bound cost-profile preparation", "profile and approved training")
        for s in notebook["cells"][0]["source"]
    ]
    prepare = """PREPARE_FIT_PROPOSAL = False
if PREPARE_FIT_PROPOSAL:
    from core import spatial_colab_fit as training

    controller.audit(source_root, protocol, output)
    selected = files.upload()  # Measured setup/review/export debits, never assumed zero.
    if set(selected) != {"external-debit.json"}:
        raise ValueError("Select external-debit.json only")
    external_debit_path = source_root / "external-debit.json"
    artifacts.put(external_debit_path, selected["external-debit.json"])
    fit_protocol = source_root / "fit-contract.json"
    training.freeze(source_root, output, artifacts.document(external_debit_path), fit_protocol)
    fit_job, fit_binding, fit_limits = controller.source_contract(source_root, fit_protocol)
    fit_permission_verified = False
    print("Fit proposal:", fit_binding)
    print("Projected seconds:", fit_job["payload"]["projection_seconds"])
    print("Remaining seconds:", fit_limits.seconds, "— later fitting permission required")
"""
    permit = """VERIFY_APPROVED_FIT = False
if VERIFY_APPROVED_FIT:
    selected = files.upload()  # Later explicit fitting approval; profile approval is insufficient.
    if set(selected) != {"fit-permission.json"}:
        raise ValueError("Select fit-permission.json only")
    fit_permission = source_root / "fit-permission.json"
    artifacts.put(fit_permission, selected["fit-permission.json"])
    controller.permission(fit_job, fit_binding, fit_permission)
    fit_permission_verified = True
"""
    run = """RUN_APPROVED_FIT = False
DEVELOPMENT_PACKAGE_SHA256 = ""
if RUN_APPROVED_FIT:
    if not fit_permission_verified:
        raise ValueError("Separate fit permission must precede development upload")
    artifacts.identity(DEVELOPMENT_PACKAGE_SHA256)
    selected = files.upload()  # Exactly twenty development crops, no reserved pixels.
    if set(selected) != {"development-cache.json"}:
        raise ValueError("Select development-cache.json only")
    development_packet = source_root / "development-cache.json"
    artifacts.put(development_packet, selected["development-cache.json"], handoff.CACHE_LIMIT)
    handoff.install_development(cache, split, development_packet, DEVELOPMENT_PACKAGE_SHA256)
    fit_output = source_root / "fit-output"
    fit_result = controller.run(source_root, fit_protocol, fit_output, cache=cache.root,
                                permission_path=fit_permission, profile_out=output)
    controller.audit(source_root, fit_protocol, fit_output)
    print("Execution:", fit_result["status"])
    if fit_result["status"] == "completed":
        for cell in fit_result["worker"]["cells"]:
            print("Seed", cell["seed"], "final train/development AUC:",
                  cell["final"]["train"]["auc"], cell["final"]["development"]["auc"])
"""
    export = """DOWNLOAD_VERIFIED_TRAINING = False
if DOWNLOAD_VERIFIED_TRAINING:
    checked = controller.audit(source_root, fit_protocol, fit_output)
    if checked["status"] != "completed":
        raise ValueError("Keep unavailable evidence private; do not claim a complete fit")
    archive = fit_output / "verified-training.zip"
    checksum = checked["output_files"][archive.name]["sha256"]
    handoff.verify_evidence(archive, checksum, fit_binding, max_members=4097)
    files.download(str(archive))
    for cell in checked["worker"]["cells"]:
        for checkpoint in cell["checkpoints"]:
            relative = "work/fit/" + checkpoint["name"]
            path = fit_output / relative
            expected_sha = checked["output_files"][relative]["sha256"]
            if handoff.stream_hash(path, fit_limits.file_bytes) != expected_sha:
                raise ValueError("Checkpoint changed before export")
            files.download(str(path))
    files.download(str(fit_output / "closed.json"))
"""
    for index, code in enumerate((prepare, permit, run, export), 6):
        notebook["cells"].append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(True),
                "id": f"native-{index}",
            }
        )
    notebook["metadata"]["colab"]["name"] = NOTEBOOK_NAME
    a.put(out / NOTEBOOK_NAME, a.encode(notebook), a.MIB)
    a.put(out / SOURCE_NAME, a.read(out / "sources.json", 4 * a.MIB), 4 * a.MIB)
    return {
        **binding,
        "native_fit_execution": False,
        "notebook": NOTEBOOK_NAME,
        "source_packet": SOURCE_NAME,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(a.encode(build(Path(__file__).parent.parent, args.out)).decode())
