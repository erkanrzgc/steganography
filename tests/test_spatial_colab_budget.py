"""Metadata and generated software fixtures; no GPU, native pixel or fitting attempt."""

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core import colab_artifacts as a
from core import colab_execution as c
from core import spatial_colab_budget as b
from core import spatial_colab_fit as f
from steganography import research_colab_train as cli

ROOT = Path(__file__).parent.parent
METADATA = ROOT / "benchmarks/png-bmp-colab-consumed-profile-metadata-20261107.json"
DEBIT = {"schema": "colab-fit-external-debit-v1", "complete": True,
         "seconds": b.MIN_EXTERNAL_SECONDS, "output_bytes": b.MIN_EXTERNAL_BYTES}


@pytest.fixture
def sealed(tmp_path):
    out = tmp_path / "consumed-profile"
    a.directory(out)
    a.put(out / "profile-metadata.json", a.read(METADATA, b.PACKET_LIMIT), b.PACKET_LIMIT)
    return out


def test_actual_consumed_metadata_requires_no_torch_or_image_reads(sealed, monkeypatch):
    monkeypatch.setattr(c, "audit", lambda *args: pytest.fail("old controller must not execute"))
    result, checksum = b.accepted_profile(ROOT, sealed)
    assert checksum == b.PROFILE_SHA
    assert result["worker"]["native_optimizer_updates"] == 0
    assert result["worker"]["generated_optimizer_updates"] == 6
    payload, limits = f.proposal(ROOT, sealed, DEBIT, amended=True)
    assert payload["schema"] == b.SCHEMA
    assert payload["prior_output_bytes"] == b.PROFILE_BYTES + b.MIN_EXTERNAL_BYTES
    assert limits.seconds == 1370.7024599469999
    assert limits.output_bytes == 499821368
    assert payload["projection_seconds"] == pytest.approx(1311.131131086032)
    assert limits.seconds + payload["prior_seconds"] == 2700
    assert f.contract(payload, limits).mode == "RGB"
    target = sealed / "fit-contract.json"
    f.freeze(ROOT, sealed, DEBIT, target, amended=True)
    job, binding, _ = c.source_contract(ROOT, target)
    f.bind_profile(ROOT, job, sealed)
    assert not (ROOT / ".colab-process-attempts").exists()
    for permission in (None,):
        with pytest.raises(ValueError, match="permission"):
            c.permission(job, binding, permission)


@pytest.mark.parametrize("fault", ["schema", "files", "file_type", "source_type",
                                  *b.FILES, "source"])
def test_metadata_cannot_be_substituted(fault):
    packet = a.document(METADATA, b.PACKET_LIMIT)
    if fault == "schema":
        packet["schema"] = "anything"
    elif fault == "files":
        packet["files"]["extra"] = "untrusted"
    elif fault == "file_type":
        packet["files"] = []
    elif fault == "source_type":
        packet["files"]["prior-fit-source.py"] = None
    elif fault == "source":
        packet["files"]["prior-fit-source.py"] += "\n# changed\n"
    else:
        packet["files"][fault] += " "
    with pytest.raises(ValueError):
        b.validate_packet(ROOT, packet)


def test_package_is_exclusive_and_reproducible(tmp_path):
    packet = a.document(METADATA, b.PACKET_LIMIT)
    evidence = tmp_path / "evidence"
    a.directory(evidence)
    for name, text in packet["files"].items():
        a.put(evidence / name, text.encode())
    target = tmp_path / "profile-metadata.json"
    checksum = b.package(ROOT, evidence, evidence / "contract.json",
                         evidence / "prior-fit-source.py", target)
    assert checksum == a.digest(a.read(METADATA, b.PACKET_LIMIT))
    with pytest.raises(FileExistsError):
        b.package(ROOT, evidence, evidence / "contract.json",
                  evidence / "prior-fit-source.py", target)
    target.unlink()
    target.symlink_to(METADATA)
    with pytest.raises(ValueError, match="symlink"):
        b.package(ROOT, evidence, evidence / "contract.json",
                  evidence / "prior-fit-source.py", target)


@pytest.mark.parametrize("fault", ["engine", "additional_source", "removed_source", "fit_ast"])
def test_source_compatibility_preserves_entire_old_engine(tmp_path, monkeypatch, fault):
    packet = a.document(METADATA, b.PACKET_LIMIT)["files"]
    original = json.loads(packet["contract.json"])
    actual = c.source_hashes(ROOT)
    code = (ROOT / b.FIT_PATH).read_text()
    if fault == "engine":
        actual["core/spatial_native_training.py"] = "0" * 64
    elif fault == "additional_source":
        actual["core/unreviewed.py"] = "0" * 64
    elif fault == "removed_source":
        del actual["core/spatial_native_training.py"]
    else:
        code += "\ndef unexpected_training_change():\n    return 1\n"
    (tmp_path / "core").mkdir()
    (tmp_path / b.FIT_PATH).write_text(code)
    monkeypatch.setattr(c, "source_hashes", lambda root: actual)
    with pytest.raises(ValueError):
        b.compatibility(tmp_path, original, packet["prior-fit-source.py"])


@pytest.mark.parametrize("fault", ["old_debit", "bytes", "prior", "used", "profile", "amendment",
                                  "cumulative", "projection", "environment"])
def test_amendment_cannot_reset_old_debits_or_relax_other_limits(sealed, fault):
    payload, limits = f.proposal(ROOT, sealed, DEBIT, amended=True)
    if fault == "old_debit":
        payload["setup_debit"]["seconds"] = b.BASE_EXTERNAL_SECONDS
    elif fault == "bytes":
        payload["setup_debit"]["output_bytes"] -= 1
    elif fault == "prior":
        payload["prior_seconds"] -= 1
        limits = dataclasses.replace(limits, seconds=limits.seconds + 1)
    elif fault == "used":
        payload["prior_output_bytes"] -= 1
        limits = dataclasses.replace(limits, output_bytes=limits.output_bytes + 1)
    elif fault == "profile":
        payload["profile_receipt_sha256"] = "0" * 64
    elif fault == "amendment":
        payload["resource_amendment"] = "unreviewed"
    elif fault == "cumulative":
        limits = dataclasses.replace(limits, seconds=1800 - payload["prior_seconds"])
    elif fault == "projection":
        payload["projection_seconds"] = limits.seconds
    else:
        payload["environment"]["cudnn"] = None
    with pytest.raises(ValueError):
        f.contract(payload, limits)


def test_old_fit_permission_does_not_approve_new_allowance(sealed):
    target = sealed / "proposal.json"
    job = f.freeze(ROOT, sealed, DEBIT, target, amended=True)
    binding = a.digest(target.read_bytes())
    fixture = {"schema": "colab-fit-permission-v1", "user_approved": True,
               "kind": "native_fit", "contract_sha256": binding,
               "profile_receipt_sha256": b.PROFILE_SHA,
               "resource_acknowledgement": f.profile.RESOURCE_ACK}
    path = sealed / "generated-fixture-permission.json"
    a.put(path, a.encode(fixture))
    with pytest.raises(ValueError, match="permission"):
        f.approval(job, binding, path)
    fixture.update(schema="colab-fit-permission-v2", resource_amendment=b.AMENDMENT,
                   cumulative_seconds=b.SECONDS)
    path.write_bytes(a.encode(fixture))
    f.approval(job, binding, path)
    fixture["cumulative_seconds"] += 1
    path.write_bytes(a.encode(fixture))
    with pytest.raises(ValueError, match="permission"):
        f.approval(job, binding, path)


@pytest.mark.parametrize("age,elapsed", [(None, 1), (-1, 1), (1, float("nan")),
                                       (100, 80), (0, 180), (True, 1)])
def test_later_gpu_setup_requires_actual_age_and_cannot_consume_more(age, elapsed):
    with pytest.raises(ValueError):
        b.setup_age(age, elapsed)


def test_setup_age_and_larger_complete_debits_are_conservative(sealed):
    assert b.setup_age(10, 100) == 110
    debit = {**DEBIT, "seconds": b.MIN_EXTERNAL_SECONDS + 10,
             "output_bytes": DEBIT["output_bytes"] + 1000}
    payload, limits = f.proposal(ROOT, sealed, debit, amended=True)
    assert limits.seconds == b.SECONDS - payload["prior_seconds"]
    assert limits.seconds < 1370.70246


def test_cli_amendment_freezes_only_a_proposal(sealed, capsys):
    debit = sealed / "external-debit.json"
    a.put(debit, a.encode(DEBIT))
    target = sealed / "cli-proposal.json"
    assert cli.main(["freeze-fit", "--protocol", str(target), "--profile-out", str(sealed),
                     "--setup-debit", str(debit), "--cumulative-amendment"]) == 0
    assert "permission_missing" in capsys.readouterr().out
    assert cli.main(["fit", "--protocol", str(target), "--out", str(sealed / "never"),
                     "--cumulative-amendment"]) == 2
    assert not (sealed / "never").exists()


def test_delivered_amended_notebook_is_source_only_and_has_no_profile_route(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "builder", ROOT / "scripts/build-colab-amended-fit-notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    out = tmp_path / "delivery"
    builder.build(ROOT, out)
    notebook = a.document(out / builder.NOTEBOOK_NAME)
    sources = ["".join(cell["source"]).replace(
        "/content/spatial-colab-amended-fit-source-20261107", str(tmp_path / "installed")
    ) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "RUN_APPROVED_PROFILE" not in "\n".join(sources)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(sources))
    script = """import json,sys,types
from pathlib import Path
packet=Path(sys.argv[1]).read_bytes();calls=[]
files=types.SimpleNamespace(upload=lambda:calls.append('source-only') or {sys.argv[3]:packet})
google=types.ModuleType('google');colab=types.ModuleType('google.colab');colab.files=files
google.colab=colab;sys.modules['google']=google;sys.modules['google.colab']=colab
ns={}
for source in json.loads(Path(sys.argv[2]).read_text()):exec(source,ns)
assert calls==['source-only'] and 'torch' not in sys.modules
assert ns['ALLOCATION_REQUEST_AGE_AT_BOOTSTRAP_SECONDS'] is None
assert not ns['RUN_APPROVED_FIT'] and not ns['VERIFY_APPROVED_FIT']
assert not ns['DOWNLOAD_VERIFIED_TRAINING']
assert not (ns['source_root']/'.colab-process-attempts').exists()
assert not (ns['source_root']/'cache').exists()
print('metadata-only-no-profile-no-pixels-no-model-no-attempt')
"""
    result = subprocess.run(  # noqa: S603 -- owned metadata-only notebook fixture
        [sys.executable, "-I", "-B", "-c", script, str(out / builder.SOURCE_NAME),
         str(cells), builder.SOURCE_NAME], capture_output=True, text=True,
        timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "metadata-only-no-profile-no-pixels-no-model-no-attempt" in result.stdout
