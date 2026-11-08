"""One immutable Colab profile, preserved debits and a prospective resource amendment."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from core import colab_artifacts as a

SCHEMA = "colab-rgb-fit-data-v2"
AMENDMENT = "colab-rgb-cumulative-2700s-20261107-v1"
SECONDS = 2700
OUTPUT = 768 * a.MIB
PROFILE_SHA = "326b09839a7c57b22652305d98fc24a998c256ff65c8239e38c2db1dbeb5c448"
CONTRACT_SHA = "37d2f589a90ac29dcafea3fc238217022597acc3ea084d1b1f8eb6273a35a33b"
FILES = {
    "closed.json": PROFILE_SHA,
    "contract.json": CONTRACT_SHA,
    "independent-replay.json": "10d5329737615d9cd4b42496d3172e8fbf753e27f05bddd50753e959738d86ca",
    "profile-ledger-export.json": (
        "a7ad73ae16da31f064caeb4d2d690556a52e696e602e80f77c2b1455e11324d1"
    ),
}
PACKET = "colab-consumed-profile-metadata-v1"
PACKET_LIMIT = 256 * 1024
BASE_EXTERNAL_SECONDS = 950
FUTURE_SETUP_SECONDS = 300
FUTURE_CLOSE_SECONDS = 150
MIN_EXTERNAL_SECONDS = BASE_EXTERNAL_SECONDS + FUTURE_SETUP_SECONDS
MIN_EXTERNAL_BYTES = 68 * a.MIB  # Old 64 MiB plus four MiB for new metadata/evidence.
CHARGED_SECONDS = 79.29754005300015
PROFILE_BYTES = 234181832
FIT_PATH = "core/spatial_colab_fit.py"
CHANGED = {FIT_PATH, "steganography/research_colab_train.py"}
ADDED = {"core/spatial_colab_budget.py"}
FIT_ADAPTERS = {"contract", "proposal", "freeze", "approval", "bind_profile"}


def compatibility(root: Path, original: dict, previous_fit: str) -> dict:
    """Compare immutable sources and protected fit AST; never import old source."""
    from core.colab_execution import source_hashes

    expected = original["source_sha256"]
    actual = source_hashes(root)
    if set(actual) != set(expected) | ADDED or any(
        actual[name] != checksum for name, checksum in expected.items() if name not in CHANGED
    ):
        raise ValueError("profile source compatibility mismatch")
    if a.digest(previous_fit.encode()) != expected[FIT_PATH]:
        raise ValueError("original fit source hash mismatch")

    def protected(source):
        nodes = ast.parse(source).body
        return [
            ast.dump(node, include_attributes=False)
            for node in nodes
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            or node.name not in FIT_ADAPTERS
        ]

    if protected(previous_fit) != protected(a.read(root / FIT_PATH, a.MIB).decode()):
        raise ValueError("protected fitting engine changed")
    return {
        "unchanged_source_files": len(expected) - len(CHANGED),
        "changed_adapters": sorted(CHANGED),
        "added_resource_service": sorted(ADDED),
        "protected_fit_ast_unchanged": True,
    }


def validate_packet(root: Path, packet: dict) -> dict:
    if set(packet) != {"schema", "files"} or packet["schema"] != PACKET:
        raise ValueError("consumed profile metadata schema")
    entries = packet["files"]
    if (not isinstance(entries, dict) or set(entries) != set(FILES) | {"prior-fit-source.py"}
            or not isinstance(entries["prior-fit-source.py"], str)
            or len(entries["prior-fit-source.py"].encode()) > a.MIB):
        raise ValueError("consumed profile metadata files")
    for name, checksum in FILES.items():
        raw = entries[name]
        if (not isinstance(raw, str) or len(raw.encode()) > 128 * 1024
                or a.digest(raw.encode()) != checksum):
            raise ValueError("immutable consumed profile hash mismatch")
    original = json.loads(entries["contract.json"])
    compatibility(root, original, entries["prior-fit-source.py"])
    # These byte-exact receipts include the completed command, independent numeric
    # replay and external ledger. The fixed debits retain the session-target overrun.
    # No old worker runs and no checkpoint/pixel packet is transported here.
    return json.loads(entries["closed.json"])


def accepted_profile(root: Path, out: Path) -> tuple[dict, str]:
    value = a.document(out / "profile-metadata.json", PACKET_LIMIT)
    return validate_packet(root, value), PROFILE_SHA


def package(root: Path, evidence: Path, original_contract: Path,
            original_fit: Path, target: Path) -> str:
    entries = {
        name: a.read(original_contract if name == "contract.json" else evidence / name,
                     128 * 1024).decode()
        for name in FILES
    }
    entries["prior-fit-source.py"] = a.read(original_fit, a.MIB).decode()
    packet = {"schema": PACKET, "files": entries}
    validate_packet(root, packet)
    return a.put(target, a.encode(packet), PACKET_LIMIT)


def validate(payload: dict) -> None:
    from core.spatial_colab_fit import finite, setup_debit

    debit = payload["setup_debit"]
    setup_debit(debit)
    if (payload["resource_amendment"] != AMENDMENT
            or payload["profile_receipt_sha256"] != PROFILE_SHA
            or payload["profile_contract_sha256"] != CONTRACT_SHA
            or debit["seconds"] < MIN_EXTERNAL_SECONDS
            or debit["output_bytes"] < MIN_EXTERNAL_BYTES
            or finite(payload["prior_seconds"]) != CHARGED_SECONDS + debit["seconds"]
            or payload["prior_output_bytes"] != PROFILE_BYTES + debit["output_bytes"]):
        raise ValueError("amended cumulative budget must preserve every prior debit")


def setup_age(age_at_bootstrap: float | None, elapsed: float) -> float:
    """A later operator must supply the observed allocation age, never default zero."""
    from core.spatial_colab_fit import finite

    age = finite(age_at_bootstrap) + finite(elapsed)
    if age >= FUTURE_SETUP_SECONDS - FUTURE_CLOSE_SECONDS:
        raise ValueError("prospective GPU setup reservation exhausted; close runtime")
    return age
