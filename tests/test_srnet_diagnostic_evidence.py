"""Published train-only BN contrast is not deployable inference or accuracy."""

import hashlib
import json
from pathlib import Path


def test_real_diagnostic_protocol_immutable_state_and_no_accuracy_promotion():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/srnet-train-diagnostic-20261007.json"
    record = json.loads(path.read_bytes())
    assert (
        record["protocol_sha256"]
        == hashlib.sha256(
            (root / "docs/SRNET_TRAIN_DIAGNOSTIC_PROTOCOL.md").read_bytes()
        ).hexdigest()
    )
    assert record["protocol_commit"] == "2cbca3a"
    assert record["probe_pairs"] == 32 and record["state_unchanged"]
    assert (
        not record["validation_pixels_loaded"]
        and not record["deployed"]
        and not record["calibrated"]
    )
    assert record["qualification"] == "unavailable"
    assert (
        record["model_sha256"] == "b48faf464c9116fa1778d9cc411e03d154f54f3d31c2fc260cce79424b6acd51"
    )
    assert len(record["cells"]) == 4 and len(record["layers"]) == 26
    assert all(
        c["pairs"] == 8 and c["native"]["in_sample_label_accuracy"] == 0.5 for c in record["cells"]
    )
    assert sum(c["native"]["saturated_scores"] for c in record["cells"]) == 56
    assert sum(c["batch_statistics"]["saturated_scores"] for c in record["cells"]) == 0
    assert (
        sum(c["batch_statistics"]["pairs_with_different_decisions"] for c in record["cells"]) == 32
    )
    assert record["layers"][0]["native"] == record["layers"][0]["batch_statistics"]
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
