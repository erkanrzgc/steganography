"""Published control remains synthetic training evidence, never detector accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np

from core import srnet_multibatch, srnet_positive, srnet_sanity
from core.srnet_sampling import epoch_pairs

ROOT = Path(__file__).resolve().parents[1]


def test_public_control_hashes_content_schedules_metrics_and_oracles():
    raw = (ROOT / "benchmarks/srnet-positive-control-20261007.json").read_bytes()
    report = json.loads(raw)
    audit = json.loads((ROOT / "benchmarks/srnet-positive-audit-20261007.json").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == audit["control_report_sha256"]
    assert audit["control_report_sha256"] == (
        "605f31b19a0440234114dc71835ff5dba18f21a3c53a72fda72bac137acb2c63"
    )
    protocol = (ROOT / "docs/SRNET_POSITIVE_CONTROL_PROTOCOL.md").read_bytes()
    assert hashlib.sha256(protocol).hexdigest() == report["protocol_sha256"]
    assert report["protocol_sha256"] == srnet_positive.PROTOCOL_SHA
    pixels, samples = srnet_positive.generate()
    assert report["selected_rows"] == samples
    assert report["epoch_pair_schedule"] == [
        epoch_pairs(samples, seed=20261012, epoch=e)[1] for e in range(20)
    ]
    assert report["epoch_batch_schedule"] == [
        srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=e)[1] for e in range(20)
    ]
    logits = np.array(report["stored_bn_singleton_train_logits"])
    final = srnet_sanity.metrics(logits, samples)
    assert final == report["stored_bn_singleton_train_metrics"]
    assert srnet_positive.objectives(report["epoch_training"], final) == report["sanity_objectives"]
    assert report["sanity_objectives_passed"] and audit["numerical_gates_passed"]
    assert len(audit["independent_oracles"]) == 6
    assert [r["row"] for r in audit["independent_oracles"]] == [0, 1, 2, 12, 13, 14]
    assert all(r["passed"] and r["decisions_equal"] for r in audit["independent_oracles"])
    assert audit["model_sha256"] == report["model_sha256"]
    assert report["optimizer_updates"] == sum(r["updates"] for r in report["epoch_training"]) == 160
    assert report["generated_tensors_not_jpeg_or_steganography"]
    assert report["sampler_format_and_method_tags_are_synthetic"]
    assert not report["real_dataset_loaded"] and not report["validation_pixels_loaded"]
    assert report["accuracy_qualification"] == audit["accuracy_qualification"] == "unavailable"
    assert not report["deployed"] and not report["primary_detection_changed"]
    assert b"/home/" not in raw and b"/tmp/" not in raw
    assert pixels.shape == (24, 1, 256, 256)
