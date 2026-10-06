#!/usr/bin/env python3
"""Read-only native/NumPy replay of the frozen train-only sanity control."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import srnet, srnet_model, srnet_multibatch, srnet_reference, srnet_sanity  # noqa: E402
from core.srnet_sampling import epoch_pairs, source_id  # noqa: E402
from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_pixels import load_pixels  # noqa: E402
from steganography.research_srnet_evaluate import oracle_rows  # noqa: E402
from steganography.research_srnet_sanity import CACHE_SHA, MANIFEST_SHA, PROTOCOL_SHA  # noqa: E402


def audit(config_path, result_dir):
    config, _ = read_document(config_path)
    report, report_sha = read_document(result_dir / "sanity.json")
    if (
        report["status"] != "completed"
        or report["protocol_sha256"] != PROTOCOL_SHA
        or report["manifest_sha256"] != MANIFEST_SHA
        or report["train_cache_sha256"] != CACHE_SHA
        or report["validation_pixels_loaded"] is not False
        or report["in_sample_only"] is not True
        or report["deployed"] is not False
        or config["manifest_sha256"] != MANIFEST_SHA
        or config["cache_sha256"] != CACHE_SHA
    ):
        raise ValueError("frozen completed train-only sanity bindings required")
    values, samples, descriptor = load_pixels(
        Path(config["manifest"]),
        Path(config["cache"]),
        checksum=CACHE_SHA,
        split="train",
        _float=True,
    )
    indices = srnet_sanity.select(samples)
    selected = [samples[i] for i in indices]
    metadata = [
        {
            **{k: s[k] for k in ("sha256", "lineage", "quality_factor", "label", "method")},
            "source_id": source_id(s["source_group"]),
        }
        for s in selected
    ]
    if (
        report["selected_rows"] != metadata
        or descriptor["data_sha256"] != report["train_data_sha256"]
    ):
        raise ValueError("selected train identities/cache mismatch")
    paired = [epoch_pairs(selected, seed=20261012, epoch=e)[1] for e in range(50)]
    batched = [
        srnet_multibatch.epoch_batches(selected, seed=20261012, epoch=e)[1] for e in range(50)
    ]
    if report["epoch_pair_schedule"] != paired or report["epoch_batch_schedule"] != batched:
        raise ValueError("sanity schedule mismatch")
    model = srnet_model.load_model(result_dir / "model.npz", checksum=report["model_sha256"])
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    if any(int(v) != 400 for k, v in arrays.items() if k.endswith("num_batches_tracked")):
        raise ValueError("sanity BN counters mismatch")
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        logits = np.array([srnet.float_logits(model, values[i : i + 1])[0] for i in indices])
        final = srnet_sanity.metrics(logits, selected)
        if not all(
            np.allclose(final[k], report["stored_bn_singleton_train_metrics"][k], atol=1e-6, rtol=0)
            for k in final
        ):
            raise ValueError("stored model singleton replay mismatch")
        gates = srnet_sanity.objectives(report["epoch_training"], final)
        if gates != report["sanity_objectives"]:
            raise ValueError("sanity objectives mismatch")
        independent = []
        for i in oracle_rows(selected):
            reference = srnet_reference.reference_logits(
                arrays, values[indices[i] : indices[i] + 1]
            )
            independent.append(
                {
                    "sha256": selected[i]["sha256"],
                    **srnet_reference.compare(logits[i : i + 1], reference),
                }
            )
    finally:
        torch.set_num_threads(previous)
    differences = []
    pairs, _ = epoch_pairs(selected, seed=20261012, epoch=0)
    for c, s in pairs:
        delta = values[indices[s]].astype(np.float64) - values[indices[c]]
        differences.append(
            {
                "cover_sha256": selected[c]["sha256"],
                "stego_sha256": selected[s]["sha256"],
                "input_difference_rms": float(np.sqrt(np.mean(delta**2))),
            }
        )
    return {
        "schema_version": "srnet-tiny-sanity-audit-v1",
        "sanity_report_sha256": report_sha,
        "model_sha256": report["model_sha256"],
        "native_singleton_rows": 24,
        "bn_updates_verified": 400,
        "independent_forward_audit": independent,
        "input_pair_differences": differences,
        "passed": all(a["passed"] for a in independent),
        "validation_pixels_loaded": False,
        "accuracy_qualification": "unavailable",
        "primary_detection_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.result)
    write_json(args.out, result)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
