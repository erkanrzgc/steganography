#!/usr/bin/env python3
"""Complete real four-row accounting, not training or accuracy evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import srnet_multibatch  # noqa: E402
from core.srnet_sampling import source_id  # noqa: E402
from steganography.research_features import read_document, selected_samples  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_srnet_plan import plan_training  # noqa: E402

PROTOCOL_SHA = "c84a1357195fdc59650679ed9878eef66d1f3c89784a69addd90beec894b031b"
MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"
CACHE_SHA = "828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028"


def check(condition, message):
    if not condition:
        raise ValueError("SRNet multipair audit failed: " + message)


def audit(manifest_path: Path, cache: Path, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("multipair audit output must be fresh and non-symlink")
    check(
        hashlib.sha256((ROOT / "docs/SRNET_MULTIPAIR_PROTOCOL.md").read_bytes()).hexdigest()
        == PROTOCOL_SHA,
        "protocol checksum",
    )
    manifest, digest = read_document(manifest_path)
    check(digest == MANIFEST_SHA, "manifest checksum")
    samples = selected_samples(manifest, "train")
    validation = {s["sha256"] for s in selected_samples(manifest, "validation")}
    check(len(samples) == 2985 and len(validation) == 765, "split sizes")
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "manifest": str(manifest_path),
        "manifest_sha256": MANIFEST_SHA,
        "cache": str(cache),
        "cache_sha256": CACHE_SHA,
        "epochs": 1,
        "seed": 20261012,
        "batch_recipe": srnet_multibatch.RECIPE,
    }
    plan = plan_training(config, out / "plan.json")
    batches, record = srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=0)
    check(len(batches) == 1580 and record == plan["epoch_batch_schedule"][0], "bound grouping")
    source_counts, cells, seen, identities = Counter(), Counter(), set(), []
    for batch in batches:
        rows = [samples[i] for i in batch]
        check([r["label"] for r in rows] == ["cover", "stego", "cover", "stego"], "label order")
        check(rows[0]["source_group"] != rows[2]["source_group"], "source separation")
        check(all(r["sha256"] not in validation for r in rows), "validation leakage")
        for c, s in ((rows[0], rows[1]), (rows[2], rows[3])):
            check(
                all(c[k] == s[k] for k in ("source_group", "lineage", "quality_factor")),
                "pair identity",
            )
            source = source_id(s["source_group"])
            source_counts[source] += 1
            cells[(source, s["quality_factor"], s["method"])] += 1
            seen.add(s["sha256"])
        identities.append([r["sha256"] for r in rows])
    check(
        seen == {s["sha256"] for s in samples if s["label"] == "stego"},
        "complete original coverage",
    )
    check(set(source_counts.values()) == {1580} and len(source_counts) == 2, "source balance")
    check(sorted(cells.values()) == [395, 395, 395, 395, 790, 790], "context balance")
    ordered_hash = hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode()
    ).hexdigest()
    check(ordered_hash == record["ordered_batch_sha256"], "independent order hash")
    result = {
        "schema_version": "srnet-multipair-accounting-v1",
        "passed": True,
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "837cb79",
        "manifest_sha256": MANIFEST_SHA,
        "train_rows": len(samples),
        "validation_hashes_excluded": len(validation),
        "plan": plan,
        "plan_sha256": hashlib.sha256((out / "plan.json").read_bytes()).hexdigest(),
        "batches_audited": len(batches),
        "original_stegos_covered": len(seen),
        "ordered_batch_sha256": ordered_hash,
        "pairs_per_source": dict(source_counts),
        "real_training": "unavailable",
        "accuracy_metrics": "unavailable",
        "deployed": False,
        "primary_detection_changed": False,
    }
    write_json(out / "evidence.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.cache, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
