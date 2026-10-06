"""Independent real training-pair accounting; no fitting or accuracy claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.srnet_sampling import epoch_pairs  # noqa: E402
from steganography.research_features import read_document, selected_samples  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_srnet_plan import plan_training  # noqa: E402

MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"


def check(condition, reason):
    if not condition:
        raise ValueError("SRNet schedule audit failed: " + reason)


def audit(manifest: Path, cache: Path, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("audit output must be fresh and non-symlink")
    document, digest = read_document(manifest)
    if digest != MANIFEST_SHA:
        raise ValueError("frozen manifest mismatch")
    samples = selected_samples(document, "train")
    validation = {s["sha256"] for s in selected_samples(document, "validation")}
    check(len(samples) == 2985 and len(validation) == 765, "split sizes")
    sources = sorted({s["source_group"] for s in samples})
    check(len(sources) == 2, "source count")
    _, cache_sha = read_document(cache)
    protocol = Path(__file__).resolve().parents[1] / "docs/SRNET_SAMPLING_PROTOCOL.md"
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for number, source in enumerate([None, *sources]):
        source_id = "source-" + hashlib.sha256(source.encode()).hexdigest()[:16] if source else None
        plan_path = out / f"scope{number}.json"
        config = {
            "manifest": str(manifest),
            "manifest_sha256": digest,
            "cache": str(cache),
            "cache_sha256": cache_sha,
            "epochs": 10,
            "seed": 20261012,
            "training_source_id": source_id,
        }
        plan = plan_training(config, plan_path)
        chosen = [s for s in samples if source is None or s["source_group"] == source]
        originals = Counter(
            (s["source_group"], s["quality_factor"], s["method"])
            for s in chosen
            if s["label"] == "stego"
        )
        source_cells = {
            s: [k for k in originals if k[0] == s]
            for s in {sample["source_group"] for sample in chosen}
        }
        multiple = math.lcm(*(len(cells) for cells in source_cells.values()))
        required = max(
            max(originals[k] for k in cells) * len(cells) for cells in source_cells.values()
        )
        quota = ((required + multiple - 1) // multiple) * multiple
        expected = {k: quota // len(source_cells[k[0]]) for k in originals}
        for epoch, saved in enumerate(plan["epochs"]):
            pairs, _ = epoch_pairs(chosen, seed=20261012, epoch=epoch)
            replay, _ = epoch_pairs(chosen, seed=20261012, epoch=epoch)
            check(np.array_equal(pairs, replay), "deterministic replay")
            observed, identities, seen = Counter(), [], set()
            for cover, stego in pairs:
                c, s = chosen[cover], chosen[stego]
                check(c["label"] == "cover" and s["label"] == "stego", "pair labels")
                check(
                    c["sha256"] not in validation and s["sha256"] not in validation,
                    "validation leakage",
                )
                check(
                    all(c[k] == s[k] for k in ("lineage", "source_group", "quality_factor")),
                    "pair context",
                )
                observed[(s["source_group"], s["quality_factor"], s["method"])] += 1
                seen.add(s["sha256"])
                identities.append([c["sha256"], s["sha256"]])
            check(observed == expected, "hierarchical counts")
            check(
                seen == {s["sha256"] for s in chosen if s["label"] == "stego"},
                "complete pair coverage",
            )
            order_sha = hashlib.sha256(
                json.dumps(identities, separators=(",", ":")).encode()
            ).hexdigest()
            check(order_sha == saved["ordered_pair_sha256"], "ordered identity")
            check(saved["pairs"] == quota * len(source_cells), "pair count")
            check(saved["rows"] == 2 * len(pairs), "row count")
            saved_counts = {
                (c["source_id"], c["quality_factor"], c["method"]): c["sampled_pairs"]
                for c in saved["cells"]
            }
            check(
                saved_counts
                == {
                    ("source-" + hashlib.sha256(k[0].encode()).hexdigest()[:16], k[1], k[2]): count
                    for k, count in expected.items()
                },
                "published cell counts",
            )
        records.append(
            {
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "passed": True,
                "plan": plan,
            }
        )
    record = {
        "schema_version": "srnet-sampling-audit-v1",
        "manifest_sha256": digest,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "train_rows": len(samples),
        "validation_rows_excluded": len(validation),
        "scopes": records,
        "epochs_audited": 30,
        "passed": True,
        "real_training": "unavailable",
        "accuracy_metrics": "unavailable",
        "deployed": False,
        "primary_detection_changed": False,
    }
    write_json(out / "audit.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.cache, args.out)
    print(json.dumps({"passed": result["passed"], "epochs_audited": result["epochs_audited"]}))
