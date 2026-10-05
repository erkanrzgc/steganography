# Low-payload weighting results — 2026-10-05

One complete fixed run after preregistration commit `56fe36b`, with no search
or selected rerun. The earlier parity model is preserved. This is controlled
LSB replacement on real BOSSbase covers, not a blind real-world-method test.
No model is installed or used by `analyze` / `detect`.

## Data and objective

Same corpus, same 816 training and 184 validation original lineages, same
684 features and train-only normalization. All six controlled derivatives
follow their original's split. Cached artifacts were rehashed/contract-checked;
the original images were audited in the previous corpus run, not reread here.
All held-out native baselines remain frozen.

Only the training objective changes: both 5% positive methods receive weight
4, other rows weight 1. Weighted negative/positive masses are 816/9,792 and
positive BCE class weight is 1/12. Seed, learning rate, epochs, optimizer and
fixed .5 threshold remain unchanged. See [frozen protocol](SPATIAL_WEIGHTING_PROTOCOL.md).

## Every cell, including regressions

Each cell uses 184 stego and the same 184 covers. Old/new denote the previous
unweighted parity model and this weighted model, not deployed detector scores.

| Method / rate | AUC old → new | Recall old → new | New BA | New TP / FN | New ECE |
| --- | --- | --- | --- | --- | --- |
| Sequential 5% | .935078 → .949817 | .380435 → .679348 | .823370 | 125 / 59 | .262958 |
| Sequential 20% | .996987 → .995304 | .961957 → .961957 | .964674 | 177 / 7 | .135330 |
| Sequential 40% | .998789 → .997342 | .978261 → .978261 | .972826 | 180 / 4 | .137552 |
| Scattered 5% | .906486 → .924917 | .146739 → .298913 | .633152 | 55 / 129 | .249404 |
| Scattered 20% | .995894 → .991700 | .972826 → .967391 | .967391 | 178 / 6 | .175375 |
| Scattered 40% | .999114 → .997785 | .994565 → .983696 | .975543 | 181 / 3 | .138469 |

Shared covers: **TN 181 → 178, FP 3 → 6, FPR 1.6304% → 3.2609%**.
Higher-payload AUC drops in all four cells; scattered higher-payload recall
also drops. ECE worsens in five cells; scattered-5% ECE improves from .275259
to .249404 but remains far above the .05 gate.

Paired 200-replicate 95% bootstrap intervals include:

- Scattered-5% AUC [.897262, .953107], recall [.233560, .369701].
- Sequential-5% recall [.597690, .750000].
- Shared FPR [.010870, .065217].

All six cells and intervals are in the
[portable evidence record](../benchmarks/spatial-weighted-development-20261005.json).
These are per-model intervals, not a significance test of paired improvement.

## Numerical and independent verification

CPU ONNX versus PyTorch on all 1,288 validation rows: maximum logit difference
**zero**; probabilities derived from both logits with the same PyTorch sigmoid
also match exactly. Decisions agree at .5, strict absolute 1e-6 and relative
zero. The ONNX graph outputs logits, not a separately exported sigmoid.
Independent NumPy float64 affine/normalization replay
matches ONNX logits exactly at batch sizes 1, 17 and 1,288. Independent
pairwise AUC and confusion checks agree for all six cells.

An additional NumPy float32 sigmoid check at 1e-7 failed on three rows by
1.192093e-7. The preregistered 1e-6 tolerance passes with unchanged decisions;
probabilities derived with the common PyTorch sigmoid match exactly. This auxiliary rounding
difference is retained, not described as exact NumPy probability equality.

Cached research-run duration: 4.44 seconds with two CPU threads. It excludes
image feature extraction and is **not** end-to-end detector or CTF performance.

## Interpretation and next gate

Weighting raises low-payload recall, but neither low-rate cell reaches 80%;
false alarms now exceed 3%; calibration fails in every cell. Scattered-5%
still misses 129/184 stego files. All cells remain **experimental**: one
source, 184 pairs/cell, already inspected validation, unknown camera/device,
controlled methods rather than named upstream encoders.

Keep these models local. Next work needs stronger low-payload representations
and a preregistered calibration/operating-point assessment with separated
lineage roles; moving a threshold alone is not a sensitivity fix. Qualification
requires untouched independent-source validation and the full sample/gate
requirements, including named-method corpora. Never select a model from this
validation and present it as a newly blind held-out success.

Raw report SHA: `a51aa744b4907f794357d482f1649bff112197247ed682bfd49360d22d6e12df`.
Protocol SHA: `33c9aa4d81ad6095c9f4ab6b77c9fbb3401a35245ec5e35cbb6db592937bf911`.
Source/license inventory: [dataset catalog](DATASET_CATALOG.md).
Raw corpora, features, checkpoints and local configs remain outside Git.
