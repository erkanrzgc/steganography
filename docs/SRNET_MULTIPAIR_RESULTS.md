# Two-source/four-row SRNet control: implementation and accounting

The explicit control is implemented; real training and accuracy are unavailable.
No model is deployed and primary detector behavior is unchanged. The earlier
failed pilot and train-only normalization diagnostic remain published unchanged.

Frozen protocol: [SRNET_MULTIPAIR_PROTOCOL.md](SRNET_MULTIPAIR_PROTOCOL.md),
commit `837cb79`, SHA-256
`c84a1357195fdc59650679ed9878eef66d1f3c89784a69addd90beec894b031b`.
Portable evidence: [full accounting](../benchmarks/srnet-multipair-accounting-20261007.json).

## Opt-in contract

Set `"batch_recipe": "two-source-two-lineage-pairs-v1"` explicitly in both
`research srnet-plan` and `research srnet-fit` configuration files. Keep the
existing checksum-bound paths and optimizer settings. Omitting the field keeps
legacy v1 plans/cards and two-row training. A mismatched/unknown recipe fails;
four-row training requires exactly two declared sources.

Each minibatch contains cover/stego from one lineage per source, ordered
`[cover, stego, cover, stego]`. Source/quality/method balancing and paired order
are unchanged. Consecutive pairs are grouped without truncation or padding.
Plan/card v2 add ordered batch hashes and optimizer-update accounting; the
trainer reconstructs the complete bound schedule before updates. BN counters
count four-row updates, not constituent pairs. Complete evaluation obtains the
recipe from the checksum-bound plan and verifies model/card/BN accounting.
It still uses all-stage eval and stored BN statistics; no paired-BN inference.
The earlier two-row train-only diagnostic is not enabled for v2 cards.

## Complete real schedule audit

`scripts/audit-srnet-multipair.py` checks the frozen manifest and complete float
train cache, independently audits all batch identities/labels/source/lineage/
quality/order, and checks exclusion of every validation hash. It does not load
validation pixels or train a model.

- Train rows: 2,985; validation hashes excluded: 765.
- Complete epoch: 3,160 pairs, 6,320 presented rows, 1,580 four-row updates.
- Each source: 1,580 pairs; BOSS Q75/Q95 × JUNIWARD/UERD: 395 per cell;
  ALASKA declared-unknown quality × JUNIWARD/UERD: 790 per cell.
- All 1,990 original train stegos covered; oversampling is not new scene data.
- Plan SHA-256: `1fc01723c4f10d717bf7350e7c982d22f6adf58d32bcbda8be1fd5860a3ecea9`.
- Batch order SHA-256: `5e5cf224073c3c77e31cc0161a3fe5896faceee8a60113f165cb7e694ad23c2a`.

Generated tests exercise real gradient updates, exact original pixel mapping,
BN counts, isolated fitting, v2 complete evaluation and v1 compatibility.
They are regression/readiness evidence, not real detector accuracy.

Verification on Python 3.11: 1,149 tests passed, total coverage 94.89%; the
batch helper, training, plan, fit and eval services each reached 100% statement
coverage. Ruff, mypy (116 files), diff checks and wheel/sdist builds passed.
The wheel contains the new service/helper but no model weights or corpus.
Other Python versions and a full real-data ONNX replay were not rerun here.

## Remaining gates

Next run the frozen one-epoch bounded real job, then complete stored-BN eval
and independent NumPy replay. Timeout is failure, not permission to truncate or
extend the frozen budget. Publish all failed cells as well as successful ones.
Both training context and source/optimizer counts change relative to the prior
BOSS-only pilot, so this is not an isolated causal BN experiment. Declared
source separation is not verified camera/perceptual independence; the reused
small development validation cannot establish blind cross-source support.
