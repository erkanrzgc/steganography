# Matched BN/GN learning pilot: failed detection improvement

Both frozen arms completed on the user's RTX 5060 Laptop GPU on 2026-10-10.
Every method/context cell remains at **50% balanced accuracy**. Changing
normalization alone did not improve fixed-threshold singleton detection in this
small experiment. Neither checkpoint is deployed or qualified.

## Scope and controls

The [preregistered protocol](JPEG_NORM_LEARNING_PROTOCOL.md) uses 480 audited real
JPEGs from ALASKA2, BOSSbase and BOWS2. Each arm fits 360 rows /72 originals and
probes 120 rows /24 different originals, with all derivatives kept together.
The probe was already inspected in earlier controls and all rows have the
training role. It is **not blind validation**, cross-source held-out evidence,
or a representative estimate of deployed accuracy. ALASKA payload/quality are
unknown; BOSS/BOWS controlled JUNIWARD/UERD derivatives use 0.2 bpnzAC.

Both models start fresh with the same learned parameters, seed, ordered
four-row schedules and Adamax settings. Each completes eight fixed epochs of
144 updates (1,152 total). Only final-epoch singleton outputs are compared;
no best epoch, threshold, source or method was selected from probe outcomes.
BN delegates to the unchanged historical optimizer; GN uses a separate tagged
26-layer eight-group architecture. Existing detector behavior is unchanged.

## All ten cells

Each cell contains eight covers and eight stegos. Covers are shared across
method cells, so cell counts must not be pooled as independent samples.
Decision uses the larger of the two raw logits, not a fitted threshold.

| Origin | JPEG quality | Method | BN balanced accuracy | GN balanced accuracy | BN FPR | GN FPR | BN loss | GN loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALASKA2 | unknown | JUNIWARD | 50% | 50% | 37.5% | 25% | 0.693719 | 0.693146 |
| ALASKA2 | unknown | UERD | 50% | 50% | 37.5% | 25% | 0.693539 | 0.693144 |
| BOSSbase-1.01 | 75 | JUNIWARD | 50% | 50% | 25% | 12.5% | 0.694257 | 0.693149 |
| BOSSbase-1.01 | 75 | UERD | 50% | 50% | 25% | 12.5% | 0.694253 | 0.693147 |
| BOSSbase-1.01 | 95 | JUNIWARD | 50% | 50% | 25% | 12.5% | 0.694290 | 0.693148 |
| BOSSbase-1.01 | 95 | UERD | 50% | 50% | 25% | 12.5% | 0.694286 | 0.693148 |
| BOWS2 | 75 | JUNIWARD | 50% | 50% | 62.5% | 12.5% | 0.696123 | 0.693146 |
| BOWS2 | 75 | UERD | 50% | 50% | 62.5% | 12.5% | 0.696046 | 0.693147 |
| BOWS2 | 95 | JUNIWARD | 50% | 50% | 62.5% | 12.5% | 0.696006 | 0.693148 |
| BOWS2 | 95 | UERD | 50% | 50% | 62.5% | 12.5% | 0.696001 | 0.693148 |

All cells have TP equal to FP in each arm: changes in positive calls affect
covers and stegos equally. GN's lower loss and fewer false alarms come with
lower recall, not improved balanced accuracy. Near-chance final training loss
also gives no evidence of effective separation: BN
0.702917 → 0.693155;
GN 0.716799 → 0.693147.

## Physical execution and independent replay

BN took 177.27s; GN took 203.42s. Each used under
1.0 GiB peak allocated and 1.35 GiB peak reserved process GPU memory, with
4 GiB allocator and 8 GiB host caps, strict FP32, no AMP/TF32 or CPU fallback.
Singleton/four-row parity passes for both. These are measured pilot durations,
not full-corpus timing or cloud cost projections.

Portable original reports: [BN](../benchmarks/jpeg-norm-learning-bn-20261010.json)
and [GN](../benchmarks/jpeg-norm-learning-gn-20261010.json).
The [independent scalar audit](../benchmarks/jpeg-norm-learning-independent-metrics-20261010.json)
binds original report, manifest and preprocessing-audit SHA-256 values, verifies
matched initialization/schedules/optimizer and independently recomputes all
20 confusion matrices and stable logistic losses from the 120 raw singleton
outputs in each arm using Python scalar math. It does not reuse the scoring
function, perform an independent whole-network forward pass or qualify accuracy.
Snapshots remain private; no restricted corpus or model weights are published.

## Interpretation and next learning gate

Read-only follow-up checks all 240 fit cover/stego crop pairs, with no probe
pixels or optimizer steps. Three of the 24 ALASKA/UERD fit crops are exactly
identical to their covers; all other fit pairs differ. This is partial spatial
signal coverage, not corrupt labels or proof that original JPEGs are identical.
The 256-square crop may omit changed content or the selected luma channel may
have no changes. Determining which requires a separate whole-carrier inspection.
It does not explain the other nine cells' chance outcomes. No old inputs,
labels, schedules or scores are silently rewritten. See the
[bound crop-signal report](../benchmarks/jpeg-train-crop-signal-20261010.json).
Reproduce with `core.jpeg_crop_signal.inspect_fit_signal(root, audit_path)`;
the reader re-verifies the fixed audited cache before and after all pair reads.
This is an input-difference check, not an independent IDCT or learning oracle.

This rejects a detection improvement from this specific eight-epoch GN recipe;
it does not prove that GN, the data, or the architecture can never work.
The experiment does not isolate insufficient optimization, weak residual
features, crop/payload signal or limited data diversity as the sole cause.

Before a larger expensive fit, freeze a train-only real-pair learnability control:
check that audited matched cover/stego inputs differ after preprocessing, measure
whether a small fixed subset can actually be separated by optimization, and keep
unchanged controls and singleton evaluation. A memorization success would only
validate learnability, not real-world detection. Then integrate the bounded full
three-origin reader and separately freeze untouched validation/calibration.
Do not repeatedly tune this inspected probe or activate either failed model.
