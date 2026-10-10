# Frozen layerwise population-BN control — 2026-10-10

Freeze before execution. Use the final five-epoch model SHA-256
`71f6f5fe8730944c38d406362bf26e382f348f302b03c7f30809991db73e44b6`
and the existing independently audited 480-row three-origin timing kit.
For each origin, sort its 32 original lineage hashes: first 24 calibrate,
last eight probe. Keep every quality/method derivative with its original.
Expected 360 calibration / 120 probe rows, 72 / 24 original groups. These
are selected historical training originals, NOT untouched validation/test.

Compare the unchanged stored-BN model with a private cloned intervention.
For each of the 26 BN layers in forward order, estimate input mean and
population variance across all 360 calibration rows using bounded four-row
fetches, GPU float64 central moments and sample-count-weighted merging.
Previously calibrated upstream layers stay in eval mode. Stop the forward
at the target layer; do not update all layers using train-mode upstream BN.
Only running mean, variance and counters may change. All learned weights
and the source model remain bit-identical; zero optimizer updates.
This differs from the failed simultaneous cumulative minibatch refresh.

Score all 120 probe rows with ordinary singleton inference, no cover partner
or batch adaptation, fixed probability threshold 0.5. Publish all ten
source/quality/method cells, confusion counts and cross-entropy for both arms.
Compare four-row versus singleton logits on every probe row at atol/rtol
1e-4; mismatch is an engineering failure, never silently relaxed. Refresh
may fail or regress. No held-out/generalization/accuracy-gate claim.

CUDA only, deterministic IEEE FP32 model operations (no AMP/TF32); float64
statistics explicitly separate. Host cgroup <=8 GiB, GPU allocator <=4 GiB,
job <=1800 seconds, outer timeout 1830 seconds, CPU time <=3600 seconds,
file-size limit 32 MiB. Read/reverify bound inputs, model and execution
sources before/after. No overwrite, pickle, automatic deployment, external
downloads or cloud resources. Save only private numeric clone plus portable
aggregate evidence; never publish weights, raw media, host paths or secrets.
