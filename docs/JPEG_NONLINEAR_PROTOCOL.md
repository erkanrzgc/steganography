# Fixed nonlinear residual-feature comparison — 2026-10-05

Freeze this protocol and implementation before training. This is a bounded
64-unit ReLU feature network with a linear skip, **not a raw-residual CNN,
SRNet, DCTR or an installed/context-invariant detector**. It tests learned
interactions in existing pooled residual/content/quantization statistics.

Reuse exactly the frozen 2,066-feature caches and complete split/row order of
`JPEG_RESIDUAL_PROTOCOL.md`: 2,985 training / 765 validation rows, ALASKA and
BOSS JPEG simulations, JUNIWARD/UERD, original lineages and Q75/Q95 derivatives.
Manifest SHA-256:
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
Train feature SHA-256:
`0ae34f4761587fd03e5074361ed8d2bb355048f454fa1a27fadd555ae592d996`.
Validation feature SHA-256:
`25409e9e36bde0ef806d74906f01de3ea5f08bc53e63d21768e306f3fce8b446`.
No downloads, regenerations, exclusions, new thresholds or held-out test access.
Previously inspected validation and shared training sources remain development,
not blind or independent-source evidence.

Fixed `residual-mlp64-v1`: logit = linear(x) + output(ReLU(hidden(x))).
Hidden width 64, no dropout/batch normalization/shape search. CPU float32
training, seed 20261007, 300 full-batch Adam epochs, LR .001, weight decay .001
(all parameters, including biases). Same train-only standardization with scale
floor 1e-4 and source/class-balanced objective. Two math threads. Inference
uses explicit float64 accumulation and float32 I/O. Versioned domain and
architecture persist in checkpoints and model cards; legacy defaults stay linear.

Cutoff .5, no calibration, early stopping or best-run selection. Publish every
source, method, quality/pooled cell, failures and regressions against the frozen
linear residual model. Unknown quality stays unavailable. Verify normalization,
independent NumPy skip/ReLU math, confusion/AUC/ECE and CPU ONNX batches
1, 17 and all 765 rows at unchanged abs 1e-6 / rtol 0 with identical decisions.
Use existing lineage-bootstrap diagnostics; do not count correlated Q variants
as independent original scenes. Model remains local research-only regardless
of outcome. No model installation, primary score or support change.
