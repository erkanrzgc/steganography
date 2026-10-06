# SRNet numerical and isolated-job readiness

Generated engineering checks only; no real-corpus fit, accuracy improvement,
independent-source qualification or installed detector is implied.

## Fixed independent numerical contract

`core/srnet_reference` is a separate NumPy float64 forward implementation from
numeric-only weights: padded correlation, stored-statistics BN (epsilon 1e-5),
ReLU, residual sums, edge-corrected stride-2 average pooling, projected shortcuts,
global mean and classifier. It never calls Torch operators or model.forward.
Input is exactly one finite little-endian float32 256-pixel Y tensor, unchanged
pixel units. Stored shapes, counters and nonnegative variance are checked.

Comparison allows absolute logit error 1e-4 plus 1e-5 times absolute reference
logit, and requires identical decisions at class-1 logit >= class-0 logit
(softmax score >= .5, including ties), with an additional absolute class-1
softmax score bound of 1e-6;
large common logit offsets cannot hide changed margins/probabilities. This is
not real calibration. It is a new SRNet floating-operation
contract, not a relaxation of existing pixel-CNN/other detector gates. It is
fixed before any real-model replay. Record failures rather than changing the
tolerance after looking at a real result. Near-threshold flips always fail.

Independent scalar convolution/edge-average fixtures and a full 256-pixel
fractional model with deliberately nondefault BN statistics exercise the
separate calculation. Nonfinite outputs fail even when stored arrays are finite.

`core/srnet_onnx` explicitly exports numeric SRNet models with eval everywhere,
opset 17, dynamic batch, pixels/logits IO and no external data. Torch's bounded
legacy exporter is opt-in; it currently emits upstream deprecation warnings.
Export is <=32 MiB, fresh/non-symlink, checked by ONNX, checksum bound. Generated
CPU replay for batches 1/2/4 agrees with native logits and decisions under the
same contract; this does not prove every future trained model exports correctly.

Before initializing ONNX Runtime, the graph preflight rejects external tensors,
custom/control/shape-generation operators, malformed weights, duplicate attrs,
unbound topology, unsafe convolution/pooling and invalid classifier dimensions.
It independently propagates worst-case batch-4 shapes and caps sum of float32
node outputs at 1 GiB. Inputs, graph bytes/nodes/weights and replay output remain
bounded. The narrow graph is not a general arbitrary-ONNX execution API. Metadata
shape declarations cannot authorize oversized native allocations. This is not
an OS/filesystem sandbox for ONNX Runtime or a verified independent decoder.

## Isolated fitting

`research srnet-fit` now calls a fixed trusted-module job runner. The direct
Python `train_srnet` function retains its cooperative optimizer checks. CLI
jobs additionally get 8 GiB virtual address space, CPU seconds 2*max_seconds+60
(hard +1), 32 MiB per file and disabled core dumps; parent wall deadline is
max_seconds+120, including startup/cache/serialization. Math threads 1–2;
OPENBLAS is one thread. Worker stdout is read at most 64 KiB, stderr discarded.
Missing optional Torch/resource limits is unavailable, never successful.

The worker checks config SHA before fitting. It loads checkout/package-owned
code explicitly, never an artifact module. After a successful exit, the parent
checks complete card/model hashes, byte limits, plan identity and completed/
non-deployed state. Failure, malformed response, symlink, timeout or corrupt
artifact cannot return a completed model; partial output remains unusable.
Reports contain static redacted failures, not host paths or worker commands.
Subprocess timeout kills and reaps the fixed child. Resource limits are not a
filesystem/network sandbox; do not execute extracted data. No downloads occur.

Real child execution on generated fractional caches exercises training and
artifact verification under the actual limits. Separate hostile-output and
mocked-limit tests cover failure paths without changing the test runner limits.

## Remaining gates

Local verification: Python 3.11.14, 1,068 tests pass, 94.72% total coverage;
runner/reference/export/preflight 241/242 statements covered. Ruff, mypy
(111 source files), whitespace and model/data-free wheel/sdist checks pass.
Python 3.12–3.14 and fresh full Docker remain unverified. Optional dependency
absence, corrupt/oversized responses, unsafe graph allocations and changed
near-threshold decisions fail explicitly; none are counted as benchmark passes.

Model/card-to-complete-validation binding is now implemented as described in
`SRNET_EVALUATION.md`, with failed gates withholding predictions. Run it on
actual real-trained weights, add full ONNX validation, and freeze a realistic
real fitting/compute/source protocol before starting the experiment. Untouched licensed
external-source evaluation remains unavailable. Numeric success cannot replace
ROC-AUC, recall, false-positive, calibration, sample-size or source gates.
