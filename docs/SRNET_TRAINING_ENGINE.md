# SRNet paired CPU training engine

Explicit research-only fitting is available through the shared service:

```sh
steganography research srnet-fit --config CONFIG.json --out FRESH_MODEL_DIR
```

This implements fitting, not an independently audited forward/export path,
paper reproduction, benchmark score or deployed detector. Verification initially
used generated fixtures. The first separate frozen real pilot has now completed;
all detector cells failed. See `SRNET_REAL_PILOT_RESULTS.md`.

## Bound input and output contract

Config requires `manifest`, `manifest_sha256`, complete train `cache`,
`cache_sha256`, previously prepared `plan`, `plan_sha256`. Optional optimizer
fields are `threads`, `max_seconds`, `learning_rate`, `weight_decay`; other keys,
including validation inputs, are rejected. Manifest/cache provenance, every
training pair including excluded-source rows, scope and every epoch order are
recomputed against the same shared schedule contract before any update. A
rehashed forged plan, wrong decoder or changed order cannot authorize learning.
Legacy uint8 caches and other architecture plans are not accepted.

Training keeps the full bounded float cache read-only. Selected indices map to
original cache rows; only each two-row pair is copied, not a second full corpus.
Each CPU float32 minibatch is (cover, stego), targets int64 [0,1]. Inputs remain
unrounded component-Y pixel units; no /255, clipping, learned label metadata,
validation normalization, shuffling outside the plan or unpaired tail batch.

Adamax, betas (.9,.999), epsilon 1e-8, foreach disabled, constant learning rate.
Defaults lr .001, weight decay .0001, two math threads, 1,800 seconds of the
optimizer section. Bounds: threads 1–2, seconds 1–1,800, lr 1e-6–.01,
decay 0–.1. Plan supplies seed and 1–50 epochs; sampler caps 40,000 pairs/epoch.
These implementation defaults alone are not a preregistered experiment; the
first real pilot freezes them separately in `SRNET_REAL_PILOT_PROTOCOL.md`.
No early stopping, best-validation checkpoint, augmentation, resume or scheduler.

Direct Python fitting has cooperative checks before/after bounded two-row
updates; cache validation/IO is outside that clock. CLI fitting now also uses
the hard-limit isolated runner in `SRNET_NUMERICAL_READINESS.md`, including
startup/cache IO in its wall deadline. Interrupted/expired/invalid fits cannot
return a completed model; incomplete outputs must not be used.

Every update checks finite two-class logits, loss, all gradients and exact
weights/BN buffers including nonnegative variance/counters. CPU Torch RNG and
thread settings are restored on success and failure. Training BatchNorm sees
only the declared pairs; final snapshots require eval everywhere and preserve
running means, variance and counters through bounded pickle-free NPZ storage.

Output: `model.npz` and additive research `model-card.json`, bound to manifest,
complete cache, plan, data, scope, per-epoch order and model SHA-256. Card records
Torch version, optimizer, timing, exact update counts and mean training-pair
loss. It says training completed, not real-world qualification; accuracy and
independent forward audit remain unavailable, calibrated/deployed false.
No host paths or copied config strings are stored. Symlinks and overwrites fail.
An incomplete directory without a valid complete card is not a usable model.

## Verification and remaining gates

Python 3.11.14: 1,034 tests pass, 94.62% total coverage; training/fit/plan code
172/172 statements covered. Final strengthened source-index tests: 39 focused
tests pass. Ruff, mypy (107 source files), whitespace and model/data-free
wheel/sdist checks pass. Other Python versions/fresh full Docker not rerun here.

Generated 256-pixel fractional fixtures exercise actual gradients, parameter
updates, source exclusion, unnormalized inputs and persisted trained BN state.
Independent forged-plan/config checks and injected NaN logits/loss/gradients/
weights/timeouts must fail without publishing a model, while restoring RNG and
threads. This is engineering regression, not real detection evidence.

Isolated long-job limits and independent numerical forward/export now have
generated readiness checks in `SRNET_NUMERICAL_READINESS.md`. Actual real NumPy
evaluation and separately frozen pilot fitting are now documented in
`SRNET_REAL_PILOT_RESULTS.md`; full ONNX and longer/balanced fitting remain pending.
Untouched licensed external-source evaluation is still unavailable. All prior
failed results stay published, and no primary analyzer or installed model changes.
