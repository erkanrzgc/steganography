# Numeric epoch checkpoint core — engineering only

Status update, 2026-10-10: an explicit isolated controller and final-development
validation service now use this core; see `EPOCH_PILOT_VERIFICATION.md` and
`CUDA_EPOCH_RESUME_RESULTS.md`. The original core-only slice's scope and
verification below are historical. The real five-epoch pilot completed and
failed all six detection cells; see `JPEG_EPOCH_LEARNING_RESULTS.md`.
Physical resume parity is not detection qualification.

The shared four-row optimizer supports an explicit `segment` contract:
`binding` (SHA-256 of the complete caller-bound plan), `stop_epoch` (exclusive
stop within the complete planned epoch count), `resume` (verified numeric
snapshot or null), and `sink` (a passive snapshot collector). Legacy callers
omit it and retain the existing return contract, arithmetic and defaults.
`core.srnet_stream_training.fit` accepts the same optional contract. Accumulated
eight-row controls are deliberately unsupported by this checkpoint version.

Every segment completes whole epochs. A timeout or failed update is unusable;
it does not publish a partially completed epoch or automatically retry. A resume
starts at the exact next epoch; it restores model/BN state, all Adamax first/
infinity moments and scalar steps, and Torch CPU/CUDA RNG bytes. The complete
ordered four-row schedule, seed, settings, backend, Torch version and opaque
plan binding must agree. All optimizer and BN counters equal the cumulative
scheduled updates. No state is inferred from filenames or silently reset.
Fetch/schedule callbacks must be deterministic apart from explicitly captured
Torch RNG; Python/NumPy global RNG state is not captured. Production block
readers and independently seeded schedules satisfy that requirement.

## Numeric file contract

`core.srnet_checkpoint.save/load` uses an explicit SHA-256-bound stored NPZ,
not pickle, compressed archives, executable code or arbitrary model classes.
Exact named-array keys, float32/int64/uint8 types and architecture-fixed shapes
are verified before NumPy allocation. The loader rejects duplicate/unknown
members, ZIP symlinks, compression, oversized or malformed headers/data,
objects, negative infinity-norm moments, invalid BN values and wrong counters.
Metadata is strict, <=64 KiB; entire archive and declared numeric data <=96 MiB,
<=512 members. Files and ancestors must not be symlinks; output is fresh and
never overwrites an existing file. Restoring RNG uses the exact Torch build and
byte count; malformed runtime state errors leave the shared engine's caller
RNG/thread/precision policies restored. Files are never extracted or executed.

Actual producer metadata has no host path, key, password or environment secret;
the snapshot contains private numeric training state, not a portable public
accuracy report. Do not commit/distribute checkpoints or put them in the base
wheel. Caller plan bindings must include data/audit/decoder, all schedules and
all executed sources, including `core/srnet_checkpoint.py`.

## Scope and remaining gates

This slice supplies the core and numeric persistence, not an end-to-end
isolated resumable CLI, trained real model or accuracy gain. Library callers
must supply existing hard OS resource bounds and a whole-job deadline. A sink
should only collect state; publication must wait for the controller's deadline
and before/after source/input checks. The existing CLI is not silently changed
to resume or bypass its failed five-epoch single-job budget gate.

Next implement/verify a separate source/input-bound per-epoch job controller,
96 MiB checkpoint file bound, immutable parent/checkpoint chain and exact
per-job completion checks. Keep every job <=1800s, host RAM <=8 GiB and CUDA
allocator <=4 GiB; no automatic restart, model downloads or deployment. Freeze
an independent generated physical CUDA resume-parity protocol before execution,
then a full-source train-only learning/evaluation protocol before real fitting.
Earlier single-job timing failures and real detector failures remain unchanged.

## Local verification — 2026-10-10

Python 3.11.14 / CPU Torch on Kali: all 54 checkpoint tests pass, including
exact full-versus-disk-resumed model/BN/optimizer/loss/RNG comparisons, malformed
runtime RNG rejection and hostile archive preflight. The full suite passes
1,719 tests, with one actual-CUDA test explicitly skipped as unavailable and
32 existing ONNX deprecation warnings, in 301.75s. Total line coverage is
95.53%; checkpoint, shared optimizer and block training files are all 100%.
Ruff, mypy (141 files) and whitespace checks pass. Fresh wheel/sdist builds
contain 151/303 members; inspection excludes private caches, keys, raw tensors
and model weights. Only this local Python version was exercised in this slice;
the multi-version CI and physical GPU resume experiment remain separate gates.
CPU emulation of CUDA checkpoint branches is control-flow evidence only.
