# Three-origin JPEG sampling contract

The opt-in `core.srnet_diversity_sampling.epoch_batches` service implements
`srnet-three-source-quality-method-four-row-v1`. This is metadata scheduling,
not a new trained detector or a ready-to-run cloud training command.

## Bounds and identity

Input must contain exactly three declared origins, training JPEG rows only,
unique bound byte identities, and complete cover/JUNIWARD/UERD families per
original lineage and quality context. One original lineage cannot be assigned
to multiple origins. All quality variants stay attached to their original.
Camera, scene and transformed-duplicate independence remain unverified.

The explicit metadata cap is 15,000 training rows. The acquired population's
prospective 12,222 training rows exceeds the historical 12,000-row contract:
7,380 existing rows plus 807 BOWS originals times six derivatives. This new
cap does not alter `TrainBlocks`, cache-byte, resident-memory, GPU allocator,
process-time or historical plan bounds. Validation rows are not accepted.

## Exposure

Reuse the shared deterministic pair service, equalizing source exposure then
source-local quality/method cells. Every original matched pair is included;
small contexts are oversampled, never large contexts silently truncated.
Every quality has two method families, so each source quota is even. The
three-source cyclic stream groups into AB / CA / BC pairs of pairs, providing
four-row batches containing distinct declared origins and original lineages.
All three unordered source combinations have equal batch counts. The existing
40,000-pair guard remains, and source balance does not mean flat global cell
balance or balanced independent cameras.

Returned arrays are immutable; pair and batch records bind their complete
ordered byte identities. No source metadata, global NumPy RNG, images, weights
or checkpoints are changed. With the intended source/context counts, one
epoch schedules 9,864 pairs and 4,932 optimizer updates, not the old 3,288.
These counts are independently tested using generated metadata; actual BOWS
JPEG derivatives have not yet been produced or trained.

## Compatibility and remaining gates

Historical `srnet_scale_sampling`, two-source grouping, frozen source closures,
old checkpoint plans and failed accuracy records remain unchanged. The new
service is deliberately not wired into the old trainer or its CLI. No download,
paid cloud provisioning, threshold change or model deployment is implicit.

Next implement a separately versioned preparation/index/audit/reader contract
and validate the complete real three-origin population. Freeze an unchanged
normalization control and single-file-compatible intervention before learning.
Remeasure physical GPU timing for the increased schedule; do not reuse the old
two-source eligibility estimate. Cloud execution also needs a verified kernel
resident-memory bound, explicit spending cap and checkpoint backup. Preserve
WIFD as reserved; previously inspected development scores cannot select models.

## Verification — 2026-10-10

Full regression against final application code: 1,830 passed, one actual-CUDA
hardware skip, 32 existing ONNX deprecation warnings, 457.61 seconds; total
coverage 95.66%. The full run collected the first twelve new tests; three extra
quality-lineage/budget/cycle tests were added afterward and all fifteen passed
separately, together with the old pair service's tests (54 focused passes).
Lint, types (147 application files) and whitespace checks pass. New sampler
line coverage is 100%. Wheel/sdist inspection (157/316 members) excludes private
corpora, caches, keys and model weights. A lint invocation during the package
build scanned its temporary sdist copy and failed on duplicate test paths;
after the build cleaned that temporary tree, the complete root lint passed.
Only Python 3.11 was exercised locally; no new physical GPU, other Python
versions, full Docker, preparation or real-learning gate is claimed.
