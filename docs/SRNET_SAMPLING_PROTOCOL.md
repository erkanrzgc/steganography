# SRNet training-schedule preparation — 2026-10-06

Freeze this preparation contract before auditing the reused real corpus. This
is not a fitting, loss, accuracy or independent-source qualification protocol.
No optimizer schedule, trained model or deployable detector is implied.

## Fixed inputs

Reuse the exact `JPEG_FLOAT256_PROTOCOL.md` manifest, unchanged 2,985 train
rows and checksum-bound unrounded Y/256 cache. Validation (765 rows) is never
loaded by the schedule service. Preserve previous failed models/results.
Use seed 20261012, ten epochs; evaluate all-source and each of the two existing
single-source controls. A source-exclusion control is not an untouched-source
test: this is previously inspected development data with limited metadata.

## Deterministic pair contract

`srnet-source-quality-method-pairs-v1` groups train JPEG rows by declared
source, cover lineage and declared quality (unknown remains its own cell).
Exactly one cover and both JUNIWARD/UERD derivatives are required per group.
Duplicate hashes, absent/extra methods or covers, non-training rows, invalid
quality and incomplete excluded-source groups are rejected. All original
matched pairs must appear at least once in each epoch; no dropped tail.

Let source s have K_s quality/method cells and maximum original cell size M_s.
L = lcm(K_s). Per-source quota is the smallest multiple of L at least
max_s(K_s * M_s). Each source gets that quota; its cells receive quota/K_s
pairs apiece. This balances sources FIRST, contexts WITHIN source, not flat
cells across origins. Small cells are deliberately oversampled; report that
duplication, do not mistake it for more independent scenes or training data.

Independent local NumPy RNG uses SeedSequence([seed, zero-based epoch]). Sort
source IDs lexically, cell keys by repr; randomly permute cells and repeatedly
shuffle each cell without replacement to its quota, interleave cells, then
sources. Emit ordered (cover, stego) indices; the intended CPU minibatch is
one pair, labels [0,1], float32 pixel units. No division by 255, metadata as
model features or validation-driven weighting. Save ordered pair SHA-256 and
original/sampled counts per cell in every epoch. Global NumPy RNG is unchanged.

Limits: 4,000 rows, eight declared sources, 40,000 pairs/epoch, 50 epochs,
seed unsigned 32-bit. Expansion beyond limits fails, never truncates. All
configuration keys are explicit and allowlisted. Full float-cache integrity
is checked before a plan is published; symlinks/overwrites are forbidden.
Reports include hashes and opaque source IDs, never absolute host paths.

## Independent preparation audit

On all three scopes and ten epochs, independently regroup original train
metadata and verify cover/stego labels, lineage/source/quality agreement,
complete original positive coverage, exact source and within-source cell
counts, and independently recomputed ordered SHA-256. Audit replays must be
deterministic and exclude every validation hash. Save all epoch records and
protocol/implementation identities. Failures remain visible and stop the
next gate; no weighting adjustment after inspecting results.

Next separate gates: provenance-bound training/card, fixed compute/optimizer
schedule, independent forward/export parity, real fitting and licensed unseen
source evaluation. Sampler completion cannot satisfy any detection metric.
