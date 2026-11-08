# Existing WSL connection and private cache handoff — 2026-10-28

The existing pinned-key SSH tunnel is restored. Read-only inventory confirms
WSL kernel 6.6.87.2, RTX 5060 Laptop (8,151 MiB), driver 617.42, Python 3.12.3,
Torch 2.14.0+cu130 and CUDA runtime 13.0, with CUDA availability true.
The observed occupied VRAM is 1,869 MiB; this is a point-in-time observation.
The nonlogin PATH lacks `nvidia-smi`; the existing WSL binary supplies the
inventory without installing or changing a driver/package/key.

The [handoff receipt](../benchmarks/png-bmp-native-wsl-handoff-20261028.json)
verifies a private copy of the existing 118-original cache: all 119 files,
21,591,572 logical bytes, including the unchanged hash-bound manifest and
all raw pixel hashes/extents. Source groups/attribution remain intact and
all roles stay unassigned. The host verifier only reads bounded existing
bytes: it decodes no images and constructs no model. Files are owner-only.

The destination relative to the WSL user's home is
`steganography-gpu/.benchmark/png-bmp-native-cache-handoff-20261028/cache`.
Kali's original cache and all 244 frozen application/document bindings are
unchanged. The cache was copied, not regenerated. Existing WSL worktrees,
checkpoints and consumed units were not changed. Raw pixels remain private.

This administrative check and local transport are separate from GPU profiling,
external acquisition and fitting. Preserve the original cache's unavailable
final-memory acceptance; successful transport does not upgrade that result.
Every historical transfer/metadata/exposure ledger remains unchanged.
