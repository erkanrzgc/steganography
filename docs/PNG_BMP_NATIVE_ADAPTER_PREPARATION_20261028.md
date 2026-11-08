# Native spatial data adapter preparation — 2026-10-28

This continuation prepares software and reviews already published metadata.
It does not start a native fit, GPU profile, download or consumed experiment.
Actual source roles remain unassigned. The software/group-review stage opened
metadata and generated fixtures only. The later separately bound
[local cache preparation](PNG_BMP_NATIVE_CACHE_RESULTS_20261028.md) decoded the
118 admitted crops without a model or optimizer. Cache contents verify;
strict final-memory-peak acceptance is unavailable. No reserved corpus or
private model was opened. The existing generated-only control remains consumed.

## Known ancestry and inventory review

The [metadata-only report](../benchmarks/png-bmp-native-group-feasibility-20261028.json)
binds snapshot 016, SHA-256
`9f22a91dcc57046931d87c51586f9f13d70c80d988d6b9bad4153905c35acc19`.
The service merges transitive group, normalized author and normalized Commons
page connections globally across modes. An independent pairwise graph traversal
reproduces every component. Actual metadata has 75 known components and no
component shared between modes; that does not rule out unknown common origins.
Generated fixtures additionally verify a transitive cross-mode connection.

| Native mode | Originals | Known groups | Largest ten groups, originals | Minimum known groups covering 80 | Missing to 128 originals /16 groups |
|---|---:|---:|---:|---:|---:|
| L | 13 | 6 | 13 | unavailable | 115 /10 |
| RGB | 105 | 69 | 41 | 44 | 23 /0 |

These are capacity bounds, not a split assignment or a proof of a joint
train/development/reserved partition. The historical ten-group /80-original
train proposal remains infeasible. Adding twenty-three RGB originals can
raise its current 41-original upper bound only to 64. Future grouping must
be specified separately; total original counts alone do not establish it.
The historical draft, all holds/exclusions, closed reserved datasets and
exhausted transfer/metadata ledgers remain unchanged.

## Reusable data adapter

`core.spatial_native_data` owns immutable source/cover references, grouping,
hypothetical proposal validation, raw-pixel cache preparation and paired FP32
batches. It has no network, optimizer, Torch import or GPU entry point.

- Input sources require CC0/CC BY attribution, evidence/source/crop hashes,
  native L/RGB mode and unassigned roles. Duplicate IDs, originals and decoded
  crops are rejected. The catalog holds at most 128 originals per mode /256
  total; cache metadata is bounded to 2 MiB.
- Cache preparation accepts an explicit list of hash-bound `*-cover.png` or
  `*-cover.bmp` files only. Each must already be the fixed 256×256 crop. It
  verifies the encoded hash, native mode/depth and decoded crop hash without
  rescaling, converting RGB to L, normalizing intensity or using old 32-byte
  admission controls as training covers. PNG/BMP encodings represent one
  original and cannot both enter as independent sources.
- Caches contain bounded raw HWC uint8 bytes and a hash-bound manifest, with
  no pickle, executable data or attacker-controlled array header. The 64-MiB
  aggregate output limit is checked before creating the output. Existing or
  partial output directories cannot be overwritten; symlinks and nonregular
  inputs fail closed. Failed preparation remains partial, not ready.
- Loading validates manifest fields, provenance and byte accounting. Each
  requested source is read with an exact byte limit and checked against its
  crop hash before constructing a small array. Sources remain in their native
  mode and full 0–255 alphabet, including naturally odd plane-0 samples.
- Native controls use a distinct `native-spatial-scattered-40p0-v1` identity:
  source original hash, mode and recorded uint32 payload seed determine random
  payload and scatter order. Byte-rounded 40% plane-0 embedding and the scalar
  decoder reuse the existing shared services. Controls preserve all upper
  seven bits and every sample outside the selected positions. Payload identity
  does not depend on catalog position or a hypothetical role.
- Batches contain one to four distinct original/control pairs of one mode,
  ordered cover/control, FP32 NCHW with int64 labels. They contain no optimizer
  logic. A separate future native runner must bind the globally merged ancestry
  components and validate its proposal before opening role-specific data;
  original provenance group labels alone are not that binding.

Generated fixtures use full-alphabet PNG/BMP covers in both modes. Independent
NumPy packing verifies payloads separately from the scalar decoder; tests also
cover ancestry leakage, corrupt/truncated/oversized data, incorrect hashes,
native mode mismatch, symlinks/FIFOs, overwrite and pre-I/O output bounds.
All 37 focused tests pass with 100% statement coverage. The complete run has
4,044 passes /one README-length failure /two CUDA skips, 97.15% total coverage.
After a README-only repair, all fifteen documentation/cloud-removal tests
pass with unchanged application/test hashes. The
[software receipt](../benchmarks/png-bmp-native-adapter-verification-20261028.json)
retains both outcomes; the original full command is not relabeled as passing.

## Readiness

| Component | Status | Remaining work |
|---|---|---|
| Metadata grouping/capacity bounds | hazır | Does not prove unknown ancestry absent or assign roles |
| Native-shaped cache/pair/batch software | hazır | Verified with generated fixtures only |
| Actual native cache contents | hazır | 118 admitted covers, 236 scalar recoveries and 31 independently replayed FP32 batches |
| Actual cache final-memory acceptance | unavailable | Preserve failed closing observer; no regeneration/retry |
| Source-complete corpus and grouped split | eksik | Inventory deficits and historical ten-group infeasibility above |
| Native optimizer runner and numeric continuation | eksik | Bind actual data/role/order/exposure and existing numeric state services |
| Native integrated I/O and full-fit budget | unavailable | Need a separately bounded, approved zero-update measurement contract |
| Native model accuracy or deployment | unavailable | No native fitting or qualification performed |

The user may restore the existing WSL connection independently. Starting WSL
or a tunnel grants no new GPU attempt. The completed generated profiles/fits
are not rerun. No fresh acquisition allowance is inferred from this adapter,
the earlier draft or disk cleanup.
