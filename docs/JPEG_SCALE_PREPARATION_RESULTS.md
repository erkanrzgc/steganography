# Complete JPEG expansion preparation — 2026-10-08

The [frozen protocol](JPEG_SCALE_PREPARATION_PROTOCOL.md) completed all 16
blocks, not a truncated subset. The [preparation summary](../benchmarks/jpeg-scale-preparation-20261008.json)
binds execution base `e7f43e1`, source/script/cache identities and the full local
index SHA-256. The [independent audit](../benchmarks/jpeg-scale-independent-audit-20261008.json)
reread every JPEG and streamed all tensor checksums/finite-value checks.

| Declared origin | Prepared JPEG rows | Train rows | Validation rows | Original lineages |
| --- | ---: | ---: | ---: | ---: |
| ALASKA2 | 2,991 | 2,448 | 543 | 997 |
| BOSSbase-1.01 | 6,000 | 4,932 | 1,068 | 1,000 |
| Total | 8,991 | 7,380 | 1,611 | 1,997 |

These are all eligible originals from the new expansion. ALASKA copies preserve
original bytes, roles and cover/JUNIWARD/UERD families. Three unchanged
cover/stego lineages remain excluded as whole groups. JMiPOD remains acquired
but is explicitly excluded from this matched-family corpus: no matching BOSS
simulator in the pinned recipe. It is not silently advertised as trained.

BOSS uses the unchanged conseal 2025.11 recipe: Q75/Q95, JUNIWARD/UERD,
simulation input .2 bpnzAC and seed 20261006. Two qualities and stego descendants
of one original are correlated, not independent new scenes. ALASKA quality,
payload, camera/device and scene independence remain undeclared/unverified.
Train/validation roles preserve all original ancestry. WIFD and six prior
acquisitions have zero identity overlap and stay excluded; the old 3,750-row
prepared corpus/caches remain untouched and were not merged into this index.

## Independent evidence and limits

- All 8,991 JPEGs: byte/hash/format/512-square geometry, complete family/source
  membership, original roles, disjoint hashes and unsplit block ancestry checked.
- All 2,000 BOSS cover JPEGs: independently re-encoded from PGM ancestry.
  All 4,000 simulated stegos: independently checked for nonzero +/-1 DCT changes,
  exact declared change counts, unchanged quantization and .2 bpnzAC metadata.
  This does not independently rerun each simulator's cost/probability decisions,
  establish actual encoded message length, or prove payload extraction.
- All 32 split cache files: checksum/row/crop/shape/decoder contracts and every
  float's finiteness/range checked by bounded streaming reads. Four independent
  scalar IDCT sums in each of nine metadata-selected contexts pass; maximum
  observed difference after float32 conversion is 0 (fixed tolerance 2e-4).
  These 36 sampled pixel values are not a full independent pixelwise replay.

Unrounded block-aligned center256 Y crops retain the existing decoder contract:
jpeglib 1.0.2, libjpeg 6b, NumPy 2.4.6, no clipping/rounding/resizing/augmentation.
Tensor bytes 2,356,936,704; JPEG bytes 663,910,882; complete directory before
index write 3,154,875,454 bytes. This includes manifests/context feature artifacts,
not just JPEG/tensor bytes. All raw/derived images and tensors remain local.

Preparation took 1,746.49 seconds on the CPU-only host, eight logical CPUs,
about 15 GiB RAM, two isolated single-thread simulation workers / two cache
workers. This is research preparation timing, not CTF latency or training speed.
The service checks its deadline at block boundaries; the mandatory external
3,610s process-group timeout supplies the strict whole-job wall bound. Decoder
workers retain 15s and simulation workers 90s bounds. No parent/worker is left
running. Direct library callers must provide an equivalent hard parent bound.

## Training is still separate

No model was fitted, exported, installed or evaluated. Real-data failures and
chance-level prior results remain unchanged. Prepared-row counts, coefficient
checks and numerical preprocessing replay are not accuracy/qualification.
WIFD is still an unprepared, reserved unseen-origin candidate, not a passed
cross-source test. Its native high-resolution/MPO format needs a separate
bounded preprocessing/evaluation plan; no primary-frame conversion occurred.

The old whole-array trainer/sampler caps stay at 4,000 rows. The complete block
index is deliberately **not** accepted as a legacy training input. Next implement
a versioned minibatch/streaming reader and trainer, propagate the whole-job
deadline into workers and snapshot execution sources at job start, preregister
adequate exposure/learning gates, then run learning and independent-source tests.
GPU/cloud access remains unconfirmed; no paid/cloud training or upload occurred.
More data alone does not establish that the earlier weak-signal learning failure
has been repaired.

Verification: Python 3.11, 1,478 tests passed, total coverage 95.28%. Focused
layout/preparer/simulator coverage >=95% (layout 100%, orchestration 98%);
independent audit 236/241 statements (97.93%). Ruff, mypy (132 source files),
diff checks and wheel/sdist build pass. Other Python versions, GPU fitting and
full Docker E2E are unverified for this slice.
