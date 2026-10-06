# Unrounded JPEG preparation results — 2026-10-06

Full preparation completed on all unchanged **3,750 originals**: 2,985 training
and 765 validation rows, exact prior split/order/lineages. All original hashes
rechecked. Independently computed SciPy full-raster inverse DCT plus declared
crop agrees exactly at float32 on all 18 preregistered examples: cover/JUNIWARD/
UERD from ALASKA and BOSS Q75/Q95 in each split. Frozen tolerance 1e-4 absolute
/ zero relative, measured maximum 0. No tolerance adjustment or corpus rerun.

[Frozen protocol](JPEG_FLOAT256_PROTOCOL.md) / implementation commit `467f02f`;
[portable evidence](../benchmarks/float256-preparation-20261006.json) contains
all row/example identities, cache hashes, decoder and actual execution records.
The same native coefficient parser is shared; mathematical IDCT/crop is
independent, **native-parser correctness is not independently proved**.

| Split | Rows | Raw bytes | Extraction seconds |
|---|---:|---:|---:|
| Train | 2,985 | 782,499,840 | 44.86 |
| Validation | 765 | 200,540,160 | 12.27 |

Total raw data 983,040,000 bytes, 57.13 s orchestration with four workers.
Times exclude independent audit, not a full-training latency forecast. Existing
512×512 JPEGs have crop origin (128,128), width/height 256; no EXIF rotation,
RGB grayscale conversion, resampling, clipping, rounding or recompression.
Component Y and actual component-table assignment preserved. Libjpeg 6b /
jpeglib 1.0.2, NumPy 2.4.6; version selection does not prove identical binaries
on other hosts. Raw floats bind actual reconstruction here by SHA.

Training/validation cache SHA:
`828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028`,
`1c8c25244f204aaa0cad981d032241c06f7535b31bd03d7c547818dc046ae3a6`.
Raw tensor SHA:
`fd27c6da9869f4bea8bf7b97edeffd8ebae519824cd23d15b029aba25ceaf392`,
`81eecbd2b6bb2c7a54593dd86b89a1b7493b967955f844d1c4d843671913438a`.
Raw 1 GiB-per-split bound is separate from unchanged generic/uint8 limits. No
native/decoded/cache symlink, overwrite, executable artifact or model download.
Data remain local/ignored, not packaged or uploaded to the repository.

## Interpretation and next gates

Implementation verification: 955 tests pass on Python 3.11.14, 94.48% total
coverage; decoder/network/cache/persistence coverage 336/337 statements.
Lint, type checking, whitespace and model-free wheel/sdist checks pass.

Preparation passes; real SRNet training/accuracy remain unavailable. No untouched
source acquired, camera/device independence or ALASKA quality/payload supplied.
BOSS remains simulated and small, validation inspected, JMiPOD excluded.
All previous failed CNN/JRM results and primary detector behavior unchanged.
This is not reproduction of the paper's complete dataset/training/decoder.

Balanced sampler preparation now passes the independent accounting checks in
`SRNET_SAMPLING_RESULTS.md`. Next required work: provenance-bound SRNet training/card,
independent numeric/export parity and a frozen compute/data schedule, followed
by explicit licensed untouched-source evaluation. Do not infer accuracy from
lossless cache preparation or synthetic BN model-state roundtrips.

To replay the audit with the local explicitly prepared files:

```sh
venv/bin/python scripts/audit-float256.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --source .benchmark/jpeg-context-multisource-20261005 \
  --cache-root .benchmark/srnet-float-preparation-20261006 \
  --out .benchmark/srnet-float-preparation-20261006/audit-replay.json
```

Output must be fresh; no corpus extraction/regeneration required for audit.
