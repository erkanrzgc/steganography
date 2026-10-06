# JPEG pixel-residual preparation — 2026-10-05

**Preparation verified; detection accuracy remains unavailable.** The
[preprocessing/network contract](PIXEL_RESIDUAL_PREPARATION_PROTOCOL.md) was
committed as `a6d0f2b` before real extraction. [Portable evidence](../benchmarks/pixel-residual-preparation-20261005.json)
contains cache hashes, decoder versions, audit examples and actual timing.
No real-corpus CNN has been fitted, exported, calibrated, installed or deployed.

All original 3,750 JPEGs and train/validation identities remain unchanged.
No downloads, new scenes, label/source-dependent inputs, regeneration, exclusions
or test access. Exact center 128×128 Pillow-L crops, no resizing or EXIF rotation:

| Split | Rows | Shape | Raw bytes | Extraction time |
| --- | ---: | --- | ---: | ---: |
| Train | 2,985 | N×1×128×128 uint8 | 48,906,240 | 12.89 s |
| Validation | 765 | N×1×128×128 uint8 | 12,533,760 | 3.28 s |

Whole extraction 16.18 s, four bounded workers on the previously documented
eight-vCPU Ryzen 9 8945HX Linux VMware guest. This is cached-corpus preparation,
not training, CTF latency or per-image detector timing. Decoder reports Pillow
12.2.0 / JPEG API codec 6.2; this is not a binary-level cross-platform decoder
equivalence guarantee. Exact cached-byte SHA-256 is retained.

Independently rehashed every original file and validated complete descriptor/data
identity, split, dimensions, dtype, exact length and SHA. Eighteen real examples
cover both splits, origins, families and declared quality variants: bounded
independent full-raster slicing agrees exactly with cached crops. Both oracles
share Pillow's native decoder; geometry/indexing parity is not an independent
JPEG algorithm proof or an accuracy gate.

The custom optional CNN learns on three fixed high-pass residual maps, not the
old 2,066-dimensional summary vector. Independent NumPy sliding-window filter
math agrees with Torch; an actual optimizer/gradient smoke step leaves fixed
filter buffers unchanged. Bounded eval inference matches direct forward and
batch-1 outputs at unchanged absolute 1e-6 tolerance. These generated-fixture
tests and random/smoke weights are **not real-world model performance**.
The building block has no model file loader or deployed analyzer integration.

Security tests include malformed frames, oversized/truncated input and output,
failed/timeout workers, integrity/region/decoder/shape tampering, giant declared
dimensions, symlinks, overwrite, cache limits and split deadlines. Incomplete
files never acquire a complete descriptor; no pickle or extracted code executes.
Raw uint8 caches are separate from generic feature JSON/checkpoint contracts;
existing 64 MiB JSON/4,096-dimension model limits remain unchanged.

Python 3.11: 824 tests, total coverage 94.19%. New pixel/network/cache code covers
190/191 statements (99.48%). Ruff, mypy (97 application files), diff checks and
wheel/sdist build pass. Wheel includes pixel services, no raw caches, models,
corpus or vendored upstream packages; Torch remains optional. Python 3.12–3.14,
fresh full Docker E2E and real-corpus CNN training/inference are unverified.

Explicit preparation CLI is in the frozen protocol. Replay the complete audit
without training, with fresh output:

```bash
python scripts/audit-pixel-preparation.py \
  --manifest .benchmark/jpeg-context-multisource-20261005/manifest.json \
  --corpus .benchmark/jpeg-context-multisource-20261005 \
  --experiment .benchmark/pixel-residual-preparation-20261005 \
  --out FRESH_AUDIT.json
```

Only the central crop is represented, so noncentral payloads can be missed.
CMYK/small/non-JPEG inputs are unsupported by this research contract, not clean
files. Source/quality metadata can still influence pixel content indirectly;
no context invariance is established. Existing inspected ALASKA and simulated
BOSS validation remains development; camera/device/perceptual independence,
ALASKA payload/quality metadata and JMiPOD evidence are not added.

Next slice: bounded minibatch training and numeric model persistence, frozen
settings, all-source plus both source-exclusion controls on the same rows,
numeric/export audits and complete failure reporting. Keep the failed
[JRM transfer results](JRM_TRANSFER_RESULTS.md). Untouched licensed sources,
calibration, sufficient scene counts and blind CTF/release qualification remain
separate unmet gates. Primary detection/verdicts are unchanged.
