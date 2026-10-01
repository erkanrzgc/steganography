# Frozen small external-source check — 2026-10-01

Declared before examining outcomes. Detector implementation: `3afcba7`.
No analyzer, score threshold or model is changed by this experiment.

- Source: all 24 numbered images from https://r0k.us/graphics/kodak/ .
  This curator describes a belief that Kodak allowed unrestricted use, not a
  formal verified license grant. Keep downloads local; publish no images/models.
- Check original-file hashes against the frozen BOSSbase pilot. This detects
  exact overlap, not perceptual similarity or undocumented camera/scene overlap.
  Camera/device grouping is unavailable; do not claim camera independence.
- Generate both independent Steghide/BMP and OpenStego/PNG variants per image,
  retaining RGB dimensions. Rotate payloads 256, 1024 and 4096 bytes by image
  index: eight images per size per method. Reuse the same payload for each
  image's two methods. There are 24 original lineages, 48 method-specific pairs,
  96 analyzed files, NOT 48 independent original images.
- Independently extract all 48 stegos with upstream tools and demand exact
  payload matches before interpreting detector results.
- Existing balanced analysis, AI off, threshold 70; host tool availability is
  recorded. Report per-method metrics and paired-lineage bootstrap intervals.
  No training, threshold search or calibration uses these test observations.
- The 30 generated CTF cases are optional controlled regression, not a blind
  test. Known-password cases are explicitly labeled. No 120-case gate claim.
- This source is too small for support-cell certification. The old BOSSbase
  baseline remains frozen and is not replaced. Do not average unlike sources,
  code revisions or method mixes into a headline accuracy claim.

Commands (activate the provisioned environment first):

```sh
python scripts/fetch-kodak-pilot.py --out .benchmark/kodak-20261001 \
  --reserved-manifest .benchmark/pilot-v1/manifest.json
# In full Docker, with current code read-only at /app and data at /data:
python -m steganography.benchmarking.pilot generate --source /data/kodak-20261001 \
  --out /data/kodak-pilot-20261001-serial --pairs 24 --both-methods --workers 1
python scripts/verify-pilot-groundtruth.py /data/kodak-pilot-20261001-serial
# In the base host environment, not full Docker:
python -m steganography.benchmarking.pilot detect \
  --source .benchmark/kodak-pilot-20261001-serial --out .benchmark/kodak-detection-20261001
```

Execution note: the initial four-worker generation stopped on missing external
generator output. Its incomplete directory was retained, not evaluated. A fresh
serial run used the same images, payload policy and tools; all expected payloads
were independently checked. No images were selected or dropped based on scores.
Docker used read-only root, non-root user, network disabled and bounded process/
temporary storage. Generation used one worker/128 PIDs; verification and CTF used
256 PIDs, with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Concurrent validation
and tests mean observed latency is not a dedicated performance-gate measurement.
