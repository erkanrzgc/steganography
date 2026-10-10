# Multi-carrier originals acquired and independently audited — 2026-10-10

Added **900 original RGB PNGs and 2,000 environmental WAVs**. This diversifies
spatial-image and audio research beyond gray covers and spoken digits. It is
not new stego ground truth, model fitting or measured accuracy improvement.
The existing failed JPEG/WAV results stay unchanged; no model is deployed.

| Source | Originals | Train | Development validation | Untouched test | Original media bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| DIV2K HR train | 800 | 800 | 0 | 0 | 3,533,759,878 |
| DIV2K HR validation | 100 | 0 | 100 | 0 | 449,507,452 |
| ESC-50 | 2,000 | 1,200 | 398 | 402 | 882,088,000 |

DIV2K train/validation are one origin, not two independent camera populations.
ESC has 1,524 upstream original-recording groups and fifty environmental
categories. Categories are context, not labels indicating steganography.
Whole original groups and all subsequent derivatives must preserve their role.

## Complete independent audit

An auditor importing no acquisition service reread all 2,900 new originals
and 10,225 original cover rows across the ten bound historical manifests.
It independently reconstructed media/PCM hashes, geometry, native identities,
complete counts, roles, recording groups, license evidence and full archives.
No new/old exact-byte or decoded-original overlap, new duplicate, or
original-recording train/validation/test overlap was found. Audit duration:
227.34 seconds on this local environment, not a model inference benchmark.

This cannot exclude undocumented transformed duplicates, scene/camera overlap
or preexisting payloads. Unknown camera/device provenance stays unknown;
candidate cover roles are not a certificate that original files are clean.
Previously inspected development sets are not fresh blind tests. The new ESC
test role was byte/provenance-audited only, never fitted or detector-scored.

## Failures are retained

All first acquisitions failed before media download because our reserved-manifest
guard incorrectly required a lineage field on frozen old BOSS/Kodak originals.
The [separate legacy adapter](MEDIA_DIVERSITY_LEGACY_RETRY.md) accepts only
those two exact manifest hashes, excludes original SHA identities and leaves
old metadata unchanged. All other manifests still require both identities.

The subsequent native ESC import downloaded a valid pinned archive but failed
before writing WAVs: four upstream recording IDs span folds; two cross our
validation/test roles. Preserve the [failure record](../benchmarks/esc50-native-fold-failure-20261010.json).
The [separately frozen retry](ESC_ORIGINAL_GROUP_RETRY_PROTOCOL.md) uses the
exact cached archive/CSV and assigns every original recording's fragments to
its highest native fold. Two clips move validation to test, never held-out
to training. Native metadata/bytes remain untouched; assigned_group_fold is
separate. Default strict acquisition continues rejecting cross-fold IDs.

## Limits, licensing and provenance

Archives stream to exclusive private outputs under the original byte/time
limits. ZIP directory allocation, member counts, expansion, RGB geometry and
PCM frames are bounded; no upstream script is installed/executed, no symlink
followed and no output overwritten. Failed partial artifacts remain private.
The grouped import retains a copied archive so the new directory is auditable
without depending on the failure directory. Checksums are locally observed,
not upstream signed. Failed original attempt and completed retries stay distinct.

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) permits academic research only;
original owners retain copyright. [Pinned ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)
declares CC-BY-NC-3.0, with ESC-10 separately CC-BY. Preserve its documented
preprocessing/bandlimiting limitations. No raw corpus, derivatives or learned
weights are published; citation alone is not permission for redistribution.

Portable records:

- [Acquisition](../benchmarks/media-diversity-acquisition-20261010.json), SHA256
  `45d8ecc3c2a3c41e85b607dd49ba5f4430dedfb117980c436f6092e9413b81ce`.
- [Independent audit](../benchmarks/media-diversity-independent-audit-20261010.json), SHA256
  `3f6f5839bf0bba04f98085812475644ca61e61ee12612a0e575d8d0149778d24`.

Acquisition implementation hashes bind the actual worker revisions: DIV2K
workers used `74c9279`; the grouped ESC retry used `de2c884`. Do not substitute
the later on-disk source hash for code already loaded by a running downloader.
Original protocols, worker source, manifests, archive, CSV and usage evidence
are separately bound. JSON reports contain no absolute paths or secrets.

## Next learning work

See [method-specific data/learning gaps](METHOD_DATA_PLAN.md). RGB originals
enable lineage-bound PNG/BMP/channel experiments and controlled JPEG/GIF
derivatives, not native GIF method qualification. Environmental WAVs diversify
mono audio; stereo, music, compressed audio, text and containers need their
own ground truth. No dataset qualifies every method by conversion alone.

Next: versioned three-origin JPEG preparation and normalization-controlled
learning, then independent RGB and two-origin WAV comparisons. Freeze method,
rate, source grouping, preprocessing, calibration and compute limits before
fitting. All failed gates remain visible; compare real per-cell false positives,
recall/calibration and exact recovery, not sound classification or training loss.
RunPod MCP authorization exists but this session exposes no RunPod tools;
physical 24 GB pod access and a spending ceiling still need verification.
No paid compute or new training occurred in this acquisition slice.

## Verification

The fresh full regression against final application code passed **1,892 tests**
with one actual-CUDA test explicitly skipped as unavailable, 32 existing ONNX
deprecation warnings and total coverage **95.73%** (842.74 seconds).
New acquisition core line coverage is **100%**. The 59 acquisition/auditor
security tests and three portable-evidence tests also pass together; those
three evidence tests were added after full-suite collection and verified
separately, not silently counted in the completed full run.

An earlier full run was deliberately interrupted when the final grouping code
changed: 1,360 passed / one CUDA skip; exit 2 is not a full regression pass.
Its log remains private. Lint, type checks (148 application files) and
whitespace checks pass. Python 3.11 exercised here; other Python versions,
full Docker and physical RunPod GPU checks remain unverified.
Fresh model-free wheel/sdist inspection (158/319 members) excludes private
corpora, caches, credentials and learned weights. No originals were uploaded.
