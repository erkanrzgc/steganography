# Third-origin originals acquired and audited — 2026-10-10

The separately frozen `BOWS_VERIFIED_LAYOUT_PROTOCOL.md` completed. This adds
**1,001 BOWS2 original lineages: 807 train, 194 development validation**.
It does not train a model, produce JPEG float caches or improve measured accuracy.
WIFD remains reserved and the failed five-epoch JPEG scores remain unchanged.

The university-hosted archive is 183,938,734 bytes; plain TAR is 263,434,240
bytes. Original PGMs total 262,421,270 bytes, all gray 512x512. The actual member
set is 1.pgm through 1000.pgm plus 3661.pgm, not 0..1000. Acquisition kept every
native image; future later-partition downloads must exclude 3661.pgm too.

All eight previous original acquisition manifests are bound. Independent audit
reread every acquired image and all 3,000 BOSS originals: no exact identity or
decoded BOSS-pixel overlap, 1,001 unique new byte/pixel hashes, unchanged lineage
split. This does not certify absence of transformed duplicates, scene/camera
overlap, hidden payloads or undisclosed upstream processing. Cameras/devices
remain unknown. BOWS is a third declared acquisition origin, not a certified
independent camera population. Adding data does not prove memorization was the
old model's cause of failure.

## Failures remain failures

The initial advertised 1,000-file guard failed before output. Its downloaded
bytes were in memory and were not retained; no original archive hash is claimed
for that attempt. A separate bounded retrieval supplied the retained archive.
The first native import then failed our incorrect 0..1000 name assumption.
Independent enumeration revealed the actual extra 3661.pgm; a new frozen
protocol imported those exact 1,001 identities into a fresh directory.
Neither failed attempt wrote original images or is retroactively passed.

Original security byte/time/member/geometry bounds remain. The separate final
protocol explicitly accounts for 1,001 image geometries rather than silently
dropping the extra image. No archive executable, symlink, sparse record or
archive path is extracted/executed. Generated filenames and exclusive writes
avoid overwrites. Archive checksum is locally observed, not upstream signed.

## Usage and next training slice

Source/citation: [TU Dresden evaluation archive](https://dud.inf.tu-dresden.de/~westfeld/rsp/rsp.html).
No explicit redistribution license was verified there. Originals, derivatives
and any learned weights stay private; citation alone is not publication permission.
RAISE was reviewed as a later non-commercial research candidate, not downloaded.

The existing full-block JPEG preparation and four-row sampler intentionally
bind the old ALASKA+BOSS recipe. Do not forge BOWS metadata as BOSS or silently
weaken old plans. Next: versioned three-origin preparation/sampling; all
derivatives remain in their original lineage; source/quality/payload controls;
single-file-compatible normalization intervention and unchanged control.
Prepare a separate untouched source/device evaluation policy before scores.

The authorized WSL GPU tunnel is currently unavailable (connection refused).
No new training is running, no fallback CPU fit is passed as GPU training and
no cloud or paid resource was provisioned. Training requires restoring that
connection after the new recipe and learning controls are verified.

## Portable evidence

- Acquisition: `benchmarks/bows-diversity-acquisition-20261010.json`, SHA256
  `9b64ee28ef81819927343476862c8acaf5d7b61421b578309ab98fa68873d7a0`.
- Independent audit: `benchmarks/bows-diversity-independent-audit-20261010.json`,
  SHA256 `5f3f945a606fd5f6cf3cffe5bfb632b2732523ae9adfecbfc80241e0974b5536`.
- Private original source manifest SHA256
  `7b0160a7f631f857f127b6299f6c2a4b9c913a3ad8c87a5562614d62f5d94014`.
- Two failed attempts: `benchmarks/bows-first-acquisition-failure-20261010.json`
  and `benchmarks/bows-native-name-failure-20261010.json`.

Generated acquisition/security tests: 20 passed, new core line coverage 99%.
Lint/type checks pass (146 application source files). An initial full suite was intentionally interrupted
after 1,259 passed / one live-CUDA skip when native-layout code changed;
that run is not a complete regression pass. The fresh complete regression
against final application code passed 1,816 tests / one actual-CUDA skip in
470.55 seconds, total coverage 95.65%, 32 existing ONNX deprecation warnings.
Two subsequently added portable-evidence tests also passed separately.
Fresh wheel/sdist inspection (156/314 members) excludes private caches, keys,
original images and model weights. Python 3.11 exercised here; other Python
versions, full Docker and new physical GPU training are not passed by this run.

## Evidence-informed next learning controls

[Original SRNet research](https://ws.binghamton.edu/fridrich/research/SRNet.pdf)
used substantially longer exposure and described curriculum training and a
high-quality-JPEG convergence difficulty. Its specific results do not transfer
to our reduced corpus or prove that extending five epochs will fix failure.
[Group Normalization](https://arxiv.org/abs/1803.08494) defines a batch-independent
alternative. That is a candidate for the required single-image-consistency
control, not proof of better steganalysis. Freeze a paired-control experiment
and independent numerical checks before fitting; never select architecture,
thresholds or checkpoints by repeatedly consulting the failed validation set.
