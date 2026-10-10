# RGB-image and environmental-audio acquisition — 2026-10-10

Scope: explicitly authorized private research acquisition, not fitting or a
claimed improvement to deployed steganalysis. No paid compute is provisioned.

## Fixed sources, roles and limits

- DIV2K official ETH HR training ZIP: exactly names 0001..0800, 800 RGB PNGs,
  assigned train. HR validation ZIP: exactly 0801..0900, 100 RGB PNGs assigned
  development validation. No low-resolution variants are counted as originals.
  These are one declared origin, not independently certified cameras/scenes.
- ESC-50 pinned upstream commit
  `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`: all 2,000 mono PCM16/44.1kHz
  five-second WAVs, with upstream CSV metadata. Folds 1..3 train, 4 development
  validation, 5 untouched test. All fragments sharing a Freesound original ID
  must occupy the same fold/role. Verify eight clips per class/fold, fifty
  classes; 1,200/400/400 role counts. Category is context, never a stego label.
- Download only fixed HTTPS origins, no redirects, credentials or upstream
  executable installation. Per-source acquisition ceiling 2,400 seconds,
  15-second socket timeout, streaming archive writes and exclusive fresh outputs.
  DIV2K train compressed cap 4 GiB; each other archive 1 GiB. ZIP directory
  <=5,000 members /2 MiB is preflighted before parser allocation. Expanded
  declarations <=6 GiB, individual selected media <=32 MiB. RGB decode
  <=8 million pixels, min dimension 256; metadata/usage evidence <=1 MiB.
- Reject traversal, absolute paths, duplicates, encrypted/nonstandard-compressed
  entries, links/devices and corrupt CRC. Never extract archive paths or execute
  archive content. Publication of a complete manifest occurs only after every
  file passes byte/geometry/PCM/unique encoded-and-decoded identity checks.
  On failure retain partial private files, not a completed manifest or automatic
  retry. For each acquisition bind all ten existing original source manifests;
  reject prior exact byte/original identities. A separate independent audit must
  reread all outputs and prior originals, checking decoded identity too.

## License and evidence

DIV2K's official page permits academic research only; images retain their
original owners' copyright. ESC-50 declares CC-BY-NC-3.0, with ESC-10 separately
CC-BY. Retain exact source-page/README and CSV bytes and SHA-256. No permission
to publish restricted originals/derivatives/weights is inferred from citation.
ESC-50's documented preprocessing/bandlimiting caveat remains; a source cover
role does not certify absence of preexisting hidden messages.

Archives and originals remain in ignored private `.benchmark/`; publish only
portable aggregate records, source links, hashes, license notes and code.
Hashes are locally observed, not upstream signed. Counts are original media,
not stego examples, method coverage, camera independence or accuracy evidence.

## Subsequent learning

RGB originals can supply explicitly lineage-bound PNG/BMP spatial experiments,
JPEG re-encoding and palette/GIF derivatives. Environmental sounds diversify WAV
experiments beyond spoken digits. Those conversions, stego generation, exact
payload oracles, method/rate cells, normalization controls, GPU execution and
model calibration require separate versioned, bounded preparation and learning
protocols. Keep validation/test roles untouched during fit. Do not conflate
sound classification scores with steganalysis or use copied Internet images as
a verified camera-source blind test. Text/container/MP3/CTF remain separate
data and detection tasks, not qualified by these two new corpora.

Primary sources:
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and
[pinned ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6).
