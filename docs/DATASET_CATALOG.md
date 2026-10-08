# Dataset catalog and publication policy

This inventory records what is actually acquired or measured, not a list of
advertised supported methods. Raw corpora remain in ignored `.benchmark/`.
Use [benchmark results](BENCHMARK_RESULTS.md) for scores and limitations.

| Source and origin | Local material | Provenance / grouping | Usage and publication status |
| --- | --- | --- | --- |
| [BOSSbase 1.01, DDE](https://dde.binghamton.edu/download/) | Frozen 1,000-cover PGM pilot; 500 BMP/Steghide and 500 PNG/OpenStego pairs | Original cover hashes; camera metadata unassigned; archive-order selection | No explicit redistribution license verified on download page; originals/derivatives stay local |
| BOSSbase 1.01 separate development | Additional 1,000 PGM originals: 816 train, 184 validation; 7,000 controlled PNG/BMP files, including 6,000 gray-plane LSB stegos | Reserved names/identities excluded; fixed seed 20261005, complete original ancestry | Same usage limits; [development results](SPATIAL_DEVELOPMENT_RESULTS.md); this is not another independent source |
| [Kodak suite](https://r0k.us/graphics/kodak/) | Frozen 24 originals; both upstream methods on each | Same original ancestry across conversions and methods; camera metadata unknown | Curator usage statement is not an independently verified license; originals/derivatives stay local |
| [ALASKA2 competition](https://www.kaggle.com/c/alaska2-image-steganalysis/data) | Frozen 4,000-file baseline, plus two disjoint 4,000-file development acquisitions | Complete cover/JMiPOD/JUNIWARD/UERD lineages, explicit split, SHA-256 and ZIP CRC; per-file camera/device/payload/QF metadata unassigned | Subject to competition rules; local research only in this workflow; no raw corpus or model redistribution authorized by this inventory |
| BOSSbase 1.01 second development expansion | New 1,000 PGM originals: 822 train, 178 validation; all 6,000 Q75/Q95 cover/JUNIWARD/UERD JPEGs prepared | Excludes all six previous acquisition manifests; byte/ancestry/coefficient/crop audit passes | Same original source restrictions; not another source group or qualified detector |
| [WIFD](https://github.com/CSCRC-SCREED/WIFD) | 200 SDR original files: 120 JPEG, 80 MPO; whole origin reserved | Pinned source/Git blob/SHA-256; ten declared camera directories, primary frames only; scene identity unverified | Pinned README declares data/code MIT; license evidence retained; no raw data published or detection measured |
| [FSDD v1.0.10](https://github.com/Jakobovski/free-spoken-digit-dataset/tree/d6938f9bf1545aa66d8489fc9f1385a7abd64282) | 3,000 actual spoken-digit WAV recordings, six speakers; 6,000 controlled LSB stegos from 1,000 test originals | Pinned commit; per-file SHA-256, CRC, speaker and PCM metadata; whole-speaker train/validation/test roles in separate experiment manifest | Upstream CC-BY-SA-4.0; attribution, license link and applicable ShareAlike/change notices required for redistribution; no audio committed |
| [StegoAppDB paper](https://arxiv.org/abs/1904.09360) | Not acquired | Intended mobile-app evaluation; no verified local camera/app mapping | Checked database endpoint inaccessible here; access and usage conditions still required |
| [BOWS2](https://bows2.ec-lille.fr/) | Not acquired | Potential additional image source, independence must be reviewed | Endpoint was unreachable from this environment; no replacement mirror or license assumed |

## Explicit expansion — 2026-10-08

An additional 4,000 ALASKA2 JPEGs (1,000 complete four-way original lineages)
and 1,000 BOSS PGM covers were acquired. Independent byte/hash/CRC/geometry,
selection/split and reserved-identity audits found no exact overlap with the
six prior acquisition manifests. Three ALASKA lineages contain unchanged
cover/stego bytes and are excluded as whole lineages from future preparation.
New eligible train originals: ALASKA 816, BOSS 822; validation originals remain
separate: ALASKA 181, BOSS 178. Acquisition split counts are not model-ready
train-row counts, and exact exclusion cannot prove scene/camera independence.
These are larger samples from existing sources, not additional source groups.
Original source restrictions remain; no raw data or model publication.

The [complete new JPEG preparation](JPEG_SCALE_PREPARATION_RESULTS.md) uses all
997 eligible ALASKA lineages and all 1,000 BOSS originals in 16 bounded blocks:
8,991 JPEGs / 7,380 train / 1,611 validation, with fresh unrounded-Y caches.
JMiPOD is explicitly excluded from this matched BOSS-family recipe. Every JPEG
and cache hash/role was independently checked, including BOSS cover ancestry,
4,000 simulated coefficient changes and nine scalar-IDCT contexts. No model was
trained or accuracy measured; the old prepared corpus stays unchanged.

The [WIFD](https://github.com/CSCRC-SCREED/WIFD) addition pins commit
`3f577edf0b14c686aa08e8d0d8ae07a83ba44f26` and verifies its MIT data/code
license evidence. First JPEG-only attempt failed on native two-frame MPO;
40 partial files remain local without a success manifest or completed count.
`WIFD_RETRY_PROTOCOL.md` separately freezes opt-in primary-frame acquisition,
with identical 20-per-camera selection and all original bounds. Native MPO
is not relabeled JPEG or counted as deployed format support. The retry completed
all 200 files and independent primary-frame/byte audit passed; see
[complete expansion results](DATA_EXPANSION_RESULTS.md). The whole WIFD
origin remains reserved from training/calibration, and camera IDs/scene
independence remain declared/unverified. No WIFD accuracy has been measured.

Counts refer to unique original lineages, not to separately counted copies in
each method comparison. Exact-hash exclusion cannot discover undocumented
transforms or camera/scene overlap. Unknown metadata remains unknown.

The [JPEG context experiment](JPEG_CONTEXT_RESULTS.md) reuses 128 of the existing
BOSS development originals, creating 768 grayscale JPEGs at Q75/Q95 with upstream
JUNIWARD/UERD simulations (0.2 bpnzAC). It combines them with ALASKA development,
not a new original source or blind corpus. Simulations are not encoded message
recovery; conseal 2025.11 lacks JMiPOD. All derivatives/weights remain local under
the original source usage limits; [conseal](https://github.com/uibk-uncover/conseal)
and inherited notices are recorded in `../THIRD_PARTY_NOTICES.md`.

The BOSSbase development corpus also underlies the fixed
[parity/residual model comparison](SPATIAL_PARITY_RESULTS.md); this adds no
new source or original lineages. Its improved ranking and FPR, low-rate recall
regressions and calibration failures are reported together, not as qualification.

## FSDD acquisition

Explicit user-requested command; no installation or analysis implicitly downloads:

```sh
python scripts/fetch-fsdd-pilot.py --out .benchmark/fsdd-v1.0.10-20261004
```

The fixed commit is `d6938f9bf1545aa66d8489fc9f1385a7abd64282`.
The downloader checks the [pinned license declaration](https://github.com/Jakobovski/free-spoken-digit-dataset/blob/d6938f9bf1545aa66d8489fc9f1385a7abd64282/README.md),
rejects redirects and symlinks, bounds compressed/expanded data to 64 MiB,
preflights ZIP member/directory counts, verifies CRC and mono PCM16/8 kHz
frames, and refuses existing outputs. It never extracts/runs upstream code.
Failures retain partial files without a success manifest; use a new directory
after inspecting a failure, not an automatic overwrite.

This acquisition downloaded 16,624,960 archive bytes and retained 21,128,848
recording bytes. A separate audit reread all 3,000 WAVs, verified their hashes,
sizes and PCM frames, and found 3,000 unique hashes. The portable
[acquisition record](../benchmarks/fsdd-acquisition-20261004.json) binds the
source commit, archive, manifest, license evidence and downloader hashes.
The recorded SHA-256 is locally computed, not a separately signed upstream hash.

The subsequent preregistered [WAV baseline](WAV_RESULTS.md) prepared and audited
6,000 marker-free sequential/scattered LSB examples at three rates. Exact
oracle recovery passed; automatic detection failed (AUC 0.50, recall 0%).
Whole-speaker roles are in the separate experiment manifest; the original
acquisition's unassigned split fields remain unchanged for provenance.

Future audio experiments must freeze speaker/original-recording groups and
payload rates **before** embedding or scoring. The upstream spoken-digit task's
train/test recommendation is not automatically our steganalysis split. Preserve
every derivative's ancestry; speakers are not six independent dataset sources.
This corpus is narrow speech data, not evidence about music, stereo, MP3 or
arbitrary audio methods. Existing recordings have not been certified free of
all possible steganography; “cover” is the controlled experiment's source role.

## What we publish

- Commit acquisition/evaluation code, protocols, versions, source links, license
  notes, hashes, portable metrics and unsuccessful results. No credentials,
  signed download links or absolute host paths.
- Raw image/audio archives do not enter Git by default. Citation is not a
  redistribution license. The [ALASKA project](https://alaska.utt.fr/) describes
  noncommercial/no-derivatives restrictions; review both source terms and the
  competition rules before sharing modified data or trained artifacts.
- FSDD can be considered for a separately licensed data release under its
  [CC BY-SA 4.0 conditions](https://creativecommons.org/licenses/by-sa/4.0/),
  retaining attribution and marking changes. It is not relicensed under this
  software repository's MIT license. No remote upload/release occurred here.
- Label measurements precisely: synthetic regression, real-media controlled
  embedding, same-source validation, independent-source test, or blind CTF.
  Do not present these as interchangeable “real-world accuracy”.
