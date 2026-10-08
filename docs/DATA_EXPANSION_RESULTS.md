# Verified data expansion — 2026-10-08

The user-authorized [frozen acquisition](DATA_EXPANSION_PROTOCOL.md) and
[separately frozen native-format retry](WIFD_RETRY_PROTOCOL.md) completed.
The independent [portable audit](../benchmarks/data-expansion-audit-20261008.json)
reread all **5,200 files / 2,399,521,328 bytes**. No exact identity or upstream
member overlap was found with six prior acquisition manifests or between
these three acquisitions. This does not prove perceptual/scene independence.

| Acquired source | Files / original lineages | Eligible train originals | Separate validation originals | Reserved evaluation originals |
| --- | ---: | ---: | ---: | ---: |
| ALASKA2 | 4,000 / 1,000 | 816 | 181 | 0 |
| BOSSbase-1.01 | 1,000 / 1,000 | 822 | 178 | 0 |
| WIFD | 200 / 200 | 0 | 0 | 200 |

ALASKA contains cover/JMiPOD/JUNIWARD/UERD files for every original. Three train
lineages contain unchanged cover/stego bytes (five mislabeled unchanged stego
files); the audit marks all three original groups for exclusion from future
preparation, without deleting or modifying acquired files. They are not added
to the 1,638 eligible new training originals. Development validation originals
(359) are never merged into training. ALASKA/BOSS are larger samples of existing
origins, not new independent source groups or verified camera populations.

## New reserved origin and unsuccessful attempt

[WIFD upstream](https://github.com/CSCRC-SCREED/WIFD) is pinned to commit
`3f577edf0b14c686aa08e8d0d8ae07a83ba44f26`; its pinned README declares MIT
licensing for data and code. License/README Git identities and each image's
Git blob identity/SHA-256/size were verified. The
[portable acquisition record](../benchmarks/wifd-acquisition-20261008.json)
binds protocol, selection, script, source manifest, license and metadata hashes.

Identical 20-per-camera selection produced **120 native JPEGs and 80 native
MPOs**, from ten declared camera directory IDs. 120 files declare one frame,
60 declare two and 20 declare three. Only primary frames were decoded; original
bytes were retained, MPO filenames and metadata stay MPO. Secondary frames were
not decoded or certified. Acquisition adds no application MPO detector or
model-ready preprocessing support. Device identities remain declared, and
scene/cover-cleanliness independence is unverified.

The first JPEG-only attempt failed on native two-frame Canon EOS M6 MPO despite
matching upstream bytes/blob identity. Its
[failure record](../benchmarks/wifd-jpeg-only-failure-20261008.json) preserves
original selection/protocol/script identities, 40 local partial files and
315,183,023 retained bytes. Those partial copies are **not** included in the
5,200 completed-file count. The retry used a new directory, no camera/sample
exclusion, no overwrite, no relabeling and no format conversion.

The whole WIFD origin remains excluded from training/calibration. These 200
cover-role images are an unseen-origin evaluation candidate, not >=1,000
cover + >=1,000 stego over two independent sources or passed qualification.
No WIFD embedding, model evaluation or accuracy measurement occurred.

## Model and compute status

No new model was trained, installed, exported or scored. Previous failed
real-data results stay unchanged. These acquired files have not entered the
old prepared manifest/cache; acquisition is not accuracy gain.

Next: versioned, bounded whole-lineage preparation; independently verified
method/payload provenance; reviewed streaming/exposure schedule; and an adequate
training-only learning gate before new validation or blind-source testing.
The current BOSS JPEG job cap (128 originals) and trainer cap (4,000 rows) must
not be silently bypassed, nor may the expanded corpus be silently truncated.
Host observed CPU-only Torch, eight logical CPUs and about 15 GiB RAM. NVIDIA
GPU/Kaggle/Colab access remains a user choice; no paid or cloud job was opened.

Raw corpora/partial files remain ignored and local. ALASKA competition and BOSS
usage restrictions remain in force; source citation is not permission to publish
their images or derived weights. Only portable code/protocol/evidence is pushed.

Verification: 1,388 tests passed, total coverage 95.22%; targeted acquisition/
audit coverage 380/389 statements (97.69%). Ruff and mypy (130 source files)
passed, wheel/sdist built. Wheel inspection found no corpus, benchmark, checkout
script or model artifact. This host exercised Python 3.11 only; Python 3.12–3.14,
GPU training and full Docker E2E were not verified by this slice.
