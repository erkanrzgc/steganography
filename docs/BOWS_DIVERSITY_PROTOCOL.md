# BOWS2 source expansion — frozen 2026-10-10

Explicit user request: increase original-source diversity, not duplicate
derivatives or claim that memorization is already proven. Acquire exactly
1,000 original grayscale PGM images from the author-hosted TU Dresden archive:
https://dud.inf.tu-dresden.de/~westfeld/rsp/bows2-1g.tar.gz
Source context: https://dud.inf.tu-dresden.de/~westfeld/rsp/rsp.html

This archive is a new declared origin, not proof of independent cameras/scenes
or clean covers. Camera/device/payload metadata remain unknown. The hosting
page offers an evaluation corpus but no explicit redistribution license was
verified. Keep originals/derivatives private; record page evidence and citation,
no raw data or learned weights in Git/wheels. Locally computed archive hashes
are not upstream signatures. RAISE is a later non-commercial research option,
not acquired here; its full RAW corpus is too large for this first slice.

Fixed HTTPS URLs, no redirects/credentials, total wall 600 seconds, socket
timeout 15 seconds, compressed limit 200 MiB, decompressed limit 320 MiB,
member limit 2,000, per-file 300 KiB, total decoded pixels <=1000*512*512.
Strict PGM L/512x512, no symlinks/hardlinks/sparse/PAX/long-name records,
absolute/traversal/backslash paths or duplicate names/images. Never use TAR
extract/extractall; generated output filenames, fresh nonsymlink directory,
exclusive writes, no retries/overwrite or execution of downloaded content.
Preflight the complete bounded TAR before creating any image output.

Exclude every SHA-256 and lineage in all provided original acquisition
manifests; require the six ALASKA/BOSS acquisitions plus reserved WIFD and Kodak.
Any exact overlap fails acquisition, not silent dropping/replacement. Decode
and record pixel identities as well; camera/scene identity still unproven.
Split all derivatives with their original lineage: SHA256("bows2:20261010:"+
original_hash), fraction <0.8 train, otherwise development validation. No
score-dependent sampling; acquire every original from this archive.

Independent audit rereads source-page/archive bindings, original bytes,
geometry, pixels, duplicate/old identities and exact split before eligibility
for preparation. No training/validation model pixels are prepared by this
acquisition. Preserve the failed five-epoch JPEG model scores. Next preparation
requires source-balanced sampling, single-file-compatible normalization
controls and an untouched evaluation policy. Acquiring images is not accuracy.
