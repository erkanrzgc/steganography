# Frozen full-expansion JPEG preparation — 2026-10-08

Acquisition audit SHA-256:
`62b0c8218797c365b697af1068a6c9c1c896e94395430f91c2e818cc0f3eb28a`.
ALASKA development source SHA-256:
`7df0e4a5a4d64576b6c6a2edcae1eadf8b46a3afa54fe6e1855fd79c7d525626`.
BOSS development source SHA-256:
`9ae58e33f8e71e72a918ed393ecb3f7900f36d3428f3a4347e42018fc84adfe8`.
Reserved WIFD acquisition source SHA-256:
`303995e90fbeb90a52e5a6ee99063741092690b5b47d50c2703c66bdf0959f9c`.

Prepare all 997 nonquarantined ALASKA originals and all 1,000 new BOSS originals,
not a result-selected subset. ALASKA retains original JPEG bytes and original
splits, using cover/JUNIWARD/UERD for the existing matched-family recipe.
JMiPOD remains acquired but explicitly excluded: no matched BOSS simulator in
the pinned conseal recipe. Preserve all three quarantined ALASKA groups.
BOSS keeps existing conseal 2025.11 / Q75,Q95 / JUNIWARD,UERD / 0.2 bpnzAC /
seed 20261006 simulation and coefficient round-trip checks, not encoded message
recovery or authentic unknown payload rates. No source labels/filenames become
model inputs. Camera/device/ALASKA quality/payload remain undeclared.

Deterministic ALASKA original order: SHA256("scale:20261008:" + lineage).
BOSS order stays SHA256("jpeg-context:20261006:" + original SHA256).
Each source is divided into original-lineage blocks of at most 128, never
splitting correlated qualities/methods or silently truncating a source.
Expected 16 blocks: eight ALASKA (seven 128 + 101), eight BOSS (seven 128 + 104).
Keep all development train/validation roles. Expected JPEG rows: 8,991 total,
7,380 train / 1,611 validation, representing 1,997 original lineages.
The whole WIFD origin and all six prior acquisitions remain excluded.
The old 3,750-row prepared corpus/caches stay untouched and are not merged in
this slice. Existing trainer/sampler 4,000-row caps stay unchanged.

Fresh nonsymlink output; no overwrite, symlink/hardlink copies, execution,
download, training or model installation. Metadata <=64 MiB, JPEG <=existing
MAX_IMAGE_BYTES, full output <=8 GiB. Recheck source/selection/reserved bindings
and every copied/generated hash/size, complete recipes, exact block membership
and global identity/split isolation. Two existing isolated simulation workers,
90s wall /2 GiB /80s CPU each. Optional float-cache extraction uses the unchanged
block-aligned center256 unrounded Y-IDCT decoder, <=4 JPEGs per worker/15s;
bounded blocks, not an all-corpus tensor. Record decoder versions and hashes.

Overall preparation: 3,600s service deadline, external 3,610s hard process-group
wall bound, parent address space 8 GiB and CPU 7,200/7,201 seconds, no core dumps.
Failure retains partial blocks without a whole-job success index; individual
completed blocks do not establish full-source preparation. No resume/overwrite
or automatic resource extension. Independent audit must reread all JPEGs and
stream cache hashes, prove complete membership/roles, and replay metadata-only
first rows without consulting detection scores. No accuracy measurement.

A complete index is a versioned data/preprocessing contract, not a passed
learning gate or compatible input for the capped whole-array trainer. A later
explicit streaming trainer/GPU/exposure protocol is required. CPU-only host;
do not open paid/cloud training or publish inherited-restricted images/weights.
