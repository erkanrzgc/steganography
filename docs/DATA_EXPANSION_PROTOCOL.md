# Frozen acquisition expansion — 2026-10-08

User explicitly authorized additional useful datasets. Acquire disjoint
ALASKA2/BOSSbase development selections through the existing bounded range
downloaders, preserving all previously published tests and development data.
First slice: 1,000 complete ALASKA2 lineages (4,000 JPEGs) and 1,000 BOSS
originals. They are more data from existing sources, not new independent sources.
Keep inherited local-research/redistribution limits and unknown metadata.

Add a new image origin, [WIFD](https://github.com/CSCRC-SCREED/WIFD), pinned
commit `3f577edf0b14c686aa08e8d0d8ae07a83ba44f26`. Its pinned README states
data and code are MIT licensed. Validate the license's Git blob identity and
retain license/attribution evidence. Download only SDR JPEGs, not flat-field
references, scene exposure bursts, RAW files or executable repository content.

Initial WIFD slice: first 20 SDR JPEGs per eligible declared camera, ordered
by SHA256("wifd:20261008:" + upstream relative path). No score-based selection.
Camera IDs come from upstream directory names and remain declared metadata,
not independently proven device identity or scene independence. Reserve the
entire WIFD origin from training/calibration until a separate evaluation plan;
this is an unseen-source candidate, not a passed or adequately sized test.
20-image selection may be expanded by an explicit 120-per-device acquisition;
previous identities must be excluded, never overwritten or double-counted.

Fixed official API/raw hosts, pinned tree and Git blob hashes, no redirects,
bounded metadata 8 MiB, member 32 MiB, aggregate requests/data 16 GiB,
at most 2,000 files/30,000 tree entries, three attempts/request, four workers,
1,800s wall and 8 GiB address-space limits. Validate JPEG decode and <=32M
pixels, SHA-256 and upstream Git SHA-1 identities. Fresh nonsymlink directories,
generated filenames, no extraction or code execution. Interrupted downloads
retain partial files but no success manifest. No full-repository fallback.

Independent audit rereads every acquired file, sizes/hashes/CRC where available,
formats/dimensions, all reserved identities, selection and grouping. Publish
portable source/license/hash/count and failure records, not raw corpora,
credentials, absolute paths or signed URLs. Acquisition is not accuracy gain.
GPU availability is checked separately; do not provision/pay for cloud resources.
