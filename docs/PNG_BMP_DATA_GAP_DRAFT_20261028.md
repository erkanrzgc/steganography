# Native corpus gap and next acquisition draft — 2026-10-28

**Non-executable planning only.** No metadata request, media download, source
retry, role assignment or budget reset follows from this document. Preserve
snapshot 016, the completed 512-MiB expansion, every held/excluded source and
the incomplete older historical ledger. The new software preparation opens
no native or reserved images.

| Domain | Existing originals | Existing author groups | Missing originals to 128 | Missing groups to 16 |
|---|---:|---:|---:|---:|
| Native L | 13 | 6 | 115 | 10 |
| Native RGB | 105 | 69 | 23 | 0 |

These are inventory deficits, **not a guarantee of a feasible training split**.
The old draft's 10/3/3 groups with 80/24/24 originals assumes eight eligible
originals in each selected group. In the current RGB inventory, the ten largest
groups contain only **41 originals**: 8, 8, 5, 5, 3, 3, 3, 2, 2, 2. Even
adding all twenty-three missing originals to the best groups raises that
upper bound only to 64, below eighty. Reaching 128 total therefore cannot by
itself make the historical ten-group training proposal executable.

A later split proposal should retain 80/24/24 originals as a candidate while
allowing variable group counts, or preregister a distinct exposure/grouping
intervention. Check exact feasibility after complete ancestry review; never
split one author/upstream package across roles or count PNG/BMP derivatives
as independent originals. Authors shared between modes must retain one
global role. Actual native roles remain unassigned and all existing reserved
datasets stay closed. Do not edit the historical draft to imply it ran.

## Proposed next preparation stage

First use already cached source/license metadata. A separately approved
metadata-only discovery stage could then reserve up to eighty new requests,
each at most 2 MiB (160 MiB response ceiling), with exclusive pre-I/O intent,
evidence hashes, server cooldowns and no automatic retry. This is a **new
proposed** allowance; all eighty earlier requests remain consumed.

Prioritize native photographic 8-bit L and at least ten additional L author
groups; also identify the RGB deficit and verify global split feasibility.
Retain CC0/CC BY assertions with source, author, versioned license and original
hash/extent. Keep fixed crop/texture gates and at most eight originals per
author/mode. Drawings, rendered scenes, RGB-to-L conversion, unapproved
microscopy, uncertain source origins and held licenses do not fill the gap.
BOWS2/DIV2K and reserved data are not automatically imported.

Only after reviewing a concrete frozen candidate manifest can a separate raw
allowance be chosen: reserve the sum of declared extents plus one overflow
byte per attempted original, including every failure. At the existing 16-MiB
per-file ceiling, 138 originals alone have a worst-case reservation of
2,315,255,946 bytes, before replacement candidates or failures. This arithmetic
is neither a requested download nor an approved 2.16-GiB budget. Native L
availability and source diversity remain uncertain; do not promise the target
from a byte allowance or silently extend one. New acquisition code/limits
must be independently source-bound and verified; do not restart the old
collector or reuse a previously attempted upstream SHA-1.
