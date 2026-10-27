# Native PNG/BMP admission protocol — 2026-10-23

This separate continuation follows the user's instruction to proceed with
source review, data preparation and a concrete training plan. It permits one
bounded acquisition of the nine originals listed in
`benchmarks/png-bmp-native-reviewed-20261023.json`, followed by unassigned
pipeline preparation and read-only inspection. It grants no optimizer update,
new GPU profile, role assignment or reserved-data access. The completed Colab
profile and its sixteen-file source closure remain immutable.

## Source and resource scope

The manifest selects three native 8-bit L and six native 8-bit RGB PNG files.
All nine keep their published Commons SHA-1, byte extent, description-page
revision, credited author, original URL and per-page evidence hash. Metadata
review is preserved separately. The original two-candidate preparation list
is historical: its example graphic is unsuitable as natural imagery and its
volleyball photograph has an alpha channel.

CC0 and CC BY 2.0/3.0/4.0 are accepted with their actual license version and
canonical license URL. The separate `core.spatial_commons_native` service
extends the license policy without editing the frozen capacity implementation.
Attribution retains the title, author, page and license; every prepared
artifact describes its crop, format and generated LSB modification. The
official [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) and
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) terms were checked.
Commons file licenses and public review markers are recorded as published;
private permission tickets are not accessed or independently certified.

One acquisition command may reserve at most **64 MiB**, with at most sixteen
originals and 32 MiB per original; this selection fixes exactly nine files
and **40,191,158 reserved bytes**, including one overflow probe per file.
Only HTTPS `upload.wikimedia.org` originals are allowed, without redirects,
query parameters or thumbnails. Metadata API tracking parameters are removed
from the canonical original URL; the expected original hash/size is retained.
The existing 30-second per-response network timeout and bounded reads apply.
No automatic download retry or replacement source is permitted. Reserve the
full transfer bound on disk before I/O; failed and partial attempts stay
charged. Raw files and prepared images remain on private local disk.

The prior-transfer anchor hashes the known historical acquisition records
and the original Colab preparation receipt. Its `historical_bytes` remains
null: the earlier cumulative raw-transfer ledger is incomplete. This new
bounded allowance neither invents a complete cumulative total nor resets
old transfer, JPEG, WAV, fitting or GPU-profile budgets. Metadata-only HTTP
responses are recorded separately from raw-media transfer.

## Admission and checks

The exact IHDR must declare native PNG grayscale/color type 0 or truecolor
type 2, **8-bit**, with no alpha or palette. Checking the header precedes
Pillow decoding because Pillow can silently downcast RGB16 to RGB8. Retain
the existing single-frame, orientation, 16-million-pixel, encoded-byte and
symlink restrictions. No resize, grayscale conversion or alpha removal is
permitted for admitted inputs.

Review groups conservatively merge each credited author's complete collection,
including upstream Flickr streams where known. File-page derivation/source
links are inspected, and exact raw hashes and decoded crop duplicates are
rejected. A later contact-sheet and pairwise perceptual-hash review records
possible similar images; such heuristics cannot prove absence of common
ancestry. Unknown capture/processing history remains unknown. The package is
a small admission pilot, not a certified clean or representative training set.

The fixed top-left 256×256 crop precedes embedding. Reuse the existing
vectorized 32-byte plane-0 control and independent scalar decoder. Emit one
cover and one control in PNG and BMP, verify exact decoded pixels, and keep
every derivative in the original author group with **unassigned** role.
Prepared output is bounded by 256 MiB; existing outputs and symlinks are
rejected. Do not open prior reserved media or assign a new held-out split.

The explicit interfaces are:

```sh
python scripts/spatial-native-admit.py validate --manifest benchmarks/png-bmp-native-reviewed-20261023.json
python scripts/spatial-native-admit.py acquire --manifest benchmarks/png-bmp-native-reviewed-20261023.json --out local-originals --allow-download --prior-ledger prior-transfer-anchor.json
python scripts/spatial-native-admit.py prepare --manifest benchmarks/png-bmp-native-reviewed-20261023.json --originals local-originals --out local-prepared
```

Freeze the code/manifest/protocol hashes before acquisition. Generated tests
must pass, including RGB16 rejection before decoding, original hash/extent,
payload/pixel checks, duplicate crops, common-author grouping, symlinks,
overwrite, resource bounds, prior-ledger debit and explicit opt-in. Run Ruff,
mypy and the unchanged full coverage gate; new service coverage targets 95%.
Reports distinguish preparation, dataset suitability and training readiness.

## Next training plan

The admission pilot fixes observed quantities only. A later draft must state
the required independent groups per domain, matched original/control counts,
whole-origin splits, two model seeds, batch, epochs and exact updates. The
existing T4 forward/backward timing supplies only that measured component.
Optimizer/evaluation/I/O and full wall-clock costs stay unmeasured. No real
fit, optimizer measurement or additional GPU command is authorized here.
