# Native PNG/BMP admission results — 2026-10-23

**ready / hazır, admission pilot only:** six hash-verified originals,
**three native L + three native RGB**, now produce 24 PNG/BMP artifacts in
six conservative author groups. All twelve independently decoded generated
payloads, single-LSB checks and exact decoded pixel/hash checks pass. Every
role remains **unassigned**. No optimizer, fitting, new GPU attempt or
reserved-media opening occurred. This is not a representative training set.

| Original | Native mode | Credited author | License | Status |
|---|---|---|---|---|
| Female Back | L | Elements Of This World | CC BY 2.0 | ready |
| Leading lines at fountains abbey | L | Rob Ellis from Leeds, England | CC BY 2.0 | ready |
| MichaelBirkin Profile2015 | L | Red Peak Group | CC BY 3.0 | ready |
| Vista del atardecer de Cusco desde el Abra Corao | RGB | Naturegia | CC BY 4.0 | ready |
| Vigna delle Sanzioni | RGB | Vigna delle Sanzioni | CC BY 4.0 | ready |
| Jatindrapur karac bagan | RGB | AkdAnnoy | CC BY 4.0 | ready |
| UNE CUNDINAMARCA FOTOGRAFIA EN EL PARQUE | RGB candidate | RS599 | CC0 | unavailable: first acquisition stopped before storing it; cause not retained |
| Monti Sibillini da Smerillo-banner | RGB candidate | P feliciati | CC BY 4.0 | unavailable: HTTP 429; no retry |
| Huancar Rock, sandboard en el cerro del Huancar | RGB candidate | Mary Lennon Mendez | CC BY 4.0 | not attempted |

The [reviewed source manifest](../benchmarks/png-bmp-native-reviewed-20261023.json)
and [source-review record](../benchmarks/png-bmp-native-source-review-20261023.json)
retain each full title, author, Commons page/revision, original URL, license
URL, upstream SHA-1, size and evidence hash. CC BY 2.0/3.0 handling is in a
separate service; the frozen capacity/admission-v1 sources are unchanged.
Published Flickr review markers and the public Red Peak Group permission
marker were recorded. Private permission-ticket contents were not accessed
or independently certified.

Metadata discovery checked **237 distinct file pages** in twelve bounded
API responses, totaling **1,102,869 metadata bytes**. Its
[receipt](../benchmarks/png-bmp-native-metadata-discovery-20261023.json)
distinguishes metadata from media. The earlier example graphic was excluded
as non-natural imagery; the earlier volleyball candidate declares alpha and
was excluded without downloading media. A black-and-white appearance alone
did not qualify a file as native L: the Allen Diba candidate declares RGB.

The [first acquisition failure](../benchmarks/png-bmp-native-first-acquisition-failure-20261023.json)
preserves its unavailable status and full **40,191,158-byte** reservation.
The later [unattempted-source amendment](PNG_BMP_NATIVE_UNATTEMPTED_AMENDMENT_20261023.md)
made one request each for four previously unattempted sources, adding
15,116,236 reserved bytes. Three passed; one returned HTTP 429. No failed
source was retried or replaced. Cumulative conservative debit is
**55,307,394 /67,108,864 bytes**. Confirmed successful original bytes total
**19,546,814**; the exact received count for the first failed request is
unknown and remains covered by its reservation. Historical cumulative
transfer bytes remain null, with old receipts hashed into the prior anchor.

The [actual original inventory](../benchmarks/png-bmp-native-original-inventory-20261023.json)
adds verified original SHA-256 and native modes. The pre-acquisition manifest
retains its original `not_downloaded` snapshot; current acquired status lives
in the inventory. The [prepared inventory](../benchmarks/png-bmp-native-prepared-20261023.json)
lists all 24 artifact hashes, source groups, credits/licenses and changes.
The [attribution sidecar](PNG_BMP_NATIVE_ATTRIBUTION_20261023.md) preserves
full source titles, author credits, license links and modification notices;
the same file accompanies the prepared images on local disk.
Prepared image output is **2,323,003 bytes /256 MiB**. Raw and prepared media
remain private on the local working disk; no originals or weights are
published or uploaded to Drive.

Independent inspection rechecked original SHA-1/SHA-256, PNG IHDR depth/type,
complete PNG CRC verification, six unique raw/crop hashes, all 24 artifact
hashes, twelve scalar payload recoveries and unchanged upper seven bits.
All fifteen full-image dHash comparisons exceeded the review threshold of
eight differing bits; the contact sheet showed distinct visible subjects
and scenes without an obvious duplicate. These heuristics and visual review
cannot prove absence of shared capture ancestry, hidden payloads or previous
processing. See [independent inspection](../benchmarks/png-bmp-native-independent-inspection-20261023.json).

The [fixed-crop quality record](../benchmarks/png-bmp-native-crop-quality-20261023.json)
also matters: the Michael Birkin crop is mostly white background (sample
standard deviation 3.233; pooled entropy 2.025 bits), and the Female Back crop
is mostly pale wall (standard deviation 6.430). Their native-mode/payload
checks pass, but that does not establish useful training texture. They remain
pipeline fixtures; no source was silently filtered or recropped after review.

| Component | Status |
|---|---|
| Source/license/native-mode admission pilot | **ready / hazır**: six originals, six groups |
| PNG/BMP grouping, pixel and payload controls | **ready / hazır**: 24 artifacts /12 recoveries |
| Requested nine-file acquisition | **unavailable as a whole**: six complete, two unavailable, one unattempted |
| Representative native training corpus | **missing / eksik** |
| Native training loop /validated learning recipe | **missing / eksik** |
| Optimizer/evaluation/I/O and full training cost | **unavailable** |

The [next training draft](PNG_BMP_NATIVE_TRAINING_DRAFT_20261023.md) now fixes
a proposed 128 originals per domain, whole-origin grouping, two seeds per
model, batch eight and twenty epochs, with 1,600 proposed updates. It records
an inventory gap of 250 originals and unmeasured costs; training suitability
of the six pilot examples still requires review. It is not executable. No role,
retry, training or new GPU permission follows from this admission result.

Generated service QA passes 52 tests with **100% service coverage**. The
unchanged full gate passes **3,888 tests /two explicit CUDA skips** in
1,462.60 seconds, with **97.05% total coverage**. Ruff, mypy (231 files) and
`git diff --check` pass. The Python 3.12–3.14 full CI matrix remains unavailable
locally. See the [complete verification record](../benchmarks/png-bmp-native-verification-20261023.json).
