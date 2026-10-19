# Non-JPEG preparation results — 2026-10-15

This is the historical preparation-only receipt. The later, separately budgeted native linear pilot completed on October 16; see [the amended comparison results](NONJPEG_PROCESS_PREFLIGHT_RESULTS.md). That later fit does not change the preparation-only statements or missing-source states below.

Preparation and generated model checks are available; no real-data model was
trained, scored, calibrated, exported or deployed. Statistical detection and
production qualification remain **unavailable**. The frozen JPEG source closure
is unchanged. Separate non-JPEG training has not been added to its six-hour
budget. The [protocol](NONJPEG_EVENING_PROTOCOL.md) contains descriptor definitions,
limits and the proposed two-seed exposure/measurement design.

## Reused data and concrete outputs

The fresh read-only independent audit verifies all **160 original lineages**,
**4,080 artifact checksums**, **3,840 scalar payload recoveries** and 192 recipe
cells. Corpus identity remains
`a5bccf330c6a933246e1bd9ac4689e05167960939da4f2ff9771073ed8e90e2a`.
Historical speaker/recording roles and all original source/code bindings pass;
test media stayed closed. Shared PNG/BMP/channel derivatives do not increase
the independent denominator. Original corpus bytes remain 613,635,514.

Ten separate source/split feature artifacts were produced, each below the
unchanged 64-MiB limit. Eight separate input manifests bind their source, role,
original lineage and fresh audit. No filename/source/oracle values enter the
feature matrices. Each artifact is reread through the existing bound input
validator before delivery.

| Source | Train /validation rows | Local /RGB /PCM dimensions |
| --- | --- | --- |
| BOWS2 | 576 /144 | 1,548 /— /— |
| DIV2K | 2,112 /528, each spatial version | 1,548 /1,332 /— |
| ESC-50 | 288 /72 | — /— /760 |
| FSDD-v1.0.10 | 288 /72 | — /— /760 |

The three generated FP32 model checks pass: gray SRNet 4,779,616 parameters,
RGB SRNet 4,780,768 parameters and raw/first-/second-difference PCM CNN 186,882
parameters. Every parameter has finite gradients and remains unchanged; input
gradients are finite and nonzero. There were zero optimizer updates, downloads
or checkpoint writes in these checks. Optional Torch absence is explicitly tested
as unavailable. These are initialization/gradient checks, not learned accuracy.

## Native pilot attempt and normal Unicode controls

The complete 40 PNG /20 GIF /20 stereo WAV /20 MP3 new-original targets are **unavailable**.
Commons metadata access works, but original-image requests returned HTTP 429.
The initial attempt was stopped; its partial artifacts/logs are retained. The
interrupted pre-sidecar implementation did not persist per-file source rows for
thirteen native GIFs. One subsequent metadata-only API request recovered their
page IDs, authors and licenses: all 13 saved byte streams match the API's SHA-1
and size, have recorded local SHA-256 and unique decoded identities. No original
media was downloaded again. This provides **13/20 source-bound native GIF files**,
with seven missing. Their independent ancestry remains unavailable; they stay
unlabeled and `development_unassigned`, outside all training manifests.
PNG accepted zero; WAV/MP3 acquisition was not reached. Native original counts
have not been inflated or replaced with
transcodes, replicated grayscale, mono copies or excerpts.

The acquisition implementation now stops Commons requests on HTTP 403/429/503,
persists an immediate accepted-file source sidecar and journals byte counters
around requests. The interrupted first attempt's exact transferred-byte count
is unavailable; no global one-GiB download-attestation pass is claimed for that
attempt. Its individual request and in-process byte bounds were active. No
original-media requests were retried after identifying the rate-limit problem.

All six official Unicode 17.0.0 files downloaded separately: GraphemeBreakTest,
WordBreakTest, SentenceBreakTest, NormalizationTest, BidiCharacterTest and
emoji-test. Exact completed transfer was 10,914,178 bytes, including source
bytes retained with their original copyright/license headers. Hashes and source
URLs are in the portable receipt. Controls provide 32 examples per category
for ZWJ, ZWNJ, whitespace, normalization, bidi and emoji, with overlaps between
categories. Every row preserves source hash, line number and literal codepoints;
normalization rows retain five normative columns. These are conformance
negatives, not natural-document coverage or measured false-positive rates.

## Every method's readiness

| Family | Status | Available preparation | Remaining gap /approach |
| --- | --- | --- | --- |
| PNG/BMP gray LSB | **ready** for preparation | Native BOWS2, fixed original roles, scalar payload oracle, local parity, gray SRNet skeleton | Two-seed matched learning comparison; camera independence unknown |
| PNG/BMP native RGB/channel LSB | **incomplete** | Native DIV2K, channel/plane/order matrix, local/RGB parity, RGB SRNet skeleton | Second licensed native RGB origin unavailable; compare original-matched baseline/local/channel arms |
| WAV mono PCM16 LSB | **ready** for preparation | Speech/environmental origins, historical groups, scalar oracle, 760 features, PCM CNN skeleton | Two-seed grouped learning comparison versus historical failed detector |
| WAV stereo/channel difference | **incomplete** | Existing independent left/right channel regression fixtures; native acquisition validator rejects duplicated mono | Licensed native stereo; opaque left/right/both payload oracle with unchanged other-channel bytes and maximum bit-plane distortion bounds |
| Frequency-domain audio | **incomplete** | Native inspection/regression support | Freeze actual frequency encoder, payload capacity, inverse transform, SNR/peak distortion and extraction; no PCM-LSB inference from spectrogram results |
| GIF comment/loop/application | **incomplete** | 13 source-bound native GIF files with file-level licenses; existing comment/trailer fixtures and benign metadata inspection | Seven files missing; original ancestry, opaque payload/benign metadata matrix and source-independent grouping not established |
| GIF palette/timing | **incomplete** | Existing palette/delay structural fixtures | Actual palette/timing encoder, duration/visual distortion bounds, independent extraction and benign animations |
| Text zero-width/whitespace | **incomplete** | Existing extraction/round-trip fixtures plus pinned ZWJ/ZWNJ/whitespace/emoji controls | Licensed whole-document multilingual sources and a source-independent opaque payload decoder matrix |
| Text bidi/normalization/homoglyph | **incomplete** | Pinned bidi and normalization conformance controls | Natural Arabic/Hebrew/Persian/mixed-script documents, defined encoder/independent decoder; controls alone cannot establish false-positive rate |
| Container/append/polyglot | **incomplete** | Existing native CTF recovery, benign metadata and structural regression fixtures | Opaque-message oracles and independent blind original-container custody; flag fixtures are not opaque-message evidence |
| Archive/decoder recursion | **incomplete** | Existing depth/member/ratio/byte/path/time regression limits | Licensed natural benign corpus, complete opaque non-root hash denominators and blind recovery evidence |
| MP3/compressed audio | **unavailable** | Native Layer III inventory validation design; spectrogram research design | Native licensed sources, pinned real encoder/extractor, payload/distortion oracle and full decoder validation |
| Upstream tool families | **incomplete** | Existing pinned tool/license inventory in `tools/full-image.lock.json` | Actual method-specific native positives and successful byte extraction with recorded versions/licenses; inventory is not recovery |

Fixture/parser success is never reported as statistical detection success.
The base wheel remains model-free, without a new executable requirement.

## Verification and resources

The new five core modules pass 61 scoped tests at **98.37%** combined coverage;
feature and model implementations are each **100%** covered. Independent scalar
histograms include signed PCM extrema, lag/bit ordering, quarters and single-LSB
preservation. Corrupt input, wrong hash/domain/split, symlink, overwrite and
size/time bounds are exercised. Ruff, mypy and diff checks pass.

The first full-suite attempt is **unavailable**, stopped for accumulated generated
test scratch at 4,354,943,004 bytes (about 57.2 MiB above the four-GiB non-JPEG
allocation after a burst between watchdog samples). This is not reported as a
passed resource check. Its log and resource receipt remain intact. Only completed,
passing, task-owned generated function-test fixtures were cleaned, with a private
cleanup receipt; the latest three fixture directories were retained. Real source
datasets, the prepared corpus and historical checkpoints were never removed.
The retry uses pytest's standard `tmp_path_retention_policy=failed`, preserving
failed-test scratch and removing passed function-test temporaries at teardown,
plus a larger watchdog margin. Coverage thresholds and tests are unchanged.

The full unchanged gate passes **3,073 tests /two actual-CUDA skips** in
1,492.56 seconds with **96.64% total coverage**. The successful retry's peak
task-output footprint is **793,377,955 bytes** (about 757 MiB), comfortably
below four GiB; its maximum child RSS is 2,674,536 KiB, not aggregate process
memory. Ten feature artifacts total **101,268,468 bytes**. Final Ruff, 198-file
mypy, tracked/new-file whitespace checks and all 31 frozen JPEG source hashes
pass. Only Python 3.11 was exercised locally; 3.12–3.14 remain unexecuted here.
The complete evidence is in
[`benchmarks/nonjpeg-evening-readiness-20261015.json`](../benchmarks/nonjpeg-evening-readiness-20261015.json).

Initial focused invocations
also passed their test assertions but failed the repository-wide coverage gate
because unrelated source files were intentionally unexercised. Their logs are
preserved; the separate scoped run uses a stricter 95% gate for new modules.
Full-suite verification retains the repository's unchanged 90% total gate.
