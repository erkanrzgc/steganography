# PNG/BMP Colab preparation — 2026-10-23

The portable [notebook](../notebooks/png-bmp-colab/png_bmp_capacity.ipynb),
adjacent `sources.json`, GPU environment lock and
[source-bound protocol](PNG_BMP_COLAB_PROTOCOL_20261023.md) are prepared.
Open the notebook in Colab, select Runtime Version 2026.07 with a GPU, and
provide its matching source package. Measurement is an explicit notebook
toggle. The fixed base commit plus verified overlay works without publishing
a branch. The notebook has no training or default acquisition cell.

The later [actual Pro/T4 capacity run](PNG_BMP_COLAB_CAPACITY_RESULTS_20261023.md)
completed all twelve cells in 16.034616 seconds, with zero optimizer updates.
That one profile is consumed; preserve its evidence and do not rerun it.

| Component | Status | Evidence / remaining work |
|---|---|---|
| Existing gray/RGB SRNet skeletons | **ready / hazır** | Reused unchanged; generated FP32 forward/backward tests |
| Colab handoff and pinned environment | **ready / hazır** | Notebook, deterministic source package and per-file hashes checked; exact pinned environment passed in the actual Pro session |
| Native admission service | **ready / hazır** | Generated L/RGB originals; exact pixels, whole-origin grouping and all scalar-decoded PNG/BMP controls verified |
| Commons metadata manifest | **ready / hazır** | Two file-level author/license/original URL/SHA-1/byte records; metadata only |
| Approved native gray/RGB corpus | **missing / eksik** | Ancestry review, near duplicates, native mode and natural-image suitability not yet verified; roles remain unassigned |
| Numeric interruption format | **ready / hazır** | Generated model/BN/AdamW slots, RNG, permutation, mode and consumption round trips; corrupt extent/hash/environment/target rejected |
| GPU capacity result | **ready / hazır** | Tesla T4; all twelve cells passed, 16.034616 /600 seconds; zero updates |
| Two-model/two-seed cost table | **ready / hazır** | All twelve batch/seed rows measured; original receipt package verified |
| Optimizer, evaluation and media I/O cost | **missing / eksik** | The capacity command measures only forward/backward passes |
| Real-training data size and budget | **missing / eksik** | To be frozen after actual capacity measurements and data admission; no real fit authorized |

The numerical CPU checks do not substitute for a GPU measurement. CUDA
orchestration tests use test doubles and are software tests only. Missing GPU,
OOM, unsupported runtime, partial cells or deadline must stay `unavailable`.
Keep the consumed-attempt marker and partial receipts; do not automatically
restart a disconnected Colab session. No optimizer is constructed by profiling.

Commons file pages and the metadata-only ImageInfo query were checked on
2026-10-23; the query agreed with both published upstream SHA-1/byte records:

- [Example png.png](https://commons.wikimedia.org/wiki/File:Example_png.png):
  Mvolz, CC0. This is an Inkscape example graphic and **not a natural-image
  training source**; retain it only as a metadata/pipeline candidate.
- [Volleyball landscape](https://commons.wikimedia.org/wiki/File:Volleyball_game_brings_the_landscape_to_life.png):
  Chettriabhimanyu, CC BY 4.0. Native mode, common ancestry, near duplicates
  and suitability require review before any admission. No media bytes opened.

The [candidate manifest](../benchmarks/png-bmp-commons-candidates-20261023.json)
marks both ancestry reviews pending, so acquisition and preparation reject it
even with the explicit download switch. Original SHA-256 and native mode are
not invented from metadata. No BOWS2/DIV2K import, source role assignment,
reserved-data access or download occurred. Historical transfer totals remain
incomplete and are not reset. JPEG/WAV time, snapshots and receipts are unchanged.

Final verification covers **3,836 passing tests /two explicit CUDA skips**,
**97.03% total coverage** and **98.18% across the new modules** (each >95%).
Ruff, mypy (230 files) and `git diff --check` pass. The complete run recorded
one README word-count failure before its final shortening; the failed-test
replay passes. The final notebook template additionally passes all 107
affected tests, including a regression check that lints the generated notebook.
Old coverage for the changed template was discarded before recomputing it;
the unchanged 90% overall gate still passes. The local run uses Python 3.11;
the Python 3.12–3.14 full CI matrix remains unverified locally. The later actual
capacity command passed on Colab Python 3.12.13; that is not a full CI run.

Verification is recorded in
[`png-bmp-colab-verification-20261023.json`](../benchmarks/png-bmp-colab-verification-20261023.json).
Its local-GPU-unavailable/zero-attempt fields describe the initial preparation
snapshot. The later [capacity verification](../benchmarks/png-bmp-colab-capacity-verification-20261023.json)
records the consumed actual GPU measurement separately.
First sandbox/full and intermediate interrupted logs are retained on the
local disk; an interrupted run is unavailable, not a passing gate.
