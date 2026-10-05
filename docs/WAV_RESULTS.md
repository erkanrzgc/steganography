# FSDD WAV baseline — 2026-10-05

The existing automatic detector failed all six tested method/rate cells:
**ROC-AUC 0.50 and recall 0%**. Every cover and stego received overall score zero.
Changing the threshold cannot separate tied scores. A completed experiment
and exact generation verification do not establish reliable detection.

## Preregistered source and ground truth

`WAV_PROTOCOL.md` and the runner were committed as `28f4d92` before scoring
actual recordings. The existing shared AnalysisService was used without
detector changes, balanced profile, AI off and threshold 70.

FSDD v1.0.10 supplies real spoken-digit mono PCM16/8 kHz recordings. Whole
speakers george/jackson contributed 500 original covers each to this frozen
test. Lucas/nicolas remain reserved training (1,000 originals), theo/yweweler
reserved validation (1,000 originals). No model was trained or calibrated.
This is one narrow speech dataset; speaker separation does not establish
device separation, music/stereo performance or cross-source support.

For each evaluation cover, the independent research generator produced
marker-free random payloads using sequential and scattered PCM sample-LSB
replacement at requested 0.05, 0.20 and 0.40 bits per sample. Actual rates round
down to whole bytes and are retained per sample. Original RIFF metadata and
file lengths were preserved, and sample changes were bounded to one.

All 6,000 generated payloads round-tripped exactly with a separately implemented
scalar extraction oracle and known public sample ordering. A subsequent audit
independently rederived payloads/order and rechecked recovery on all 6,000.
This is controlled embedding ground truth; native/unknown-key CTF recovery
was not evaluated. These recipes are not claims about Steghide or OpenStego.

## All method/rate results

Each row compares 1,000 stegos against the same 1,000 covers. There are **7,000
unique evaluation files**, not 12,000 independent observations.

| Placement | Requested bits/sample | TP / FN | TN / FP | ROC-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential | 0.05 | 0 / 1,000 | 1,000 / 0 | 0.50 | 50% |
| Sequential | 0.20 | 0 / 1,000 | 1,000 / 0 | 0.50 | 50% |
| Sequential | 0.40 | 0 / 1,000 | 1,000 / 0 | 0.50 | 50% |
| Scattered | 0.05 | 0 / 1,000 | 1,000 / 0 | 0.50 | 50% |
| Scattered | 0.20 | 0 / 1,000 | 1,000 / 0 | 0.50 | 50% |
| Scattered | 0.40 | 0 / 1,000 | 1,000 / 0 | 0.50 | 50% |

Recall and FPR are 0% in every cell. All numeric discrimination/recall targets
fail; zero false alarms do not compensate for missing every stego. The 200-draw
paired-lineage bootstrap yields AUC CI [0.50, 0.50], balanced accuracy [50%, 50%],
recall/FPR [0%, 0%]. These degenerate within-selection intervals follow from
the tied scores; they do not imply certainty about another source.
ECE is unavailable because heuristic scores are uncalibrated.

## Integrity, coverage and reproducibility

All required components (audio_wav, filestruct_appended, file_structure,
signatures) returned `ok` on all 7,000 files, as did optional ExifTool.
Stegseek was unavailable on the host; unrelated format analyzers were unsupported.
This measures the available native configuration. No analysis failure was
counted as a negative and no selected example or failed real run was removed.

Independent auditing reread all file sizes/hashes, found 7,000 unique hashes,
checked all 3,000 original role declarations and zero development/test ancestry
overlap, retained PCM/header bytes and bounded changes/order. Independent
positive/negative comparisons reproduced all confusion matrices/AUC and the
degenerate bootstrap intervals. The detector source fingerprint is unchanged
from the prerun reference. Source recordings and generated files remain local.

The complete generation/evaluation/summary run took **142.90 seconds**, including
53,965,940 corpus bytes. Analysis median was 0.0741 seconds, p95 0.1250 seconds
on the 8-vCPU VMware guest, Ryzen 9 8945HX, four workers, one math thread.
No full test suite ran concurrently with this evaluation. These are small
speech-file analysis timings, not CTF recovery latency gates.

Reproduce the fixed command in [WAV_PROTOCOL.md](WAV_PROTOCOL.md) with a new
output directory. The portable [aggregate](../benchmarks/fsdd-lsb-baseline-20261005.json)
contains code/protocol/source/manifest/score hashes, all cells/intervals and
audit notes. Full local scores are `.benchmark/fsdd-lsb-baseline-20261005/scores.jsonl`.
The inspected test speakers remain reserved for regression; new feature/model
development uses only the separately reserved speakers. Independently untouched
source evaluation remains required before promoting WAV support.
