# Independent pilot results — 2026-09-18

The base analysis configuration failed to distinguish the two tested external
embedding families. The full CTF recovery workflow succeeded on the 30 controlled
examples. These measure different capabilities; recovery success must not be
presented as detector accuracy.

## Data and ground truth

Downloaded 1,000 actual BOSSbase images from the official DDE archive. Each was
converted from grayscale PGM to RGB without resizing and used for one cover/
stego pair: 500 Steghide/BMP and 500 OpenStego/PNG pairs, with requested payloads
of 256, 1024 or 4096 bytes. Every one of the 1,000 stegos was independently
extracted using its upstream tool and matched its expected payload byte for byte.

Source selection is archive order, not representative random sampling. Camera
metadata is unavailable. BOWS2's checked host did not resolve; no second source
was acquired. The cross-source gate therefore remains **unavailable**.

## Detection baseline

Balanced profile, AI disabled, threshold frozen at 70 before evaluation. Base
environment had ExifTool, but no zsteg or Stegseek. No tuning or training used
these observations.

| Measurement | Result |
|---|---:|
| Clean / stego files | 1,000 / 1,000 |
| True positives / false negatives | 0 / 1,000 |
| False positives / true negatives | 0 / 1,000 |
| Recall | 0% |
| False-positive rate | 0% |
| Balanced accuracy | 50% |
| ROC-AUC | 0.499021 |
| Paired-lineage bootstrap 95% AUC interval | 0.496142–0.501800 |
| Analysis errors | 0 |

The zero FPR is not a useful success on its own: every stego was missed at the
deployed threshold. AUC near 0.5 means merely moving the threshold is unlikely
to fix this configuration. This result is specific to these methods, payloads,
conversions and source; it is not evidence about all possible steganography.

## Recovery baseline

Full Docker, non-root, read-only root, network disabled: **30/30 exact payloads**
recovered and all jobs completed. Five examples each covered Steghide/BMP,
OpenStego/PNG, base64, gzip, ZIP and base64-wrapped ZIP. The Steghide password was
provided. No unknown-password cracking success is claimed.

Overall median was 0.162 seconds and p95 was 15.733 seconds. The 20 easy decoding
examples dominate the median: Steghide's median was 15.581 seconds and
OpenStego's was 9.440 seconds. Detection and recovery shared an 8-vCPU VM, so
these are pilot observations, not the final reference-hardware latency gate.

This is not a blind suite, 30 independent method families, or the planned
120-challenge acceptance test.

## Actions and next experiment

- Fixed OpenStego preferences initialization: it previously could print an
  access error and exit zero. The Docker smoke now demands an actual output
  file and exact independent OpenStego extraction.
- Keep these results frozen. Statistical detector support for the tested
  families is unproven; do not market the base package as reliably detecting them.
- Separate development data from this baseline before implementing stronger
  spatial statistics/features or training/calibrating a model. Threshold
  selection belongs to validation data, never to this published test set.
- Acquire a second source with documented usage conditions and camera/lineage
  metadata. Evaluate the revised model on an untouched source before promoting
  any support cell. JPEG and mobile-app evaluation remain separate work.

Machine-readable aggregate: `benchmarks/pilot-20260918.json`.
Protocol and commands: `docs/PILOT_PROTOCOL.md`.
Local full results: `.benchmark/pilot-detection-v1/report.json`,
`.benchmark/pilot-ctf-v1/report.json`; upstream round-trip verification:
`.benchmark/pilot-downloads/groundtruth.json`.

No source images or payloads are committed. The source manifest and results
include hashes so the local evidence can be checked against the published
aggregate.
