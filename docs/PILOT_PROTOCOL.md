# Independent pilot protocol (frozen before first evaluation)

This is an exploratory baseline, not a support-cell certification or blind CTF
benchmark. Do not tune thresholds against its results and call the re-run a new
held-out result. Subsequent tuning needs a separate development corpus and a
previously unused final test source.

## Source acquisition

Explicit command:

```sh
python scripts/fetch-boss-pilot.py --out .benchmark/boss-pilot-1000 --count 1000
# Interrupted download: repeat with --resume (existing images are CRC-checked).
```

The fixed URL is the Binghamton DDE official BOSSbase 1.01 archive. ZIP range
reads check ETag/CRC and record each image SHA-256. There is no published SHA-256
in the retrieved catalog, so `upstream_sha256_verified` remains false. This is
integrity/provenance tracking, not an upstream signature verification.

Selection is the first 1,000 image members in archive order, chosen to limit
network traffic. It is not random or camera-stratified. Image-level camera and
device metadata are unavailable and remain null. The download page provides
no explicit license grant; preserve the source attribution and keep images
local for this research evaluation. Do not redistribute them or infer
commercial/model-distribution rights. Large corpora remain under ignored
`.benchmark/`, never in git or the wheel.

The checked BOWS2 host failed DNS resolution. ALASKA2 and StegoAppDB have not
been acquired. Consequently the independent-source gate is **unavailable**.

## Generation and isolation

Run generation in the full image with the current source tree mounted read-only
at `/app` and `.benchmark` as the writable `/data` directory:

```sh
python -m steganography.benchmarking.pilot generate \
  --source /data/boss-pilot-1000 --out /data/pilot-v1 --pairs 1000 --workers 4
```

Each original image is used for exactly one detection pair. Alternate between
Steghide/BMP and OpenStego/PNG, converting PGM to RGB without resizing. Requested
payload sizes are 256, 1024 and 4096 bytes. Requested bits per pixel do not
include framing, encryption or coding overhead and are not actual modified-bit
rates. Duplicate original hashes are rejected. Covers and all their derivatives
retain the same lineage and split. All pairs are baseline test observations;
no training or calibration is performed in this run.

The generators are independent upstream executables, not this project's
embedding code. Cryptographic randomness inside an upstream tool can make stego
bytes differ between generations; manifests record the actual bytes' hashes.

Thirty recovery examples comprise five each of Steghide/BMP, OpenStego/PNG,
base64, gzip, ZIP and base64-wrapped ZIP. A known public fixture password is
supplied for Steghide. Challenge filenames do not disclose methods. Evaluator
ground truth is outside the solver's job directory. These controlled examples
do not represent 30 distinct real-world embedding families or a blind suite.

## Evaluation

```sh
python -m steganography.benchmarking.pilot detect \
  --source .benchmark/pilot-v1 --out .benchmark/pilot-detection --workers 4
python -m steganography.benchmarking.pilot ctf \
  --source /data/pilot-v1 --out /data/pilot-ctf
```

Detection uses the existing balanced profile, AI disabled, threshold 70. Run
the initial detector evaluation on the base/native environment; record optional
tool absence in per-sample coverage. It does not measure full-image detection
accuracy. Report confusion matrices, ROC-AUC, recall, FPR, balanced accuracy,
per-method results and paired-lineage bootstrap 95% intervals (200 resamples).
Scores remain uncalibrated heuristic values, not estimated probabilities.

Recovery runs in full Docker without network and compares SHA-256 against the
exact expected payload bytes, never substring/flag-shaped matches. Missing
tools/examples count against the requested denominator. Record incomplete jobs
alongside any recovery, and median/p95 runtime. Time limits are 5 seconds per
optional tool and a cooperative 60 seconds per challenge for this pilot.

Preserve the first reports. Publish aggregate evidence and failures, not source
images or extracted payloads. No result from this pilot can satisfy the planned
1,000-cover/1,000-stego **per-cell**, two-independent-source release gate.
