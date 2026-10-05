# FSDD LSB baseline protocol — 2026-10-05

Freeze this protocol and runner before first scoring actual FSDD recordings.
Use acquired v1.0.10 source manifest SHA-256
`f4199c1e58b4f8be1ffbcb1924cdcdb980daa46b52a537f6f7c845273fbf8b4d`.
No new downloads, model training, calibration or threshold search.

Whole-speaker partition: george/jackson are frozen evaluation (1,000 originals),
lucas/nicolas are reserved training, theo/yweweler reserved validation. Keep all
originals and their derivatives in one split. Speaker separation does not prove
device separation or independent dataset sources. Never tune on evaluated files.

For every evaluation original generate six marker-free random-payload examples:
sequential and scattered signed-PCM16 LSB replacement, at requested 0.05, 0.20
and 0.40 bits per sample. Round payload down to complete bytes; record actual
rate, payload hash and changed samples. SHAKE256 payload and public ordering
seed derive from 20261005, original SHA-256, method and rate. Scattered positions
use Python Random.sample without replacement. Every sample changes by at most
one; original RIFF header/metadata bytes and file length are preserved.

The research generator does not use the application's Carrier embed/extract.
A separately implemented scalar PCM/byte-assembly oracle checks exact payload
recovery for every generated example, with known ordering. This establishes
controlled embedding ground truth, not native CTF recovery or upstream
Steghide/OpenStego compatibility. Do not inject flags, markers or length headers.
Keep source audio and generated derivatives local under the FSDD usage conditions.

Score the 1,000 original covers once and all 6,000 stegos with unchanged shared
AnalysisService, balanced, AI off, threshold 70. Each method/rate cell compares
1,000 stegos with the same 1,000 covers. Required native coverage is audio_wav,
filestruct_appended, file_structure and signatures. Any analysis error, missing
required component or failed sample invalidates the cell, never counts as clean.
Record optional coverage, input hashes, runner/detector/protocol fingerprints,
full per-file scores and failure classes without paths/secrets from exceptions.

Publish all six confusion matrices, ROC-AUC, balanced accuracy, recall and FPR;
95% percentile bootstrap intervals use 200 draws of complete cover/stego
lineages, seed 20261005. No threshold recommendations or tie-order average
precision. ECE/calibration and cross-source support remain unavailable.
One speech source does not establish general audio, stereo or MP3 accuracy.

Limits: 1 MiB per WAV, 80,000 frames, 64 RIFF chunks, 3,000 originals,
1,000 evaluation lineages, 512 MiB corpus output, 8 MiB manifests,
32 MiB score artifact, four workers. Use an outer 1,800-second timeout.
Report duration/hardware; cooperative/native parser limitations still apply.
Write only exclusive nonsymlink outputs; keep incomplete attempts visible.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --kill-after=10s 1800s python -m steganography.benchmarking.wav \
  --source .benchmark/fsdd-v1.0.10-20261004 \
  --source-sha256 f4199c1e58b4f8be1ffbcb1924cdcdb980daa46b52a537f6f7c845273fbf8b4d \
  --protocol docs/WAV_PROTOCOL.md --out .benchmark/fsdd-lsb-baseline-20261005
```
