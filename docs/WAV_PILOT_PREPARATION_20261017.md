# AudioMNIST WAV training preparation — 2026-10-17

The separate AudioMNIST WAV pilot is prepared through data validation and
matched training code. The generated linear preflight passed; the CUDA capacity
preflight of the complete trainer remains unavailable because the pinned SSH
loopback endpoint is down. No source-data fit, checkpoint, accuracy result or
reserved-speaker access occurred.

The pinned source is AudioMNIST commit
`630d7dab4c040882834de6fa21baf9a60372accd`. Its repository declares MIT, and
the paper states that all 60 speakers gave written consent before recording.
The acquisition tool downloaded only the frozen 20 train and ten development
speaker groups: 400 and 200 files respectively. The 30 reserved speaker groups
remain closed. Parent-recording ancestry is unavailable, so all cuts stay
grouped by speaker. The existing incomplete cumulative transfer ledger was not
reset; this separate acquisition used 36,335,714 raw bytes under a 384-MiB cap.
All 600 source files are retained in the ignored local research-data directory;
the tracked receipt contains only bounded metadata and hashes.

Every WAV passed mono, 48-kHz, uncompressed PCM16 and 160–220,500-frame checks.
Raw-file and decoded-PCM hashes are unique. Lengths range from 17,132 to 47,478
frames (median 29,809). Each source has one cover and eight deterministic
sequential/scattered 5%/40% LSB-plane variants. Independent payload recovery
passed for all 4,800 variants, with no derivative files retained. The 3,600
train rows also produce finite `[0,1]` 760-feature vectors; the in-memory
matrix shape is `(3600, 760)` and was not written to disk.

The matched arms are the existing 760-feature linear reference and the native
FP32 1D residual CNN using raw PCM and first/second differences. Both use the
same two seeds, 20 epochs, row permutations, virtual batch of 64 (last batch 16),
SGD at 0.01, weight-only L2 0.0001, equal cover/stego mass, and fixed threshold
0.5. Each arm plans 144,000 row presentations and 2,280 updates; the full
comparison plans 288,000 presentations and 4,560 updates. Development reports
paired ROC-AUC, balanced accuracy, recall, false-positive rate and ECE, with
uncertainty clustered by the ten speakers. No OOF fold or threshold search is
used. The CNN pair has a shared one-hour GPU ceiling; the linear pair has a
separate ten-minute CPU ceiling. The JPEG budget is unchanged.

The source-bound generated linear preflight used synthetic WAVs at the 600
observed frame lengths from preparation metadata. Both seeds completed 72,000
row presentations and 1,140 updates each; all eight evaluation cells were
computed, in 44.82 seconds, with no checkpoint. Its receipt is
[here](../benchmarks/wav-audiomnist-linear-generated-preflight-20261017.json).
It establishes the linear code path and CPU allowance only; its synthetic
metrics are not model accuracy.

The [training protocol](../benchmarks/wav-audiomnist-training-protocol-20261017.json)
requires a generated CUDA run for both model arms. The previous remote attempt
refused `127.0.0.1:22240` before starting any remote unit; local Torch and CUDA
are unavailable. The failed attempt remains
[recorded here](../benchmarks/wav-audiomnist-preflight-attempt-20261017.json).
Reconnect using [`WSL_GPU_RECONNECT.md`](WSL_GPU_RECONNECT.md) before rerunning
the CUDA preflight. No source-data fit is authorized by the protocol, and
training readiness and accuracy remain **unavailable**.

Ruff and targeted mypy pass; the 14 focused WAV tests pass. After installing
the missing cryptography and piexif packages, the complete suite began and
reached 119 passed /23 optional skips with no failures, then was interrupted
after 194 seconds. Focused WAV coverage is 38% (65% in `wav_pilot.py`, 27% in
`wav_pilot_training.py`); the CUDA-dependent trainer paths remain untested
locally. Full mypy still reports 23 dependency/NumPy typing errors in 12 files.

The source repository describes 30,000 spoken-digit files from 60 speakers and
the paper records the consent and capture details. See the
[official repository](https://github.com/soerenab/AudioMNIST/tree/630d7dab4c040882834de6fa21baf9a60372accd)
and [paper](https://doi.org/10.1016/j.jfranklin.2023.11.038).
