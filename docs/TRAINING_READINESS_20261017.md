# Training readiness — 2026-10-17

**Training is not ready to start.** Owned GPU access and generated model checks
pass, but the proposed new WAV experiment still lacks accepted media, a granted
separate budget, its matched trainer and a frozen execution protocol. No fitting,
media acquisition, threshold search or checkpoint creation occurred.

## Hardware and generated model checks

The existing pinned SSH connection reaches the owned RTX 5060 Laptop GPU
(8,151 MiB). PyTorch 2.14.0+cu130 /CUDA 13.0 initializes successfully. Its
PyTorch UUID hash matches the historical frozen device identity.

Native grayscale/RGB 256×256 SRNet skeletons and the three-channel PCM residual
CNN pass FP32 forward/backward checks on CUDA. The audio checks cover 160,
48,000 and 220,500 samples. Each of the five cases runs three generated passes:
finite logits/input/parameter gradients, unchanged parameters and zero optimizer
updates. AMP and TF32 are disabled. Independent scalar differences, signed
extremes and a single PCM LSB survive the CPU-to-CUDA input transformation.

Peak allocated CUDA memory is **606.30 MiB**; the probe is bounded by a
180-second owned unit, four-GiB host/allocator ceilings and zero swap. Actual
probe time including imports is 5.97 seconds; peak host RSS is 1,647,284 KiB.
BatchNorm buffers belong only to disposable generated models; no state is saved.
These checks demonstrate execution, not detection accuracy.

The first generated probe stopped before any model case because the audit
hashed NVIDIA's GPU-prefixed UUID while the frozen inspector hashes PyTorch's
UUID representation. The failed log/source hashes are retained. A separately
bound correction uses the existing inspector's exact representation and passes;
no historical GPU/source binding was changed.

## Data and integrity

- All five frozen source closures remain unchanged: 18 original native,
  27 amended native, 35 JPEG, 30 WAV diagnosis and four local-document sources.
  The local-document runtime's additional 27 bindings also match.
- All 14 existing model files retain their recorded SHA-256 digests.
- Seven historical **train** feature artifacts pass hash, source, split,
  dimension, ordered-row, finite `[0,1]` and controlled-cell checks. The 684-value
  spatial prefixes agree across arms. Training independence remains 32 BOWS2
  lineages, 32 DIV2K lineages, 31 ESC recording groups and two FSDD speakers.
  Validation manifest metadata confirms disjoint groups; validation feature
  vectors, raw carriers and reserved media remain closed. No OOF fold is refit.
- The 16 local PDF/TXT/MD/ZIP originals /48 controlled artifacts pass checksums,
  all 16 independent payload oracles, matching-length whitespace controls and
  four retained license-file hashes. All roles stay `development_unassigned`;
  recovery still requires the known original extent.
- The 23-suffix inventory remains intact. It does not imply a representative
  dataset or a trainable detector for every format.

Cached AudioMNIST README, MIT license and 60-speaker metadata match their pinned
hashes at commit `630d7dab4c040882834de6fa21baf9a60372accd`. The draft's hash-sorted
20/10/30 speaker roles are disjoint, and its proposed 400 train /200 development
paths are unique and consistently grouped. No media was requested: actual file
existence, native headers, original hashes and file-specific source acceptance
remain unavailable. Parent-recording ancestry is still unknown; cuts stay
grouped by speaker.

## Exposure and capacity

The unchanged draft proposes two seeds ×20 epochs ×3,600 training rows:
**144,000 CNN row presentations**, with 64-row virtual batches and 2,280 proposed
optimizer updates across both seeds. None of these updates was executed.

At the maximum supported 220,500-sample input, the larger of two warm generated
forward/backward timings is 0.03991 seconds. Applying that timing to every
proposed presentation gives **5,747 seconds /95.8 minutes**, before optimizer,
input transfer, extraction/features, evaluation or checkpoint costs. This
maximum-length scenario exceeds the draft's proposed one-hour GPU ceiling.
It is not a forecast for the unacquired recordings or a complete capacity gate;
their actual duration distribution and complete trainer costs are unavailable.
The comparison therefore has no established in-budget execution capacity.

## Software verification

The fresh focused run passes **149 tests** (`--no-cov`); Ruff, 210-file mypy and
`git diff --check` pass. The fresh complete unchanged coverage gate passes
**3,321 tests /96.81% total coverage**, with **two explicit local CUDA skips**.
Those JPEG CUDA/export tests remain unavailable on the CPU host; the separate
remote native generated checks do not turn their skips into passes. Local
Python verification covers 3.11; 3.12–3.14 remain unavailable locally.

The full gate runs under a four-GiB /zero-swap owned unit with fresh disk-backed
temporary storage, failed-only fixture retention and a two-GiB output watchdog.
Its independent child exit is zero. The observer records 1,013.62 seconds,
126,029,429 bytes peak task output and 2,673,420 KiB peak child RSS. Tracked native
storage is approximately 775 MiB, within the four-GiB allocation; six GiB remain
reserved for JPEG. No dataset or checkpoint is deleted. The unit is stopped
after successful verification.

The corrected generated GPU probe's measured time is separate from the first
failed control attempt, whose exact duration is unavailable (180-second owned
watchdog). Per-check measurements do not certify exact cumulative control time.
The earlier completed training allowances are unchanged.

## Required next work

1. Reconcile the incomplete historical raw-transfer ledger or establish an
   explicit separate acquisition scope. No earlier limit is reset, and neither
   the draft nor these checks grant download or training budgets.
2. Acquire and accept only the approved native source files with license,
   source/hash/header and speaker-role bindings; keep reserved media closed.
3. Implement and verify the matched source-specific trainer, then bind a
   separate source/data/exposure/compute protocol with a complete capacity gate.
   Preserve the proposed exposure unless a separate intervention changes it.

JPEG's completed allowance and historical results remain unchanged. No new model
is qualified or deployed. See the non-executable
[WAV draft](WAV_NEXT_PROTOCOL_DRAFT.md) and the complete
[format data inventory](ALL_FORMAT_DATA_STATUS.md). The portable
[check receipt](../benchmarks/training-readiness-20261017.json) records source,
artifact, log and failed/corrected-probe hashes.
