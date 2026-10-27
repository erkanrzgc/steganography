# PNG/BMP Colab capacity results — 2026-10-23

**ready / hazır:** the one generated capacity profile completed all twelve
gray/RGB × two-seed × batch 2/4/8 cells on a Tesla T4 in the verified Pro
account. The command took **16.034616 /600 seconds**, with **zero optimizer
updates**. All twelve single-LSB, finite-gradient and unchanged-parameter
checks passed. No real media was opened or downloaded, and no fitting,
checkpoint or deployment occurred. The attempt is consumed; do not rerun it
automatically or use the remaining allowance for larger batches or tuning.

The [frozen contract](PNG_BMP_COLAB_PROTOCOL_20261023.md), existing native
gray/RGB SRNet skeletons and sixteen-file source closure are unchanged.
Inputs were generated 256×256 FP32; AMP and TF32 were disabled. Each cell
used one warm-up followed by three timed forward/backward passes, without
constructing an optimizer.

| Runtime / software | Observed value |
|---|---|
| Colab Runtime Version selected in UI | 2026.07 |
| GPU | Tesla T4 |
| Python | 3.12.13 |
| Torch / CUDA / cuDNN | 2.11.0+cu128 /12.8 /91900 |
| NumPy / Pillow | 2.0.2 /11.3.0 |
| Gray / RGB parameters | 4,779,616 /4,780,768 |

The table reports the median of three forward/backward intervals. VRAM
figures are Torch allocator peaks per cell, in MiB. Process RAM is a
cumulative high-water mark; its maximum was **1,152.24 MiB**. Maximum
allocated/reserved VRAM was **1,608.18 /2,102 MiB**.

| Model | Seed | Batch | Forward/backward median (ms) | Peak allocated VRAM (MiB) | Peak reserved VRAM (MiB) |
|---|---|---|---|---|---|
| Gray | 2026102301 | 2 | 69.129 | 445.97 | 530 |
| Gray | 2026102301 | 4 | 88.193 | 796.47 | 904 |
| Gray | 2026102301 | 8 | 192.763 | 1602.42 | 2102 |
| Gray | 2026102302 | 2 | 42.680 | 446.34 | 544 |
| Gray | 2026102302 | 4 | 87.515 | 796.47 | 904 |
| Gray | 2026102302 | 8 | 192.439 | 1602.42 | 2102 |
| RGB | 2026102301 | 2 | 42.757 | 447.60 | 544 |
| RGB | 2026102301 | 4 | 88.831 | 798.60 | 916 |
| RGB | 2026102301 | 8 | 191.379 | 1608.18 | 2100 |
| RGB | 2026102302 | 2 | 43.316 | 447.60 | 544 |
| RGB | 2026102302 | 4 | 89.781 | 798.60 | 916 |
| RGB | 2026102302 | 8 | 192.886 | 1608.18 | 2100 |

Batch 8 is the largest tested batch. These timings include neither optimizer
steps, evaluation nor media I/O and cannot be treated as full-training time.
The first gray batch-2 cell was slower; every original interval is retained,
without selecting a replacement run. Setup, idle GPU time and Colab compute
units were not measured by the 600-second command.

The supervisor reports **18,566 bytes /1 MiB** of output. The exported JSON
files total 18,356 bytes; the remaining 210 bytes belong to the unexported
worker stdout log. Reported output arithmetic was replayed, but that log's
content was not independently downloaded. The verified transport package is
22,259 bytes, with SHA-256
`31bce1a611bd2e0e6aaee97218a81640e40aa2b1b06dc469bcfcde22fa88e85c`.

Evidence is preserved in the [original receipt package](../benchmarks/png-bmp-colab-capacity-receipts-20261023.json),
[measurement and cost table](../benchmarks/png-bmp-colab-capacity-20261023.json)
and [independent verification record](../benchmarks/png-bmp-colab-capacity-verification-20261023.json).
All fifteen packaged file hashes, sixteen source hashes, twelve separate
cell receipts, thirty-six intervals and cost-table medians agree. The source
package SHA-256 remains
`aacf402ba829f7ee6843c1695873ebe4ab96f060f21767bc511b6682874b408f`.
The notebook serialized its executed protocol using `json.dumps(protocol)`;
the [exact executed bytes](../benchmarks/png-bmp-colab-capacity-protocol-20261023.json)
have SHA-256 `27623ddd6a6248634b305720a7754a7215ae614c1232fca6f304375e79a8f837`.
Its semantics and source bindings equal the original formatted protocol;
the original file and its digest are preserved.

An earlier wrong-account setup stopped at the strict environment gate on
Torch 2.10.0+cu128. Its [unavailable preflight](../benchmarks/png-bmp-colab-first-preflight-20261023.json)
is retained: no capacity command started, zero profile attempts were consumed
and its runtime was disconnected. The successful Pro run used the required
Torch version. Final browser inspection showed no active sessions; the
remote notebook was saved with `RUN_MEASUREMENT = False`.

| Component | Current status |
|---|---|
| Portable notebook, source package and pinned GPU execution | **ready / hazır** |
| Two-model/two-seed capacity and cost table | **ready / hazır** |
| Native admission and generated numeric checkpoint QA | **ready / hazır** |
| Approved native corpus and ancestry/native-mode review | **missing / eksik** |
| Optimizer, evaluation and media I/O costs | **missing / eksik** |
| Real-training data size, recipe and total budget | **missing / eksik** |

The [preparation and software verification](PNG_BMP_COLAB_PREPARATION_20261023.md)
remains valid; its initial local-GPU-unavailable receipt is a historical
snapshot. Commons candidates remain unassigned and reserved data stays
closed. The next real-training contract needs admitted source groups, fixed
data quantities, splits, payloads, optimizer/exposure and evaluation costs
before a training budget can be frozen. This measurement grants no real fit,
new media acquisition or historical transfer/JPEG/WAV budget reset.
