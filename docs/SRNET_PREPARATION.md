# SRNet-style research preparation

This is an independently written architectural implementation informed by
[Boroumand, Chen and Fridrich's paper](https://ws.binghamton.edu/fridrich/research/SRNet.pdf),
not copied upstream code, pretrained weights or reproduced published scores.
No accuracy improvement, supported cell or deployed detector is claimed.

## Structural contract

The model uses learnable convolutions rather than a fixed high-pass stem.
Its seven front stages retain spatial resolution. Residual additions have no
following ReLU; reduction uses parallel pooled and projected paths.

| Stages | Channels | Operation |
|---|---|---|
| 1–2 | 64, 16 | Conv/BN/ReLU |
| 3–7 | 16 | Two convs, identity shortcut, no pooling |
| 8–11 | 16, 64, 128, 256 | Two convs + 3×3 average pool; 1×1 stride-2 shortcut |
| 12 | 512 | Two convs, global average |
| Classifier | 2 | Bias-free linear logits |

He-normal convolution initialization, .2 convolution bias, .01 Gaussian
classifier initialization. Implementation-specific BN epsilon 1e-5, PyTorch
momentum .1, edge average pooling excludes padding; these choices do not assert
binary equivalence with the authors' implementation. All parameters CPU float32.
4,779,616 learned parameters, about 19.12 MB of parameter tensors alone; this
is not an estimate of training activation/optimizer memory.

The paper uses 256×256 inputs and unrounded JPEG decoding. Our bounded wrapper
accepts 128/256 square grayscale uint8 tensors in pixel units; existing Pillow
128 crops are rounded and **are not the paper's JPEG preprocessing**. Wrapper
support for those tensors does not qualify either input representation.

## Explicit readiness command

```sh
steganography research srnet-preflight --out FRESH_REPORT.json
```

Runs a generated two-row 128-pixel fixture through one Adamax update, confirms
finite gradients and an updated first convolution, then compares singleton
versus batched evaluation. Frozen BN evaluation is mandatory for every nested
module; live BN in inference is rejected. Tolerance 1e-5 absolute on logits,
plus equal argmax decisions. These checks are synthetic regression/engineering
readiness only: no dataset reads, training metric, ROC-AUC, extraction or
generalization evidence. No weights are saved or installed.

Fixed subprocess, two math threads, 90-second wall timeout, 60/61-second CPU
limit, 4 GiB virtual address limit, 64 KiB file output and core dumps disabled.
Unsupported resource limits or missing Torch are unavailable, not successful.
Failed/timeout/malformed/oversized workers produce explicit redacted failure;
CLI returns nonzero unless completed. Paths cannot overwrite or use symlinks.
Output is a portable readiness record, never a detector verdict. This is
resource containment, not a filesystem or network sandbox.

## Required next gates

Local generated readiness completed: finite gradients, first convolution
updated and singleton/batch decisions equal; maximum logit difference
7.63e-6 within 1e-5. In-worker preparation/update/eval took 1.12 s, excluding
Torch import/startup. [Portable readiness record](../benchmarks/srnet-preparation-20261006.json)
explicitly marks real training/accuracy unavailable. Python 3.11.14: 907 tests
pass, 94.38% total coverage, new architecture/readiness code 132/133 statements.
Ruff, mypy (101 files), diff check and model-free wheel/sdist build/inspection
pass. Old TUI test now waits for its queued screen update, not only the report
file; the original assertions remain. Other Python versions/fresh full Docker
unverified here.

Before a real fit: bounded versioned 256-pixel/unrounded JPEG preprocessing;
safe numeric persistence including BN running statistics/counters; independent
forward/export parity; training-only balanced paired batches and a frozen
schedule/data/source protocol. Eval must use persisted running statistics, not
the current input batch or validation labels. CPU cost measured here is a tiny
readiness check, not a realistic full-training runtime forecast.

Untouched licensed external-source acquisition remains unavailable. All previous
failed CNN/JRM results stay published; a larger architecture alone supplies no
accuracy evidence. Wheel remains model-free and optional research dependencies
remain opt-in. No changes to carriers, analyzers, CLI analysis, API or TUI verdicts.
