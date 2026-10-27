# PNG/BMP Colab preparation and zero-update capacity contract — 2026-10-23

This separate contract prepares existing `core.nonjpeg_models` gray/RGB SRNet
skeletons without modifying any frozen JPEG source. It authorizes software QA,
generated checkpoint tests and a single generated capacity measurement under
the source hashes in `notebooks/png-bmp-colab/protocol.json`. No real fit,
download, role assignment, deployment or historical budget reset is implied.

## Reproduction

Open `notebooks/png-bmp-colab/png_bmp_capacity.ipynb` in Colab. Select Runtime
Version **2026.07**, GPU, and supply the adjacent `sources.json`. The notebook
checks its fixed SHA-256, clones base commit
`2be950e8af0984bc012bc89d3ca072cea01c24f3`, and exclusively adds the reviewed,
hash-bound source overlay. Existing base files must match byte for byte.
This is a portable handoff, without needing an unpublished commit or branch.
No CPU research requirements are installed. Python 3.12.13, NumPy 2.0.2,
Pillow 11.3.0, Torch 2.11.0+cu128 and CUDA 12.8 are required. GPU model,
cuDNN and software versions are recorded. Unavailable or retired runtime
versions must be reported unavailable rather than silently migrated.

The public environment details come from Google's
[versioned runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)
and the linked [immutable GPU package snapshot](https://github.com/googlecolab/backend-info/blob/ec53f93fe44a9cab118dc5451527b18e36e1a643/pip-freeze.gpu.txt).
See [Drive I/O guidance](https://research.google.com/colaboratory/faq.html): work
on the local VM disk and export one hash-verified receipt package for storage.
No Drive mount or cloud AI provider is introduced.

## Measurement

Run explicitly, once, from the source tree:

```sh
python scripts/spatial-colab.py profile --protocol colab-protocol.json --out /content/spatial-capacity
```

The supervisor allows **600 total seconds**, including source/environment
checks, worker startup, all twelve cells and receipt creation (a two-second
receipt margin is reserved). It kills and waits for the worker on timeout.
A protocol-digest attempt marker is created exclusively in the output parent
before the worker. Existing output/marker means consumed; no automatic retry,
batch fallback or continuation in another directory is authorized. Keep the
marker and all partial records across runtime interruptions; do not erase them
to retry. Colab VM termination cannot guarantee final receipt persistence.

Two domains × seeds 2026102301/2026102302 × batches 2/4/8, in that fixed order.
Inputs are generated uint8, converted exactly to FP32 in the original 0–255
scale, 256×256, one or three native channels. No AMP, TF32, resize, color
conversion or optimizer is used. Each cell has one warm-up and three timed
forward/backward passes, synchronized around GPU kernels. Every pass requires
finite nonzero input gradients and finite parameter gradients. Parameters must
remain bit-identical; training-mode BN buffers may change and are discarded.
No gradient accumulation or checkpointing reduces memory use. Parameter
comparison copies contribute to peak memory and are explicitly part of the
measured QA workload. Per-cell VRAM allocated/reserved peaks, process cumulative
RAM high-water mark, cell duration, GPU/software and output bytes are recorded.
GPU absence, mismatch, OOM, nonfinite gradients, partial cells and deadline
are `unavailable`; previous complete cells remain partial evidence only.

Cost rows cover both models and both seeds at each batch. Median measured
forward/backward time is not full-training time. Optimizer state/steps,
evaluation, media decode/I/O and warm-up amortization remain unmeasured.
Dataset size, epochs, a real-training budget and accuracy qualification stay
missing until a separately approved source/data/exposure/grouping protocol.

## Data admission

`benchmarks/png-bmp-commons-candidates-20261023.json` is a **metadata-only**
candidate list, not a representative corpus or certified clean media. It
records each original URL, page, author, CC0/CC BY license, upstream SHA-1,
byte count and whole-source ancestry. SHA-256 is added only after an explicitly
authorized acquisition verifies the SHA-1 and size. A license entry alone
does not prove native mode or suitability. Unknown related origins require
manual ancestry review; no automatic split is available. Reproductions,
processed images and same-original crops must share a parent group. Exact
raw hashes and decoded crop duplicates are rejected. Near-duplicate and
semantic-origin review remains a human admission requirement.

The explicit acquisition interface exists but is disabled by default:

```sh
python scripts/spatial-colab.py acquire --manifest reviewed-commons.json --out local-originals --allow-download --prior-ledger existing-transfer-ledger.json
python scripts/spatial-colab.py prepare --manifest reviewed-commons.json --originals local-originals --out local-prepared
```

The prior ledger contains `historical_bytes` (integer or null). Its original
hash and incompleteness are retained; an unknown historical total stays null.
An exclusive reservation receipt precedes network I/O and survives failure.
It conservatively charges each declared file size plus its one-byte overflow
probe before any transfer; rejected/partial transfers never reset this debit.
Acquisition permits only HTTPS upload.wikimedia.org originals with no redirects,
bounded by declared per-file size, 32 MiB/file and 256 MiB/command, 128 sources.
This does not authorize execution of acquisition in this preparation session.
BOWS2 and DIV2K are not copied or imported. No historical ledger is reset.

Admission requires single-frame native 8-bit L/RGB PNG/BMP without alpha,
palette, alternate EXIF orientation or image conversion. Bounds are 16 million
pixels and 32 MiB encoded input. The fixed top-left 256×256 crop happens before
embedding. Generated 32-byte plane-0 sequential controls are stored alongside
cover variants in both PNG/BMP and independently scalar-decoded after encoding.
Controls are pipeline fixtures, not representative steganography or learning
evidence. Every artifact stays **unassigned** in its original ancestor group;
the service cannot open reserved splits or assign training/test roles.

## Numeric interruption format

`core.spatial_colab_state` stores bounded raw numeric arrays preceded by an
exact schema and externally verified SHA-256. No pickle, executable objects,
archive extraction, symlink or overwrite. Exact model/BN arrays, explicit
scalar FP32 AdamW slots/options, Python/NumPy/Torch CPU/CUDA RNG, complete data
permutation/cursor and cumulative time/output/update/presentation debits are
included, together with uniform model train/eval mode. Restoring clears stale
gradients and prevalidates RNG encodings and target tensor dtypes. All headers,
shapes, dtypes, total extents and finite values are
checked before array allocation; incompatible source/data/environment and
truncation fail closed. This is a format choice, not an authorized recipe.
Only generated states and generated optimizer slots are used for round-trip
QA. Real-model or real-data checkpoints are not created. Future real fitting
needs a separately approved recipe, exposure, source/data bindings and budget.
