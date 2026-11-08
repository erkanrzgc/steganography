# Colab readiness control correction — 2026-11-06

The free Tesla T4 allocation already succeeded. The first metadata inventory
remains unavailable and its same-allocation closing audit remains preserved;
neither is rerun. The software defect was coupling resource discovery to the
GPU query: an inaccessible cgroup identity ended the whole inventory before
reading the card.

## Implemented behavior

`core/spatial_colab_readiness.py` now exposes independent resource and hardware
observations. Its frozen dataclasses carry metadata only. A parent-segment,
malformed, missing, symlinked or inaccessible cgroup identity returns
`unavailable`; the path is not normalized, joined or traversed. The GPU query
still runs when explicitly selected and may report visible hardware. The
combined result is `partial_resources_unavailable`, never strict resource
acceptance or fitting readiness.

The [Linux kernel namespace documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#the-root-and-views)
explains that namespace-relative cgroup identities may contain parent segments.
That explains the observed format; it does not make the corresponding host
filesystem path safe or prove that the controller is accessible or delegated.
No namespace, mount, cgroup limit or WSL acceptance rule is changed.

The fixed NVIDIA query retains a ten-second parent timeout, five CPU seconds
and an eight-KiB kernel output-file limit. Its output is validated and error
text is not included in reports. Package metadata inspection never imports
Torch. CPU-only is the default; a GPU query requires an actual boolean opt-in.
Exclusive metadata receipts reject overwrite. The sixty-second elapsed check
is a diagnostic deadline, not a replacement for a training resource supervisor.

## Portable delivery

`notebooks/png_bmp_colab_readiness_v3_20261106.ipynb` embeds exactly three
model-free service files and the existing project LICENSE. A fixed base-commit
identifier records their repository origin; this notebook reconstructs only
the exact hash-bound source overlay, without claiming a full Git checkout.
Both package and per-file hashes are checked before exclusive writes. Unexpected
files, changed hashes, a symlink parent and existing destination are rejected.
All computation stays in the shared service; the notebook only installs the
reviewed sources and calls the service.

Select Python 3 /CPU /Runtime Version 2026.07. `CHECK_GPU = False` remains the
default. No runtime allocation, pip installation, Drive mount, native cache,
checkpoint, source acquisition, optimizer or model operation is performed by
the code. This software preparation grants no new GPU/profile attempt or native
fit. The full Colab execution supervisor and a measured optimizer/I/O contract
remain distinct work described in the portability review.

## Verification

All 44 generated tests pass with **100%** statement coverage of the new
service. They include the actual observed parent-segment condition with an
independent card query, v1/v2 metadata, immutable observations, wrong versions,
missing tools, malformed/oversized/secret-bearing query output, timeout,
symlink, exclusive-attempt and deadline cases. Seven separate source-bootstrap
checks verify exact source installation and corruption/overwrite/path rejection.
Notebook syntax and CPU-default settings pass. Ruff and 243-file mypy pass.

The fresh full CPU gate passes **4,136 tests /two explicit Kali-CUDA skips**,
with 32 warnings and **97.194602%** total statement coverage. The new service
retains 100% coverage. All 466 frozen application/test input hashes verify;
the original 464 remain unchanged. Existing provisioned dependencies, one
BLAS/OpenMP thread, disk-backed TMPDIR/basetemp, eight-GiB RAM and zero swap
are used. Pytest reports 1,425.44 seconds; its Python wrapper records
1,427.431714 seconds and exit zero. The complete log/coverage hashes and
source bindings are retained; no earlier gate is relabelled.

The QA unit is inactive /PID zero after transient-unit unloading. Its closing
MemoryPeak is `[not set]` and its independent kernel start/exit values are
unavailable; no final QA memory peak or kernel interval is claimed and the
suite is not repeated to reconstruct them. This software gate does not change
any older resource acceptance. See the
[verification receipt](../benchmarks/png-bmp-colab-readiness-v3-verification-20261106.json).

## Actual CPU-only Colab validation

The new self-contained notebook runs successfully in a fresh 2026.07 CPU
runtime. Both code cells complete; package metadata matches Python 3.12.13 /
NumPy 2.0.2 /Pillow 11.3.0 /Torch 2.11.0+cpu. The same parent-segment cgroup
condition now produces `partial_resources_unavailable` with reason
`parent_identity`, while the GPU field correctly stays `not_requested`.
The metadata service takes 0.0574666 seconds. This is software execution and
an honest unavailable-resource result, not strict training readiness.

The notebook is saved and the CPU runtime disconnected/deleted after
197.487040 /300 observed seconds including image startup. No GPU request,
NVIDIA query, model, native-data transfer or old attempt relaunch occurs.
Both code cells match the delivered sources byte for byte; the result's
SHA-256 replays from the ordinary-UI notebook export. Account-bearing
screenshots and the full export stay private. See the
[redacted CPU result](../benchmarks/png-bmp-colab-readiness-v3-cpu-validation-20261106.json)
and [source bindings](../benchmarks/png-bmp-colab-readiness-v3-bindings-20261106.json).
The adjacent plain JSON source package exactly matches the notebook's embedded
four-file source/license package, making it reviewable without decoding it.
