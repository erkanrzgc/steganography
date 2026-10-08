# Full-block streaming readiness — complete, not a detection score

The [frozen protocol](SRNET_STREAM_READINESS_PROTOCOL.md) was committed before
real execution at `31e969c66d0d93b199fb4cf2ec7cf814e2030b10`; its SHA-256 is
`d71091f499bba7f8d10f61eef0e08895154acec5b09e9da9fef9a8e1bac50f22`.
[Portable complete evidence](../benchmarks/srnet-stream-readiness-20261008.json)
is an exact copy of the local worker report, SHA-256
`c35ce235e1af58f1632ca2d19227525d97f5a494a07fa2601136282312dda336`.
Raw data, configuration paths, tensors and any generated test weights remain
local. No real model was trained, installed, calibrated or deployed.

## Observed whole-corpus readiness

All 16 training block caches passed full length/hash/finite verification:
1,934,622,720 bytes, read in at most 1 MiB verification chunks. All 7,380
unique train rows / 1,638 original train lineages were then read through the
complete planned four-row epoch, without truncation to the old 4,000-row cap.
All 1,611 validation rows had role/family metadata inspected, but **no
validation JPEG, descriptor or pixel tensor was opened**. The prior full
independent preparation audit stays bound by checksum, not rerun as detection.

Exactly 6,576 pairs / 13,152 row presentations / 3,288 **planned** optimizer
updates, seed 20261008. No optimizer ran in this real readiness job. Source
balance produces 3,288 pairs per origin. Each BOSS quality/family cell retains
all 822 original pairs; each ALASKA unknown-quality/family cell retains all
816 original pairs, oversampled to 1,644. Correlated repetitions are not
additional training scenes or independent validation samples.

- Ordered pair SHA: `8f1d15c66361f87703eb1c287acf3486601705b8b4f926293df7fb2dace7ed23`.
- Ordered four-row SHA: `2819b9db1099f95b5beca4ea50b488e58a9428596a9eae2259d9cd872d24936c`.
- Ordered streamed tensor SHA: `38a8794d44f3edde43cd38e6b577b982f06b4a021e7d61ceccb0700a011ff91a`.

Maximum returned batch allocation: **1,048,576 bytes**. This is not a total
process RSS claim: metadata, hashing temporaries and (during actual fitting)
Torch/model/activations/optimizer allocations are additional. The new reader
does not load/copy a whole-corpus float tensor or memory-map the entire dataset.

Worker wall duration: **7.801804s**, Linux 6.8.11-amd64, eight logical CPUs,
16,739,250,176 bytes host RAM, CPU-only Torch 2.14.0+cpu; CUDA unavailable.
The full regression suite ran concurrently, so this is an engineering run,
not an isolated training-throughput or CTF latency benchmark. All 12 execution
dependency hashes were captured before data loading and verified unchanged
before publication; the exact hashes, decoder and limits are in the evidence.

## Numerical and safety checks

Generated 9-row two-source fixtures, not real accuracy data, exercise complete
cross-block mapping. Streaming and legacy four-row fitting yield **exactly
equal** losses, all model parameters, running BN state and counters, with the
same four optimizer updates. RNG and caller thread counts are restored.
Additional generated direct/isolated fitting verifies checksum-bound plans,
numeric artifact reload, complete cards and rejection of post-save source
mutation (partial weights retained without a complete card).

Adversarial tests cover symlinks, nonblocking FIFO rejection, metadata/row/
byte limits, incomplete lineage families, crossed roles, malformed indices,
forged tensor/cache/decoder contracts, NaN/oversized values, truncation,
mutation after verification, omitted planned rows, propagated deadlines,
redacted worker failures, timeout/response limits and artifact forgery.
No raw dataset download, cloud operation, permissive cap change or deployment.

Focused verification: 164 tests pass; new reader 98.85%, scaled sampler and
stream fit adapter 100%, orchestration 99.34% statement coverage.

Final handoff verification: **1,556 tests pass**, 32 warnings, 291.23s, total
statement coverage **95.38%**. Ruff, mypy (136 checked files) and
`git diff --check` pass. Wheel/sdist build succeeds; archive member inspection
finds no raw `.benchmark`/credential files or `.f32`/`.pt`/`.pth`/`.onnx` data or
weights. Only Python 3.11 was verified locally; Python 3.12–3.14, CUDA fitting
and full Docker E2E remain unverified, not passed by implication.

## What remains

Real detection remains unqualified; all prior chance-level failures stay
published. **Ready data and correct streaming are not a passed learning gate.**
This slice implements a CPU-only training path with the existing 1800s ceiling;
it is not a verified CUDA trainer or an adequate full-corpus learning run.
Historical CPU timing suggests approximately 62 minutes for one new complete
epoch (an extrapolation, not measured), before multiple epochs and evaluation.
No predictably incomplete real fit was started and no timeout was extended.

Next preregister adequate train-only exposure and a learning gate, then choose
explicit local compute or a user-controlled GPU workflow. No paid resources
or restricted-dataset uploads will be provisioned automatically. Independent
single-file numerical replay and a separately bounded, blind-source evaluation
must follow any successful fit. Reserved WIFD needs its own high-resolution/MPO
preprocessing policy; its 200 covers cannot meet the 1,000+1,000 qualification
cell. Historical preparer remaining-deadline/source-snapshot hardening is still
pending; its frozen provenance is not rewritten by this new streaming contract.
