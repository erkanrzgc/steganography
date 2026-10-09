# Physical WSL CUDA preflight and local transfer — 2026-10-09

The original [generated GPU report](../benchmarks/cuda-wsl-generated-probe-20261009.json)
was fetched from the user's WSL checkout over key-authenticated local SSH with
a host key verified against the user's terminal fingerprint. Its exact SHA-256
is `27d89db7a1f8ab4aea8fff80c820f133cfc3029d5fb3134e14ff803f7c979d11`;
all 14 execution-source hashes match checkout `d2610f0`. The separately frozen
[generated protocol](CUDA_GENERATED_PROBE_PROTOCOL.md) remains unchanged.

## Completed physical generated preflight

- NVIDIA RTX 5060 Laptop, compute capability 12.0, Torch 2.14.0+cu130 /
  CUDA runtime 13.0, Ubuntu/WSL2 and Python 3.12.3. The Kali VM remains CPU-only.
- Exactly one generated four-row FP32 optimizer update, seed 20261008;
  19.58524116 seconds including initialization and CPU/GPU comparison.
- Same returned weights, singleton stored-BN forward maximum difference
  `0.000091552734375`; finite/allclose gate passes with atol/rtol `0.0001`.
  These are two Torch forwards, not an independent mathematical oracle or
  bitwise CPU/GPU optimizer equivalence.
- Actual process PyTorch allocator peaks: allocated 951,980,544 bytes
  (0.886601 GiB), reserved 1,434,451,968 bytes (1.335938 GiB). Allocator cap
  4 GiB; kernel cgroup-v2 resident RAM bound 8 GiB. No TF32/AMP/fallback.
  Allocator measurements do not bound total driver/library/other-app memory.

No real corpus used, no real model trained/exported/deployed, no inference
accuracy measured. This one generated update is not a throughput benchmark
or evidence that a complete real epoch fits the 1800s ceiling. The previous
[Kali unavailable record](CUDA_BACKEND_VERIFICATION.md) is preserved unchanged;
CPU/emulated test coverage is not relabeled as physical CUDA verification.

## Private local transport

The already audited prepared tree was copied into a fresh owner-only WSL
staging directory on the same computer; no cloud/public upload. Recursive
rsync dry-run checksum comparison reports no changed/created/deleted files:
9,082 regular files, 1,049 directories, 3,155,045,461 file bytes. Neither tree
contains symlinks. Rsync transport checksums are not claimed to be SHA-256;
native input bindings and train tensors are checked separately by SHA-256.
Copying/comparing all files reads validation bytes solely for transport
integrity, not validation predictions, decoder/model selection or tuning.

## Completed native train-only check

The [original native report](../benchmarks/wsl-stream-readiness-20261009.json),
SHA-256 `934d2c7fc382342c5cd42a9a3e01689bbf0169a4916cf980e2e6e6b518405523`,
completed in 9.693818406 seconds under the frozen
[local readiness protocol](WSL_TRANSFER_READINESS_PROTOCOL.md), SHA-256
`783566337998c88f38f9ebffdb84c54f1fbb2bc5bbd25795d002654f3990c860`.
The protocol was committed at `45c9fd1` before native execution; execution
source hashes still match `d2610f0` (documentation-only changes).

- Every bound training cache/tensor verified by SHA-256 and finite-value
  checks: 7,380 unique rows, 1,638 original train lineages, 1,934,622,720
  training tensor bytes. All 13 start/end execution-source hashes match.
- One exact source/Q/method-balanced schedule: 6,576 pairs, 3,288 **planned**
  updates, 13,152 presentations; all six cells covered, none silently dropped.
- Ordered tensor SHA-256
  `38a8794d44f3edde43cd38e6b577b982f06b4a021e7d61ceccb0700a011ff91a`
  exactly matches the historical Kali CPU readiness run. Pair/batch schedule
  hashes also match. Returned batch allocation <=1 MiB, not a process RSS claim.
- Native check opens no validation cache/tensor/JPEG; only validation metadata.
  The separate complete-tree transport comparison does read validation bytes,
  as disclosed above. No optimizer executed, real model trained or accuracy
  measured; CUDA metadata availability does not make CPU tensor I/O a GPU
  throughput measurement. Concurrent Kali tests prevent any speedup claim.

A separate adequate-exposure train-only learning protocol, stored-normalization
learning gate and independent held-out qualification remain required. Previous
chance-level detector results and default primary verdicts are unchanged.

## Checkout verification

Final Kali/Python 3.11.14 run: **1,600 tests pass, one actual-CUDA test explicitly
skipped**, 32 warnings, 237.47 seconds, total coverage **95.43%**. Three new
tests check immutable recorded reports/protocol hashes and their disclosure/
accounting contracts; they are not live hardware tests. Focused CUDA/stream
tests: 119 pass, one explicit live-CUDA skip; evidence tests: three pass.
Ruff, mypy (138 files) and `git diff --check` pass. Wheel and sdist builds pass;
148/295 archive members respectively, no raw corpus/tensors/numeric weights,
SSH credentials or provider-specific memory. Python 3.12.3 on WSL has actual
generated preflight, pip dependency check and native I/O evidence only, not a
full-suite pass. Python 3.13–3.14/full Docker remain unverified.
