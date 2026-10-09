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

Native train-only check is **pending** under the frozen
[local readiness protocol](WSL_TRANSFER_READINESS_PROTOCOL.md), SHA-256
`783566337998c88f38f9ebffdb84c54f1fbb2bc5bbd25795d002654f3990c860`.
Its CUDA metadata check will not itself execute a GPU optimizer. A separate
adequate-exposure real-learning protocol and independent held-out qualification
remain required; prior chance-level detector results are unchanged.
