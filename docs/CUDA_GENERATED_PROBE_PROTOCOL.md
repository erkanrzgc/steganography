# CUDA generated probe v1 — fixed before hardware execution

This is a generated hardware/resource/forward-parity preflight, **not** a real
training, extraction, detection accuracy or cross-source qualification gate.
No datasets, downloads, model installation or cloud jobs enter this protocol.
User hardware: reported RTX 5060 Laptop on Windows 11; not yet observed from
the Kali/VMware development guest. GPU absence is `unavailable`, never success.

Run only the fixed module `steganography.research_cuda_probe`, fresh output,
60s internal deadline / 90s isolated-parent wall timeout / 120s CPU / 1 MiB
per-output file / no core dumps. Kernel cgroup-v2 ancestor RAM bound <=8 GiB
required instead of the CPU-only virtual-address limit. No unbounded retry.

Source snapshots before/after execution, exact device `cuda:0`, supported
compiled GPU architecture, deterministic CUBLAS environment, FP32 IEEE only,
TF32/AMP off, no CPU fallback. PyTorch allocator cap is min(4 GiB, 70% of GPU
memory), not a claim about total driver/system GPU use. Preserve caller RNG,
threads and CUDA policy settings. Unsupported deterministic kernels/OOM fail.

Seed 20261008. Four generated float32 256x256 rows: smooth sine/cosine base,
base plus 8-level checkerboard, transposed base, transposed base plus that
checkerboard. Pixel units unchanged, no clipping/resize. Exactly one four-row
Adamax optimizer update using the production numerical engine, existing .001
learning rate/.0001 weight decay. No model weights written or deployed.

Return the trained generated model to CPU, compare CPU and GPU singleton
stored-BN logits on row0 using those same weights, finite values and
`allclose(atol=1e-4, rtol=1e-4)`. Report maximum difference, generated loss,
one-update accounting, actual allocator peaks, execution metadata/source hashes
and duration. Do not claim CPU/GPU bitwise learning equality or an independent
math oracle: both forwards use Torch. A passing preflight only permits a
**separate preregistered** adequate-exposure real-learning workflow.

Unit tests with emulated APIs or CPU-redirected transfers test control flow
only. The live hardware test is explicitly skipped/unavailable on this host,
not counted as passed. A failed/unavailable worker emits a separate unavailable
record, with no complete hardware-success artifact and no fallback fitting.
