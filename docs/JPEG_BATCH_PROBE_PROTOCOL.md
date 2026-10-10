# Generated genuine-batch capacity protocol v1

This is a hardware feasibility check, not an accuracy or real-data learning
experiment. Before any GPU update, freeze this protocol and its source closure.
Historical curriculum engines, source-bound reports and deployed detectors stay
unchanged. No model or dataset downloads, exports, or cloud spending.

Each of 8, 16 and 32 rows runs in a separate new process. Fixed seed 20261012,
generated 256x256 grayscale wave/checker inputs, paired alternating targets,
SAME-pool/BN-epsilon research architecture, two genuine Adamax updates. LR .001,
weight decay .0001, betas .9/.999, epsilon 1e-8, foreach disabled. No gradient
accumulation; it would not reproduce larger-batch BatchNorm statistics.

Use the unchanged CUDA-only FP32 deterministic policy: no AMP, TF32 or CPU
fallback, allocator at most 4 GiB, enforced cgroup-v2 host limit at most 8 GiB,
two CPU threads. Worker deadline 120 seconds, parent timeout 180 seconds,
CPU limit 160/165 seconds, zero core dumps, 64 KiB output file limit. The fixed
worker spawns no children; parent timeout kills and reaps it. Each report is
exclusive and symlink-free. Missing prerequisites, timeout or allocator OOM
are unavailable, never success; other worker errors fail closed. Do not raise
memory limits or retry automatically after OOM.

Report all attempted batch sizes, losses, allocated/reserved GPU peak and
elapsed time with pre/post source hashes. Models are discarded. Generated
inputs cannot qualify mathematical parity, independent accuracy, or support.
Select the largest completed size only as a candidate for a separately frozen
real-data experiment: successful two-update profiling does not guarantee a
longer real-data job fits. Re-test with its actual input/preprocessing contract.

Explicit invocation inside an 8 GiB systemd user scope, with fresh output:

```text
python -m steganography.research_batch_probe --rows 8 --out FRESH.json
```

Run 16 and 32 separately; no all-at-once allocation. Local CPU tests use a
small independent test network for update validation and never count as a
physical GPU result. An unavailable SSH endpoint means no GPU check started.
