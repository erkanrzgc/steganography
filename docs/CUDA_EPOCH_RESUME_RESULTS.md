# Original physical epoch-resume probe — 2026-10-10

Preregistered protocol SHA-256
`2b15db4b1322cb5ef9884a7e842525e523470e47da30a5e8d20e7ad9851696c5`
was committed/pushed at `3e242a7` before actual WSL execution. The original
report `benchmarks/cuda-epoch-resume-probe-20261010.json` has SHA-256
`c00264d64a8522ff275694fb6823ff1b2a55450c55d42a2071a2733518c61813`;
all 16 executed sources matched that checkout when retrieved over pinned SSH.

RTX 5060 Laptop / Torch 2.14.0+cu130: 128 generated updates, exact uninterrupted
versus saved/restored model/BN/Adamax/RNG/loss state. Both fits reconstructed
models in the same isolated process; cross-process/runtime portability is not
claimed. Runtime 30.3357s, steady interval p95 0.09043310s. The separately fixed
per-epoch estimate is 714.6881s for 3,288 updates, eligible against 1800s.
Five epochs together would estimate 3093.4403s and remain ineligible as a
single job. Previous failed single-job profiles are not overwritten/relabelled.

Each actual epoch job still requires complete immutable input/source checks
and its own deadline, 8 GiB resident cgroup / 4 GiB CUDA allocator limits.
This result permits attempting the separately preregistered real-data pilot;
it is not real learning, a detector score, independent math or a speedup claim.
