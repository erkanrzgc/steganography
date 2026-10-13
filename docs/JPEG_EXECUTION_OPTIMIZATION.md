# JPEG execution optimization gate

This is an execution/safety-check revision, not a new learning algorithm or
permission to reinterpret historical failed fits. Preserve all older reports,
audits and snapshots. New execution hashes require a fresh complete audit.
Model/optimizer equations, original identities, paired view/D4 schedules,
32,768 presented rows, two seeds, precision and learning gates stay unchanged.

## Measured bottleneck

Generated native RTX 5060 /Torch 2.14.0+cu130 stage profiling of `9cdd31f`
measured 0.66157 seconds/update in reference-Adamax: 0.44516 in the optimizer.
The old validator independently transferred parameter, gradient, moment/norm
and step scalars to the host. Consolidating only outer guards did not remove
this optimizer bottleneck. These generated timings are not detection scores.
Instrumented stage totals include extra synchronization and are not a complete
end-to-end production estimate; measure uninstrumented whole updates too.

## Preserved validator contract

Before any parameter update: require dense FP32 gradients/parameters, exact
slot names/shapes/dtypes, every finite element, nonnegative infinity norms and
scalar finite integral step >=1. CUDA reduces these flags on device and copies
one boolean vector per device, preserving per-check failure classification.
CPU uses serial boolean reads. No tensor floating-point update is fused,
reordered or omitted. Negative-zero/NaN/Inf/missing-gradient/state cases must
retain rejection and atomic parameter safety.

## Physical gate before fitting

Run fixed separate generated processes for each of legacy/reference-Adamax/
convolution-L2/genuine-16. Preserve the trusted `9cdd31f` optimizer source
(SHA-256 `64587efac86c01effe265aa8870d4ef80f4d0a2aad91389fd06cb41631add740`).
At seed 20261012, compare eight actual CUDA updates per side with the old
serial guards versus the new complete guards. Require bitwise equal losses,
every model/BN tensor and optimizer slot. Export only hashes/differences, no
weights. A mismatch or resource failure blocks fitting; never relax tolerances.

Time generated four-row workloads separately: eight warmups, 64 instrumented
updates for bottleneck location, then 64 whole updates with production guards
and no extra stage synchronization. Conservative trial estimate:
`whole_seconds_per_update * 8192 * 1.25 + 60`, including a fixed reader/evaluation
allowance. It must fit the unchanged 1800-second trial ceiling before claiming
the monolithic trial is eligible. A generated estimate is not a guarantee of
real-data completion. Keep any estimate/failure visible; do not shrink exposure
or choose an easier source to force eligibility.

Each diagnostic process: <=300/301 CPU seconds, 280-second internal deadline,
330-second systemd hard stop, zero core dumps, 1 MiB output cap. Kernel host
RAM <=8 GiB and Torch allocator <=4 GiB, two threads, deterministic IEEE FP32,
no AMP/TF32/CPU fallback. No corpus, export, download or cloud spending.
Do not run probes concurrently with the real fit. Once the numerical/timing
gate passes, create fresh audit/config/output bindings and preserve unavailable
controls; real completion and development accuracy remain separate outcomes.
