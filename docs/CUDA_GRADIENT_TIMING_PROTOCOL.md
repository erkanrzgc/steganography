# Consolidated-gradient CUDA timing — frozen before physical execution

This is a separate measurement of changed execution, not a revision of
`CUDA_THROUGHPUT_PROTOCOL.md` or its immutable ineligible result. The only
optimizer-engine change is aggregation of finite-gradient boolean checks on
explicit CUDA: every gradient element remains checked, missing gradients fail,
full numeric-state validation still runs after each optimizer update. No
floating-point gradient/optimizer arithmetic, CPU execution or limits change.

Before execution, require generated CPU exact loss/state equivalence of the
old per-parameter and aggregate checks, missing/NaN/infinity tests, host scalar
read-count verification and the full regression suite. Those are CPU tests,
not CUDA numeric parity or hardware performance evidence.

Use the existing explicit isolated `research_cuda_profile` service, fresh
output, unchanged generated four-row wave/checker FP32 input, seed 20261008,
64 updates, discard first 16 inter-fetch intervals and retain exactly 47.
Bind all 16 sources before/after to this new checkout. IEEE FP32, no TF32/AMP/
fallback, deterministic CUDA, cgroup-v2 RAM <=8 GiB, allocator <=4 GiB, internal
180s/parent 210s/CPU 360s/output 1 MiB/core zero; model discarded, no corpus.

Keep the exact prospective decision: five epochs *3288 updates =16440;
estimate =120 +2 * steady-interval p95 *16440. Eligible only if <=1800s.
Otherwise no fit, epoch reduction, safety-factor change or raised job ceiling.
Even an eligible estimate needs a separately preregistered full-source
stored-normalization train-only learning/evaluation workflow before fitting.

Compare descriptively with the first report, not as a controlled speedup claim:
Windows applications, thermal/power state and concurrent Kali work are not
matched or isolated. A single generated timing cannot prove long-run real I/O
performance, CUDA update equivalence, mathematical correctness, learning or
accuracy. Preserve both complete original reports and every failed gate.
