# Generated CUDA timing and fixed-fit budget gate — frozen before profiling

No real data, validation inference, model export/installation or accuracy
measurement enters this gate. Its purpose is to decide whether a separately
preregistered five-epoch real learning control is worth attempting within the
unchanged 1800s ceiling. It does not guarantee long-run thermal performance.

Use the explicit fixed service `research_cuda_profile`, fresh output, exactly
64 generated four-row FP32 optimizer updates, seed 20261008. The generated
wave/checker inputs match the first preflight's shape and pixel units. Run the
unchanged shared optimizer including finite gradients and full numeric-state
validation on every update; do not time only asynchronous kernel submissions.
Capture monotonic timestamps at each production input fetch. Discard the first
16 of the 63 inter-fetch intervals; retain all 47 subsequent intervals. The
production engine synchronizes loss/gradient/state before the next fetch.

Snapshot all 16 dependency sources before/after, bind the report to those
hashes, reject source mutation. CUDA only, no fallback, IEEE float32,
TF32/AMP off, deterministic algorithms; kernel cgroup-v2 RAM <=8 GiB and
PyTorch allocator <=4 GiB as in the existing policy. This is an allocator
bound, not total driver/library memory. Internal deadline 180s, isolated parent
210s, CPU 360s, output file 1 MiB, no core dump. The generated model is discarded.

Fixed prospective budget: five epochs * 3,288 updates = 16,440 updates.
Estimated time = 120s fixed overhead + 2 * steady-interval p95 * 16,440.
Only an estimate <=1800s makes the fixed five-epoch candidate eligible to
attempt. Otherwise report insufficient budget; do not lower the safety factor,
change epochs based on scores, override the ceiling or run a predictably
incomplete fit. Short generated timings need not match real I/O, image-content
workload, other Windows GPU users or long-run thermal/power behavior.

Passing this timing gate is not passing a learning gate. Before real fitting,
freeze the exact full-source schedule and train-only stored-normalization
learning objectives, independently test the evaluation workflow and use fresh
bound plans. No validation-based tuning or support qualification.
