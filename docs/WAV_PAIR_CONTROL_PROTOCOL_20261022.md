# Fresh generated PCM pair control — 2026-10-22

This separate CPU intervention permits **one zero-update profile followed,
only if all checks pass, by one generated learning attempt**. It implements
the user's request to proceed with the stated profile/conditional-learning
sequence. It does not authorize real AudioMNIST fitting, reserved-media
access, downloads, checkpointing, automatic retry or an old v3 restart.
The prior design snapshots and every failed/incomplete receipt stay intact.

## Source and data before learning

Freeze the exact specification and complete 34-file source closure in
`benchmarks/wav-pair-control-protocol-20261022.json` after the generated tests,
Ruff and mypy pass. Full unchanged coverage >=90%, new code >=95% and
source bindings must pass before either execution phase. Sources include the
unchanged old generator, bounded decoder, opaque oracle and scalar AUC,
together with the new service, adapter, tests and this document.

Generate 600 fresh mono PCM16 WAVs at 48 kHz using the unchanged speech-like
construction, an independent fixed generator seed **2026102201** and the
original hash-bound per-role length schedule. There are 400 train /200 fresh
development originals, in 20/10 generated groups of 20. The new sources must
have unique raw and decoded-PCM hashes and zero overlap with the old corpus
or the other split. No real WAV path is opened; only frame metadata is read.
This is new data from the same generator family, not independent natural
speech. No old dataset roles are reassigned.

The profile extracts all nine rows per original in fixed source/recipe order:
cover, then sequential/scattered ×5/40% ×planes 0/1. Existing materialization
must verify all **4,800** opaque payload recoveries. Only the ten fixed
pair-energy candidate values enter the model. Labels, group IDs, source
hashes, oracle information, filenames and sample metadata stay outside the
feature array. No normalizer is fitted. Both role matrices are FP32, finite
[0,1], shape 3,600×10 and 1,800×10. Their NPY bytes (pickle forbidden), row
lineage/order digests and fresh manifest are hash-bound before learning.

Freeze a separate fit-binding receipt containing the protocol SHA, successful
profile SHA, a conservative profile debit and exactly one allowed fit.
The debit is ceil(independent systemd monotonic phase duration)+1 seconds;
it must cover the worker-reported interval and stay within 300 seconds. It binds the complete manifest and
both arrays transitively. A changed hash, domain, split, dtype, range or row
order fails before an optimizer is constructed. The profile's original seed
and manifest are never replaced after seeing a metric. Development feature
extraction is allowed during profiling; no development predictions, AUC or
label-based choice is computed in that phase.

## Profile: zero optimizer updates

CPU only: Torch **2.14.0+cpu**, NumPy **2.4.6**, two threads, deterministic
algorithms, FP32. Construct the 418-parameter 10→32→2 ReLU head with smoke
seed 2026102202, then execute 20 forward/backward batches of 64 cached train
rows. No optimizer is constructed. Gradients must be finite and every
parameter must remain identical. The full native length schedule is used
for all source generation, hash/oracle checks and feature extraction; this
is not a short-waveform throughput substitute. Retain extraction/smoke times,
startup-inclusive process time, cgroup/RSS peaks and cache hashes.

Before allowing a fit, require the conservative projection
`profile_elapsed + (smoke_elapsed/20)*2280*10 + 60 <= 1800` seconds.
The factor ten and 60-second scoring/finalization reserve are fixed here;
this projection is a resource gate, not a measured full training duration.

The profile gets **300 seconds**, with five seconds reserved for evidence.
Its exact elapsed time is debited before the fit phase. Missing resources,
timeout, changed parameters, extraction/oracle or source failure means
unavailable and forbids fitting. A resource pass is not learning evidence.

## Learning recipe fixed before results

The same untrained candidate family is used; this is not the old raw CNN.
For each seed **20261016, 20261017**, start a newly initialized FP32 head.
AdamW uses lr=0.001 and weight decay=0.0001 for weight matrices, zero for
biases, betas=(0.9,0.999), eps=1e-8, amsgrad=false, foreach=false and
fused=false. No scheduling, dropout, BatchNorm, AMP, warm start or hyperparameter
search. Only CPU execution is authorized by this contract.

Train for **20 epochs**, exactly **72,000 row presentations /1,140 optimizer
updates per seed**. NumPy's row permutation seed is model_seed XOR 0x4E4A5047,
with one sequential generator across epochs. Each actual batch has 64 rows,
the last batch of every epoch 16. Unlike v3's scalar waveform accumulation,
this fixed feature head computes its batch in one matrix operation. The
objective is summed unreduced cross entropy times class weights 4.5 for
cover /0.5625 for stego, divided by actual batch size. Finite gradients are
required before every update. Both full schedules are required even if the
first seed misses its learning target. No early selection or retry.

Save a numeric completed-fit receipt immediately, then all train/development
metrics for that seed before beginning the next. Preserve an atomic confirmed
callback-update lower bound throughout; incomplete work never substitutes
for a finished second seed. Do not save weights or optimizer checkpoints.
Evaluate in inference mode. Score direction is softmax class 1 (stego).
Use the bound integer-half-win scalar AUC; retain all eight cells and unrounded
metrics plus score digests. There is no threshold search or accuracy claim.

Also report a fixed, untrained comparator: cached FP32 global plane-0 feature
(column 0), higher equalization = stego. Report every cell for both splits,
without choosing its direction or threshold from results. This comparator
cannot rescue a failed head. The old v3 scores are from another corpus and
cannot be called a matched comparison; no v3 reference refit is authorized.

## Limits, acceptance and failure

The profile and fit have a **combined 1,800-second CPU wall allowance**,
including imports, generation, all extraction/oracles, fits, inference and
receipt assembly. Subtract the measured profile debit from the fit deadline;
reserve five seconds internally for final evidence. Each phase runs in a
separate user systemd unit with MemoryMax=2G, MemorySwapMax=0, CPUQuota=200%,
Restart=no, KillMode=control-group and TimeoutStopSec=5. The outer profile
watchdog is 310 seconds; the fit watchdog is its remaining eligible time plus
10 termination-only seconds. Do not extend exposure into watchdog grace.

Owned generated WAVs total at most 48 MiB; arrays/metadata/logs and retained
outputs keep total new experiment output below **64 MiB**. Each artifact/log
is capped at one MiB. Native inputs have at most 220,500 samples; no derived
WAVs are retained. Temporary generated files are confined to the profile's
exclusive owned directory and normally removed; historical data/checkpoints
are never deleted. Verification scratch is measured separately, not silently
charged to or used to reset a historical experiment ledger. This CPU allowance
adds no JPEG time and does not reset the incomplete raw-transfer ledger.

After process exit, independently verify systemd monotonic phase durations.
The authoritative decision passes the conservative profile debit plus full
fit-unit duration to the shared acceptance function; elapsed time outside
the total budget remains unavailable even if a worker printed a pass.

After both full schedules, require training AUC >=0.60 and both sequential
and scattered 40%-plane-0 development AUCs >=0.70, **in each seed**. All eight
train/development cells, exact exposure, source/cache bindings and in-bound
resources are required. Missing or incomplete evidence is **unavailable**;
a complete below-target head is **failed_learning_gate**; a passing pair is
**passed_generated_only**. Every outcome keeps real_fit_authorized=false
and production_qualified=false. No new native performance follows.

The pair summary loses order within quarters and can merge a one-bit change
when counts swap; this known limit stays explicit. A fresh generated pass
would establish only this control's learning result. Another real or
generated fitting intervention needs its own frozen contract; real AudioMNIST
additionally needs explicit approval and an accepted full generated control.
