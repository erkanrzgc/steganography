# PNG/BMP generated training preparation — 2026-10-28

**Software preparation only. No new GPU experiment, native fitting or download.**
The existing Windows 11 /Ubuntu-24.04 WSL2 endpoint was reconnected using its
previous dedicated key and pinned host key. Read-only inventory confirms the
RTX 5060 Laptop, compute capability 12.0, driver 617.42, Python 3.12.3 and
Torch 2.14.0+cu130 /CUDA 13.0. NVIDIA-SMI is available at WSL's standard
library location even though it is absent from the shell PATH. No driver,
Ubuntu, package environment or historical training job was reinstalled.
The [connection receipt](../benchmarks/png-bmp-wsl-connection-20261028.json)
is an identity/environment check, not a capacity measurement.

## Implemented software

`core.spatial_training_data` owns a bounded generated corpus and the candidate
scattered plane-0 payload. Native-shaped inputs are 256×256 HWC uint8 with
one or three channels. Payload size is `floor(floor(samples*2/5)/8)` bytes;
PCG64 supplies a fixed recorded scatter permutation. Vector embedding is
checked by an independent scalar, big-endian decoder. Only the chosen plane-0
samples can change. FP32 batching preserves the original 0–255 integers and
keeps each original/control pair together; the maximum is four pairs/eight
images. No resize, RGB-to-L conversion or input media reader is provided.

Generated source IDs are separated by mode, data seed, split and source index.
Increasing a corpus count does not invent new identities for existing sources.
Train and development generators have disjoint namespaces. The generator uses
**even-valued random covers**, an artificial positive-control distribution.
It is not natural media or evidence of native steganalysis performance.
The existing 32-byte sequential admission fixtures remain unchanged.
The [generated manifest](../benchmarks/png-bmp-generated-corpus-20261028.json)
freezes 208 proposed learning-corpus originals plus twelve separate profile
originals. All 220 source/group namespaces, cover hashes and control hashes
are unique, and all 220 scalar payload checks pass. Only metadata/hashes are
persisted; no generated image or model is saved by this manifest preparation.

`core.spatial_generated_training` owns the generated-only step/evaluation/
checkpoint API. It reuses both unchanged SRNet skeletons, scalar FP32 AdamW
at 1e-4 and the existing bounded numeric state format. Plans bind the full
generated data manifest, model seed, epoch schedule, batch size, device and
source digest. Epoch tails retain paired labels without repeating or dropping
an original. Scores use stored BatchNorm state and the shared tie-aware AUC
calculation; scoring does not update parameters or buffers.

CPU execution scopes/restores Python, NumPy and Torch RNGs, threads and FP32
policy. CUDA execution delegates to the existing strict CUDA policy: no
CPU/AMP fallback, IEEE FP32, deterministic algorithms, bounded cgroup RAM and
the existing at-most-4-GiB allocator cap. The new CUDA path has **not run**.
Neither old JPEG CUDA evidence nor the T4 capacity profile proves this new
generated workload's costs or learning rule.

Numeric checkpoints carry model/optimizer/BN state, RNGs, the complete epoch
order and cumulative completed-update/presentation/time/output counters.
The saved checkpoint includes itself in its cumulative output-byte debit.
Restore rejects changed environment/source/data/device/plan, forged order,
mid-batch cursor, inconsistent counters, exhausted time and output limits.
Symlinks and overwrite are rejected by the unchanged artifact/state services.
Nonfinite gradients/state and deadlines invalidate a training session; an
update completed before a subsequent failure stays charged. A late checkpoint
file is retained as unavailable rather than reported ready.

The API is **not an experiment supervisor or native-data fit adapter**.
An independent supervisor must still bind source/data/exposure, enforce total
wall/RAM/swap/output limits, keep an external monotonic consumption ledger,
reject stale checkpoint rollback, and prevent retries. Checkpoint availability
does not authorize resumption. The metadata elapsed time is sampled before
writing; independent accounting must include checkpoint I/O and process startup.

## Verification and limits

The focused run passes 57 generated tests: payload/group/shape/security cases,
actual small CPU AdamW steps for both skeletons, exact uninterrupted/resumed
model/optimizer equality across reconstructed model instances in the same
CPU runtime/process, RNG restoration, scoring immutability, invalid
checkpoint/debit rejection, output bounds and deadlines. These are software
fixtures, **not a preregistered full learning control**. The first focused run's
single fixture failure is retained: a symlink test reused an already-saved
session and reached the correct fresh-session guard first. Only that fixture
was reordered. A fractional-sample rejection test and two end-of-operation
deadline tests were subsequently added and pass; the corresponding completion
checks reject evaluation/checkpoint I/O that finishes beyond the deadline.
Focused statement coverage is 100% for data preparation and 99.34% for the
training core; the only uncovered statement seeds the unexecuted CUDA device.
The full software gate passes **3,945 tests /two explicit CUDA skips**, with
**97.07% total coverage**, in 1,289.68 seconds. Ruff and 233-file mypy pass;
see the [verification receipt](../benchmarks/png-bmp-generated-preparation-verification-20261028.json).
The full gate ran in Kali's provisioned Python 3.11.9 CPU environment. The
physical host's Python 3.12.3 was checked read-only; the new code has no
physical CUDA test pass, cross-process resume acceptance or 3.13–3.14 local
matrix result. The two CUDA tests are unavailable on Kali, not hardware passes.
All earlier verification/failure records remain unchanged.

The [generated-control draft](PNG_BMP_GENERATED_CONTROL_DRAFT_20261028.md)
defines the proposed next experiment. Its executable supervisor, frozen source
closure and execution allowance still need a distinct final contract before
any run. The [data-gap draft](PNG_BMP_DATA_GAP_DRAFT_20261028.md) preserves the
completed acquisition and explains why inventory count alone does not make
the old fixed-group split feasible. All 118 native originals remain unassigned;
reserved data and the exhausted transfer/metadata ledgers stay unchanged.

| Component | Status |
|---|---|
| Existing WSL GPU identity/environment | ready; read-only verification |
| Generated pair/step/scoring/numeric-resume software | ready; CPU software QA |
| New PNG/BMP CUDA execution and measured full-step costs | unavailable; not attempted |
| Full generated control supervisor and final acceptance/debit contract | missing; draft only |
| Native grouped corpus and fit adapter | missing; roles unassigned |
| Native fitting or deployment | not performed |
