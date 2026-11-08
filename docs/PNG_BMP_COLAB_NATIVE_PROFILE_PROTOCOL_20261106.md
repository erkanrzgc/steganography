# Colab RGB native cost-profile proposal — 2026-11-06

Status: **provider-managed RAM revision; software verification passed**.
The user rejected the inherited fixed eight-GiB host-memory ceiling. This
revision removes that ceiling from the Colab proposal. The earlier resource
approval question is obsolete; it is not an approval for the revised command.
No native fitting command is registered by this contract.

The prospective command contract is
`benchmarks/png-bmp-colab-native-profile-prospective-contract-20261106.json`,
SHA-256 **d791914942d070f37b03c9bb5c3710fa7ce5f539d977f909f5b98fe7c912a865**.
The reviewed source packet is
`notebooks/png_bmp_colab_native_profile_sources_20261106.json`, SHA-256
**77e3e8dc5e144c1d9cf94089d1ffccd945c9c0c90fb64271871c747d38ec8e90**.
The base Git commit is `2f41bfe2e95febcaab80f80ab4875c72073c3a32`; the explicit
overlay hashes identify the added implementation. No claim is made that the
overlay was already committed there. The actual executed v1 generated CPU
notebook/source packet and its old QA records remain immutable historical
snapshots; this is a distinct native-profile extension.

## Source, environment and input

`core.spatial_colab_profile` reuses the existing immutable model/data/AdamW/
numeric services. Its scoped profile view prevents native optimizer steps,
native checkpoint writes, development/reserved scoring and persisted native
scores. Generated optimizer access additionally requires the generated source
namespace. Its train-timing implementation avoids the application's JPEG/AUC
import chain, allowing the source packet to operate without application CLI
dependencies. No old WSL guard, JPEG file, model skeleton or cache is rewritten.

The runtime is 2026.07: CPython 3.12.13 /Linux, NumPy 2.0.2, Pillow 11.3.0,
Torch 2.11.0+cu128 /CUDA 12.8, Tesla T4 and expected driver 580.82.07. These
expectations come from the retained earlier inventory; current allocation and
compatibility remain unverified. cuDNN must report a positive numeric version;
the complete actual environment is frozen before model creation and replayed
against both numeric states. No package installation or CPU-lock application
occurs. Missing/mismatched hardware, versions or deterministic APIs are
`unavailable`, with no CPU/other-GPU/account fallback or automatic retry.

The separate CUDA policy uses IEEE FP32, disabled AMP/TF32, deterministic
algorithms with errors enabled, a fixed CUBLAS workspace and at most a 4-GiB
Torch allocator. It restores precision, allocator, RNG and thread state on
exit. It does not invoke or modify the old strict WSL cgroup policy. The
precision API is documented in the primary
[PyTorch 2.11 CUDA notes](https://docs.pytorch.org/docs/2.11/notes/cuda.html).

The exact public metadata proposal retains 64/20/21 RGB sources across
38/12/19 known ancestry components, plus thirteen unassigned L sources across
six components. The canonical split SHA-256 is
`852b36d9bfee693c3be175ab5365d160a3e9d5c6da08e9fa1a2a0add552abefa`;
cache metadata SHA-256 remains
`8da0ac4ac98b7b0e9e3613b02b58c3cb076adb9456c74ef29c90f6a70ee1dfd2`.
Every actual source role remains unassigned. Known grouping does not rule out
unknown ancestry. No source, license, hold or historical reserved role changes.

Only the 64 train raw uint8 crops may be copied after bound permission:
12,582,912 pixel bytes, plus the original license/provenance manifest.
The JSON transport packet is bounded to 32 MiB, with whole-packet and every
pixel-file hash checked; no regeneration or decoding of originals is needed.
Development, reserved and L pixel files are omitted entirely. Metadata for all
sources remains present to verify global ancestry and unassigned roles.
No actual cache packet has been created or uploaded by software preparation.

## One proposed profile and its resource policy

One command has **240 seconds total**, including verification/imports, worker
execution, reaping, independent numeric replay, evidence packing/hash checks
and a conservatively charged **10-second closing reserve**. The old WSL/T4
cost and remaining allowances are not inherited.

- Two fixed model seeds: 2026102301 /2026102302; RGB 256×256 FP32, batch eight.
- Three native forward/backward passes per seed, unchanged native parameters,
  empty native optimizer state and **zero native optimizer updates**. Train-only
  evaluation records timing, never AUC/scores. This gives 48 native backward
  and 256 train-evaluation presentations.
- Three generated AdamW updates per seed: **six generated updates /48
  presentations**. Two generated numeric checkpoints are written and restored
  within the worker. These are software/cost controls, not a learning gate.
- The parent independently reads both numeric states using NumPy, validates
  every numeric array/environment/source/data/order binding and replays all
  six intent/commit consumption records. It does not read native pixels or
  import Torch for this independent audit.
- One private ZIP_STORED evidence package contains the generated states,
  generated fixtures, logs and contract/measurement receipts; no actual native
  pixels are in the output directory/package. The archive is streamed and
  verified without extraction/execution. Central-directory size/count is bounded
  before ZIP member allocation; symlinks/special files, traversal, compression,
  duplicate names, size/hash/CRC corruption and oversize reservations fail.

The native v2 contract uses `provider-managed-ram-with-headroom-watch-v1`.
There is **no custom DATA or RSS ceiling**: both contract fields are zero,
the worker inherits the runtime's limits, and `prlimit --data` is omitted.
Hard Linux CPU <=240 seconds, CORE=0 and per-file <=128 MiB limits remain.
CUDA virtual address space is not capped as if it were RAM. Before launch and
while the worker runs, `core.colab_memory` reads kernel `MemTotal` and
`MemAvailable`; reported available memory at or below ten percent of the
initial reported capacity stops the command. Missing/inconsistent observations
or changed capacity also make the command unavailable. Cgroup counters are
recorded separately as metadata. Kernel-reported capacity is **not a verified
tenant limit**; the monitor does not promise an isolated or hard RAM bound.
The runtime/provider's own limits remain in effect. RSS/HWM and swap are still
sampled; nonzero observed swap or a persistently unavailable observer terminates
the worker. A default
kernel wall timer, parent-death signal and parent watchdog prevent ordinary
interruption from leaving the worker running. Source-bound exclusive attempt
records prevent another output directory retry in the retained workspace.

**Hard cgroup resident-RAM and zero-swap enforcement remain unavailable.**
The headroom/swap monitor does not satisfy or repair the old WSL contract.
`wait4` reports the exact child's kernel CPU time
and maximum RSS, including inherited process-creation memory; it is not a
whole-process-tree or isolated-model memory peak. Parent packing/audit buffers
are bounded by file/manifest sizes; aggregate cgroup RAM is not measured or
enforced by this proposal. The recorded VRAM values are Torch allocator peaks,
not the entire driver's GPU-memory footprint.

Total command output is <=256 MiB, including original generated artifacts,
the verified archive and closing records; a per-file 128-MiB bound also applies.
Packing reserves duplicate bytes plus ZIP headers before writing. A deadline,
OOM, incompatible environment, failed numeric audit or invalid closing result
is unavailable and consumes the intent. Late final writes invalidate the earlier
closing receipt. Preserve partial commits/checkpoints and every failed receipt;
no remaining time grants a retry. Retain the ledger/evidence outside an erased
Colab runtime; local exclusivity is not cross-runtime durable storage.

## Reviewable execution surfaces

`steganography.research_colab_profile` exposes only metadata `freeze`, explicitly
permitted `cache-package`, `profile`, and read-only `audit`. Metadata freeze can
target the prospective Colab interpreter from Kali; execution still requires
an exact current interpreter match. The CLI never creates a permission file.
Only an external, later explicitly approved `colab-profile-permission-v1`
receipt may authorize the exact contract hash and the resource acknowledgement
`hard_cgroup_ram_and_swap_not_enforced`. Fixture permissions have different
source hashes and are not production permissions.

`notebooks/png_bmp_colab_native_profile_20261106.ipynb` defaults to CPU/source
setup. Its permission/media/profile/export switches are all false. The setup
uploads only reviewed source and creates the same prospective contract without
importing Torch. An external approved permission is verified in a separate
step before a train-cache upload can be enabled. Work stays on `/content`;
Drive is not mounted. Any eventual GPU runtime allocation/start/close interval
must be recorded separately from the 240-second command. Software preparation
does not start such an allocation or assert a current free-GPU entitlement.

## Acceptance and remaining authority

The revised focused software gate passes 149 tests, including an isolated generated CPU
profile with application imports forbidden, default-notebook execution with one
source-only upload and no Torch/media/GPU/attempt, native-step refusal, exact
metadata roles, generated state replay, Linux process failures and hostile ZIP
tests. The new memory observer has 100% coverage; all affected modules exceed
95%. Ruff /249-file mypy pass. The fresh 478-input full gate is separately
source-bound. Preserve the first fixture checksum-limit failure and
all intermediate lint/type/coverage evidence.

A separate read-only integration reuses the existing generated CPU states,
performs no optimizer step, opens no actual media and imports no Torch. Both
502-array CPU states and six commits replay. The verified 25-member archive
is 115,939,468 bytes; original artifacts plus archive total 231,871,115 bytes
against 268,435,456, with a 1.217173-second observed copy/audit/pack interval.
This validates the real-size CPU evidence path, not T4 performance or GPU-state
size. The actual GPU command retains its own reservation/closing checks.

The existing native cache retains its original `unavailable_final_memory_peak`
resource acceptance despite unchanged verified contents. No software result
relabels it. No real media acquisition, GPU execution, model upload/deployment,
native role assignment or historical restart follows from this preparation.

A successful future profile supplies new Colab optimizer/evaluation/I/O cost
evidence and a conservative projection. Its shared candidate 640-update /
5,120-presentation recipe, factor 1.5 and nominal 1,800-second /768-MiB
full-fit ceilings are proposals, not a guaranteed bound or approved allowance.
No native fitting worker exists here. A later fit needs its own source/data/
environment/exposure/resource contract, the independently accepted profile,
cumulative profile/setup/closing debits and explicit approval. Keep reserved
pixels closed and every old transfer/metadata/experiment budget unchanged.

## Verification-source amendment

The first complete gate remains failed: 4,267 passes /one process-observer
failure /two CUDA skips, with 97.21% coverage. The kernel had already removed
the killed child while its proc file was being read, returning ESRCH
(`ProcessLookupError`). The repaired reader treats ENOENT/ESRCH as a disappearing
child while still requiring `wait4` reaping and final kernel resource evidence.
Two explicit regression cases cover both errors. Permission receipts also now
require the literal boolean `true`, rejecting numeric `1`.

The repaired focused gate passes 134 tests. Only the controller and its two
test files change; data, recipe, environment expectations and resource ceilings
are identical. The old prospective contract/preparation are retained as v1
snapshots; the exact old source packet/notebook and failed full evidence remain
private. Neither prospective version consumed a physical GPU attempt. The
476-input repaired full gate is separately bound in the verification-repair
folder; do not relabel the first failed command as passing.

The repaired full gate now passes **4,270 tests /two explicit Kali-CUDA skips**,
32 warnings and **97.212052%** total statement coverage. Controller/profile/
handoff/CLI coverage is 97.391304% /98.901099% /96.629213% /98.039216%.
All 476 source hashes match; Ruff and 248-file mypy pass. Pytest/wrapper take
1,508.10 /1,509.702084 seconds. The successful QA unit is inactive /PID zero
with no restart. Its final MemoryPeak and independent kernel interval remain
unavailable after unloading; preserve the first observer and do not rerun to
reconstruct them. See `benchmarks/png-bmp-colab-native-profile-verification-20261106.json`.
That completed gate belongs to the historical fixed-memory implementation,
retained with its v2 prospective contract/preparation and private exact source
packet. It is not relabelled as verification of the provider-managed revision.

## User-directed RAM revision

The fixed eight-GiB host DATA/RSS limits were inherited from the local setup;
they were not a Colab requirement. The user's objection supersedes that
proposal and the associated unanswered approval question. The separate
four-GiB Torch allocator, model/data/split/exposure, 240-second command and
256-MiB output remain unchanged. The historical generated CPU process contract
retains its explicit fixed fixture limits and consumed execution record.

Fifteen new memory tests cover 4/8/16/32/96-GiB reported capacities, low
headroom, changing/malformed/missing metadata and symlink rejection. The first
focused revision had 147 passes /two obsolete fixture expectations; both were
repaired without changing application behavior. The next focused run passes
all 149 tests. The first full launch had an absent temporary-directory parent:
182 passes /528 setup errors before interruption, recorded as unavailable QA,
not a code pass. The corrected full launch uses fresh disk-backed directories
and the identical 478 input hashes. Keep both first logs.

See `benchmarks/png-bmp-colab-provider-memory-amendment-20261106.json` for
the source/policy change. Physical execution remains a separate step; this
software revision creates no production permission or train-cache package.

The revised full gate passes **4,285 tests /two explicit Kali-CUDA skips**,
32 warnings and **97.205087%** total statement coverage. Memory observer /
controller /profile /handoff /CLI coverage is 100% /96.685083% /98.901099% /
96.629213% /98.039216%. Ruff /249-file mypy and `git diff --check` pass;
all 478 frozen input hashes match and the original 466 inputs are unchanged.
Pytest/wrapper take 1,390.04 /1,391.702300 seconds. The successful QA unit is
inactive /PID zero /zero restarts. Its final post-exit MemoryPeak and independent
kernel interval are unavailable after unloading; no rerun reconstructs them.
See `benchmarks/png-bmp-colab-provider-memory-verification-20261106.json`.
The delivered source packet, native v2 contract and notebook hashes independently
agree; the 216-entry source packet includes the memory observer. All notebook
execution flags remain false. No new physical GPU result is claimed by CPU QA.
