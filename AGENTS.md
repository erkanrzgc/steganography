# Project memory

This file is the provider-independent handoff contract for maintainers and
automation working on this repository. Read it before changing code.

## Architecture boundaries

- `core/` owns reusable application services and immutable domain types.
- `modules/` contains carrier/analyzer plug-ins. Third-party analyzers using the
  existing `Analyzer.analyze(Path)` contract must continue to work.
- CLI, API, TUI, and reports call the shared service layer; they do not contain
  independent detection or extraction logic.
- Untrusted files, archives, decoder output, and external-tool output are always
  bounded. Never execute an extracted artifact or follow an artifact symlink.
- Cloud AI providers and uploads have been removed. Do not reintroduce them
  through old compatibility options. Historical triage cannot affect primary
  verdicts, calibration, or benchmark gates; local ML research stays explicit.
- The base wheel remains model-free and does not require external executables.
  Model and dataset downloads require an explicit user command.

## Compatibility rules

- Preserve payload v2/v3 extraction and the public `Carrier`/`Analyzer` plug-in
  contracts.
- Preserve existing `/v1` and `/v2` routes. New report fields are additive.
- `confirmed` requires a verified project marker or successful extraction.
- Missing required coverage produces `inconclusive`, never a clean verdict.
- Reports never contain absolute host paths, passwords, steg keys, environment
  secrets, or an unredacted secret-bearing command.
- Do not write project memory beneath `.codex`, `.agents`, `.claude`, or another
  provider-specific directory.

## Verification

Run from the repository root using a fully provisioned environment:

```text
ruff check .
mypy core modules report api steganography ui registry.py config.py cli.py
pytest
```

The PR gate is Python 3.11–3.14 and total coverage >= 90%; new analysis, CTF,
and report code targets >= 95%. Real-dataset jobs may be skipped only as
`skipped`/`unavailable`, never passed.

CI's lightweight version matrix uses `pytest --no-cov`; explicit optional
dependency skips are expected. The separate CPU `research` job provisions
`requirements-research-ci.txt` and runs the complete unchanged coverage gate.
Quality checks also need Torch/ONNX to check the source-bound export contracts.

If `/tmp` is a small tmpfs, use a fresh disk-backed pytest `--basetemp` and
`TMPDIR`. Preserve the first failed run's log; filesystem exhaustion is an
unavailable verification attempt, not a code pass. Do not delete datasets or
checkpoints to fix a full temporary filesystem.

## Active milestone

PNG/BMP Colab preparation is separate from the frozen JPEG/WAV experiments.
See `docs/PNG_BMP_COLAB_PROTOCOL_20261023.md` and its preparation status.
The portable notebook binds the existing gray/RGB SRNet skeletons, Runtime
Version 2026.07 and a generated 600-second zero-optimizer profile. Native
Commons candidates remain unassigned with pending ancestry/native review.
No new media was downloaded and no real fitting is authorized. Numeric
checkpoint QA uses generated states only; it does not authorize a recipe.
The one actual Pro/T4 capacity profile completed all twelve cells in
16.034616 seconds with zero updates; see
`docs/PNG_BMP_COLAB_CAPACITY_RESULTS_20261023.md`. This attempt is consumed;
do not rerun it automatically. Optimizer/evaluation/I/O and full-training
budgets remain unmeasured; no real fit follows from this capacity result.
Keep all historical budgets, reserved data and consumed attempts unchanged.

A later explicit continuation permits the separate native Commons admission
pilot, bound by `docs/PNG_BMP_NATIVE_ADMISSION_PROTOCOL_20261023.md` and its
unattempted-source amendment. Six originals (three L /three RGB), six author
groups and 24 PNG/BMP artifacts pass twelve independent payload checks;
all roles remain unassigned. Preserve the first partial unavailable acquisition
and later HTTP 429, with no retries. Conservative cumulative raw-transfer
debit is 55,307,394 /67,108,864 bytes; the older historical total stays null.
See `docs/PNG_BMP_NATIVE_ADMISSION_RESULTS_20261023.md`. No further download,
GPU attempt or real fit follows from the completed pilot. The 128-originals
per-domain training plan is a non-executable draft with missing data, adapter,
learning control and optimizer/evaluation/I/O cost; it does not grant fitting.
The native admission full software gate passes 3,888 tests /two CUDA skips,
97.05% total coverage; the new admission service has 100% coverage.

A later continuation permits data-only expansion under
`docs/PNG_BMP_NATIVE_EXPANSION_PROTOCOL_20261023.md`: a separate 512-MiB
reservation ceiling, native photographic L/RGB PNG sources, fixed crop
texture gates and at most eight originals per author group per mode.
The private source-bound paced schedule preserves every intent/reservation,
respects server cooldowns and never retries a previously requested SHA-1.
Keep actual roles unassigned, microscopy excluded pending the user's
preference, existing reserved data closed and both historical ledgers intact.
This is collection/preparation only; no new GPU attempt or real fitting.

The original expansion parent was interrupted after batch-1057. Batch-1058
has an attempt marker but no source intent/reservation; keep it unavailable
and skip that source. The user's 2026-10-24 continuation permits the bound
unattempted-only parent in
`docs/PNG_BMP_NATIVE_UNATTEMPTED_CONTINUATION_20261024.md`, not a retry.
Its user service `png-bmp-native-unattempted-20261024.service` has a four-hour
deadline, 2-GiB memory /zero swap and no restart. Preserve all 84,115,935
previously reserved expansion bytes and the old 55,307,394-byte prior debit.
All eighty metadata requests are consumed. Original source code, the frozen
catalogue, source holds and all fitting restrictions remain unchanged.
The user subsequently requested a stop. The continuation service is inactive
with no automatic resume. Preserve the stop receipt in
`benchmarks/png-bmp-native-expansion-user-stop-20261024.json`. The last fully
reviewed subset is snapshot 007: 13 L /79 RGB originals, 368 PNG/BMP artifacts
and 184 scalar payload checks. Later acquisitions remain unreviewed and
unassigned; do not admit them or restart collection automatically.

The user's 2026-10-26 explicit continuation supersedes that stop for data-only
preparation. Offline snapshot 008 reviews the backlog: 13 L /87 RGB originals,
400 PNG/BMP artifacts and 200 independent payload checks. Preserve the three
holds and both montage/illustration exclusions. The fresh source-bound parent
and service are documented in `docs/PNG_BMP_NATIVE_USER_RESUME_20261026.md`.
They skip every prior attempt/log and SHA-1 intent/result, keeping all
248,412,639 expansion bytes and the separate 55,307,394-byte debit at resume.
Debit 4,143 seconds from the old continuation: the new service allows at most
10,257 seconds of the original four hours, with 2-GiB memory /zero swap and
no restart. The original 512-MiB ceiling and exhausted metadata budget remain.
Sixteen generated resume checks pass with 100% coverage. New acquisitions
still require independent review; no role assignment, GPU attempt or fitting.

This resumed collection/preparation is now complete at its transfer ceiling.
The service is inactive /PID zero with no restart. Snapshot 016 has 118
reviewed originals (13 L /105 RGB), 472 PNG/BMP artifacts and 236 independent
scalar payload checks; no unreviewed candidate remains. Preserve all three
holds and both exclusions. Final replay closes 192 requests /189 raw files
with the complete reservation chain. Expansion debit is 536,421,895 of
536,870,912 bytes, plus the separate old 55,307,394 bytes; the older total
stays null. The unused 449,017 bytes cannot fit the next bound source.
Conservative old-plus-new continuation time is 7,035 /14,400 seconds; none
of the remaining time grants a retry or new download. Exact post-exit peak
memory is unavailable after transient-unit unloading; preserve the failed
observer and earlier kernel samples instead of rerunning collection.
See `benchmarks/png-bmp-native-expansion-closeout-20261026.json` and the
updated results. Fresh full QA passes 3,888 tests /two CUDA skips, 97.05%
coverage; all 59 additional generated checks pass. The corpus still lacks
115 L /23 RGB originals and ten L groups for the draft target. Keep roles
unassigned, reserved data closed, metadata exhausted and fitting unauthorized.
This attempt is consumed; no automatic collection, GPU attempt or real fit.

The existing local GPU host is Windows 11 /Ubuntu-24.04 WSL2 with the user's
RTX 5060 Laptop /8-GB VRAM. `docs/WSL_GPU_READINESS_RESULTS.md` already records
physical CUDA verification with Python 3.12.3 /Torch 2.14.0+cu130 on 2026-10-09;
later local probes and real JPEG fits are complete historical experiments.
Read that record, `docs/WINDOWS_GPU_TRAINING.md` and `docs/WSL_GPU_RECONNECT.md`
before proposing GPU/WSL installation. This Kali workspace is the separate
CPU-only VMware guest; its missing CUDA device says nothing about existing
host installation. The dedicated `steganography-wsl` SSH alias/key/known-host
files remain; the initial 2026-10-26 check found the port-22240 listener absent.
Restore the existing connection rather than reinstalling Ubuntu, replacing
keys or rerunning a consumed experiment. Current host driver/package state
needs read-only revalidation after reconnect. See the corrected
`docs/PNG_BMP_LOCAL_READINESS_20261026.md`. No new PNG/BMP GPU profile, media
acquisition or fitting was executed by this hardware assessment. Preserve
the CPU CI, frozen Colab environment and every previous local result/budget.

On 2026-10-28 the user restored the existing WSL tunnel. Read-only inventory
confirms RTX 5060 Laptop /capability 12.0, driver 617.42, Python 3.12.3 and
Torch 2.14.0+cu130; see `benchmarks/png-bmp-wsl-connection-20261028.json`.
This is not a new GPU profile. New generated-only software lives in
`core/spatial_training_data.py` and `core/spatial_generated_training.py`:
byte-rounded forty-percent scattered plane-0 controls, paired FP32 batches,
bounded AdamW step/scoring and numeric checkpoint continuation. Fifty-seven
focused software tests pass, including actual small CPU optimizer steps and
exact gray/RGB interrupted-state parity. These are QA, not a full learning
control or native accuracy result. See the dated preparation document.
The new generated-control and data-gap documents are non-executable drafts;
the actual source-bound supervisor/acceptance/independent debit contract is
still missing. Do not run a GPU profile, full generated control or native fit
from these drafts or rerun any old job. Existing real media stays unassigned.
The largest ten RGB groups hold only 41 originals; 23 more images cannot
produce the old ten-group /80-original training partition. A future split
needs independent group-feasibility review, without changing historical data
or assigning real roles. All prior transfer/metadata ceilings remain consumed.
The new preparation full gate passes 3,945 tests /two explicit Kali-CUDA
skips, 97.07% total coverage; Ruff and 233-file mypy pass. New data/training
modules have 100% /99.34% statement coverage. The proposed generated manifest
contains 220 sources /220 payload checks with disjoint source/group namespaces;
no image/model was persisted by manifest generation. See
`benchmarks/png-bmp-generated-preparation-verification-20261028.json` and
`benchmarks/png-bmp-generated-corpus-20261028.json`. Preserve the first focused
fixture failure. This is generated CPU software QA, not a full learning gate,
physical CUDA execution or permission to fit or acquire real media.

A later user continuation permits the separate generated-only intervention in
`docs/PNG_BMP_GENERATED_CONTROL_PROTOCOL_20261028.md`, conditional on its full
software gate, source/environment freeze and deployment checks. The new
`spatial_control` /`spatial_control_io` /`spatial_control_supervisor` services
implement four fresh profile cells (sixteen generated updates), conservative
cost projection and, only if eligible, four fresh fits (1,600 updates).
Its separate allowance is 1,800 seconds /8-GiB RAM /zero swap /4-GiB allocator
and 512-MiB output, with one exclusive attempt marker and no retry. All actual
learning inputs remain generated in memory; PNG/BMP transport is separately
measured. The earlier draft and preparation results remain historical snapshots.
This conditional continuation never authorizes native fitting or acquisition.
Record the actual outcome before any later action; never relaunch a consumed
attempt or treat generated-only accuracy as native qualification.

That one generated intervention is now complete: **passed_generated_only**.
All four cells completed 1,600 learning updates /12,800 presentations, plus
sixteen profile updates; 1,616 commits and eight numeric checkpoints verify.
Final generated development AUCs are .723958 /.706597 (L) and .763889 /.769097
(RGB), with train AUC 1.0 in every cell. All sixteen AUCs also replay through
an independent pairwise calculation; 1,533 parent observations and all 1,528
nonnull progress anchors verify.
Kernel execution is 1,356.435690 seconds; conservative time through independent
closeout is 1,598.898706 /1,800 seconds. Peak host memory is 3,182,301,184 bytes;
allocator allocated/reserved peaks are 3,849,946,112 /4,089,446,400 bytes.
The complete private WSL package is 469,630,247 logical bytes, below 512 MiB.
Preserve the failed direct post-exit cgroup observer: the released empty
cgroup is unavailable, while retained systemd and parent memory peaks agree.
All eight checkpoint hashes and 243 staged files pass a separate closing
audit. Numeric models remain private in WSL; the unit is inactive /PID zero,
with no restart. This attempt is consumed; no remaining time grants a retry.
See `docs/PNG_BMP_GENERATED_CONTROL_RESULTS_20261028.md` and its full receipts.
Full software QA passes 4,008 tests /two explicit CUDA skips, 97.13% coverage;
63 focused tests pass, and all four new modules exceed 95%. No native media,
download, role assignment, real fit or deployment follows from this pass.
The corpus deficit, RGB group feasibility and native cache/training adapter
remain unresolved. Native integrated I/O and its full-training allowance are
unavailable; any native fit needs a separately frozen contract and approval.

A later user continuation permits native data-adapter software preparation
and offline group review. `core/spatial_native_data.py` now provides immutable
unassigned source references, transitive group/author/page components, proposal
leakage checks, bounded hash-verified raw uint8 caches and paired full-alphabet
FP32 scattered controls. Thirty-seven generated-only tests pass with 100%
module coverage. The independent metadata graph replay retains 13 L /105 RGB
originals across 6/69 groups; eighty RGB originals require at least 44 current
groups and the historical ten-group capacity is still 41. Actual roles remain
unassigned. See `docs/PNG_BMP_NATIVE_ADAPTER_PREPARATION_20261028.md`.
The separately frozen data-only cache then decoded 118 admitted canonical
PNG crops, preserving all original/reserved media: 21,591,572 cache bytes,
236 scalar payload recoveries and 31 independently replayed FP32 batches,
with zero optimizer/GPU/network use. All 244 bindings and cache hashes verify.
Its final memory peak is unavailable: preserve the failed closing observer
and `[not set]` result. The earlier worker sample is 85,831,680 bytes before
the independent audit, not a final peak. The kernel journal bounds the unit
to 9.769115 /180 seconds under guarded <=512-MiB RAM /zero swap. The unit is
inactive /PID zero; this attempt is consumed and must not be regenerated.
Strict resource acceptance is `unavailable_final_memory_peak` despite verified
cache contents. See `docs/PNG_BMP_NATIVE_CACHE_RESULTS_20261028.md`.
Full software QA retains 4,044 passes /one README-length failure /two explicit
CUDA skips, 97.15% coverage. A README-only repair passes all fifteen related
tests with unchanged application/test hashes, verifying 4,045 unique tests;
do not relabel the original full command as passing. Ruff /238-file mypy pass.
Preserve the first generated worker failure on empty namespace source files.
A native optimizer supervisor, feasible split, integrated I/O cost and full-fit
allowance remain missing. Restoring the
WSL tunnel does not permit a consumed profile/fit to restart. Preserve all old
source/environment bindings, holds, reserved data, failed logs and budgets.

The existing WSL tunnel is restored again; read-only inventory confirms the
same RTX 5060 /617.42 /Python 3.12.3 /Torch 2.14.0+cu130 CUDA environment.
The private native-cache handoff verifies all 119 files /21,591,572 bytes in
`steganography-gpu/.benchmark/png-bmp-native-cache-handoff-20261028/cache`,
relative to the WSL user's home. No original cache was regenerated; all 244
bindings stay unchanged. See `docs/PNG_BMP_WSL_HANDOFF_20261028.md`. The
nonlogin PATH's missing nvidia-smi was resolved using the existing binary;
no package/driver/key change. Transport preserves unassigned roles and the
cache's unavailable final-memory acceptance; it grants no historical retry.

The later user continuation permits the separate RGB pilot preparation in
`docs/PNG_BMP_RGB_PILOT_PROTOCOL_20261028.md`: exact metadata-only 64/20/21
RGB proposal across 38/12/19 ancestry components; thirteen L sources remain
unassigned. The new native training/pilot/supervisor services require full
software QA and source/environment freeze before one fresh 240-second GPU
cost profile. It has zero native updates and six generated optimizer updates,
with two generated numeric checkpoint restores and live cgroup peak evidence.
The candidate native recipe is two seeds /twenty epochs /640 updates, but its
1,800-second /768-MiB cumulative allowance must be measured and explicitly
approved against the accepted profile before execution. Do not create an
approval from a preparation request. No historical retry, cache regeneration,
new download or reserved-pixel access is permitted. Actual source roles stay
unassigned; proposal roles are a separate access overlay.

The fresh RGB software gate now passes 4,092 tests /two explicit Kali-CUDA
skips, 97.18% total coverage; all five new data/training/pilot/supervisor/CLI
modules exceed 95%. Ruff /242-file mypy pass; a separate 51-test focused run
also passes. Preserve the initial tool-server interruption, the optional-Torch
CI-contract failure (215 passes /one failure, then interrupted), its repaired
test sources and all focused failures. No earlier command is relabelled.
All 464 application/test input hashes and the old 244 cache bindings verify.
See `docs/PNG_BMP_RGB_PILOT_PREPARATION_20261031.md`. The successful local QA
unit's final post-exit MemoryPeak is unavailable; no final peak is claimed.
The outer profile controller also binds parent wall time, evidence transport,
unit stop and a five-second closing reserve in an exclusive `closed.json`;
only that final profile may support later approval. The current WSL port-22240
listener remains absent: source/environment freeze and physical cost are still
unavailable, with no fresh GPU attempt consumed or native role rewrite.
EVREN's public H200 /141-GB notebook interface and 10-CR/hour single-GPU
training tariff were reviewed separately in `docs/EVREN_PLATFORM_REVIEW_20261031.md`.
Actual account runtime access, notebook pricing and environment compatibility
remain unverified. This review performs no authentication, upload or job and
does not convert the WSL-bound contract into an EVREN execution allowance.

A later user-authorized inspection of the existing EVREN browser session
confirms that ordinary training is accessible but `/workspace` redirects to
access denied for the current account. GPU reservations explicitly provide no
SSH or direct server access; custom SRNet execution remains unavailable there.
The Kaşif programme advertises early Workspace access, without a demonstrated
guarantee. No project, notebook, application, reservation, upload or job was
submitted; no credentials/tokens were extracted. Keep account screenshots and
identifiers private. See the same EVREN review document for this later finding.

On 2026-11-06 the existing WSL connection is restored and the unchanged
Python 3.12.3 /Torch 2.14.0+cu130 /driver 617.42 /RTX 5060 environment and
native cache manifest verify again. The first source-only e5c2ea05... package
cannot reach the existing user bus because SSH lacks its two environment
variables. Preserve that read-only failure; no launch/attempt marker exists.
The prospective amendment in `docs/PNG_BMP_RGB_DEPLOYMENT_ENVIRONMENT_AMENDMENT_20261106.md`
sets only process-local user-bus context. Protocol 024f575186008867d66ab9e4760c858faa53f516d6fd85f4771e8ae20494a618
and its fresh source-only package preserve the first package and unchanged 464
application/test files. Keep the local staging-receipt collision and independent
verification of the already-staged files; do not restage or relabel the failure.

That one fresh RGB profile now completes: **profile_ready**, zero native /six
generated optimizer updates, 48 native backward /256 train-only evaluation
presentations and two generated numeric states. Each state's 503 arrays,
source/data/environment/order and three-update debit replay independently;
all six intent/commit pairs verify. Native parameters stay unchanged and no
native scores are recorded. Worker time is 34.515326 seconds, WSL kernel time
42.777769, independent controller kernel time 48.680403 and conservative
command debit 52.546351 /240 seconds. A separate read-only CPU audit charges
four more seconds, for 56.546351 total prospective-budget debit. Live cgroup
memory.peak and retained systemd peak agree at 2,370,605,056 bytes; allocator
allocated /reserved peaks are 3,849,421,824 /3,995,074,560 bytes. The complete
private package is 119,847,933 logical bytes /354 files, below 256 MiB.
Both units are inactive /PID zero with no restart. The profile is consumed;
no automatic rerun, native fit, source acquisition or deployment follows.
Preserve the old cache's unavailable final-memory acceptance and all old ledgers.
See `docs/PNG_BMP_RGB_PILOT_CAPACITY_RESULTS_20261106.md` and its receipts.

The new fit proposal stays **proposal_only_pending_user_approval**: two fresh
RGB seeds /twenty epochs /640 updates /5,120 presentations, unchanged learning
rule and 64/20/21 grouped split. All actual roles remain unassigned; twenty-one
reserved RGB and thirteen L originals stay closed. Conservative fit projection
is 1,717.298493 seconds. Keep the nominal 1,800-second /768-MiB cap; account
for the extra audit by reducing a later controller's cumulative cap to 1,796
seconds, and reserve four MiB by reducing its output cap to 764 MiB. After the
immutable profile debit, native fit allowance is 1,743.453649 seconds (systemd
1,743) /683,999,060 bytes; estimated margin is only 26.155156 seconds.
Approval must bind the amended protocol, final closed-profile SHA-256
10451b458524bddc28f94c39ef8ac7e87f82ea479db1669d3f034262078dbc41,
canonical split and these exact limits. No approval receipt exists. The user's
request to reconnect or inspect EVREN is not approval for that native fit.

Later on 2026-11-06 the user explicitly selects EVREN for future GPU work:
local GPU load makes the laptop unusable. This supersedes local fitting as the
next action. Do not launch another local GPU profile/fit or spend unused local
allowances unless the user explicitly changes that preference. The completed
profile and all private checkpoints/receipts remain preserved; closing inventory
finds both WSL units inactive /PID zero and no CUDA compute process. The local
fit proposal is now `deferred_user_selected_evren`; its prior bytes are retained.
EVREN preparation and authorized account inspection may continue, but the WSL
controller, profile costs and protocol do not authorize an EVREN job. Establish
custom workspace permission, backend/resource capabilities and a separately
bound source/environment/cost contract before credit use or native fitting.
Keep all actual media unassigned and existing reserved pixels closed.

EVREN follow-up preparation on the same date reopens the existing browser
session and confirms `/workspace` still redirects to access denied. No cloud
job, upload, reservation or application is submitted. A reviewed, unsent access
request is in `docs/EVREN_WORKSPACE_ACCESS_REQUEST_DRAFT_20261106.md`; backend
handoff conditions are in `docs/PNG_BMP_EVREN_PREPARATION_20261106.md`.
The new `notebooks/png_bmp_evren_readiness_20261106.ipynb` is CPU-only metadata
inspection: syntax and local execution pass, Torch is not imported and there
are no GPU calls or model/optimizer creation. It starts no runtime and grants
no fitting authority. Screenshots, account identifiers and credit observations
stay private. All 464 verified application/test files remain unchanged, so no
long software gate was rerun for this document/notebook-only preparation.

The user subsequently clarifies that EVREN AI Workspace is currently limited
to the 29-person early-access programme. Treat this as a service availability
constraint, not an ordinary account permission fault. Do not keep proposing
that an access request will enable general Workspace use. The unsent access
request is deferred; no application or message is authorized by this correction.
Custom SRNet execution on EVREN remains unavailable until an officially
supported custom-code route is accessible. The ordinary training registry
contains built-in model families and no observed SRNet option; do not replace
the frozen SRNet recipe silently. Keep the user's no-local-GPU preference.
A different cloud backend or a different model needs an explicit user choice
and its own source/environment/resource contract before execution.

The user's subsequent 2026-11-06 message selects Google Colab after the
EVREN early-access clarification. Colab supersedes EVREN as the next GPU
backend; the preference against local GPU load remains active. See
`docs/PNG_BMP_COLAB_RETURN_20261106.md`. The new CPU-only readiness notebook
checks package metadata without importing Torch or running GPU operations.
Preserve the consumed 2026-10-23 Pro/T4 attempt and 2026-11-06 WSL profile;
neither is a new Colab optimizer/I/O measurement or fitting approval. A new
Colab supervisor/source/environment/cost contract is required before a fresh
GPU intervention. No real cache/model upload or acquisition follows from the
backend choice. Keep all roles, reserved pixels and historical debits intact.

The separate Colab CPU session now completes with Runtime Version 2026.07,
Python 3.12.13 /NumPy 2.0.2 /Pillow 11.3.0 /Torch 2.11.0+cpu and
`metadata_ready_only`. Preserve its first incorrect GPU-build comparison as
unavailable; the corrected CPU metadata check is separate. Cgroup metadata is
unavailable and the subscription screen shows no active Pro entitlement;
neither resource enforcement nor a current GPU allocation is established.
The readiness notebook alone was uploaded, saved and the CPU runtime closed.
No cache/model upload, package install or GPU profile was performed. See
`benchmarks/png-bmp-colab-return-readiness-20261106.json`. All 464 application/
test hashes still match the passed full QA; the long suite was not rerun.

The user's later free-Colab continuation permits one separate read-only T4
allocation/inventory. That allocation succeeds: Tesla T4 /15,360 MiB VRAM /
driver 580.82.07 with the expected 2026.07 GPU package metadata. The first
inventory remains `unavailable_inventory`: its cgroup identity contains parent
segments and is correctly rejected. Preserve that first result and do not join
or normalize its unsafe path. A separately sourced same-allocation closing
audit retains the failure and performs the first/only NVIDIA query; it does
not relaunch the worker or request another card. Both executed source cells
and receipt hashes replay exactly from the ordinary-UI notebook export. The
runtime is disconnected/deleted within 294.536867 /300 observed seconds.
No Torch import, tensor/model/optimizer operation, native upload, package
installation or new GPU profile occurs. This hardware inventory is consumed;
do not rerun it automatically or mistake its metadata for resource acceptance.
See `docs/PNG_BMP_COLAB_GPU_INVENTORY_RESULTS_20261106.md` and the concrete
`docs/PNG_BMP_COLAB_RGB_PORTABILITY_20261106.md`. The WSL-specific actor,
cgroup-v2 guard and independent systemd timing need a distinct Colab executor.
All old trials, budgets, reserved pixels and 464 fully tested input hashes stay
unchanged. Preserve prelaunch v1/v2 source corrections and the first Ruff
failure; no source correction is labelled as an old GPU attempt retry.

The subsequent Colab readiness correction is implemented in the new model-free
`core/spatial_colab_readiness.py`. Resource discovery now returns immutable
unavailable metadata for parent-segment/inaccessible cgroup identities while
an explicitly requested bounded NVIDIA query completes independently. It never
normalizes that unsafe path or claims resource enforcement. Forty-four generated
CPU tests pass with 100% new-module coverage; seven source-bootstrap checks and
an isolated delivered-source CPU import/receipt roundtrip also pass. Ruff and
243-file mypy pass. The self-contained v3 notebook defaults to CPU and embeds
exactly three service sources plus LICENSE with package/per-file hashes.

The new v3 notebook completes an actual 2026.07 CPU-only Colab validation:
`partial_resources_unavailable`, matching pinned CPU package metadata and the
same `parent_identity` reason. The service takes 0.0574666 seconds. Both executed
cells and the receipt hash replay exactly from its ordinary-UI export. The CPU
runtime is closed after 197.487040 /300 observed seconds, with zero GPU/NVIDIA
requests and no models, media/cache/checkpoint upload or historical relaunch.
This fixes the metadata control flow, not the missing Colab training supervisor.
Keep the first GPU inventory unavailable and all old acceptance/budgets intact.
See `docs/PNG_BMP_COLAB_READINESS_FIX_20261106.md` and its v3 CPU/binding receipts.

The fresh complete CPU software gate now passes 4,136 tests /two explicit
Kali-CUDA skips, 32 warnings and 97.194602% total statement coverage. The new
readiness service has 100% coverage; Ruff and 243-file mypy pass. All 466
input hashes verify and the original 464 remain unchanged. It uses existing
provisioned dependencies, one BLAS/OpenMP thread and disk-backed TMPDIR/basetemp
under eight-GiB RAM /zero swap. Pytest takes 1,425.44 seconds; its wrapper
records 1,427.431714 seconds and exit zero. The QA unit is inactive /PID zero
after transient-unit unloading. Preserve its unavailable final MemoryPeak and
independent kernel interval; do not rerun to reconstruct them. Full source/log/
coverage/exit evidence is retained in `.benchmark/png-bmp-colab-readiness-v3-20261106`
and the public v3 verification receipt. This completes the metadata-control
software correction, not the separate Colab training executor. No new GPU
profile, native fitting, cache upload, download or historical retry follows.

The later continuation implements the generated-only process software in
`core/colab_execution.py`, with a thin `research_colab_execution` CLI and
reproducible CPU notebook delivery. Only `generated_qa` is registered; there
is no native worker or arbitrary-command route. Exclusive source/environment
contracts and workspace attempt ledgers, hard Linux DATA/CPU/file/optional-AS
limits, parent RSS/output sampling, kernel wall/parent-death termination and
post-exit `wait4`/hash audits are covered by 61 focused generated CPU tests.
Both new modules exceed 95% statement coverage. RAM watching/DATA/AS do not
provide the old cgroup resident-RAM/zero-swap contract; keep those unavailable.
The kernel RSS peak includes inherited process-creation memory.

This new controller completes one actual Colab 2026.07 CPU-only smoke task:
zero optimizer updates, exit 0/reaped, 1.454632 seconds including closing
reserve, kernel peak 107,876,352 bytes and zero observer gaps. The runtime is
disconnected/deleted within 167.814153 /300 observed session seconds. Exported
original cells and all preclosing output hashes replay; the reconstructed
contract hash confirms Python 3.12.13 /CPython /Linux. Preserve its consumed
receipt. The public notebook subsequently wraps literals/sorts two imports
for lint; source packet bytes and bootstrap AST are unchanged, with a passing
related delivery test. Keep the exact executed original privately. The fresh
full gate's original 470-source binding is retained; only that presentation
builder changes, while all 469 application/test inputs remain unchanged.
Preserve the first targeted-coverage import failures, proc-transition fixture
failure, lint failures and the additional paused-parent observer's incorrect
nonzero-exit/invalidation expectation. Its retained API result is correctly
unavailable, SIGALRM/reaped; no second paused-parent run reconstructs its missing
kill timestamp. See `docs/PNG_BMP_COLAB_PROCESS_PREPARATION_20261106.md` and
`benchmarks/png-bmp-colab-process-cpu-validation-20261106.json`.
The new Colab CUDA policy/native profile adapter and measured optimizer/I/O
cost remain missing. This software preparation creates no GPU/profile/native
permission, fit approval, media upload/acquisition or historical retry.

Its fresh full gate passes 4,197 tests /two explicit Kali-CUDA skips,
32 warnings and 97.194842% total coverage. Controller/CLI coverage is
97.278912% /96.551724%; Ruff /245-file mypy pass. Pytest/wrapper time is
1,308.36 /1,310.161493 seconds. The QA unit is inactive /PID zero with no
restart; preserve unavailable final MemoryPeak and kernel interval after
unloading. See `benchmarks/png-bmp-colab-process-verification-20261106.json`.
The user's subsequent nonstop continuation permits the next native Colab
adapter software preparation and generated CPU QA. It does not approve a
native fit or rerun a consumed GPU/profile attempt.

That software extension now lives in `spatial_colab_profile` and
`spatial_colab_handoff`, with a thin `research_colab_profile` CLI and a new
CPU-default notebook. The separate native profile schema requires an external
source/resource-bound permission; no actual permission was created. Native
optimizer steps/checkpoints, nontrain scoring and saved native scores are
forbidden by the profile view. Generated namespaces alone allow six AdamW
updates and two numeric restores; the parent independently validates arrays,
environment/data/order and all six commits without Torch or native pixel reads.
Private evidence packing reserves duplicate bytes and validates ZIP directories
before allocation, hashes/CRC, paths, regular members and total bounds, without
extraction/execution. Source-only and train-only cache transports are explicit;
no actual cache package, upload, GPU runtime or native job was started.

The prospective Colab contract SHA is
`7d5550729a9ba777ca8658fc0374e206967fdb0ca4e15856526198ac37d1aa0f`;
reviewed source packet SHA is
`caa92688920318ec559ec6af030f62160dcbd482679c8f24735fccbfd46926dd`.
See `docs/PNG_BMP_COLAB_NATIVE_PROFILE_PROTOCOL_20261106.md` and the prospective
contract/preparation receipts. It proposes one 240-second T4 command, including
independent audit/archive and ten-second closing reserve: hard DATA eight GiB,
eight-GiB worker RSS watch, four-GiB Torch allocator, 128-MiB per file and
256-MiB cumulative output. Hard cgroup resident RAM/zero swap and aggregate
parent/worker peak remain unavailable; this different prospective policy needs
explicit acknowledgement, not a retroactive relaxation of the WSL rule.
Only 64 train RGB crops /12,582,912 pixel bytes may be transported after bound
permission; all 20 development /21 reserved RGB and thirteen L pixels stay
closed. The exact old proposal/cache hash, actual unassigned roles, original
cache's unavailable final-memory acceptance and every old budget stay unchanged.
The expected 2026.07/T4 package/driver metadata comes from the earlier consumed
inventory, not a new CUDA validation. A positive actual cuDNN version is frozen
before models and checked in both numeric states. The notebook creates no
permission and defaults all permission/media/GPU/export switches to false.

All 132 focused software tests pass; new controller/profile/handoff/CLI modules
exceed 95% coverage. This includes real generated RGB CPU steps/restores and
an isolated delivered profile with application imports blocked, plus source-only
default notebook execution with no Torch/media/GPU/intent. Emulated CUDA or
native dispatch tests never qualify physical hardware. Preserve the first
fixture checksum-reader failure and intermediate lint/type failures. The new
476-file full gate is running in user unit
`steganography-colab-native-profile-qa-20261106.service`; logs/source binding/
coverage are under `.benchmark/png-bmp-colab-native-profile-20261106`.
Ruff and 248-file mypy pass. Only the new process controller changes relative
to its earlier completed software snapshot; all original 466 inputs and the
exact executed v1 notebook/source packet are unchanged. Record this full gate
before any later GPU action. No native fitting worker or full-fit approval is
implemented; the 640-update cost projection remains an unapproved candidate.
An additional read-only generated CPU evidence integration reuses both saved
states: 502 arrays each /six commits verify with no Torch import, optimizer
step or actual media access. Its 115,939,468-byte archive and 231,871,115-byte
total fit the 256-MiB envelope in 1.217173 observed seconds. This is not T4
cost/size evidence; preserve the source-bound full gate and physical-result gap.

The first full native-profile software gate fails one proc observer test:
4,267 pass /one failure /two CUDA skips, 97.21% coverage. Preserve that command,
log and source binding. A killed child disappeared during the proc read with
ESRCH; the controller now treats ENOENT/ESRCH as disappearance and still requires
`wait4` reaping/kernel evidence. Two added regressions pass. Permission parsing
also requires literal boolean true, rejecting numeric 1. The repaired focused
gate passes 134 tests; recipe, data and all resource ceilings are unchanged.
Old prospective v1 contract/preparation and exact private source packet remain.
The current contract SHA is
`cce2cdce2623db31d872867f9e9d79450620ec6816e57740dec579111650302b`;
source packet SHA is
`4d3fccc4d9893e1030452a1c855d766e76bcc98201fc846daff786c19d052fce`.
No actual permission or physical attempt exists for either version. The fresh
476-source repaired full gate runs in
`steganography-colab-native-profile-repaired-qa-20261106.service`, with evidence
under `.benchmark/png-bmp-colab-native-profile-20261106/verification-repair`.
Its source amendment names the controller/two test files. Record its final
result without relabelling the failed original or starting a GPU automatically.

The repaired full gate now passes 4,270 tests /two explicit Kali-CUDA skips,
32 warnings and 97.212052% total coverage. Controller/profile/handoff/CLI
statement coverage is 97.391304% /98.901099% /96.629213% /98.039216%; Ruff and
248-file mypy pass, all 476 source hashes match. Pytest/wrapper elapsed time
is 1,508.10 /1,509.702084 seconds. The repaired QA unit is inactive /PID zero
with no restart; final MemoryPeak/kernel interval are unavailable after unloading.
Preserve the first observer and failed original, without a reconstruction rerun.
See `benchmarks/png-bmp-colab-native-profile-verification-20261106.json`.
The current cce2... contract /4d3f... source packet are software-ready, with
explicit physical profile/resource/data-transfer permission still missing.
No actual train-cache package or upload, new GPU allocation/attempt, native
optimizer step or fit/deployment follows automatically. The next question
may seek permission for that one fresh 240-second T4 profile with the exact
64-train overlay and acknowledged DATA/RSS-watch policy; it never approves a fit.

That exact profile/data-transfer/resource-policy question has now been sent;
the answer is pending. A preselected UI option or elapsed time is not approval.
Do not create the production permission file, copy actual train pixels, upload
media or allocate a fresh GPU until the user explicitly accepts these bounds.

The user subsequently rejects the inherited fixed eight-GiB RAM ceiling.
That resource-policy question is now obsolete, not an execution approval.
The native Colab v2 contract removes custom DATA/RSS caps (both fields zero;
no prlimit --data), inheriting provider/runtime limits. `core/colab_memory.py`
observes kernel MemTotal/MemAvailable and stops at ten-percent headroom;
kernel capacity is not a verified tenant limit. Cgroup counters remain
informational. CPU/file/output/time, observed-swap stop and the four-GiB Torch
allocator remain. Do not change historical WSL/generated-process contracts.
The current prospective contract SHA is
`d791914942d070f37b03c9bb5c3710fa7ce5f539d977f909f5b98fe7c912a865`;
source packet SHA is
`77e3e8dc5e144c1d9cf94089d1ffccd945c9c0c90fb64271871c747d38ec8e90`.
Old fixed-memory proposal/preparation are preserved as v2 snapshots.
The corrected focused run passes 149 tests; Ruff /249-file mypy pass.
Preserve the first 147-pass/two-fixture-expectation failure and first full
QA's missing-basetemp-parent interruption (182 passes /528 setup errors).
Fresh full QA uses the identical 478 frozen inputs in
`steganography-colab-provider-memory-complete-qa-20261106.service`, with logs
under `.benchmark/png-bmp-colab-provider-memory-20261106/full-qa`.
Record that gate independently; the old 4,270-test pass belongs to old code.
No production permission, actual pixel package/upload or GPU attempt is
created by the RAM correction. Native fitting still has no registered worker.

The provider-managed revision's full gate now passes 4,285 tests /two explicit
Kali-CUDA skips, 32 warnings and 97.205087% total coverage. Memory observer /
controller /profile /handoff /CLI statement coverage is 100% /96.685083% /
98.901099% /96.629213% /98.039216%. Ruff /249-file mypy /diff checks pass;
all 478 frozen inputs match and all original 466 remain unchanged. Pytest /
wrapper take 1,390.04 /1,391.702300 seconds. The successful QA unit is inactive /
PID zero with zero restarts. Its final MemoryPeak and independent kernel
interval are unavailable after unloading; preserve the first observation and
do not rerun to reconstruct them. See
`benchmarks/png-bmp-colab-provider-memory-verification-20261106.json`.
The independent source/metadata delivery audit checks all 216 packet entries,
the exact native v2 contract, disabled notebook switches and 64 unassigned
train-proposal metadata rows, without actual pixel reads. This software pass
removes the inherited host cap; it creates no new runtime/profile or native fit.

`v0.6`: shared analysis context, PNG/BMP native analysis, bounded CTF playbook,
portable JSON/HTML/SARIF/bundle reporting, and initial CLI/API/TUI surfaces.
See `docs/ROADMAP.md` for current status and later milestones.

The user's later request to finish preparation carefully and faster authorizes
the complete Colab fitter software and generated CPU QA. The new service is
`core/spatial_colab_fit.py`, with a thin training CLI and complete 2026-11-07
notebook. `native_fit` requires an accepted source-matching physical profile,
complete external debits and a later distinct profile-bound fitting permission.
It inherits the provider-managed host policy; no eight-GiB RAM cap returns.
The nominal 1,800-second /768-MiB allowance debits profile/closing/external
work before fitting and additionally projects per-epoch development inference.
One marker keyed by profile receipt forbids reusing it with an altered fit
contract. All actual roles remain unassigned; the 64/20/21 RGB overlay and
thirteen L sources are unchanged. Development transport requires fit permission;
reserved pixels stay closed. No production permission or actual pixel packet
is created by this preparation, and no GPU allocation/profile/fit is executed.

The prepared recipe retains two seeds /twenty epochs /640 updates and records
loss/development AUC every epoch. Numeric states at 5/10/15/20 per seed replay
independently with all commits, scores and order/debit/environment bindings.
Training report archives omit numeric files; eight states export individually
under the closing inventory. The fit receipt is bounded to four MiB; explicit
training ZIP member bounds are 4,097, while the profile default stays 513.
The historical WSL/native engine, model skeletons and every old budget remain.
See `docs/PNG_BMP_COLAB_TRAINING_PROTOCOL_20261107.md`.
The source packet SHA is
`8677a46f3fb7840b01b46b2b1355a286c9d30296ccf271519fe54bc57d8e300a`;
new prospective profile contract SHA is
`37d2f589a90ac29dcafea3fc238217022597acc3ea084d1b1f8eb6273a35a33b`.
Keep all earlier notebook/protocol/source packets as historical snapshots.

The corrected focused gate passes 225 tests; fitter/CLI coverage is
99.218750% /95.744681%, controller/handoff 96.314496% /97.260274%.
Ruff /251-file mypy /diff pass. Twelve actual generated CPU SRNet steps,
four numeric states and twenty independently replayed AUCs are software QA,
not a full control or native result. The default full notebook is tested
source-only, with no Torch/media/GPU/intent. Preserve the first 216-pass/two-
failure command (tuple serialization and archive count), its repaired
218-pass command and all lint/type evidence. Full QA is frozen over 482
inputs in `steganography-colab-training-complete-qa-20261107.service`, with
private evidence in `.benchmark/png-bmp-colab-training-preparation-20261106`.
Record its outcome before any physical action. No fitting approval follows
from the user's preparation request or a software pass.

The first full training-preparation QA was interrupted prospectively to fix
a live inventory race with atomic progress-file renames. Preserve its log,
original 482-source binding and the subsequent tmp_path stash KeyError during
SIGINT teardown; no final count/wrapper-exit receipt exists. It stays unavailable.
Only live nonhash size sampling skips an ENOENT entry; closing hashed inventory
still fails on disappearing evidence, with symlink/type checks unchanged.
The new regression brings focused QA to 226 passes in 143.40 seconds;
controller coverage is 96.359223%, other new-module figures are unchanged.
The fresh final full QA is in
`steganography-colab-training-final-qa-20261107.service`, with evidence under
`.benchmark/png-bmp-colab-training-preparation-20261106/verification-final`.
The current notebook/prospective contract use the 8677a46f... /37d2f589...
source bindings above. Preserve the exact earlier unexecuted candidate privately.

Final Colab training-preparation QA passes 4,362 tests /two explicit Kali-CUDA
skips, 32 warnings and 97.220138% total coverage. Fitter/controller/handoff/CLI
coverage is 99.218750% /96.359223% /97.260274% /95.744681%. Ruff /251-file
mypy /diff checks pass; all 482 frozen hashes match and all original 466 inputs
are unchanged. Pytest/wrapper take 2,175.47 /2,177.758398 seconds. The successful
unit is inactive /PID zero /zero restarts. Its final MemoryPeak and independent
kernel interval are unavailable after unloading; preserve the first observer
and never rerun to reconstruct them. See
`benchmarks/png-bmp-colab-training-verification-20261107.json` and the final
preparation receipt. Software preparation is complete. A separately permitted
physical profile must supply actual compatibility/cost; only an eligible
profile can freeze a reviewable fit proposal for later explicit approval.
The same frozen source already implements that transition and the full fit.
No production permission, actual native packet/upload, physical profile or
native optimizer update was performed by this preparation. Keep the nominal
budget/ineligibility rule, prior attempts/ledgers, reserved media and actual
unassigned roles unchanged; no automatic source/budget relaxation or retry.

The user's subsequent "siradakii" permits the one separate train-only Colab T4
cost profile, now consumed. Status is
`completed_command_operator_close_target_missed`: independent command acceptance
passes in 79.297540 /240 seconds with zero native /six generated updates, two
503-array generated states and unchanged native parameters. Only 64 training
crops were uploaded; development/reserved/L pixels stayed closed. All 34 archive
members, six intent/commit pairs and exported notebook cells replay. The closed
receipt SHA is
`326b09839a7c57b22652305d98fc24a998c256ff65c8239e38c2db1dbeb5c448`.
The GPU session took 957.495231 seconds, exceeding the 600-second operator
closing target; preserve that failure and the corrected pre-execution UI paste.
The runtime is deleted and the session manager confirms no active sessions.
Do not rerun this profile or treat command time as entire allocation time.
See `docs/PNG_BMP_COLAB_PHYSICAL_PROFILE_RESULTS_20261107.md` and its receipts.

The complete two-seed/twenty-epoch fit projection is 1,311.131131 seconds,
including epoch development scoring. Conservative session/handoff charges add
950 seconds and 64 MiB outside the command: cumulative debit is 1,029.297540
seconds, leaving only 770.702460 of the nominal 1,800. Time is ineligible;
storage alone is eligible. The 2,700-second cumulative /768-MiB suggestion is
non-executable: the current validator still requires 1,800 seconds. A separately
reviewed source-compatible resource amendment and explicit later fit approval
are missing; never forge smaller debits, reset budgets, relaunch the profile
or create a fit permission from this continuation. Future setup remains charged.
All 482 tested inputs remain unchanged; no long software gate was repeated.
Hard cgroup RAM/zero-swap, whole-tree RAM and whole-driver VRAM peaks remain
unavailable. All old cache acceptance, roles, reserved pixels, transfer ledgers
and the no-local-GPU preference remain unchanged.

The next user continuation permits preparation of a distinct 2,700-second
cumulative resource amendment. `core/spatial_colab_budget.py` pins the consumed
closed/profile/audit/ledger hashes in a 76,080-byte metadata packet. It checks
all 215 immutable old sources and the protected fit AST; only five fit resource/
permission adapters and the thin CLI change. No old worker/checkpoint/pixel is
recreated or executed. Existing profile/fit markers and the controller are
unchanged. The v1 cumulative allowance stays ineligible; v2 requires complete
unchanged prior charges plus 300 seconds for future setup/export/closing and
four MiB more external evidence. This is not a budget reset.

The exact fit proposal SHA is
`7a45cab6774388e9eb156af90eb4c10c962f522a6a2ad6678853fb88e65d7b94`;
source packet SHA is
`a465948b02d85321f802535143ddccb0a0d781f7bb8f029b8b99427013b951f1`.
Of 2,700 seconds /768 MiB, prior/reserved work consumes 1,329.297540 seconds
and 305,485,000 bytes. The new command allows 1,370.702460 seconds /499,821,368
bytes, with unchanged 20-second command closing and provider-managed RAM.
The 1,311.131131-second estimate fits, retaining only 29.571329 seconds beyond
the validator's additional thirty-second margin. Future observed allocation
age must be supplied; setup is below 150 seconds, preserving 150 for export/
runtime deletion. Keep any missed closing target as a failure, with no retry.

See `docs/PNG_BMP_COLAB_CUMULATIVE_AMENDMENT_20261107.md`, the new disabled
fit-only notebook and exact unapproved v2 template. No profile-launch route
exists in that notebook; default cells are metadata-only with no Torch, model,
pixel or marker. The first unexecuted draft's 180-second future reservation is
superseded prospectively by 300 seconds. Preserve all draft/source/lint failures,
the first 22-pass/ten-failure mutable-test-debit command and its corrected
34-pass command with a failing whole-project focused coverage exit. The
260-test focused gate passes; the new service has 100% focused statement
coverage. The first full launch is unavailable after interruption for builder
import/literal lint corrections: SIGINT was absorbed by a tested supervisor,
then the scoped QA unit was killed. Its log includes the resulting test failure
and has no final pytest count. Do not relabel it. The corrected 522-input full
gate runs in `steganography-colab-budget-corrected-qa-20261107.service`; record
its outcome before GPU action. The corrected source-only 34-test gate, Ruff
and 252-file mypy pass. No new GPU allocation, native media access or optimizer
step occurs in this preparation. Actual native fitting still requires explicit
user approval of that exact contract and resource acknowledgement.

The corrected cumulative-amendment full gate is complete: 4,396 passes /two
explicit Kali-CUDA skips /32 warnings, 97.225442% total coverage. New resource
service coverage is 100%; fitter/controller/handoff/CLI are 99.256506% /
96.116505% /97.260274% /95.833333%. All 522 source hashes match; 480 previous
inputs remain unchanged, with only the declared two adapters modified. Ruff
and 252-file mypy pass. Pytest/wrapper take 1,400.59 /1,402.060753 seconds.
Preserve the unavailable final MemoryPeak: the retained exited unit's observer
still reports `[not set]`. Its 270 samples include a maximum 8,589,934,592-byte
live sample, not a final peak. The live kernel did verify eight-GiB RAM /zero
swap for CPU QA only. Current unit timestamps are zero; independent kernel
interval is unavailable and current monotonic time predates the old start.
Never compare those clock domains or rerun to reconstruct evidence. The unit
is inactive /PID zero /zero restarts. Verification and preparation receipts
are in the public `png-bmp-colab-cumulative-amendment-*-20261107.json` records.
Status is ready_for_explicit_fit_approval. No production permission, GPU
allocation, real-media pixel access or native update was performed. The later
generic "devam" continues this preparation, not approval of the unpublished
contract. Actual fitting still requires the exact v2 contract/resource approval.

The user's subsequent "devam et" directly answers the explicit final question
approving the linked 2,700-second cumulative contract /two seeds /twenty epochs /
640 native updates. This is now fitting approval for the exact 7a45cab6... v2
contract, not another preparation-only continuation. The private permission
and approval receipt are in `.benchmark/png-bmp-colab-native-fit-20261107`.
All 522 tested input hashes still match. One existing train packet is reused;
the permission-gated twenty-source development packet is 5,246,282 bytes,
SHA f359a4cb4460b17fb75e52a74da566e2b884d406f4ad8b3efe9a4f651b7c5ae6.
No acquisition or reserved/L pixel access follows. Actual roles stay unassigned.
The private operational notebook embeds that approved permission, takes source
and a bounded independent allocation timestamp in one upload, and then takes
only train/development packets after permission verification. It prints launch
timing and exports the original ledger anchors; source services and the frozen
contract are unchanged. An isolated first-three-cell execution verifies the
exact source/contract/permission with no Torch, native pixels, model or marker.
This is deployment input QA, not a fresh GPU attempt or optimizer update.
Record the forthcoming single allocation/fit outcome before further action;
no automatic retry, old-profile relaunch, local GPU work or budget reset.

That one approved Colab deployment is now consumed:
`unavailable_prelaunch_setup_reservation_exhausted`. The T4/2026.07 allocation
request and first three cells complete; the second setup-age guard rejects
after both permitted train/development cache installs, before the launch print,
controller/output directory or any native model/optimizer work. There are zero
worker launches/updates/checkpoints and no native scores. Bootstrap allocation
age is conservatively 81.929555 seconds; full request-to-confirmed deletion is
252.836224 seconds (254-second conservative upper bound). The ordinary account
session manager confirms no active sessions. No fresh NVIDIA query or GPU
profile occurred. The exact rejection elapsed argument was not printed.
See `docs/PNG_BMP_COLAB_NATIVE_FIT_DEPLOYMENT_RESULTS_20261107.md` and its receipt.

All 84 approved RGB crops were transported and hash-checked; reserved 21 RGB /
13 L pixels stay closed and actual roles remain unassigned. The original cache
was not regenerated; only its ephemeral working copy was installed. Preserve
the unavailable complete-worker/environment/memory receipts. Independent UI
notebook replay verifies counts 1–4 /unexecuted export cells and all six ASTs.
The first literal comparison failed because Colab's show/hide-code command
inserted one leading `# @title` comment per cell; stripping only that comment
restores exact source. No source packet/contract/model change occurred. All
522 tested hashes still match; no long software suite was repeated.

Charge the entire reserved 300 seconds: cumulative prior/external debit remains
1,329.297540 seconds /305,485,000 output bytes. The prelaunch 150-second target
failed; deleting within the separate 300-second envelope does not pass it.
The nominal unentered 1,370.702460-second command grants no new allocation or
retry. Do not relaunch the consumed deployment, reuse its private approval to
create another attempt, discard its spent time, or silently expand the window.
Any corrected intervention needs a separate source-bound deployment/resource
contract and explicit approval, including this failed debit. Provisioning and
browser transfer/export must receive realistic reservations. Offline corrective
preparation may continue; no replacement GPU job or source acquisition follows
automatically. Keep all historical results, native scopes and no-local-GPU
preference unchanged.

The user then challenges the operator's artificial setup limit and explicitly
requests completion of the already-approved training ("hacim su hazirliklar
tammasa su egitimi tamamlya"), followed by a continuation after an interrupted
assistant turn. This supersedes the earlier setup/deployment restriction for
one corrected deployment; do not ask for the same training approval again.
The new external deployment envelope is in
`.benchmark/png-bmp-colab-native-fit-resume-20261107/deployment-contract.json`.
It retains the exact 7a45cab6... native command /a465948b... source packet,
1,370.702460-second native timer and output bounds, unchanged two-seed/twenty-
epoch/640-update recipe, original approval and all 522 tested source hashes.
Only the operational notebook's external setup/transport guards change.
An isolated first-three-cell check passes with no Torch/pixels/model/intent.

The old 2,700-second record is retained as the inner fitting subledger, no
longer claimed as the complete current external deployment budget. Its entire
1,329.297540-second /305,485,000-byte debit, including the failed 300-second
deployment, stays charged. The new explicit parent envelope independently
accounts for current setup/transfer/export, with an operator-selected generous
5,400-second cumulative hang watchdog and two-GiB combined-output ceiling.
This number is not quoted as an exact user-requested limit. Its prelaunch /
post-export watchdogs are 1,800 /900 seconds. No historical budget is reset,
failed attempt relabelled, profile rerun, new original acquisition or local GPU
work occurs. Private source/notebook/deployment/permission bindings are frozen
before allocation. Record this new deployment's actual outcome and independent
closing debit. Actual roles remain unassigned; reserved 21 RGB /13 L stay closed.

That corrected native RGB fitting is now complete: execution completed,
learning result `failed_learning_gate`. Both seeds finish twenty epochs /
640 native updates /5,120 training presentations, forty epoch records and eight
numeric states. Final train AUCs are .505615234375 /.5146484375; development
AUCs .5175 /.5275. Last epoch losses .6943890191614628 /.6924730055034161
remain near chance; both seeds fail all three fixed learning conditions.
No further epochs, tuning, qualified detector or deployment follows.
See `docs/PNG_BMP_COLAB_NATIVE_RGB_RESULTS_20261107.md` and the public receipt.

Actual T4 /580.82.07 /Python 3.12.13 /NumPy 2.0.2 /Pillow 11.3.0 /
Torch 2.11.0+cu128 /CUDA 12.8 /cuDNN 91900 matches the frozen environment.
Native command debit is 569.256510 /1,370.702460 seconds; request-to-confirmed
deletion is 920.654108 seconds. All previous 1,329.297540 seconds stay charged;
round new external work to 372 seconds and reserve sixty local evidence seconds,
giving 2,330.554050 /5,400 cumulative seconds. Complete native output is
461,183,415 bytes, with 459,326,099 bytes in eight numeric states. Conservative
combined output including transport/private copies/reserves is 1,790,263,775
of two GiB. Kernel worker RAM peak is 1,525,960,704 bytes, including inherited
creation memory. Whole-tree RAM, whole-driver VRAM and an allocator peak for
this unchanged fitter are unavailable; do not reuse the old profile peak as
a training peak. Native hard cgroup RAM/zero-swap remains unenforced.

The independent local NumPy-only replay verifies all 1,329 archive members,
eight hashes/states, 640 commit pairs and 48 AUC records; its receipt exactly
matches the original VM audit. Replay takes 3.607338 seconds, with no Torch or
native pixel reads. Actual inference exposure is 2,192 rows across 46 scoring
passes; final development records reuse epoch twenty. All six ordinary-UI
exported code-cell ASTs match with counts 1–6 and no errors. Preserve harmless
Colab title annotations and browser reconnects; neither restarts the worker.
The closed receipt SHA is
`a8f3be7b1044c538b86f35ace0c539e313ead90a1b250a9b2ddcffcf91a0feb2`;
archive SHA is
`55c0d9a0bd49898aee08dd6f860ee63dd2acd48f3d2fe4895aa716352bd03920`.
All downloads and audit complete before runtime deletion; account session
manager confirms no active sessions. Private outputs are in
`.benchmark/png-bmp-colab-native-fit-resume-20261107`; retain every state,
permission, ledger, raw notebook and source/data binding. Preserve the initial
invalid public-summary helper and its corrected publication; no GPU rerun.
All 522 software input hashes remain unchanged, so no long gate is repeated.
The updated README passes all fifteen existing cloud-removal/scope checks.
Actual roles stay unassigned; 21 reserved RGB /13 L remain closed, no cache
is regenerated and no new media is acquired. This actual native attempt is
consumed; no automatic retry, profile, fitting extension or deployment.

The user's subsequent root-cause request permits a read-only native RGB
diagnosis, now complete. See `docs/PNG_BMP_NATIVE_RGB_DIAGNOSIS_20261107.md`
and its aggregate receipt. Existing saved development pairs order correctly
15/20 and 20/20, but image-specific score offsets overwhelm the weak embedding
response: midpoint SD /pair-difference RMS is 123.42 /43.65. Paired ordering
is not standalone accuracy; retain the failed global AUC/gate. All 64 train
crop bytes match the transferred packet, all explicit payload checks pass,
and all 105 parameter tensors change between epochs 5 and 20 in both models.
Validated final numeric arrays reproduce GPU train scores on CPU within
3.58e-7; batch-statistics-only scoring stays weak at .512939 /.538330 AUC.
Two gradient-only CPU batches have finite/nonzero gradients and unchanged
parameters; temporary BatchNorm buffers are restored. No optimizer is created
or called. The old generated even-valued covers permit an artificial parity
cue absent from native covers; do not treat their pass as native qualification.
The separate causal contributions of 64 originals, 320 updates per seed and
learning recipe remain unidentified; more data alone is not a proven fix.
CPU investigation takes 72.484640 seconds /2,036,563,968-byte kernel process
RSS peak, no GPU/network/checkpoint writes. All eight original hashes and
522 software bindings remain unchanged. Only train pixels and saved development
scores are inspected; reserved/L/development pixels stay closed. No follow-up
fit, tuning, acquisition or budget reset follows from this diagnosis.

On 2026-11-08 a continuation prepares a metadata-only data/update comparison
and completes one new read-only CPU replay of the eight existing native RGB
states. All saved train AUCs stay between .503418 and .514648. Seed 2026102302
improves paired ordering from 43/64 to 61/64 while source-offset/response ratio
falls from 261.90 to 23.78; seed 2026102301 is not monotonic. No checkpoint
demonstrates high-train/low-development overfitting. See
`docs/PNG_BMP_NATIVE_CHECKPOINT_TRAJECTORY_20261108.md`. The replay uses only
64 fitted train originals /64 scalar payload recoveries /1,024 CPU inference
rows, zero backward/optimizer/GPU/network/checkpoint writes. Service time is
78.357453 seconds and kernel process RSS peak 701,538,304 bytes. Independent
stdlib replay verifies 32,768 pairwise comparisons and all eight original hashes;
all 522 software bindings remain unchanged. Reserved/L/development pixels stay
closed. This is not a historical Colab rerun or environment-bound resume.

`docs/PNG_BMP_NATIVE_DATA_UPDATE_STUDY_20261108.md` fixes a non-executable
proposal: nested eight/64 train originals across 6/38 whole ancestry components,
the two original seeds freshly initialized, endpoints 320/1,280 updates in
four new trajectories, prospectively 5,120 updates /40,960 presentations and
eight states. Metadata-only selection precedes checkpoint replay. Four complete
order schedules and all counts replay independently; no actual roles change.
The 113.85-minute linear historical proxy is not measured cost or an allowance.
The existing native plan's twenty-epoch limit must not be monkey-patched or its
schedule replaced to execute this proposal. A distinct runner/supervisor,
generated QA/full software gate, source/environment/cost freeze and separate
concrete execution authority are still missing. No new fit/profile, acquisition
or historical budget reset is authorized by this preparation or a generic
continuation. Keep the user's Colab/no-local-GPU preference and all old trials.

For the user's requested independent second opinion, a self-contained Turkish
review context is in `docs/PNG_BMP_NATIVE_RGB_SECOND_OPINION_20261108.md`.
It contains verified metrics, uncertainties, fourteen verbatim code excerpts
and the proposed study's explicit execution gaps; it is not a full chat export.
The builder/bindings stay private in `.benchmark/png-bmp-second-opinion-20261108`.
The note also observes all-zero initial positive-class probability arrays in
both historical train/development cells. Cold-start BN/logit saturation is an
open review question, not a proven explanation of the later failed fit.
No external assistant upload, account sharing, media/state disclosure or new
optimizer/GPU work occurred. All 522 software inputs remain unchanged.

The dated JPEG four-arm/two-seed matrix and both read-only diagnoses are
complete; see `docs/JPEG_CONTROL_RESULTS.md`. No candidate passed the combined
learning rule; production qualification is unavailable and none was deployed.
Preserve the 16 private snapshots and historical cancelled receipts; any next
training intervention needs a
separately frozen source/data/exposure protocol, not a blind restart.

The separate second-round JPEG refinement protocol is frozen in
`docs/JPEG_REFINE_PROTOCOL.md`. Use its shared warm-up/branch ancestry and
cumulative six-hour/ten-GiB limits; unavailable GPU/preflight is not learning
evidence. Historical source bindings and snapshots remain immutable.

The separately frozen numerical preflight amendment is documented in
`docs/JPEG_REFINE_NUMERICS_PROTOCOL.md`. Preserve the original failed gate;
use the complete amended source closure and explicit prior time/storage debits.
It changes validation only, never the training recipe or learning rule.

Native linear fitting is separately frozen in
`docs/NONJPEG_LINEAR_COMPARISON_PROTOCOL.md`: two seeds, grouped train-only
CV/thresholds, 20 epochs, independent 30-minute CPU /32-MiB output allowance.
It does not authorize native deep-model training or consume extra JPEG time.

The separately bound native process-path correction is specified in
`docs/NONJPEG_PROCESS_PREFLIGHT_PROTOCOL.md`. Keep the failed zero-update
attempt and original source bindings immutable; debit its time/storage before
one amended attempt, including read-only replay. Never relax carrier symlink
checks or automatically retry a failed amended fit.

The amended native pilot is complete: 14 numeric models and 480 cells replayed,
with 20,200 real optimizer updates including CV. WAV balanced accuracy remains
0.50 and spatial changes are mixed; no detector is qualified or deployed.
The amended JPEG refinement also completed all 28,672 updates without passing
its combined source/seed rule. Preserve both completed experiments and every
failed receipt; any further fitting needs a separately frozen intervention.

The separate WAV read-only diagnosis is specified in
`docs/WAV_READONLY_PROTOCOL.md`: train feature artifacts only, existing model
inference and cached development metrics at their original thresholds. Never
refit missing OOF folds, open reserved media or sweep validation thresholds.
Preserve all 14 model digests; diagnosis does not authorize another fit.
This diagnosis completed four audio models /32 train cells with zero updates;
all 14 digests are unchanged. See `docs/WAV_READONLY_RESULTS.md`.
`docs/WAV_NEXT_PROTOCOL_DRAFT.md` is not executable: AudioMNIST review is
metadata-only, parent-recording ancestry is unavailable, and its cuts must stay
grouped by speaker. The original cumulative raw-transfer ledger is incomplete;
do not reset it or treat a draft allowance as permission to download or fit.
The bound AudioMNIST pilot (`docs/WAV_PILOT_TRAINING_PROTOCOL_20261017.md`)
first failed its generated CUDA preflight on a missing linear `seed` receipt
key (keep that record `unavailable`). The receipt amendment
(`docs/WAV_PILOT_RECEIPT_AMENDMENT_20261019.md`) passed one generated preflight
within all limits. Real fitting stays unauthorized until explicitly approved.
A generated positive control then showed the v1 CNN does not learn; draft v3
(`core/wav_pilot_v3.py`) also failed its preregistered gate at reduced exposure.
Do not fit real AudioMNIST until a recipe passes a full-exposure control.
`docs/WAV_FULL_CONTROL_PROTOCOL_20261021.md` defines the next source-bound,
read-only acceptance contract. Its verifier requires both full seed schedules,
disjoint generated splits, all eight cells and resource evidence. At the
acceptance-contract freeze, the full generated runner/supervisor was missing
and needed its own frozen binding;
the acceptance contract alone authorizes neither a GPU attempt nor real fitting.
A separate full runner is now implemented in `core/wav_control_runner.py` and
bound by `docs/WAV_CONTROL_RUNNER_PROTOCOL_20261021.md`. Its generated corpus
manifest is frozen with zero updates. It permits one bounded generated control
only after the full software gate and deployment checks; it never authorizes
real fitting. Its v2 evidence amendment preserves completed seed evidence.
The full software gate passed 3,528 tests /two CUDA skips, 96.93% coverage;
new modules have 100% coverage. The one full generated attempt is unavailable:
1,937 confirmed updates as a lower bound, one complete scored seed and
797/1,140 confirmed second-seed updates before the frozen deadline. The full
first seed also fails the learning rule: train AUC 0.585101 and 40%-plane-0
development AUCs 0.557425/0.549725. Independent systemd monotonic time agrees
with the supervisor; the human wall-clock discrepancy is not a timer fix or
retry authorization. See `docs/WAV_FULL_CONTROL_RESULTS_20261021.md`.
Never relaunch it automatically; any further fit needs a separately frozen
intervention. Real fitting is unauthorized. Keep all prior acceptance, QA and
failed-control records intact.

The next untrained representation is `core/wav_pair_candidate.py`: ten
full-alphabet pair-energy statistics, with an optional 418-parameter FP32
head. Its 26 generated scalar/gradient tests pass with 100% module coverage;
the full gate passes 3,554 tests /two CUDA skips, 96.94% coverage. `docs/WAV_PAIR_CANDIDATE_DRAFT_20261022.md`
is a non-executable design, not a new training recipe or retry authorization.
There is no optimizer, cache/training adapter or candidate accuracy result.
Pair-count swaps can merge a one-LSB change; universal raw-PCM equivalence
is not claimed. See `docs/WAV_PAIR_CANDIDATE_VERIFICATION_20261022.md` for
this limit and the adaptive-development-data reuse caveat.
Do not replace the old recipe-bound gate with this domain; any fit needs a
separately frozen contract and real AudioMNIST remains unauthorized.

A separate pair-control service now exists in `core/wav_pair_control.py`,
bound by `docs/WAV_PAIR_CONTROL_PROTOCOL_20261022.md` and its 34-file protocol.
This later contract permits one CPU zero-update profile followed only on
success by one fresh generated two-seed fit. Preserve the earlier untrained
design snapshot, every old failed control and all historical source bindings.
Its independent 30-minute /2-GiB /zero-swap /64-MiB allowance adds no JPEG time
and resets no transfer ledger. A profile failure forbids fitting; any attempt
failure forbids automatic retry. Real media, downloads, checkpointing and real
AudioMNIST fitting remain unauthorized by this separate generated contract.
That one intervention is now complete and **failed_learning_gate**: both seeds
completed 1,140 updates /72,000 presentations, with target development AUCs
0.541750/0.520375 and 0.546025/0.495800 against 0.70. All 4,800 payload checks
passed; independent charged time is 30.556292 seconds. Full QA passes 3,598
tests /two CUDA skips, 96.97% coverage; both new modules exceed 95% coverage.
See `docs/WAV_PAIR_CONTROL_RESULTS_20261022.md`. Keep this complete failure;
do not rerun or tune it automatically. Any next fit needs a newly frozen
intervention; real AudioMNIST fitting remains unauthorized.

The pair-control read-only diagnosis is complete; see
`docs/WAV_PAIR_DIAGNOSIS_RESULTS_20261022.md` and its separately frozen protocol.
Only the 3,600-row train cache and metadata were opened, with zero updates.
All 80 feature/cell AUCs agree between independent methods; both initial
gradient identities and unchanged-parameter checks pass. Q0 plane-0 train
AUCs 0.909031/0.753063 show signal survives caching. Plane-1 dominates the
initial signal-gradient projection (93.67%/89.57%), supporting a scale/
competition hypothesis, not a proven full-training cause. No trained weights,
loss trajectory or actual gradient history exist; do not refit to reconstruct
them. The next raw-versus-standardized matched comparison in
`docs/WAV_PAIR_NEXT_INTERVENTION_DRAFT_20261022.md` is non-executable. It grants
no new fit, download, retry, native-media access or real-data permission.

A later, separate matched scaling comparison is implemented in
`core/wav_pair_comparison.py`, bound by the 42-file contract in
`docs/WAV_PAIR_COMPARISON_PROTOCOL_20261022.md`. The user's continuation
authorizes one conditional generated CPU intervention under that new contract:
fresh generator seed 2026102204, raw versus train-only z-score input, both
original model seeds and four full fits /4,560 updates. The old draft remains
non-executable by itself. Require the full software gate, successful zero-update
profile and frozen normalization/cache/debit binding before fitting. Preserve
the matched initialization/order, all epoch/step telemetry and score arrays
for independent AUC replay. Its separate 30-minute /2-GiB /zero-swap /64-MiB
allowance adds no JPEG time and resets no transfer ledger. No retry, checkpoint,
native-media opening, download or real AudioMNIST fit is authorized.
This one comparison is complete: all four fits /4,560 updates /288,000
presentations. Raw failed; train_zscore **passed_generated_only** with target
development AUCs 0.984600/0.950725 and 0.979150/0.945625. The old raw CNN
attempt remains unavailable and the earlier pair run remains failed.
All 4,800 payload recoveries, 72 independent score-AUC replays and resource
limits pass; independent charged time is 31.407780 seconds. Full QA passes
3,662 tests /two CUDA skips, 96.99% coverage; new modules exceed 95%.
See `docs/WAV_PAIR_COMPARISON_RESULTS_20261022.md`, including weak/regressing
scattered 5%-p1 cells. Do not rerun or tune the completed comparison. This
pass belongs to the standardized pair-feature domain, not the old CNN-bound
native pilot. Any native fit needs a separately frozen source/data/domain/
exposure/grouping protocol and explicit approval; no model was deployed.

The separate native standardized pair-head contract is now implemented in
`core/wav_pair_native.py` and `docs/WAV_PAIR_NATIVE_PROTOCOL_20261022.md`.
The current continuation permits software QA and one cached train/development
zero-update preparation profile, not real fitting. Preserve the 20/10 speaker
roles and all 30 reserved groups; no download or historical budget reset.
Its two-seed /2,280-update CPU fit adapter requires later explicit approval
bound to the protocol/profile and independent profile time debit. Do not
create an approval receipt from this preparation request or a generated pass.

That one native preparation profile completed: all 600 originals /4,800
payload checks, 20 backward batches and unchanged parameters, zero optimizer
updates. Full QA passes 3,729 tests /two CUDA skips, 97.00% coverage;
new modules exceed 95%. Independent profile debit is 24 seconds of the
separate 1,800-second allowance. See `docs/WAV_PAIR_NATIVE_PREPARATION_20261022.md`.
Do not rerun the profile automatically. Actual native fitting is still
pending explicit user approval; no approval receipt exists for this protocol.

On 2026-10-23 the user explicitly approved the one native two-seed CPU fit.
The bound approval and original supervisor completed both full seeds:
2,280 actual optimizer updates /144,000 presentations, with 40 train epochs,
eight actual gradient/step records and all saved score arrays. The exploratory
target failed: required development 40%-p0 AUCs are 0.478800/0.486625 and
0.466650/0.468700 against .70; pooled AUCs are .616641/.622978, with .55/.42
cover FPR. All 36 AUCs and 80 metric/cluster/paired-interval checks replay;
all 49 source bindings and 14 old model digests remain unchanged. Charged
profile+fit time is 30.059356/1,800 seconds. No checkpoint or deployment;
reserved speakers remain closed. See `docs/WAV_PAIR_NATIVE_RESULTS_20261023.md`.
The earlier pending-approval records are historical snapshots. This one fit
is consumed; never relaunch it automatically or use remaining time for tuning.
Any further fit requires a distinct frozen intervention and explicit approval.

All built-in extension data states are recorded in
`docs/ALL_FORMAT_DATA_STATUS.md`. The local PDF/TXT/MD/ZIP preparation protocol
is frozen in `docs/ALL_FORMAT_DATA_PROTOCOL.md`; source/license-bound local
documents and controlled variants stay `development_unassigned`, grouped by
whole upstream package. They are not representative detection benchmarks or
certified clean files. No download, role assignment or fitting is authorized
by the local preparation contract.
The local pilot completed 16 documents /48 controlled artifacts across four
package groups; all payload/hash checks pass and all roles remain unassigned.
The complete inventory has 23 suffixes, including CSV/LOG. See
`docs/ALL_FORMAT_LOCAL_RESULTS.md`; retain the original README-gate failure
and explicit repair receipts. No representative all-format dataset or new
training qualification follows from this preparation.

Full-image tool changes must update `tools/full-image.lock.json`, preserve
upstream license notices and run the hardened checks in
`docs/FULL_IMAGE_VERIFICATION.md`. Inventory presence is not recovery evidence.
Never publish Stegseek wordlist cracking stdout (it can expose recovered keys).

## Handoff checklist

1. Preserve unrelated working-tree changes and avoid history rewrites.
2. Update architecture, roadmap/status, benchmark protocol, and changelog when
   a slice changes their contract.
3. Add independent fixtures and adversarial limit/security tests.
4. Record optional dependency/tool absence as coverage, not success.
5. Run focused tests, the full suite, lint, type checking, and `git diff --check`.
6. Keep commits small and state any unverified environment-dependent checks.

## Interaction and response rules

* Yanıtlarını kısa, net ve doğrudan ver.
* Yaptığın rutin işlemleri adım adım anlatma.
* Açıklama yapmadan önce mümkünse doğrudan işlemi gerçekleştir.
* Ben istemedikçe bariz veya basit şeyleri açıklama.
* Terminal komutlarının çıktısını önemli bir hata yoksa tekrar etme.
* Araç ve terminal kullanımını gereksiz yere anlatma.
* Kod değişikliklerinden sonra yalnızca önemli değişiklikleri belirt.
* Bir görev tamamlandığında sonucu en fazla 3 kısa maddeyle özetle.
* Uzun giriş, sonuç ve gereksiz bağlam yazma.
* İşlem başarılıysa uzun açıklama yapma.
* Sorun yoksa final cevabını mümkünse 5 satırdan kısa tut.
* Bir hata varsa yalnızca hatanın nedenini ve gerekli çözümü açıkla.
* Kod yazarken veya düzenlerken açıklama yerine uygulamaya öncelik ver.

On 2026-11-08, at the user's request, `main` history was rewritten to drop
co-author trailers from ten commit messages; trees, authors and dates are
unchanged. Sixteen hashes changed (1c22583 onward, e.g. 2be950e→9627e37,
2f41bfe→c3b1731). Historical receipts, docs and `BASE_COMMIT` constants keep
their original hashes as executed records; resolve them with
`benchmarks/commit-hash-map-20261108.json`. Never add co-author trailers.

The 2026-11-08 parity-input diagnosis is complete; see
`docs/PNG_BMP_RGB_PARITY_PROBE_RESULTS_20261108.md`. A training-free WS
estimator separates all 64 train pairs (AUC 1.0) and dev pairs (.9525). The
one approved seed-2026102301 run (Colab unavailable, executed on local CPU, no
GPU) adds only three LSB-parity channels: train AUC .51 -> .98, loss .63 -> .14,
but development AUC .62 with chance pair ordering, so the gate fails on
overfitting. Follow-up runs (fresh payload per epoch, lr 1e-3, both original seeds)
reach development AUC .74/.77; seed 2 misses train AUC .85, so the two-seed
gate is not passed. All four CPU attempts are consumed; see the results doc.
Forty epochs with a 1e-3 -> 1e-4 decay after epoch 30 then pass the gate on
both seeds (train .91/.98, development .82/.81); private final states are
saved. On 2026-11-09 the user approved the one pre-registered reserved
evaluation: both frozen states score AUC .868/.866 (20/21 pairs each) on the
21 reserved RGB originals, passing the .70 rule; WS scores 1.0. The reserved
set is now consumed for this recipe; never re-evaluate tuned variants on it
as unbiased evidence. All six CPU attempts are consumed. `modules/lsb_replacement.py`
now feeds a heuristic `weighted_stego_payload` signal in `image_bitplane`.
An LSB-matching fine-tune (`research/spatial_matching_probe.py`) lifts
development matching 20%/10% AUC to .78/.74 and .66/.70 but misses its .80
gate; 64 originals cap progress. On 2026-11-09 the user approved dataset
expansion and downloads: DIV2K/BOSSbase are kept privately under
`.benchmark/external-datasets-20261109`, and GPU training uses
`research/external_rgb_training.py` with the DIV2K Colab notebook; see
`docs/PNG_BMP_RGB_EXTERNAL_TRAINING_20261109.md`. Never redistribute them.
The consumed Colab budget tests reject any added or changed `core/` or
`steganography/` source (frozen closure). Put post-freeze experiments in the
new `research/` package and analysis helpers in `modules/`; type-check with
`mypy research` as well. Retiring that live-tree freeze needs user approval.
