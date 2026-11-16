# Handoff history (archived 2026-11-17)

This is the complete former `AGENTS.md`, archived verbatim when the handoff
was condensed. It records every historical attempt, failure, budget and
source binding. Current working rules live in `AGENTS.md`.

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
On 2026-11-11 the CPU rich-model detector `research/rich_color_detector.py`
reaches DIV2K validation matching AUC .99/.97/.90 at 40/20/10% (CNN .76/.69/
.63). Native development stays at about .70-.75 (cover-source mismatch).
See `docs/PNG_BMP_RGB_RICH_DETECTOR_20261111.md`; models stay private.
The consumed Colab budget tests reject any added or changed `core/` or
`steganography/` source (frozen closure). Put post-freeze experiments in the
new `research/` package and analysis helpers in `modules/`; type-check with
`mypy research` as well. Retiring that live-tree freeze needs user approval.
On 2026-11-12 the user's 80%-per-method goal produced
`docs/DETECTION_EXPANSION_20261111.md`: SRM/DCTR/WAV research features, new
analyzers (stegano recovery, container fields, invisible Unicode, network
covert channels, video containers and lossless frames) and a zsteg
false-positive fix. Report every miss with its payload. Private corpora live in
`.benchmark/{boss-gray-rich,real-tools}-20261111` and
`.benchmark/{wav-lsb,network,video,boss-academic}-20261112`. F5 (matthewgao
build) and OpenStego 0.8.6 under `.benchmark/tools-20261111` may be run
(user approval 2026-11-12).

## Cross-assistant handoff (2026-11-13)

This section lets any assistant (Claude, Codex, ChatGPT or another) continue
the work without private chat memory. It overrides generic tool defaults.

### Working rules

- **Language**: answer the user in short Turkish (see the interaction rules above).
- **Commits**:
  - One commit per file or folder.
  - The message starts with `update <path>:` and briefly says what changed. It
    is not a few words, and it has **no capital letters anywhere**, acronyms
    included. Example: `update research/srm_features.py: add selection-channel-aware weighted co-occurrences`.
  - Never add co-author, "Generated with ..." or any assistant trailer.
  - Push only after the full QA gate passes.
  - Never rewrite or force-push already pushed history without the user's
    explicit confirmation. Whether to reword older pushed messages to the
    `update ...` style is still an open question to the user.
- **Data and privacy**:
  - Never redistribute DIV2K, BOSSbase or ALASKA2 data, or derived weights;
    they stay under `.benchmark/` (gitignored).
  - The ALASKA2 holdout (1,000 lineages) stays locked unless the user approves.
  - Keep EVREN screenshots and account identifiers private.
  - Reports never contain absolute host paths or secrets.
  - Never publish Stegseek cracking stdout.
  - Do not delete datasets, checkpoints, logs or failed-run evidence.
- **Frozen closure**: do not add or change `core/` or `steganography/` sources
  (the consumed Colab budget tests reject it). Put new work in `research/`
  (experiments) and `modules/` (analyzers); type-check `research` too.
- **Hardware**: no local GPU work. GPU runs go through Google Colab, operated by
  the user in their browser. The user runs `sudo` in their own terminal; never
  ask for a password.
- **Environment**: the project interpreter is `venv/bin/python` (Python 3.11;
  includes conseal 2025.11 and jpeglib). The system `python3` lacks the
  research dependencies.
- **Full QA gate**:
  - `venv/bin/ruff check .`
  - `venv/bin/mypy core modules report api steganography ui registry.py config.py cli.py research`
  - `venv/bin/python -m pytest -q -p no:cacheprovider --basetemp <fresh disk-backed dir>`,
    with `TMPDIR` on disk too.
  - The full run takes about 30 minutes. Keep the first failed log.
  - The last pass (2026-11-12): 4,578 passed, two CUDA skips, 97.39% coverage.
  - `tests/test_cloud_removal.py` keeps `README.md` between 400 and 900 words,
    with four badges and resolvable local links.

### Goal: at least 80% per stego method

The user's goal is at least 80% balanced accuracy, on data never used in
training, for every steganography method at a stated payload. Report every
miss honestly; never inflate. Results, misses and evidence are in
`docs/DETECTION_EXPANSION_20261111.md`.

- **Passing**:
  - gray LSB on BOSSbase (all payload-specific rows)
  - stegano, OutGuess, steghide (JPEG and BMP)
  - F5 at every payload (0.05 bpnzac: 80.8%)
  - OpenStego at 0.4
  - AudioMNIST WAV replacement
  - network covert channels and video containers
- **Missing**:
  - unknown-payload gray LSB at 5% (72.1%)
  - OpenStego at 0.1 (79.8%)
  - ESC-50 WAV: replacement at 5/10%, and ±1 matching
  - AudioMNIST matching at 5/10%
  - FSDD at every payload
  - academic spatial methods at 0.4 bpp, except WOW (see below)
- **zsteg on the DIV2K test split**: 1% false alarms on clean PNG/BMP and 100%
  on stegano. It does not detect steghide BMP; that is expected.

### Benchmarks in flight (private, under `.benchmark/`)

The queue runs detached (`setsid nohup .benchmark/queue-20261113.sh`, log
`.benchmark/queue-20261113.log`), so it survives the assistant session. The
first chain, which was tied to the session, died with it on 2026-11-13.
Every script caches features per method and skips methods already in its
result JSON, so rerunning the queue resumes. A method interrupted midway
restarts from scratch.

- **Done, v2 at 0.4 bpp** (`boss-academic-20261112/run_maxsrm.py` →
  `result-maxsrm.json`):
  - Features: SRM + weighted (maxSRM-style); "both" won the train holdout
    for every method. 5,497 train images.
  - Results: HILL 76.4%, S-UNIWARD 78.9%, **WOW 80.3% (passes)**, MiPOD 75.2%.
  - v1 (plain SRM, 3,000 train) gave 70.5/75.6/73.9/71.9%.
- **Queue order**:
  1. `run_v3.py hill suniward mipod` → `run_v3.log`, `result-v3.json`. The
     0.4 bpp rows with 9,000 train images (the split plus its 500 validation
     and 3,003 earlier-used BOSS images, pixel-checked against test) and a
     5-fold out-of-fold train threshold.
  2. `boss-jpeg-academic-20261112/run.py` → `run.log`, `result.json`:
     - J-UNIWARD 0.4, UERD 0.4 and nsF5 0.05/0.1/0.2 bpnzac.
     - BOSSbase QF75; DCTR or calibrated DCTR, chosen on the train holdout.
     - 3,000 train / 1,000 test.
     - EBS is skipped: the conseal EBS simulator changes ~71% of the nonzero
       AC coefficients at 0.4, so it is not a valid embedding.
  3. `run_rate.py 0.6 <methods still below 80% after v3>` → `run_rate06.log`,
     `result-rate0.6.json`. A stated higher payload; the 0.4 rows stay
     reported.

### Unfinished work, in order

1. When the runs finish:
   - Fill the `<!-- academic-v3 -->` placeholder in
     `docs/DETECTION_EXPANSION_20261111.md` (v1 and v2 are written): the v3
     rows with a third-look adaptivity note, the JPEG section and the 0.6
     rows.
   - Update the summary table and the "have not been run yet" sentence there.
2. `README.md`:
   - Add the academic rows. Old rows were merged (868 words now), which
     leaves room under the 900 limit.
   - Rerun `tests/test_cloud_removal.py`.
3. Note the academic results in `CHANGELOG.md`.
4. Run the full QA gate, commit per file in the style above, then push.
   - Local commits not yet pushed (all of `origin/main..main`): weighted SRM
     features and tests, the ridge memory fix, this handoff, the merged README
     rows and the academic v1/v2 + zsteg doc update. Nothing is uncommitted
     except the unrelated `nano.383070.save`, which must not be committed.
   - Check the queue first: `pgrep -af queue-20261113` and
     `.benchmark/queue-20261113.log`. Never start a second copy while it runs.
5. Later:
   - a stegseek benchmark for steghide
   - SRNet on Colab for the academic methods at 0.4 bpp (CPU features plateau
     there)
   - optional extra formats (DICOM, HEIC/AVIF, PSD, fonts, PE/ELF, SQLite);
     the user has not chosen yet

### Queue closeout and verification (2026-11-14)

The detached queue above ended at 20:38:24 +03:00; do not restart it.
V3 at 0.4 bpp is HILL 76.90%, S-UNIWARD 79.95% and MiPOD 76.00%, all misses.
S-UNIWARD must not be rounded into an 80% pass. At the separate 0.6 bpp
condition the three score 83.20 /86.45 /82.20%; all pass at that payload.

The first academic JPEG stage exited 1 before test features or a result JSON:
15 empty/zero-filled derived JPEG covers predate this queue. All 2,649 old
covers (2,634 valid /15 invalid), original source and failed log are preserved.
One isolated completion in `.benchmark/boss-jpeg-academic-repair-20261113`
now exits 0 with no remaining process. Do not relaunch it or the old queue.
The worker's sole AST change is its output directory; methods, payloads,
seeds, features, split and selection are unchanged. The closing audit verifies
4,000 raw sources, all 2,649 old cached contents (including the 15 invalid
files), four source/split bindings and five metric calculations. Hard links
change inode link counts; no old media bytes were rewritten. The requested
old-directory `result.json` is a byte-identical copy of this completion, SHA
`6f0f7ec431249227ada47cf3d757b2d8e4a9dcdd0d75ed7a102c7738cf7adace`.
Keep both attempts' logs, snapshots, staging/exit receipts and closing audit.

JPEG scores are J-UNIWARD /UERD 83.80 /82.65% at 0.4 bpnzac, and nsF5
60.70 /75.15 /94.25% at 0.05 /0.1 /0.2 bpnzac. The two lower nsF5 payloads
miss; EBS stays unavailable. All results and misses are now in the detection
document, README and changelog. No GPU, download or deployment occurred.

The first new full QA finishes with 4,588 passes /two CUDA skips /32 warnings
and one teardown error, 97.39% coverage. Preserve that failed command and the
focused one-pass/one-error reproduction in
`.benchmark/academic-closeout-qa-20261113-ppi9ynip/`, including its original
570-file binding, pre-fix test snapshot and `coverage.failed.data`.
The progress-rename test's global fake `os.scandir` leaked into pytest's
temporary-directory cleanup; a `monkeypatch.context()` now restores it before
fixture teardown. This changes one test only, with no application-code change.

The fresh full gate in `.benchmark/academic-closeout-qa-20261114-q65ywhk3/`
now passes: 4,588 tests /two explicit CUDA skips /32 warnings, 97.39%
statement coverage. Ruff /268-file mypy pass; SRM and rich-color research
modules both have 100% coverage. All 570 bound inputs match at close. Pytest
takes 1,891.03 seconds; the full wrapper takes 1,899.503044 seconds. The
scanner regression and fifteen README/scope tests also pass.

It uses one BLAS/OpenMP thread, disk-backed TMPDIR/basetemp and
`tmp_path_retention_policy=failed`. Both historical 8.3-GiB QA trees, real
datasets, checkpoints and all failed logs are preserved; only the rebuildable
pip package-download cache was purged for space. Installed packages and
application sources are unchanged. Keep both full-gate records distinct.

Benchmarking, result documentation and the repaired software gate are complete;
README has 888 words. No new GPU, download, fit, tuning or deployment follows.
Use git tracking status to determine publication state and preserve the
per-file lowercase commit rules above. Do not rerun completed jobs to republish
them. Preserve the unrelated `nano.383070.save`.

### CPU rich-model improvement closeout (2026-11-14)

The user's explicit continuation selects fresh CPU rich-model development.
This step is complete. All older GPU, native Commons, transfer, holdout and
frozen-source restrictions above remain binding. No download, local GPU,
raw-media decode, native role change or ALASKA2 access occurred. Do not restart
any completed/failed parent or automatically spend its unused allowance.

Read `docs/RICH_MODEL_CPU_IMPROVEMENT_RESULTS_20261114.md` and its linked
prospective linear, nonlinear, numeric-amendment, boundary and final-evaluation
protocols. Existing SRM + weighted caches supply 9,000 historical training
sources /8,997 exact-cover-feature groups. Source pairs and duplicate feature
groups stay together: 5,999 fit /1,500 calibration /1,501 audit pairs. Earlier
models fitted this cohort; its audit is held out for these fresh models,
not untouched external qualification.

The first twelve-candidate union selects square-root Gaussian SVC, gamma
1/9,464 and C=100, for all three methods at 0.4 bpp. Audit BA improves from
raw Fisher 75.5163 /79.0806 /74.3171% to 79.2805 /82.3784 /78.3811% for
HILL /S-UNIWARD /MiPOD. Paired-group 95% intervals have positive lower
bounds; HILL and MiPOD still miss 80%. All scores, counts, pairwise AUCs,
2,000-resample bootstraps and 66 sampled predictions replay independently.
Intervals are per method, without multiplicity adjustment.

The separate fixed C=300/1,000 boundary step is calibration-only after that
audit is consumed. Eight of twelve fresh candidates fail to converge at the
fixed 100,000-iteration limit; keep them ineligible, with no retry or limit
increase. Exact-count calibration over all sixteen candidates selects C=100 /
300 /100 at gamma 1/9,464 for HILL /S-UNIWARD /MiPOD. The consumed audit is
not rescored for these choices. Independent calibration/AUC/selection replay
passes; all states remain private, source-bound numeric files without pickle.

After the final complete software gate passes, a global recipe freeze binds
all three states/thresholds before any historical test-feature access. One
inference-only evaluation uses the unchanged 1,000 BOSS cover/stego pairs:

- HILL, 0.4 bpp: 79.50% BA /AUC .888380, 1,590 of 2,000 correct; miss.
- S-UNIWARD, 0.4 bpp: 80.75% /.897665, 1,615 correct; passes this cell.
- MiPOD, 0.4 bpp: 77.65% /.874248, 1,553 correct; miss.

The common-fit raw reference is 76.00 /78.95 /74.05%: gains are 3.50 /1.80 /
3.60 points. Exploratory paired 95% gain intervals are +1.99875 to +5.15,
+0.29875 to +3.25 and +2.00 to +5.15 points. All counts/AUC/intervals and
another 66 sampled predictions replay, largest prediction difference
2.438e-10. This is a further adaptive look at the historical test, already
viewed by v1/v2/v3 and higher-payload experiments. Preserve every old row;
v3 trained 9,000 sources with a different threshold policy, so its values
are not the controlled ablation reference. No qualification or deployment.
The every-method target remains unmet at the stated misses.

New public code is bounded signed-power Fisher fitting /exact-count threshold
calibration in `research/power_fisher.py`, bounded row scoring in
`research/fld_ensemble.py`, and the external-tool launcher below. The first
float threshold helper breaks two optimal-count ties incorrectly; preserve
both original helper/test snapshots under `rich-model-software-qa-20261114`.
HILL's gamma=.1/D C=10 and MiPOD's linear C=.01 are affected. They are
unselected in the twelve-candidate union, whose winners/reference thresholds
are unchanged. The original six-candidate MiPOD choice used that threshold;
retain its observed audit without claiming exact tie-rule compliance.

Preserve the first nonlinear exit 1 after 48.043474 seconds, its completed
C=1 state and failed strict 1e-10 guard before C=10 state persistence. Its
exact real error magnitude is unavailable. Generated 12,000-row controls
reproduce dual-sum roundoff; the separately frozen correction uses a 1e-7
replay guard and records real errors. It reuses C=1 byte-for-byte and debits
the old time, without resetting the 1,200-second per-method allowance. All
boundary prediction checks pass, maximum error 2.557e-8. The first read-only
auditor also used the wrong sklearn 1.9.1 statistic-casting order; preserve
that failure and the later tie-audit failure, then the successful replay.

The first complete software gate separately passes 4,624 tests /two CUDA
skips /97.40% coverage. The post-tie-fix full attempt is unavailable after
118 tests /797.375897 seconds: parallel benchmarking inherits the coverage
collector lock in `external_tools._limits` before exec. A deliberate child
SIGABRT retains its traceback; then the parent is stopped. Core dumps are
disabled for that diagnostic. Original adapter/test snapshots match its
input hashes; no failed/interrupted command is relabelled.

`modules/external_tool_worker.py` now applies limits after startup in an
isolated interpreter and execs the resolved optional tool, avoiding Python
preexec callbacks in analysis threads. The same memory/CPU/output/FD policy
and parent timeout remain, with core dumps disabled. Launch failures produce
error coverage rather than a clean result. Frozen `core/` and `steganography/`
sources are unchanged. Twenty-seven focused tests pass with 100% new-worker
coverage, including twelve real parallel launches. Preserve the earlier
42-pass /84.25%-combined-coverage failure and initial Ruff line-length failure.
A model-free wheel verifies both launcher/adapter bytes without private data.

The fresh final complete gate passes **4,646 tests /two explicit CUDA skips /
32 warnings**, **97.4158046877721%** total statement coverage (27,971/28,713).
Pytest takes 1,674.81 seconds; wrapper time is 1,680.578804 seconds. Ruff and
270-file mypy, including research, pass. All 570 bound source inputs match
at close. Power Fisher, ensemble and launcher have 100% coverage; the adapter
has 98.35%. The unchanged complete gate is 90%; new analysis targets exceed
95%. Disk-backed TMPDIR/basetemp use passed-test retention cleanup. Preserve
all historical checkpoints/datasets/failed logs. Only the rebuildable npm
package-download cache is purged (1,043,311,878 logical bytes); installed
packages and npm logs stay intact. README has 896 words; all fifteen
README/link/scope checks pass after document-only bookkeeping, with
application/test inputs still matching the full gate.

Private evidence directories under `.benchmark/` are:
`rich-model-software-qa-20261114`, `rich-model-improvement-20261114`,
`rich-model-nonlinear-20261114` (failed),
`rich-model-nonlinear-completion-20261114`, `rich-model-boundary-20261114`,
`rich-model-final-evaluation-20261114`, `rich-model-full-qa-20261114`,
`rich-model-full-qa-tie-fix-20261114` (interrupted),
`rich-model-launcher-qa-20261114`, and `rich-model-full-qa-final-20261114`.
All parents/model workers are closed. Final inference, preparation and closing
verification conservatively bring nonlinear debits to 194.567984 /142.542859 /
170.848608 seconds of each 1,200-second allowance. Combined nonlinear,
completion, boundary and final output stays below the shared 64-MiB cap.
No unused time grants another test look, fit, guard relaxation or retry.
Use git tracking status to determine publication; preserve the per-file
lowercase commit rules and unrelated `nano.383070.save`.

### User-requested storage cleanup and fresh continuation (2026-11-14)

The user explicitly requests removing unnecessary steganography files, then
continuing development. `docs/STORAGE_CLEANUP_20261114.md` records the completed
cleanup: 27.859466552734375 GiB reclaimed, about 30 GiB available /85% used.
Only six verified passed pytest generated-scratch trees and the unused
`ci-minimal-env-20261010` environment are removed. Earlier statements about
preserving those passed temporary copies are historical pre-cleanup records.
Real datasets/features/scientific models, all QA logs/source receipts and
failed/interrupted scratch remain. Compressed file/hash inventories, sparse
extent metadata and CI package/configuration evidence are private under
`.benchmark/qa-storage-cleanup-20261114-3k9p9lsx/`. All 570 source bindings and
six predecessor/reference model hashes verify; active `venv` is intact.
No consumed experiment is rerun to recreate disposable unit-test copies.

The explicit continuation permits the fresh, bounded CPU sample-count study
in `docs/RICH_MODEL_CPU_SAMPLE_COUNT_PROTOCOL_20261114.md`, using existing
BOSS feature caches only. Keep 1,500 calibration pairs separate; the former
5,999 fit plus consumed 1,501 audit pairs become 7,500 fit pairs. Fit one
fixed C=100 /gamma=1/9,464 Gaussian model each for HILL/MiPOD; select against
the old state using calibration only. The former audit is now training,
not new held-out evidence. S-UNIWARD stays unchanged. Freeze choices before
any optional one-time historical test access; preserve its further adaptivity.
No new media, GPU, ALASKA2, native role change or old parent restart.
The new study uses fresh exclusive markers, a new 1,200-second per-method
operator cap, sampled 10-GiB RSS guard and at most eight MiB of disk output.
Matrices stay in RAM. No retry/guard relaxation follows. Verify the already
passed 4,646-test source closure; do not create another large QA tree when
application/test code is unchanged. Record the new outcome before publishing.

The fresh sample-count study is now closed in
`.benchmark/rich-model-samplecount-20261114/`; see
`docs/RICH_MODEL_CPU_SAMPLE_COUNT_RESULTS_20261114.md`. Both fresh C=100
models converge on 7,500 fit sources, and calibration selects them before
global choice/test freezes. One historical-test inference each gives HILL
80.05% (1,601/2,000 correct) and MiPOD 77.90% (1,558/2,000) at 0.4 bpp.
S-UNIWARD stays at 80.75%, without rerunning it. The former internal audit
is training in this step; no new audit accuracy is claimed.

Changes of +0.55 /+0.25 points versus the predecessor have exploratory 95%
intervals -0.45 to +1.55 /-0.90 to +1.30. Both contain zero; this is not a
statistically reliable gain or untouched external qualification. HILL passes
by one decision only; MiPOD remains below 80%. No test-based adjustment.
Independent calibration/count/AUC/bootstrap and 44 sampled state predictions
verify, maximum discrepancy 9.195e-11. Original feature-cache hashes, versions,
group separation and all 570 application/test source bindings match.

Kernel wait4 peak RSS is about 4.44 GiB, and all matrices stay in RAM.
Fit processes take about 50 seconds each, inference nine seconds each;
conservative method debits with closing audits are 73.373759 /72.885059 of
their new 1,200-second caps. Persistent output stays under two MiB, below
the eight-MiB limit; guest space remains about 30 GiB free. No old allowance
is reset, source/data download made or GPU used. All parents are closed;
no automatic retry, new grid or test look follows. Application/test code is
unchanged, so the prior passed 4,646-test gate is source-verified and reused,
not relabelled as a new run. Documentation checks are recorded separately.

### Fresh MiPOD kernel continuation (2026-11-15)

The user's latest explicit continuation permits the fresh CPU-only comparison
in `docs/MIPOD_KERNEL_PROTOCOL_20261115.md`. It uses the same 7,500 fit /1,500
calibration pairs, with no new data, GPU, native roles or ALASKA2 access.
Four fixed kernel-ridge and thirty additive-family SVC candidates compete
against the frozen 77.90% predecessor using calibration only. Choices must
freeze before any optional single selected historical-test inference; that
historical test remains adaptive, not independent qualification.

The exclusive parent/evidence directory is
`.benchmark/mipod-kernel-20261115/`. Check its `attempt.json`, `parent.log`,
`closing.json` and live processes before any action. An existing attempt must
never be duplicated or automatically relaunched. The fresh cumulative cap
is 1,200 seconds, sampled RSS 10 GiB and disk output eight MiB. Matrices
stay in RAM. Generated numeric checks and the unchanged 570-source software
closure verify before fitting; the actual passed 4,646-test gate is reused,
not relabelled as a fresh run. Record the actual closeout and every failure
before publication. All old restrictions, consumed attempts and evidence stay
binding; preserve the unrelated `nano.383070.save`.

That fresh kernel comparison is now closed and verified. All 34 candidates
are eligible; calibration selects full square-root Gaussian kernel ridge .03.
One further historical test gives MiPOD **78.75%** /AUC .885946 at **0.4 bpp**,
with TPR 84.6% /FPR 27.1%, 1,575/2,000 correct. Its +0.85-point change
against 77.90% has exploratory 95% interval -0.25 to +1.90 points, containing
zero. MiPOD remains 25 decisions below 80%; no reliable gain, qualification,
deployment or test-based adjustment is claimed. HILL /S-UNIWARD /WOW retain
their prior scores without reruns. All earlier misses and payloads stay.

Independent replay verifies all 35 calibration records, global selection,
full test counts/AUC/bootstrap and 90 direct-distance numeric predictions;
maximum discrepancy is 2.154e-9. Separate read-only scaler hash bindings
precede global choice and verify again at close. Original data hashes and
all 570 source bindings match the passed 4,646-test software gate. No new
application/test code or full-suite run; documentation checks are separate.
Parent execution plus independent closing audit takes 398.727003 seconds;
supplemental hash verification conservatively brings debit to 400.727003
of the fresh 1,200-second cap. Wait4 peak is 7,014,477,824 bytes /6.54 GiB;
this is observer evidence, not strict cgroup acceptance. Parent-close output
is 4,225,621 bytes, with no persisted feature/Gram matrices. Learned states
stay private. See `docs/MIPOD_KERNEL_RESULTS_20261115.md` and its public
numeric receipt. Parents are closed; no automatic grid, retry, new test
look, old allowance reuse, download or GPU action follows from this outcome.
All fifteen documentation checks pass; README has 897 words. Determine
publication from git tracking status and preserve the per-file lowercase
commit rules; never rerun the experiment merely to publish its records.

### MiPOD diagnosis and normalization continuation (2026-11-15)

The user's new instruction explicitly requests investigating the MiPOD miss
and attempting 80%, while asking whether the other methods all pass. They do
not: retain every payload-specific miss. Current spatial point estimates at
0.4 bpp are HILL 80.05%, S-UNIWARD 80.75%, WOW v2 80.30%, MiPOD 78.75%.
None turns repeated historical testing into untouched qualification.

The fresh CPU-only protocol is `docs/MIPOD_NORMALIZATION_PROTOCOL_20261115.md`.
Its exclusive parent/evidence directory is
`.benchmark/mipod-normalization-20261115/`; inspect existing attempt/status/logs
and processes before any action. Never duplicate or automatically relaunch it.
It diagnoses fit/calibration features only, then compares six fixed feature
normalizations and four classifiers each against the old ridge .03. Same
7,500 fit /1,500 calibration pairs; no test-driven selection, new media,
raw decode, GPU, ALASKA2 or native Commons role change. Generated checks and
all 570 bindings to the passed software gate precede fitting. Fresh cumulative
limit is 1,500 seconds, sampled 10-GiB RSS and eight MiB output, with all Gram
and CDF-reference matrices in RAM. Freeze states/scalers/choice before any
optional single selected historical-test inference; preserve all failures.
Record the actual outcome and independent closeout before publication.

The fit-derived residual-activity quartiles show strong calibration error
concentration for the old ridge: 96.16 /85.90 /74.75 /63.29% from lowest to
highest activity, using the same frozen global threshold. This is an
association, not causal proof. The 9,464-dimensional compact cache has 272
constant cover bins; it is not a reproduction of the full published maxSRMd2.
The original MiPOD paper itself reports .2038 error at 0.4 bpp for its
medium-block version under a different protocol; this is context, not a
performance ceiling. No threshold, data or payload is changed to force 80%.

The normalization intervention is now independently closed: all 24 fresh
states are eligible; calibration selects `rc8-sqrt-ridge0.03` at 80.5667%.
Its one selected historical inference gives MiPOD **79.70%** /AUC .887769,
TPR 81.4% /FPR 22.0%, 1,594/2,000 correct. It remains six decisions short
of 80%. The +0.95-point change has exploratory paired 95% interval -0.10125
to +2.00 points, including zero. All 25 calibration records, metrics/bootstrap
and 70 direct-distance predictions replay, max discrepancy 8.055e-10.
Parent plus independent closeout takes 842.469969 of 1,500 seconds, with
5,420,617 bytes at parent close; no Gram/CDF-reference matrices persisted.
Do not relaunch it or convert its unused allowance to another attempt.

A later read-only installed-source audit confirms the existing conseal
2025.11 generator uses 9x9 /degree-nine variance estimation, 7x7 Fisher
smoothing and `MiPOD_FIX_WET`. The original paper's Table I .2038 error is
its unsmoothed medium-block setting; Table II's smoothed setting reports
.2243 at 0.4 bpp. Keep that distinction in research comparisons. No simulator
setting, payload or cached media is changed to improve the headline.

The same current user instruction also authorizes the distinct planned
content-heterogeneity intervention in `docs/MIPOD_CONTENT_PROTOCOL_20261115.md`.
Its fresh exclusive directory is `.benchmark/mipod-content-20261115/`;
inspect its marker/logs/processes and never duplicate or retry it. Two/four
fit-cover-quantile partitions define local paired experts; each query gate
uses only that query's own features. Three fixed ridges per expert and
local/global weights .5/1 yield twelve fresh system candidates plus the
current global reference. Calibration alone selects the system/threshold,
with artifact choice frozen before any optional one selected test inference.
The fresh cumulative cap is 600 seconds /sampled 10-GiB RSS /eight-MiB output.
This does not reuse the closed normalization allowance. All 570 source hashes
and generated expert/gating checks pass before fitting. Record actual outcome
and independent closeout, preserving every earlier result and failure.

The content-expert intervention is now independently closed. All eighteen
expert solves and twelve systems are eligible. Calibration selects two
partitions /ridge .1 /.5 local-global mix at 80.60%, only one extra calibration
decision versus the global reference. Its frozen historical test is worse:
**78.40%**, AUC .885336, TPR 81.2% /FPR 24.4%, 1,568/2,000 correct. Gain
versus the preceding 79.70% is -1.30 points, exploratory 95% interval -2.05
to -0.54875 points. Preserve this negative finding; it does not validate the
hypothesis or permit cherry-picking a different local model on test scores.
All thirteen records, system/gate reconstructions, bootstrap and 58 numeric
predictions verify, max discrepancy 5.403e-11. Debit is 99.080078 /600 seconds,
3,220,932 bytes at close. No old allowance is reused or parent relaunched.

The separate geometry comparison under the same current research instruction
is `docs/MIPOD_HELLINGER_PROTOCOL_20261115.md`: unwhitened square-root /RC8
histogram kernels, fit-cover-median bandwidth factors .25/1/4, three ridges
and one SVC each, 24 fresh candidates plus both previous references. Keep
selection calibration-only and preserve the negative content-reference row.
The new total cap is 900 seconds /sampled 10-GiB RSS /eight-MiB output.
The first preparation fails generated-only distance/median QA before any real
fit or source-freeze completion. Keep `.benchmark/mipod-hellinger-20261115/`
unchanged, its failed source/log/exit and read-only distance diagnosis.

`docs/MIPOD_HELLINGER_NUMERIC_AMENDMENT_20261115.md` binds a source-corrected
completion in `.benchmark/mipod-hellinger-completion-20261115/`. The small
bandwidth sample now uses direct-difference SciPy pdist, avoiding positive
dot-product cancellation for identical vectors. All recipe choices, <1e-12
zero rule and numeric guards remain unchanged. Ten seconds conservatively
debit the failed 2.504-second preparation plus .410-second read-only diagnosis
and overhead, leaving 890 /900 seconds, not a fresh allowance reset. Check
its exclusive marker/logs/processes and never duplicate or relaunch it.
Record the actual corrected-source generated checks, real fits and independent
closing outcome before publication. No new data, raw decode, GPU, holdout
access, source role change or product deployment is authorized by these trials.

The corrected Hellinger completion is now independently closed. Generated
controls pass before the first real fit; all 24 fresh models are eligible.
Calibration (BA 81.8667%) selects unwhitened RC8 square-root Gaussian kernel,
gamma .006107876271462043 /ridge .01, threshold -.010646304581314325.
One frozen selected historical inference gives MiPOD **80.70%** at **0.4 bpp**,
AUC .900443, TPR 82.5% /FPR 21.1%, 1,614/2,000 correct: fourteen above the
80% boundary. Its +1.00-point change versus the global RC8 reference has
exploratory 95% interval -0.15 to +2.15 points, containing zero. Geometry,
bandwidth and ridge change together, so no single-factor cause, reliable
population gain, untouched qualification or deployment is claimed.

All 26 calibration records, both sample bandwidths, exact test counts/AUC/
bootstrap and 70 direct-distance predictions replay independently. Actual
completion time is 485.513698 seconds, conservatively 495.513698 /900 after
the ten-second old debit. Parent-close output is 4,391,195 bytes, below eight
MiB. The two earlier successful studies and this completion total 64 metric
records /198 sampled predictions and 1,437.063745 parent-debit seconds;
combined parent-close output is 13,032,744 bytes (about 12.43 MiB). Check the
public receipt for exact per-stage bytes; no feature/Gram/reference matrices
were persisted. Models and raw data remain private; about 30 GiB stays free.

At 0.4 bpp the four academic spatial historical point estimates now pass:
HILL 80.05%, S-UNIWARD 80.75%, WOW v2 80.30%, MiPOD 80.70%. Other methods
are not all passing across their payloads: retain nsF5 .05/.1 bpnzac,
OpenStego .1 bpp, unknown-payload LSB at 5%, FSDD WAV and unavailable EBS.
No earlier miss or negative content-expert row is removed. Results/diagnosis
and public evidence are `docs/MIPOD_DIAGNOSIS_RESULTS_20261115.md` and
`benchmarks/mipod-diagnosis-20261115.json`. All parents are closed; no old
allowance/result grants a new grid, retry, download, GPU run or deployment.
All 570 application/test inputs still match the passed 4,646-test software
gate, with no fresh full-suite claim. Documentation checks and publication
are recorded separately; keep per-file lowercase commits and the unrelated
`nano.383070.save` intact.

The final documentation gate passes all fifteen tests; README has 899 words,
four badges and valid local links. Source/metric/link verification again
matches all 570 full-gate bindings, 64 calibration records and 198 sampled
predictions. No model worker remains active. Determine publication from git
tracking status; the full software gate is reused with unchanged inputs,
not relabelled as a fresh run.

### Remaining floor continuation and BOSS source correction (2026-11-15)

The user's new explicit continuation requests work on the remaining sub-80%
conditions. This permits fresh bounded CPU studies on existing caches, with
unchanged payloads and no old attempt restart. Preserve every previous result,
failure and resource ledger; no downloads, raw decoding, GPU, ALASKA2 access,
native Commons role change or deployment follows. New protocols are
`docs/FLOOR_MIXED_LSB_PROTOCOL_20261115.md`, its role amendment, and
`docs/FLOOR_IMAGE_PROTOCOL_20261115.md`. All 570 software-gate sources remain
unchanged; reuse the actual passed 4,646-test gate without claiming a new run.

A newly added source-role guard catches a real legacy split error before any
new mixed calibration/test scoring. The original BOSS split shuffles file
names without grouping pixel hashes. Training row 3,044 and test row 503
share pixel content; independent cached cover-feature equality verifies it.
Train/calibration and calibration/test have no shared pixel hash. Preserve
`.benchmark/floor-mixed-lsb-20261115/` as failed, with its source/freeze/log/
exit. No model was refitted. The separate read-only audit is private under
`.benchmark/floor-boss-role-audit-20261115/`, with public aggregate receipt
`benchmarks/boss-pixel-role-audit-20261115.json`. Earlier statements claiming
these original 1,000 test sources were all pixel-disjoint are superseded by
this correction; historical scores themselves remain unchanged records.

On the remaining 999 pixel-disjoint test sources, saved-score replay gives
HILL 80.03003003%, S-UNIWARD 80.73073073%, MiPOD 80.68068068%, without any
new model or threshold. WOW exact clean metrics are unavailable because its
score vectors were not retained. Aggregate bounds are 80.28028028–80.38038038%,
not a confidence interval; both exceed 80%. Do not rerun consumed WOW fitting
to invent missing evidence. The JPEG 3,000-source cohort ends before row
3,044 and is pixel-disjoint from its 1,000-source test; check each cohort.

The mixed-LSB source-corrected completion is independently closed in
`.benchmark/floor-mixed-lsb-completion-20261115/`, under the amendment.
It reuses three numeric Fisher states, with no refitting. A single fixed
median of standardized mixed/replacement-5%/matching-5% scores is selected
on the 500-source calibration set to maximize minimum BA over all eight
kind/rate conditions. No query class/payload or paired cover chooses a model.
On the corrected 999-source test, all eight cells pass: BA ranges from
85.48548549% to 88.83883884%. Replacement/matching at 5% score 85.48548549 /
86.03603604%, with FPR 22.02202202%. High-payload scores regress relative to
the older 0-threshold detector; retain those tradeoffs instead of hiding them.
All seven/eight-cell calibration records, thresholds, AUCs, eight test cells
and eighteen direct numeric scores verify. Completion takes 3.392694 seconds,
13.392694 /120 after the ten-second old-failure/audit debit; output is below
one MiB, with no feature matrix persisted. Do not duplicate or relaunch it.

The image-floor parent/evidence directory is `.benchmark/floor-image-20261115/`.
Inspect its marker/logs/processes before action and never duplicate it. It
compares fixed source-grouped OpenStego .1 bpp and nsF5 .05/.1 bpnzac kernels,
with one global choice freeze before any selected test inference. Fresh total
cap is 1,800 seconds /sampled 10-GiB RSS /eight-MiB private output, one BLAS
thread and all matrices in RAM. Source/QC hashes may bind combined caches;
test rows never fit/calibrate models. Record its actual outcome and independent
closeout before publication. Keep the unrelated `nano.383070.save` intact.

The image-floor study is independently closed: all 72 kernel candidates are
eligible, with 210 direct-distance prediction replays (max 6.888e-10), full
calibration/test counts/AUC/intervals and unchanged 570 software hashes.
OpenStego .1 remains 79.75% (319/400). nsF5 .05/.1 score 59.35 /70.75%, below
both target and prior 60.70 /75.15% rows; keep these negative new results.
No test-based replacement, rate increase or old-row removal is permitted.
Exact resources/receipts are in `.benchmark/floor-image-20261115/closing.json`.

The fresh WAV protocol is `docs/FLOOR_WAV_PROTOCOL_20261115.md`. Its original
preparation in `.benchmark/floor-wav-20261115/` fails the ESC source-group
check before any actual fit. The legacy v2 loader ignores the already-admitted
`esc-original-group-max-fold-v1` source receipt. Four original recording IDs
span filename folds; these are different waveform clips, not exact copies.
The existing receipt assigns whole recording groups to their maximum fold:
1,200 fit /398 calibration /402 test. Preserve legacy 400-test rows separately.
The aggregate audit is `benchmarks/esc50-role-audit-20261115.json` and the
first source amendment is `docs/FLOOR_WAV_ROLE_AMENDMENT_20261115.md`.

The first corrected-source preparation in
`.benchmark/floor-wav-completion-20261115/` also fails before any actual fit:
its implementation mistakes constant `source_group='ESC-50'` for recording
identity. Preserve its source/log/exit unchanged. The second source correction
`docs/FLOOR_WAV_GROUP_KEY_CORRECTION_20261115.md` uses the actual `group_key`,
matching upstream_recording_id. The exclusive actual-fit parent is
`.benchmark/floor-wav-groupkey-20261115/`; inspect marker/log/process before
any action and never duplicate it. Its metadata-only preflight verifies:
ESC 1,200/398/402 clips across 919/295/310 recording groups; AudioMNIST
300/100/200 clips across 15/5/10 speakers; FSDD 1,500/500/1,000 clips across
3/1/2 speakers. The source/QC and generated numeric-tree controls now pass.

WAV compares nine kernel plus one fixed numeric-tree candidate for each of
sixteen previously failing dedicated cells, and one mixed tree per dataset.
Choices freeze globally before test inference. Original total cap is 2,400
seconds, with twenty-five seconds conservatively charged for both failed
preparations and diagnosis, leaving 2,375. Sampled RSS cap ten GiB, output
24 MiB, one BLAS/OpenMP thread, no raw decode or new media. Record the actual
outcome/partial failure and resource debit before any further action; do not
retry consumed candidates or reset this allowance. Public software remains
unchanged, and every old experiment, model, failed log and source ledger stays.

A distinct OpenStego augmentation is prospectively frozen in
`docs/FLOOR_OPENSTEGO_WS_PROTOCOL_20261115.md`, under the user's same explicit
remaining-floor request. It permits bounded reading of the existing .1-bpp
RGB PNG corpus for 114 neighbor/parity/weighted-stego statistics, without
password/key lookup or media modification. Same parent-separated fit/cal/test;
ten fixed new classifiers and small fixed late score mixtures use calibration
only. Its separate allowance is 1,500 seconds /sampled four-GiB RSS /eight-MiB
private output, no Gram/raw-pixel persistence. Preparation is in progress at
`.benchmark/floor-openstego-ws-20261115/`; no attempt marker or fit exists yet.
Do not infer a launch from this preparation. Preserve all BOSS role corrections,
source-clean score denominators and the unrelated `nano.383070.save`.

The WAV sequence is now independently closed in
`.benchmark/floor-wav-f256-20261115/`. The group-key-corrected parent completes
all sixty ESC dedicated candidates, then its full-feature mixed tree hits the
680-second phase limit. The first partial continuation reuses those exact
states, completes twenty AudioMNIST dedicated candidates, then its full-feature
mixed tree hits the 240-second phase limit. Preserve both failures as unavailable,
not models evaluated or passing. The separate feature-selection continuation
reuses all eighty completed states byte-identically, without refitting, and
fits the unattempted FSDD families plus distinct 256-feature mixed alternatives.
Columns are chosen only from fit-cover/stego Fisher scores, never test roles.
The full-5,433-feature references remain unavailable; no old attempt is retried.

All 166 WAV calibration records, forty measured test cells and 726 direct
numeric predictions replay (maximum error 1.475e-10), with all 570 software
bindings unchanged. None of the sixteen new dedicated cells passes 80%:
corrected ESC replacement 5/10% is 73.1343 /78.3582%; matching 5/10/20/40%
is 65.2985 /65.1741 /66.0448 /69.4030%. AudioMNIST matching 5/10% is
56.50 /59.50%. FSDD replacement 5/10/20/40% is 50.00 /49.70 /49.95 /51.10%;
matching is 50.35 /50.00 /49.45 /50.85%. All new mixed cells and false alarms
must also remain visible, including FSDD's 99.50% FPR and AudioMNIST matching
20% at 79.50% (a miss). Historical passing dedicated cells were not rerun;
legacy ESC 400-test and corrected 402-test cohorts are different.

The final WAV continuation takes 190.574979 seconds; conservative cumulative
debit including both source failures and partial parents is 1,170.574979 /2,400
seconds. Combined logical output is 18,379,294 bytes, counting retained hard
links conservatively; the last directory alone is 10,058,670 bytes at closing.
All parents are closed; unused time grants no consumed candidate retry or
new grid. Preserve the 980-second prior debit, all states, failed logs and
full-feature unavailability. Protocols/amendments are linked from
`docs/REMAINING_FLOOR_RESULTS_20261115.md`.

The first OpenStego WS parent subsequently launched and passed source/generated
checks, then its data-only feature phase reached the 600-second watchdog
(exit -9, 600.925 seconds) before any real classifier fit. Preserve the
original freeze/source/logs and failed feature phase. The prospective
`docs/FLOOR_OPENSTEGO_WS_EXECUTION_AMENDMENT_20261115.md` permits only a
source-corrected completion in `.benchmark/floor-openstego-ws-completion-20261115/`.
It uses two ordered independent-file workers with byte-identical descriptors;
model candidates, original roles, payloads, scalar oracle and guards stay exact.
Charge 650 seconds for old preparation/feature work and overhead; at most
850 /1,500 seconds remain. The parent samples summed worker/descendant RSS
under four GiB and combined old/new output under eight MiB. Inspect its
exclusive marker/log/process before action; never duplicate it. Record its
actual result, independent closing and cumulative debit before publication.

The two-worker OpenStego WS completion is now independently closed:
**87.75% BA** at unchanged **.1 bpp**, AUC .954325, TN 175 /TP 176 (351/400),
TPR 88.0% /FPR 12.5%. Calibration selects the 15-leaf /l2=1 tree among ten
new WS candidates, then a fixed 75% WS /25% SRM mixture among five systems.
Threshold 1.061469973523839 is frozen before constructing test features;
200 crop pairs represent 100 DIV2K validation parents. The paired +8.00-point
gain over the frozen 79.75% kernel has exploratory 95% interval +3.75 to
+12.00 points; BA interval 84.5–91.0%. This reused historical cohort is not
untouched cross-source qualification. No research model is deployed.

All fifteen calibration records, test metrics/parent bootstraps and forty-two
sampled numeric predictions replay (max error 1.464e-11); a separate scalar
oracle exactly reproduces 114 features on both real calibration images.
Completion wall time is 408.041839 seconds, cumulative 1,058.041839 /1,500
including the unchanged 650-second old debit. Combined parent-close output
is 6,655,173 bytes /eight-MiB cap. Feature preparation's summed sampled RSS
peak is 496,332,800 bytes; wait4 metadata is per process, not an aggregate
peak claim. The consumed parents are closed; no unused time grants a retry.

All remaining-floor results and every payload-specific miss are now in
`docs/REMAINING_FLOOR_RESULTS_20261115.md` and
`benchmarks/remaining-floor-20261115.json`. The main detection table, README
and changelog include the two solved gaps, new negative image/WAV results
and BOSS/ESC role corrections. Four older CPU result reports carry explicit
supersession notices while preserving executed numbers. Exact clean WOW
metrics remain unavailable; keep its aggregate bounds distinct from a CI.
The overall all-method/all-payload 80% goal remains unmet: low nsF5 and WAV
misses plus unavailable EBS remain. New sources or hypotheses need their
own prospectively frozen intervention, not a relaunch of these old parents.

Fresh Ruff /270-file mypy including research and all fifteen disk-backed
README/scope tests pass. README has 895 words, four badges and valid local
links. Every one of the 570 unchanged sources still matches the actual
passed 4,646-test /two-CUDA-skip /97.415805% full CPU gate; this is source-
identical reuse, never a claim of a new full-suite run. QA and publication
source/log evidence is private in `.benchmark/floor-closeout-20261115/`.
About thirty GiB remains free /85% used; no data/model/log cleanup occurred.
Inspect git tracking to determine publication; preserve per-file lowercase
subjects, no trailers, and the unrelated `nano.383070.save` untouched.

Final publication verification checks 502 public metric records, all 570
source hashes and 75 local links. All new research parents are closed.
Private verification is `publication-verification.json` in the closeout
directory; determine the final commit/push state from git tracking.

### Low-signal diagnosis and calibrated relations (2026-11-15)

The user's next explicit research/fix request permits fresh bounded CPU
hypothesis studies on already-admitted data. `docs/LOW_SIGNAL_DIAGNOSIS_PROTOCOL_20261115.md`
prospectively authorizes bounded raw fit/calibration JPEG/WAV reads, with
no historical test decoding, new media, GPU or old fit restart. Its exclusive
parent in `.benchmark/low-signal-diagnosis-20261115/` is closed verified:
6.011487 seconds /240, 1.398-GiB wait4 peak, eleven exact old feature replays,
all sampled nsF5 changes and all 570 software hashes match. Original bytes,
source roles and payloads remain unchanged. Keep its source/freeze/logs.

Among the fixed sampled fit/calibration clips, median first-difference
saturation under the old quiet-frame quantizer is 93.5435% for ESC and
89.2962% for FSDD, versus 5.4450% for AudioMNIST. These are selected-source
diagnostics, not population statistics. Sampled nicolas training clips alone
have all-even samples; other sampled FSDD speakers have about half even.
This supports a quantizer information-loss/source-shortcut hypothesis, not
an impossibility proof. The WAV paper's abstract must not be interpreted as
a universal 80% guarantee: its one-second /10%-payload table entry is 62.25%.
nsF5 at .05/.1 changes only about .56/1.30% of nonzero AC coefficients;
optimal-coding nsF5 is not the real F5 encoder's same-change-count experiment.

`docs/NSF5_CALIBRATED_RELATIONS_PROTOCOL_20261115.md` freezes a distinct compact
coefficient-domain study using already-installed sealwatch 2024.12's JRM
absolute subset, per-mode histograms, four-edge crop calibration and signed
decrement estimates. No install or full-JRM reproduction is claimed. Same
2,400 fit /600 calibration /1,000 test source roles; ten fixed new models
per unchanged payload and fixed late mixtures choose only on calibration.
Only selected calibration BA >=80% permits new test feature construction;
otherwise preserve the measured old miss as context and mark the new test
unavailable by its calibration gate. Global choice precedes any new test
feature. The exclusive new evidence directory is
`.benchmark/nsf5-calibrated-relations-20261115/`. Inspect marker/log/process
before action; never duplicate it. Fresh total cap 2,400 seconds including
five preparation seconds, <=six-GiB summed sampled RSS and 256-MiB output,
four file workers /one BLAS thread each, disk-backed temporary calibration
files and no copied raw media or Gram persistence. Record actual outcome,
independent audit and closing debit before publication or later intervention.

The original coefficient-study preparation exits 1 after 1.503 seconds:
its reused JSON reader expects Path while the new caller supplies a string.
No real feature or model exists in that failed attempt. Preserve it unchanged.
`docs/NSF5_CALIBRATED_RELATIONS_SOURCE_AMENDMENT_20261115.md` binds the separate
`.benchmark/nsf5-calibrated-relations-completion-20261115/` source correction,
with ten conservative old-preparation/failure seconds and 2,390 /2,400 seconds
remaining. A separate corrected-helper/generated-control preflight passes;
all scientific choices and real-fit guards remain unchanged. Inspect the
new exclusive marker/log/process and never duplicate or relaunch either.

The separate `docs/WAV_MODULAR_CALIBRATION_PROTOCOL_20261115.md` now freezes
a fresh unsaturated-modular/re-embedding hypothesis for the sixteen previously
failing dedicated WAV conditions. Its generated scalar histogram/context,
reference-determinism and numeric-export preflight passes. The new FSDD fit
omits nicolas's 500 upscaled-eight-bit training clips as a declared experimental
cohort overlay; no original role, file or historical result is changed or
reassigned. George/jackson fit, lucas calibrates, theo/yweweler remain test.
Source and feature changes are joint, not a single-factor causal experiment.
ESC and AudioMNIST corrected roles stay unchanged. Choices use calibration
only; calibration BA below 80% prevents new test feature construction/inference.
The exclusive parent/evidence directory is
`.benchmark/wav-modular-calibration-20261115/`; inspect marker/log/process,
never duplicate it. Fresh cap 1,800 seconds including five preparation seconds,
six-GiB summed sampled RSS /256-MiB output, at most two file workers /one BLAS
thread each and disk-backed TMPDIR. Record actual failures/gates, independent
audit and cumulative closing resources before publication. No raw media/Gram
copy, payload change, download, GPU, reserved data or deployment follows.

The second nsF5 source preparation succeeds, then training feature export
fails after 254.107 seconds at the inherited eight-MiB hard per-file limit.
No real classifier was fitted. Preserve its truncated archive as unavailable,
never reuse an unverified prefix. The first modular WAV parent is explicitly
stopped during data-only ESC preparation after 126.696 parent seconds for the
same known source issue; its real -15 exit and explicit external-stop receipt
remain separate from a watchdog result. An independent fit-only WAV feature
cost probe takes .091548 seconds for 220,500 samples, with zero fits/test reads.

`docs/LOW_SIGNAL_CHUNKED_CACHE_AMENDMENT_20261115.md` prospectively binds two
separate chunked-cache completions, retaining the eight-MiB file guard and all
feature bytes/model/source/seed/threshold definitions. At most 100 sources
per compressed part, per-part hashes and exact ordered concatenation; generated
scalar and chunk-manifest roundtrips pass before actual fitting. nsF5 now uses
`.benchmark/nsf5-calibrated-relations-chunks-20261115/`, debit 280 /2,400 seconds,
at most 2,120 remaining; combined output includes both earlier parents. WAV
uses `.benchmark/wav-modular-calibration-chunks-20261115/`, debit 150 /1,800,
at most 1,650 remaining; four independent-file workers replace two, within
the unchanged six-GiB summed RSS and 256-MiB combined-output caps. Original
files, partials, logs and roles remain. Inspect exclusive markers/logs/processes;
never duplicate a parent or restart old work. Actual outcome/independent
closing and every calibration-gate exclusion must precede publication.

Both chunked-cache interventions are now independently closed. nsF5 .05
selects a fixed 75% new /25% DCTR mixture with calibration BA 65.416667%;
it fails the predefined gate, with no new .05 test features/inference. nsF5
.1 selects 50/50 new plain-square-root coefficient kernel /frozen fair-fit
DCTR, calibration 81.00%, threshold .7342438804846576. One selected historical
test gives **77.70%**, AUC .853915, TPR 74.5% /FPR 19.1%, TN 809 /TP 745
(1,554/2,000). Still a miss: 46 decisions short, BA 95% interval 76.15–79.15%.
Paired gain versus its frozen 70.75% kernel is +6.95 points (95% +5.30 to
+8.65); against the best legacy 75.15% point it is only +2.55 points. No
paired interval is invented for unavailable legacy score vectors. Historical
evaluation remains adaptive, not untouched qualification or deployment.

All sixteen new WAV calibration choices miss 80%, so all new WAV historical
test features/inferences are excluded by the prospective gate. Keep these
`not_evaluated_calibration_gate`, never a test pass. ESC replacement 5/10%
is calibration 71.733668 /77.135678%; matching 5/10/20/40% is 61.934673 /
64.572864 /66.080402 /65.452261%. AudioMNIST matching 5/10% calibrates at
59.0 /70.0%. All eight FSDD calibration BAs are 51.6–52.3% after the matched-
bit-depth overlay and modular/re-embedding features. This joint negative
intervention does not support a single saturation/8-bit-shortcut explanation,
and does not prove an irreducible data accuracy ceiling. All earlier measured
test misses and the historical passing rows remain unchanged.

Independent closing replays thirty nsF5 calibration records, one selected
test cell, sixty-two direct predictions (max error 1.018e-10), three actual
calibration feature vectors and all source bindings. WAV closing replays
128 calibration records, all sixteen gate exclusions, 256 predictions
(max error 2.602e-11) and three full real-source descriptor replays. Source
and feature changes are joint; no single-factor or universal-floor claim.
A separate fit-cache audit verifies all 500 nicolas covers have eight zero
low bits; neither of the other 1,000 FSDD fit covers does. No test is read by
that shortcut audit. Preserve its receipt in the low-signal closeout folder.

nsF5 chunk completion takes 452.055094 seconds, cumulative 732.055094 /2,400
including the 280-second prior debit. Combined parent-close output is
140,801,014 bytes, including both earlier parents/truncated archive. WAV
completion takes 461.113286 seconds, cumulative 611.113286 /1,800 including
150 old seconds; combined output is 120,403,931 bytes including the stopped
parent. Both respect original six-GiB summed sampled RSS /256-MiB combined
output caps and the inherited eight-MiB per-file guard. All parents are
closed; unused consumed time grants no old retry. Total retained logical
evidence is about 249.10 MiB, with about thirty GiB free /85% used. No data,
model, partial or failed log was deleted. Keep every source/failure/debit.

The cause analysis, primary-paper context, calibration-only exclusions and
actual test miss are public in `docs/LOW_SIGNAL_DIAGNOSIS_RESULTS_20261115.md`
and `benchmarks/low-signal-diagnosis-20261115.json`. Main detection results,
README, changelog and the previous remaining-floor report link the findings.
Do not portray the .05 calibration gate or WAV exclusions as measured test
results. The overall 80%-per-method goal remains unmet; no payload increase,
new media/model download, GPU, reserved-data access or product deployment.

Fresh Ruff /270-file mypy including research and all fifteen README/scope
checks pass. README is 898 words with four badges; all 570 source hashes
still match the actual passed 4,646-test /two-CUDA-skip /97.415805% full gate.
This is source-identical reuse, not a fresh full-suite run. Final software,
publication and training-shortcut evidence is private in
`.benchmark/low-signal-closeout-20261115/`. Determine publication from git;
keep per-file lowercase commit messages, no trailers, and preserve the
unrelated `nano.383070.save` unchanged.

### Fresh cross-representation retry (2026-11-15)

The user's explicit retry request permits the distinct family frozen in
`docs/RETRY_FUSION_PROTOCOL_20261115.md`, not relaunch of any consumed parent.
JPEG joins existing plain square-root DCTR/coefficient features in new paired-
bootstrap Fisher ensembles and composite sum/product kernels; fifteen new
models per unchanged .05/.1 payload, fixed calibration-only score systems.
WAV compares seven fixed combinations of frozen old/new model scores per
sixteen failing dedicated conditions, with no old classifier refit. Source
roles, payloads, old results and all historical debits remain unchanged.

The exclusive new parent/evidence directory is
`.benchmark/retry-fusion-20261115/`; inspect marker/log/process and never
duplicate it. The first generated control and parent fail before actual fit;
see the source correction below. The intended total
fresh cap 1,800 seconds including five source-preparation seconds, six-GiB
summed sampled RSS /96-MiB output, inherited eight-MiB per-file guard and
disk-backed TMPDIR. No Gram/raw-media persistence. All domain choices freeze
globally before optional selected inference; calibration BA below 80% excludes
new test, and a selected old reference reuses its retained result. Only fresh
gated feature coverage may read existing test sources via unchanged bounded
workers; <=100-source hashed chunks and <=four file workers, one BLAS thread.
No download, GPU, ALASKA2/native role change or deployment follows. Independent
closing and actual outcomes must precede publication; unused old allowances
are not transferred or reset. All 570 application/test sources stay unchanged.

The first generated fusion control incorrectly uses float32 kernel operands,
while real candidates already use float64. Generated residual reaches
3.268e-6, failing the unchanged 1e-6 guard; float64 gives <=8.483e-14.
Preserve the twelve-case generated-only diagnosis. The original parent was
incorrectly launched after failed preflight and fails on its missing success
receipt after 1.503 seconds, with zero actual candidates fitted.
`docs/RETRY_FUSION_NUMERIC_AMENDMENT_20261115.md` binds only the corrected
generated dtype and strict preflight-before-launch control in
`.benchmark/retry-fusion-completion-20261115/`. All generated and calibration-
schema controls now pass before launch. Actual model/source/seed choices and
all tolerances stay exact. Charge ten prior seconds inside the original
1,800 cap, leaving 1,790; combined-output accounting includes both parents.
Inspect its exclusive marker/log/process; never duplicate either parent.

The fresh fusion completion is independently closed: nsF5 .1 reaches 77.85%
(AUC .864762), still below 80%, versus 77.70%; the +.15-point paired interval
contains zero. nsF5 .05 calibrates at 66.25% and remains gated. WAV selects
the unchanged old ESC replacement-.1 reference (78.3582% old test), with all
other fifteen new WAV conditions gated below .80. No new WAV test feature
was constructed. The unused gated raw-worker branch was not exercised.
Closing replays 152 calibration records and 114 predictions (max 4.762e-10),
with one new selected test; cumulative 176.483623 /1,800 seconds including
ten prior seconds, 7,258,804 combined parent-close bytes. Preserve both parents.

The same user retry request now permits a separate fixed sample-count study
in `docs/NSF5_SAMPLE_COUNT_PROTOCOL_20261115.md`, not use of closed time.
Metadata-only review of the existing BOSS corpus leaves 8,996 unique non-test
pixel groups, preserving the old first-3,000 order and 600 calibration roles.
Only 3,003 previously excluded non-test PGM sources were pixel-decoded for
missing hashes, not current test pixels. Fixed additional calibration yields
7,496 fit /1,500 calibration groups; original roles and all old media stay.
One new coefficient-only Gaussian ridge .01 fit at unchanged .1 bpnzac uses
existing old features plus 5,996 added-source feature pairs, with bounded
transient Q75 compression and no persisted JPEG/raw copy. Exact old raw-to-DCT
and two old feature-vector replays plus generated numeric controls pass.
No DCTR/new model family/grid, download, GPU or reserved-data access follows.
The exclusive parent is `.benchmark/nsf5-sample-count-20261115/`; inspect
marker/log/process and never duplicate it. Fresh cap 1,800 seconds including
thirty conservatively charged metadata/control seconds; ten-GiB summed sampled
RSS /256-MiB output /eight-MiB file guard, four file workers /one BLAS thread,
disk TMPDIR and no Gram persistence. Preserve all failures, source/group
checks and actual closing. New calibration BA below .80 prevents new test
inference; otherwise one frozen historical test reuses existing verified
features. No untouched qualification or deployment is claimed.

The fresh sample-count intervention is now independently closed. All
8,996 fit/calibration sources are pixel-unique and exact cover-feature hash
intersections across fit/calibration/test are zero. First-3,000 cached .1
pairs and roles stay unchanged, with 5,996 added-source pairs only. The
single 7,496-fit coefficient Gaussian ridge .01 model calibrates at
**75.333333%**, AUC .831252444, TPR 79.133333% /FPR 28.466667%, 2,260/3,000
correct. It fails the predefined .80 gate; no new historical test inference
occurs. Existing 77.85% test evidence is context, not its result or a pass.

Read-only fixed-threshold retained calibration diagnostics give 76.833333%
/AUC .848905556 on the preserved 600 and 74.333333% /.819381481 on the added
900. Threshold -.0420271442453668 never changes. Source composition and
repeated earlier model development on the old subset cannot be separated;
this is not a proof of intrinsic source difficulty or a universal ceiling.
Keep the metadata precheck's missing-hash failure in the retry closeout log;
its complete source inventory and plan precede any actual model fit.

Independent closing verifies the one calibration record, landmark median,
22 direct predictions (max error 1.196e-10), eight sampled training equations
(max residual 3.743e-11), two actual added calibration feature vectors and
all 570 software hashes. Actual completion takes 235.421302 seconds,
cumulative 265.421302 /1,800 including thirty prior metadata/control seconds;
parent-close output 66,902,300 bytes. The fit's per-process wait4 peak is
about 4.647 GiB, below its ten-GiB sampled observer cap. No RAM cgroup claim.
All parents are closed; unused consumed time grants no old retry. Together
with fusion, about 70.73 MiB logical evidence remains, no data/model/log
cleanup and about thirty GiB free /85% used. No copied raw/JPEG corpus or
Gram, download, local GPU, reserved-data access or deployment occurred.

Current results/limitations are `docs/RETRY_FUSION_RESULTS_20261115.md` and
`benchmarks/retry-fusion-20261115.json`. Keep both generated control/preflight
failures and the unchanged actual recipes/numeric guards. nsF5 .1 is 77.85%
with a +.15-point interval containing zero; other .05/WAV misses remain.
The overall 80% floor is still unmet. Every historical result and gate keeps
its original status; no test-selected fallback, raised payload or calibration
score is presented as test accuracy. Scientific code/models stay private;
all 570 application/test sources remain identical to the passed 4,646-test
full gate. Final document QA and publication evidence is private under
`.benchmark/retry-closeout-20261115/`; determine publication from git tracking,
with per-file lowercase messages, no trailers and nano.383070.save preserved.

Final retry software checks pass fresh Ruff, 270-file mypy including research
and all fifteen disk-backed README/scope tests. README has 898 words and four
badges. All 570 hashes match the actual passed full gate, with no new full-suite
claim. Both successful parents are closed; preserve unrelated nano.383070.save.

### Reference feature correction and full JPEG features (2026-11-16)

The user's new fix request permits a fresh CPU reference-parity investigation
and bounded full-feature intervention, with all earlier restrictions intact.
The original read-only audit fails its repository import before any extraction;
the separate completion preserves that failure and a missing optional psutil
parent import. Its proc-based observer completes in 6.157195 seconds,
11.157196 including five prior seconds, peak sampled RSS 315,359,232 bytes.
Two existing fit covers differ from installed sealwatch DCTR in 213/209 bins;
changing only nearest-even to reference half-away rounding removes all errors
above 1e-6, maximum 2.972e-8. IDCT already agrees within 2.576e-13.

Public `research/dctr_features.py` now defaults to reference rounding and
retains explicit `rounding="legacy"` through all file/coefficient helpers.
Historical caches/states require legacy arithmetic; never mix or rewrite them.
Fourteen generated focused tests pass, including independent convolution/phase
oracles. The fresh full QA now passes 4,651 tests /two explicit CUDA skips /
32 warnings, 97.41688424717145% coverage; corrected DCTR is 100% covered.
Ruff /270-file mypy including research and all 570 current source hashes pass.
Pytest takes 2,325.52 seconds; wrapper time is 2,338.472293 seconds. Its user unit is
`steganography-reference-feature-qa-20261116.service`, with evidence under
`.benchmark/reference-feature-full-qa-20261116/`. The unit is exited /PID zero.
Preserve unavailable final MemoryPeak and the separate retained live samples;
the older 4,646-test pass does not cover these changes.
No frozen core/steganography source changes or model deployment occur.

The new protocol is `docs/JPEG_REFERENCE_FEATURES_PROTOCOL_20261116.md`:
corrected 8,000 DCTR plus full 11,255 JRM, not the old absolute subset.
Same 2,400 fit /600 calibration /1,000 pixel-disjoint historical test pairs,
unchanged .05/.1 payloads/seeds. Eight fixed new candidates per payload use
calibration only and require .80 before new test construction. Two actual
file-worker controls verify exact old changed counts; generated numeric
kernel/ridge/SVC/reference-ensemble exports pass. Original feature/model
definitions and all prior misses remain unchanged historical records.

The first data-only parent is explicitly stopped before any real classifier
fit, with 480 complete feature rows in fifteen hashed chunks. Parent age upper
bound is 656.911455 seconds, 676.911455 with preparation. Keep its unavailable
final MemoryPeak/worker-exit evidence; systemd's post-stop success is not a
successful scientific run. Its sources/partials/preparation failures remain in
`.benchmark/jpeg-reference-features-20261116/`. An extra AST helper first
expected a new build name despite an existing definition; the shell incorrectly
started data preparation after that extra failure. A later relative-path reader
also fails. Corrected independent AST integration passes before any actual fit;
never erase either error or label the failed checks as successful preflight.

The prospective execution amendment is
`docs/JPEG_REFERENCE_FEATURES_COMPLETION_20261116.md`: a fresh
exclusive completion in `.benchmark/jpeg-reference-features-completion-20261116/`
references the stopped prefix without copying/recalculating it, replays its
embedding fractions, and extracts only remaining feature rows. The new outer
operator envelope is 7,200 cumulative seconds, 800 prior seconds charged,
6,400 remaining; the old 4,200-second contract is preserved, not relabelled.
Combined old/new output stays <=768 MiB /eight MiB per file; feature workers
use <=three-GiB sampled summed RSS, fitting <=ten GiB, six single-thread file
workers and disk TMPDIR. Actual fitting waits for the new full QA's pass and
inactive/exited unit. Inspect both marker/log/processes and never duplicate,
restart or relaunch either parent. That first completion fails preparation
before any new features/actual fits: an incorrect document filename, retained
in its source/log. `docs/JPEG_REFERENCE_FEATURES_PATH_CORRECTION_20261116.md`
binds a separate `.benchmark/jpeg-reference-features-path-corrected-20261116/`
completion. Only the document path and prior accounting change: 810 seconds
charged, 6,390 remaining inside 7,200; output includes all parents. The full
metadata/path/hash preparation passes with writing intercepted before launch.
That path-corrected parent is now closed verified; do not relaunch it.
A one-fit-cover bincount cost probe is private and not adopted; JRM library
sources remain unchanged. No GPU, download, new media, locked data, native-role
change or universal 80% claim follows from reference parity.

The path-corrected completion closes at 4,219.138375 s, cumulative
5,029.138375 /7,200 s with the 810-second prior debit; combined output across
all three parents and the cost probe is 646,354,568 bytes /768 MiB. All sixteen
candidates are eligible. nsF5 .05 selects joint ridge at calibration 65.75%
and stays `not_evaluated_calibration_gate`. nsF5 .1 selects the d_sub-1,000
51-learner FLD ensemble (calibration 82.50%, threshold 7/51); one frozen
historical inference gives **80.85%**, AUC .894063, TPR 76.0% /FPR 14.3%,
1,617/2,000 correct. Paired change versus 77.85% is +3.00 points (95% +1.50
to +4.50); BA interval 79.35–82.40% crosses 80%. Features and classifier
change together; the reused test is adaptive, not untouched qualification.
Independent audit replays 16 calibration records and 118 predictions (max
error 1.164e-10); a separate replay of retained scores matches exactly. See
`docs/JPEG_REFERENCE_FEATURES_RESULTS_20261116.md` and its public receipt.
nsF5 .05, WAV misses and unavailable EBS keep the overall 80% goal unmet.

WAV conditional-MoM preparation (`docs/WAV_CONDITIONAL_MOM_PROTOCOL_20261116.md`)
is generated-only; no raw WAV feature bank, attempt marker or classifier fit
exists in `.benchmark/wav-conditional-mom-20261116/`. Its private runner and
auditor are unexecuted drafts and two generated audits are marked invalid;
complete the numeric/source preflight and closing before any launch.

On 2026-11-17 the drafts are preserved; new `runner.py` (draft plus a
generated 300-vs-300 /d_sub-1,000 ensemble check), a matching `audit.py` and
`parent.py` are written. Generated preflight passes; a scratchpad end-to-end
rehearsal on generated WAVs passes both measured and calibration-gated paths
(16 records, 118 predictions); a write-intercepted real-source `prepare` dry
run verifies 5,670 input hashes and all roles. A private two-fit-clip size
probe (`.benchmark/wav-conditional-mom-size-probe-20261117/`) projects ~39 MB
train features; FSDD q1 first-difference valid-pair fraction is 0 on probed
clips. The exclusive parent was then launched once and is now closed verified:
408.74 s, 438.74 /3,600 s with 30 prior seconds, 116,304,461 bytes /512 MiB,
largest summed sampled RSS 0.80 GiB. All sixteen calibration choices miss .80
(ESC 63.19–68.22%, AudioMNIST 59.00 /62.00%, FSDD 51.4–54.0%), so every cell
is `not_evaluated_calibration_gate` with no new test features. The audit
replays 128 records, 768 predictions (max 4.366e-11) and three descriptors.
See `docs/WAV_CONDITIONAL_MOM_RESULTS_20261117.md`; never relaunch it.
