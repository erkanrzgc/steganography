# Colab RGB training preparation — 2026-11-07

Status: **software complete; full verification passed**. The user asks to finish
preparation faster while retaining the checks. This continuation authorizes
implementation and generated CPU QA. It creates no physical profile permission,
native fitting approval, actual media packet or cloud allocation.

The complete workflow is
[`png_bmp_colab_native_training_20261107.ipynb`](../notebooks/png_bmp_colab_native_training_20261107.ipynb),
with its named companion source packet. It covers source installation, one fresh
cost profile, a measured fit proposal, later fit permission, development transport,
training, independent replay and verified export. All execution switches default
to false. The source-upload filename matches the delivered companion; no rename
or installation command is needed. Work stays on the runtime disk; Drive is not
mounted. Keep the same runtime and its exclusive ledger through profile review
and any approved fit. Closing an erased runtime does not authorize a relaunch.

## Frozen implementation and input

The source overlay is reviewed against base commit
`2f41bfe2e95febcaab80f80ab4875c72073c3a32`; it is not claimed to be committed
there. The source packet SHA-256 is
`8677a46f3fb7840b01b46b2b1355a286c9d30296ccf271519fe54bc57d8e300a`.
The new prospective profile contract is
[`png-bmp-colab-training-profile-contract-20261107.json`](../benchmarks/png-bmp-colab-training-profile-contract-20261107.json),
SHA-256 `37d2f589a90ac29dcafea3fc238217022597acc3ea084d1b1f8eb6273a35a33b`.
Previous notebook/source/protocol versions and every consumed attempt remain
historical snapshots; this source extension is not a retry of any of them.

`core.spatial_colab_fit` reuses the unchanged SRNet skeleton, paired native
data/AdamW/order service and numeric state format. It uses the separate Colab
FP32 policy, retaining disabled AMP/TF32, deterministic kernels and the four-GiB
Torch allocator. It does not modify the WSL cgroup guard, JPEG/WAV recipes or
their budgets. The process controller registers `native_fit` separately from
`native_profile` and `generated_qa`; no arbitrary-command route exists.

Runtime Version 2026.07 expectations remain Python 3.12.13, NumPy 2.0.2,
Pillow 11.3.0, Torch 2.11.0+cu128, CUDA 12.8, Tesla T4 and driver 580.82.07.
The actual positive cuDNN version and complete profile environment must match
exactly again before fitting. No CPU lock, pip install, alternate GPU/account
fallback or package/driver change is automatic.

The data proposal remains **64 train /20 development /21 reserved RGB** across
38/12/19 known ancestry components, with thirteen L sources unassigned.
The split SHA is
`852b36d9bfee693c3be175ab5365d160a3e9d5c6da08e9fa1a2a0add552abefa`;
the original manifest SHA remains
`8da0ac4ac98b7b0e9e3613b02b58c3cb076adb9456c74ef29c90f6a70ee1dfd2`.
Actual source roles remain unassigned. The proposal is an access overlay, and
unknown ancestry is not ruled out by the known grouping.

The later profile permission can expose only the 64 train crops, 12,582,912
raw pixel bytes. A distinct fit permission is required before copying/uploading
the twenty development crops, 3,932,160 bytes. The development packet is bounded
to 32 MiB, checks every ID/extent/hash and preflights all destination paths before
writing. It refuses overwrite and symlinks. Twenty-one reserved RGB and thirteen
L pixel files stay closed. No original is decoded/regenerated and no media is
downloaded by this preparation. Public transfer metadata retains source, author,
license, original hash and crop hash for both proposed access stages.

## Profile, debit and later approval

The source-bound profile retains its 240-second command, ten-second closing
reserve, zero native /six generated updates, two generated numeric restores and
independent audit/archive. See the
[profile protocol](PNG_BMP_COLAB_NATIVE_PROFILE_PROTOCOL_20261106.md) for that
recipe. The new source closure includes the complete fitter before any profile,
so accepting a profile requires no intervening implementation change.

`freeze-fit` first audits the actual completed physical profile using its
original contract, exclusive ledger, closed receipt and every output hash.
Missing, unavailable, generated/emulated or nonphysical results are ineligible.
The caller must supply a measured `colab-fit-external-debit-v1` record with
`seconds`, `output_bytes` and literal `complete: true`, covering additional
setup/review/evidence work not charged by the profile. Unknown costs are not
silently treated as zero. Runtime allocation and idle billing time also need
their own observations; command timing is not a billing promise.

The prospective cumulative command allowance remains **1,800 seconds /768 MiB**.
The fit receives only the remainder after the complete profile output, closed
receipt and explicit external debits. Its independent closing reserve is twenty
seconds. The cost projection retains the 1.5 factor and additionally charges
nineteen extra development passes per seed: epoch development scoring is
separate from the old initial/final estimate. Each seed therefore evaluates
548 cover/control pairs: initial train/development, twenty development passes
and final train. The estimate is not a guaranteed upper bound or accuracy claim.

Eight future numeric states reserve eight times the largest measured generated
state plus one MiB per state, with sixteen MiB for remaining evidence. Time or
storage ineligibility prevents even freezing a fitting proposal. The proposal
binds exact profile/receipt hashes, source, split, full environment, recipe and
debits. `colab-fit-permission-v1` must later explicitly approve that contract,
the profile receipt and the resource acknowledgement. A profile permission is
insufficient. This preparation generates neither production permission.

## Complete training and progress

The fixed candidate has two fresh seeds, 2026102301 and 2026102302; twenty
epochs each, batch eight, RGB 256×256 IEEE FP32, scalar AdamW at 1e-4 and fixed
payload seed 2026102805. There are sixteen updates /128 presentations per epoch,
320 /2,560 per seed and **640 updates /5,120 presentations** in total. No tuning,
early stopping, development-driven schedule change or test access is automatic.

Each update has an exclusive intent and commit containing the actual loss and
consumption. Every epoch persists its mean loss and complete development scores
and AUC. The parent streams a compact seed/epoch/loss/AUC/update record to the
notebook; the worker also retains its log. Initial train/development and final
train scores are saved. The final development score is the twentieth epoch,
without an unnecessary extra inference pass.

Numeric states are saved at epochs 5/10/15/20 per seed. They carry model,
optimizer, RNG, data order and consumption bindings. The controller has no
automatic resume/retry command; a timeout, OOM, interruption or invalid audit
retains partial intents/commits/states and consumes the attempt. A second marker
is keyed by the accepted profile receipt, so changing a fit contract/debit cannot
reuse that one profile for another fit.

The exploratory learning rule stays fixed: both seeds require final train AUC
at least .85, development AUC at least .70, and last-epoch mean loss at most
.8 times first-epoch mean loss. A complete failed rule is `failed_learning_gate`,
not an unavailable execution. Neither result qualifies or deploys a detector;
the small development set and repeated visibility remain exploratory limitations.

## Resource and evidence checks

Host RAM follows the provider-managed policy: no inherited eight-GiB DATA/RSS
ceiling. Kernel capacity/availability are observed with ten-percent headroom;
kernel capacity is not asserted to be a verified tenant limit. Provider limits
remain in effect. Hard cgroup RAM/zero-swap enforcement and a whole-tree peak
stay unavailable. Observed swap, observer failure, deadline or OOM closes the
command as unavailable. CPU, file/output limits, process-group termination,
parent-death signal and kernel wall timer remain active.

The parent independently replays all 640 intent/commit records, forty epoch
records, eight numeric states and 48 score/AUC records. Average-rank worker AUC
is checked against a separate pairwise calculation, including ties and finite
scalar checks. Order, numeric environment/source/data, checkpoint schedule and
consumption must agree. The auditor imports no Torch and opens no native pixels.

Each numeric state is a separately hashed private file. The verified training
report ZIP excludes numeric files, avoiding a second copy of eight large states.
It contains the metadata/score/ledger evidence; all states are bound by the
closing inventory and exported individually. The fit receipt has a bounded
four-MiB extent and reservation for its larger inventory. Training archives
explicitly allow at most 4,097 members, checking the directory before allocation;
the historical profile default remains 513. Manifest/directory bytes, ZIP paths,
types, CRC, member hashes and cumulative output remain bounded. No archive or
numeric artifact is extracted/executed.

## Software verification and remaining measurements

The final corrected focused gate passes **226 tests**. The new fitter/CLI have
99.218750% /95.744681% statement coverage; controller/handoff coverage is
96.359223% /97.260274%. Ruff /251-file mypy and diff checks pass. One generated
CPU QA loop executes twelve real SRNet updates across two reduced schedules,
saves four numeric states and independently replays twenty AUCs. This is plumbing
QA, not a full learning control or native accuracy result. The source-only
notebook executes all default cells in isolation without importing Torch,
uploading media or creating an attempt.

Preserve the first focused run: 216 passes /two failures (direct result tuple
serialization and the profile-sized archive member ceiling). The corrected
second gate passes 218 tests; later one-profile exclusivity, nonfinite AUC,
worker-dispatch and default-notebook regressions bring the final focused gate
to 225. A live inventory correction adds the 226th regression: an atomic
progress rename may remove its temporary entry while the watchdog scans it.
Only live size sampling tolerates ENOENT; the final hashed inventory remains
strict. Symlinks and nonregular artifacts remain rejected in both cases.

The first full launch was interrupted prospectively for that correction. Keep
its log/source binding and the teardown `tmp_path` stash KeyError after SIGINT;
no final pytest count/wrapper-exit receipt exists, so it remains unavailable.
Only the controller and its new test change relative to that candidate. Its
exact source/notebook packet remains private. The final 226-test focused run
takes 143.40 seconds. The fresh 482-input full gate runs in
`steganography-colab-training-final-qa-20261107.service`; its result must be
recorded before any physical action. Preserve all intermediate lint/type evidence.

Prepared components are notebook/source delivery, model/data/optimizer runner,
progress, numeric continuation QA, independent audit, role-scoped transport and
profile-bound approval/budget enforcement. Physical T4 compatibility/cost and
native accuracy remain unavailable until their separately permitted executions.
The old cache's unavailable final-memory acceptance, every historical result,
reserved image, transfer ledger and consumed GPU allowance remain unchanged.

The final full gate passes **4,362 tests /two explicit Kali-CUDA skips**, 32
warnings and **97.220138%** total statement coverage. Fitter/controller/handoff/
CLI coverage is 99.218750% /96.359223% /97.260274% /95.744681%. Ruff,
251-file mypy and `git diff --check` pass. All 482 frozen input hashes match;
all original 466 inputs remain unchanged. Pytest/wrapper take 2,175.47 /
2,177.758398 seconds. The successful QA unit is inactive /PID zero /zero
restarts. Its final post-exit MemoryPeak and independent kernel interval remain
unavailable after unloading; no rerun reconstructs them. Keep the first
observer and interrupted original command. See
[`png-bmp-colab-training-verification-20261107.json`](../benchmarks/png-bmp-colab-training-verification-20261107.json).

Preparation is ready for a separately authorized physical profile. If its
measured cost is eligible, the same frozen source can immediately prepare the
exact fitting proposal for later approval; no implementation work is left
between these stages. A measured budget failure remains ineligible, and does
not authorize a larger allowance or another attempt.

## Later physical outcome

The separately authorized profile is now consumed. Its command and numeric
evidence verify, but the operator session closing target was missed. Accounting
for that complete session makes the current cumulative fitting time ineligible.
See [the physical result and budget observation](PNG_BMP_COLAB_PHYSICAL_PROFILE_RESULTS_20261107.md).
The possible 2,700-second successor remains non-executable; this protocol and
the current validator retain their original allowance. No fitting approval,
budget reset or profile retry follows from the measurement.

The subsequent [distinct cumulative v2 amendment](PNG_BMP_COLAB_CUMULATIVE_AMENDMENT_20261107.md)
prepares a separate reviewable 2,700-second proposal while preserving this v1
contract and every prior debit. It still requires later explicit fitting approval.
