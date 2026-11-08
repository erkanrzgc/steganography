# Colab process preparation — 2026-11-06

Historical v1 snapshot: the later, separately sourced native-profile extension
is documented in [its proposal](PNG_BMP_COLAB_NATIVE_PROFILE_PROTOCOL_20261106.md).
The executed v1 notebook/source packet and all receipts below remain unchanged.

The generated CPU process controller is implemented in `core/colab_execution.py`.
This completes the process-control software slice identified by the portability
review. It does not complete the native SRNet Colab adapter or freeze a new
native profile/fit protocol. The only registered workload is `generated_qa`;
`native_profile`, `native_fit` and arbitrary commands are rejected.

The thin `steganography.research_colab_execution` CLI exposes `freeze`, `run`
and `audit`. The CPU notebook uses the same service. It contains the exact
reviewed controller/artifact sources and MIT license, bound by per-file and
package SHA-256; the recorded base Git commit is
`2f41bfe2e95febcaab80f80ab4875c72073c3a32`. These additional sources are an
explicit overlay, not a claim that they already exist in that commit.
`python3 scripts/build-colab-process-notebook.py --out <fresh-directory>` reproduces the
delivery without a network request, installation or dataset/model download.

## Limits and evidence

The Linux parent launches one fixed source-verified Python worker through
`prlimit`; it never executes an extracted artifact or user-provided command.
The contract binds the complete local `core/` and `steganography/` Python
source inventory, actual Python implementation/version/platform and limits.
The delivered minimal notebook contains only three core sources and LICENSE.
No Torch import, model, optimizer, media file, GPU operation or descendant
worker belongs to this workload.

- DATA, CPU, per-file size, CORE=0 and optional address-space limits are hard
  kernel process limits. They apply before Python starts. DATA and AS describe
  data/virtual memory; neither is a cgroup resident RAM limit.
- RSS/HWM/swap counters and cumulative output are sampled by the parent. A
  watch violation or persistently unavailable observer terminates/reaps the
  exact child. Short exec/exit observer gaps are counted; they are not described
  as continuous resource coverage. Swap is observed, not hard-disabled.
- A default-disposition kernel wall timer terminates the worker even if the
  Python parent stops sampling. `PR_SET_PDEATHSIG` and a parent-identity race
  check prevent the fixed worker surviving its parent. Notebook interruption
  in the parent also kills and reaps the worker.
- `wait4` retains that child's kernel CPU time and maximum RSS after exit.
  The RSS peak conservatively includes process creation before `exec`; it
  must not be presented as a model's isolated memory requirement.
- Complete observed parent time includes source verification, spawn, sampling,
  reaping and closing hashes. A full closing reserve is additionally charged.
  Receipt/ledger writes must finish within that conservative debit and the
  command ceiling. If they exceed it, an exclusive invalidation receipt makes
  the earlier closing receipt ineligible for acceptance.

The shipped CPU smoke task allows 10 seconds, 128-MiB DATA, 256-MiB AS,
8-GiB RSS watch, 1-MiB per file and 8-MiB cumulative output. Its workload only
returns generated metadata. The relatively large RSS watch retains inherited
process-creation peaks; it does not allocate that memory. No old GPU or
transfer allowance is spent or reset.

The source-root `.colab-process-attempts/<contract-hash>.intent.json` is written
exclusively before execution. Moving to another output directory cannot retry
that contract in the same retained workspace. Interrupted and failed attempts
remain consumed. This local ledger is not durable across deletion of an entire
Colab runtime; retained receipts must accompany any later handoff. No restart,
resume or automatic retry command exists.

Output audits reject symlinks, special files, excess depth/count/size, changed
hashes, changed source/environment/protocol and an overwritten closing receipt.
Failed evidence stays `unavailable`; an unreadable inventory has unknown byte
count rather than a reported zero. A successful audit checks every output hash
and the independent exclusive ledger anchor. Raw worker logs remain private;
public receipts contain relative names and error classes only.

Linux semantics are documented by the primary references for
[process limits](https://man7.org/linux/man-pages/man2/getrlimit.2.html),
[wall timers](https://man7.org/linux/man-pages/man2/setitimer.2.html),
[parent-death signals](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html),
[child resource collection](https://man7.org/linux/man-pages/man2/wait4.2.html)
and [resource accounting preserved across exec](https://man7.org/linux/man-pages/man2/getrusage.2.html).
These controls do not repair or bypass the rejected Colab cgroup identity.
The old WSL strict RAM/zero-swap acceptance rule remains unchanged.

## Verification and remaining work

The focused generated CPU suite currently passes 61 tests. It uses actual Linux
processes for deadlines, hard memory/file limits, parent death, interruption,
failure/no-retry behavior and source-only delivery. Unit fixtures additionally
cover malformed contracts/counters, environmental/source drift, hostile output
types, tampering and over-budget finalization. New controller/CLI statement
coverage exceeds 95%. Preserve the initial coverage-import failures, transient
proc-counter failures and lint failures; later successful commands do not
relabel them. The full repository gate has a separate fresh source binding.
The fresh complete gate passes **4,197 tests /two explicit Kali-CUDA skips**,
32 warnings and **97.194842%** total statement coverage. Controller/CLI coverage
is 97.278912% /96.551724%; Ruff and 245-file mypy pass. Pytest takes 1,308.36
seconds, its wrapper 1,310.161493 seconds. The unit is inactive /PID zero with
no restart; final MemoryPeak and independent kernel interval are unavailable
after unloading. Preserve that observer instead of rerunning the gate.
See `benchmarks/png-bmp-colab-process-verification-20261106.json`.

The new controller also completes one actual Colab 2026.07 CPU execution:
`completed`, zero updates, child exit 0/reaped and 1.454632 seconds including
the conservatively charged one-second closing reserve. Its kernel peak is
107,876,352 bytes including spawn; the live sampled peak is 19,877,888 bytes
and there are zero missing-counter observations. The CPU runtime is deleted
after 167.814153 /300 observed session seconds. Its two exported code cells
match the original delivery exactly. Independent offline reconstruction
replays every preclosing output hash, the exact contract/environment and the
closing receipt. Python 3.12.13 /CPython /Linux matches the contract hash.
See `benchmarks/png-bmp-colab-process-cpu-validation-20261106.json`.

After that execution, presentation-only notebook lint repairs wrap the embedded
source literal/allowed tuple and sort two imports. The source packet remains
byte-identical, the bootstrap AST is identical and the run AST changes only
those independent imports. The related notebook/delivery test passes again.
Preserve the executed preformat notebook privately; do not describe the public
formatted cell text as byte-identical to that earlier export. The full run's
original 470-file input binding is retained; only the builder's presentation
code changes, and all 469 application/test inputs remain unchanged.

An additional paused-parent observer preserves its failed expectation that
the Python library must exit nonzero and invalidate a receipt whenever its
returned result is unavailable. The retained library result is correctly
`unavailable`, child exit SIGALRM/reaped, with conservative debit 2.064878
against two seconds; final writes finished before the actual two-second wall
ceiling, so no invalidation file is required. The CLI separately maps an
unavailable result to exit 2. Keep that first observer failure; no second
paused-parent attempt is run and its unretained kill timestamp is unavailable.

| Component | Status |
| --- | --- |
| Generated CPU process supervisor and audit | ready; focused software QA passes |
| Reproducible CPU notebook/source delivery | ready; isolated local delivery passes |
| Actual Colab execution of this new controller | ready; one generated CPU execution completed |
| Hard cgroup resident RAM and zero-swap evidence | unavailable; not claimed by this policy |
| New Colab CUDA policy and native measurement adapter | missing |
| Colab-bound integrated optimizer/evaluation/I/O cost | unavailable |
| Approved native fit/source/environment/resource contract | missing |

The next native implementation must explicitly define its resource acceptance,
reuse existing immutable SRNet/data/optimizer/numeric services, keep all real
roles unassigned and all 21 reserved RGB/13 L pixels closed. A fresh Colab
profile needs its own reviewed contract and permission; a native fit still
needs later explicit approval against measured cost. This generated-only
software does not create either permission. Existing Colab/WSL attempts,
cache acceptance, checkpoints, source bindings, holds and transfer ledgers
remain historical and unchanged.
