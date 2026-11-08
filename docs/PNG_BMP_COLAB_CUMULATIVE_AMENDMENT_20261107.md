# Colab RGB cumulative resource amendment — 2026-11-07

Preparation snapshot: **ready_for_explicit_fit_approval**. The later user
approval and consumed deployment are recorded in the
[deployment result](PNG_BMP_COLAB_NATIVE_FIT_DEPLOYMENT_RESULTS_20261107.md).
The 150-second setup reservation failed; this is not a fresh launch allowance.
The user's continuation permits preparing the proposed 45-minute cumulative
allowance. It does not approve an unreviewed fitting contract. The previous
1,800-second contract remains unchanged and ineligible; the new schema is
`colab-rgb-fit-data-v2`, amendment `colab-rgb-cumulative-2700s-20261107-v1`.

## Exact reviewable proposal

- [Frozen fit contract](../benchmarks/png-bmp-colab-amended-fit-contract-20261107.json)
- [Disabled fit-only notebook](../notebooks/png_bmp_colab_amended_fit_20261107.ipynb)
- [Matching source packet](../notebooks/png_bmp_colab_amended_fit_sources_20261107.json)
- [Unapproved permission template](../benchmarks/png-bmp-colab-amended-fit-permission-template-20261107.json)
- [Consumed profile metadata](../benchmarks/png-bmp-colab-consumed-profile-metadata-20261107.json)

Fit contract SHA-256:
`7a45cab6774388e9eb156af90eb4c10c962f522a6a2ad6678853fb88e65d7b94`.
Source packet SHA-256:
`a465948b02d85321f802535143ddccb0a0d781f7bb8f029b8b99427013b951f1`.

The candidate stays two seeds /twenty epochs, RGB 256×256 IEEE FP32, batch eight,
640 updates /5,120 presentations and eight numeric states. Its unchanged
64/20/21 grouped overlay retains 38/12/19 ancestry components. Actual roles
remain unassigned; reserved 21 RGB and all 13 L pixels stay closed. The old
cache's unavailable final-memory acceptance and all acquisition budgets remain.

## Complete cumulative ledger

| Item | Seconds | Bytes |
| --- | ---: | ---: |
| Immutable consumed profile, including closing | 79.297540 | 234,181,832 |
| Earlier external session/handoff charge | 950 | 67,108,864 reserved |
| New setup, export and runtime-deletion reservation | 300 | 4,194,304 reserved |
| Total charged/reserved before the new command | 1,329.297540 | 305,485,000 |
| New command allowance | **1,370.702460** | **499,821,368** |
| Cumulative cap | **2,700** | **805,306,368 (768 MiB)** |

The earlier 950-second charge preserves 890 seconds of session overhead and
60 seconds of conservative CPU handoff/audit allowance. These are not reduced
to the 0.581940-second independent numeric replay. The old 600-second operator
target was missed; this amendment retains that failed result permanently.
It grants no profile retry or historical resource pass.

The new 300-second reserve covers at most 150 seconds before fit launch and
150 seconds for evidence export/runtime deletion afterward. An initial
unexecuted draft reserved 180 seconds; review added explicit checkpoint-export
and closing time before freezing this candidate. Its old source/delivery
snapshot stays private. Software development/CPU QA is separate from this
GPU-intervention reservation and creates no learning exposure.

The measured conservative fit estimate is 1,311.131131 seconds, including
development scoring every epoch. The validator requires an additional
30-second margin; the new proposal retains **29.571329 seconds beyond that
margin**. This is eligible but narrow, not a guaranteed completion time.
Eight checkpoints/metadata reserve 484,410,704 bytes, leaving 15,410,664 bytes
of output margin. The fixed command's twenty-second closing reserve, per-file
128-MiB cap, four-GiB Torch allocator guard and provider-managed RAM policy
remain. No whole-driver VRAM, whole-tree RAM or hard cgroup zero-swap claim
is added. No global process limit is raised: the remaining command is below
the existing 1,800-second per-process maximum.

## Reusing the consumed measurement

The metadata packet is 76,080 bytes; its SHA-256 is
`58d314622b77d7f79c674c3b947ea47895ab04693380a06d51a44b9ac548465e`.
It contains the byte-exact old contract, closing receipt, independent audit,
external ledger and original fit source. Each receipt is pinned to its known
digest. In particular, the accepted closed-profile hash remains
`326b09839a7c57b22652305d98fc24a998c256ff65c8239e38c2db1dbeb5c448`.
No pixel, numeric model or original artifact is included, executed or regenerated.
No 234-MB output tree needs uploading to another runtime.

The resource service verifies all 215 unchanged sources in the old 217-file
application closure. Only the fit contract/proposal/approval adapter and thin
CLI change; one resource service is added. The original fit source hash is
checked against the original contract. AST comparison additionally preserves
every fit definition outside the five resource/approval adapter functions,
including model execution, loss/scoring, RNG, payload, optimizer and independent
training audit. Original source is parsed as data, never imported/executed.
The new contract independently freezes the complete current source closure.

The existing controller and one-fit-per-profile marker remain unchanged.
Both parent and child verify the metadata/source/debit binding before native
work. Changing a fit contract cannot reuse an already consumed fit marker.
The new v2 permission must bind the exact contract/profile and explicitly
acknowledge the amendment and 2,700-second cumulative cap. The v1/profile
permissions are rejected. The delivered template has `user_approved: false`;
no production fitting permission is created by this preparation.

## Later deployment requirements

The notebook has no profile-launch route. Its default CPU cells install
reviewed source, verify metadata and reconstruct the exact proposal without
Torch, models, media or an attempt marker. Every fit/export switch remains
false. Train/development uploads require later v2 fitting permission; no
development packet is generated now.

For a separately approved GPU launch, record the allocation request independently
and supply its observed age at first bootstrap. Missing/nonfinite age is
rejected; measured age plus elapsed setup must stay below 150 seconds both
before media upload and before execution. Otherwise close the runtime and
retain the unavailable deployment evidence; no automatic retry is authorized.
The external observer must also charge export/deletion and retain any closing
overrun. Kernel termination bounds the command, not the notebook UI or entire
provider allocation. A later account/session allocation remains unverified here.

After software acceptance, approval must cover this exact recipe, contract,
preserved failed session target, complete cumulative ledger and unavailable
hard-memory guarantees. This document creates no GPU allocation, pixel upload,
native optimizer update, detector qualification or deployment.

## Software acceptance

The corrected complete command passes **4,396 tests /two explicit Kali-CUDA
skips**, 32 warnings and **97.225442%** total statement coverage. The resource
service has 100% coverage; fitter/controller/handoff/CLI have 99.256506% /
96.116505% /97.260274% /95.833333%. All 260 focused checks and the corrected
34 metadata/notebook checks pass. Ruff and 252-file mypy pass. All 522 frozen
input hashes match; 480 of the previous 482 inputs are unchanged, with only
the declared fit/CLI adapters modified.

Pytest/wrapper take 1,400.59 /1,402.060753 seconds. Live kernel policy verifies
eight-GiB RAM /zero swap for this CPU software gate only. Final post-exit
MemoryPeak is unavailable even with the retained exited unit: the closing
observer reports `[not set]`. Preserve the 270 observations and maximum live
sample of 8,589,934,592 bytes as a nonfinal sample, never a final peak. Current
unit timestamps are zero and its independent kernel interval is unavailable;
do not compare current and earlier monotonic clock domains. The unit is now
inactive /PID zero /zero restarts. No rerun reconstructs missing observations.
None of these CPU QA limits changes the native provider-managed RAM policy.

Preserve the first 22-pass/ten-failure focused command (mutable fixture debit;
the adapter now copies that input), the corrected 34-pass command whose global
focused coverage gate still failed, the initial lint failures and the original
interrupted full command. SIGINT was absorbed by a tested supervisor, producing
a test failure; the scoped QA unit was then killed, with no final pytest count.
That command remains unavailable. The successful command above is separate.

See [verification](../benchmarks/png-bmp-colab-cumulative-amendment-verification-20261107.json)
and [component status](../benchmarks/png-bmp-colab-cumulative-amendment-preparation-20261107.json).
The exact contract, rejected-by-default template and disabled notebook are
ready for review. Actual fitting permission and a future account allocation
remain missing/unavailable; no real-media pixels were opened by this amendment.
