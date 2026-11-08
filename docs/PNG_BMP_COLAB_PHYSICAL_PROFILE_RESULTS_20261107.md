# Colab RGB physical cost profile — 2026-11-07

The separately authorized T4 command completed and its evidence independently
verified. Overall status is **completed_command_operator_close_target_missed**:
the command met its 240-second limit, while the 600-second operator session
closing target was missed. The runtime was disconnected/deleted; the ordinary
session manager subsequently confirmed no active sessions. This attempt is
consumed, with no automatic retry.

Public receipts:

- [Physical measurement](../benchmarks/png-bmp-colab-physical-profile-results-20261107.json)
- [Cumulative budget observation](../benchmarks/png-bmp-colab-native-fit-budget-observation-20261107.json)
- [Frozen preparation and recipe](PNG_BMP_COLAB_TRAINING_PROTOCOL_20261107.md)

## Execution and data scope

Runtime Version 2026.07 supplied Tesla T4, driver 580.82.07, Python 3.12.13,
NumPy 2.0.2, Pillow 11.3.0, Torch 2.11.0+cu128, CUDA 12.8 and cuDNN 91900.
The source-bound RGB SRNet used 256×256 FP32 with AMP/TF32 disabled.

Only the 64 proposed training crops were transported: 12,582,912 raw pixel
bytes in a 16,891,346-byte bounded packet. No development, reserved RGB or L
pixels were opened/uploaded. Actual source roles remain unassigned. The
original cache was not regenerated and no original media was downloaded.

There were zero native optimizer updates, 48 native backward presentations and
256 train-only native inference presentations. Native parameters were unchanged;
no native accuracy scores were recorded. Six generated optimizer updates /48
presentations produced two generated numeric states. The earlier gray/RGB
capacity measurements remain historical; this intervention measures RGB only.

## Measured resources and independent audit

| Observation | Result |
| --- | ---: |
| Worker | 37.132244 s |
| Observed outer command before closing reserve | 69.297540 s |
| Charged command including ten-second reserve | 79.297540 /240 s |
| Allocation request to confirmed deletion | 957.495231 s |
| Conservative confirmed-close upper bound | 959 s |
| Operator session closing target | 600 s — missed |
| Kernel worker peak RAM, including process creation | 1,537,445,888 bytes |
| Torch allocator allocated /reserved peaks | 1,978,490,368 /2,696,937,472 bytes |
| Complete command output | 234,181,832 /268,435,456 bytes |
| Exported verified archive | 117,088,684 bytes /34 members |

The command exited zero and was reaped. Provider-managed host RAM and observed
ten-percent kernel headroom were used; hard cgroup RAM/zero-swap enforcement,
whole-tree RAM peak and whole-driver GPU peak remain unavailable. Allocator
measurements do not establish whole-driver VRAM consumption.

The independent CPU auditor verified all 34 archive members, every closing
output hash, the external intent/close anchors, six generated intent/commit
pairs and both numeric states. Each state contains 503 arrays and three update
debits. Source, environment, data and order bindings agree. The independent
numeric result matches the original runtime audit exactly. Replay took
0.581940 seconds and imported no Torch, opened no native pixels and performed
no new updates.

The ordinary-UI notebook export preserves all nine original code cells exactly
apart from trailing whitespace, execution counts 1–9 and no error output. One
additional read-only closing cell exports the attempt ledger. An accidental
source-selection paste was corrected before contract/profile execution;
the exported cells verify that correction. It caused no profile relaunch.
Source uploads and interface handling extended the overall GPU allocation
beyond its operator target. Preserve this failure separately from the verified
command result; 79.3 seconds is not the entire GPU session duration.

## Cost projection and current budget failure

The unchanged candidate is two RGB seeds, twenty epochs each, batch eight,
640 optimizer updates /5,120 presentations. The estimate includes generated
complete-step timing, native backward/inference timing, checkpoint/I/O timing
and development scoring every epoch, using the frozen factor 1.5. It is not a
guaranteed upper bound or an accuracy result.

| Model /seed | Updates | Presentations | Projected fit including epoch development |
| --- | ---: | ---: | ---: |
| RGB SRNet /2026102301 | 320 | 2,560 | 605.722039 s |
| RGB SRNet /2026102302 | 320 | 2,560 | 705.409092 s |
| Total | 640 | 5,120 | 1,311.131131 s (21.85 min) |

The initial/final-only base estimate is 1,152.748793 seconds; the nineteen
additional development passes per seed add 158.382338 seconds.

The unchanged nominal cumulative allowance is 1,800 seconds /768 MiB. Charge
890 seconds for GPU-session time outside the observed command, conservatively
rounded from the 959-second confirmed-close bound, and 60 seconds for local
handoff/read-only evidence work. The latter is an explicit conservative debit,
not the measured numeric replay duration. Reserve a separate 64 MiB for
external transport/evidence. These charges reset no previous ledger.

Including the immutable 79.297540-second command debit gives 1,029.297540
seconds consumed and **770.702460 seconds remaining**. The 1,311.131131-second
fit estimate plus closing cannot fit: **current_time_eligible = false**.
Storage alone remains eligible: 504,015,672 bytes remain against a
484,410,704-byte checkpoint/metadata reservation.

A possible **2,700-second cumulative /768-MiB** successor would retain every
debit, leaving 1,670.702460 seconds before any future setup charge. This is
only a non-executable budget observation. The current validator requires the
original 1,800-second cumulative contract. A source-compatible resource
amendment and later explicit fitting approval are required; neither an
executable replacement contract nor a fitting permission exists. Future
deployment/upload/closing work still needs debiting. No new profile is
authorized to replace the consumed measurement.

## Bindings and preservation

| Binding | SHA-256 |
| --- | --- |
| Source packet | `8677a46f3fb7840b01b46b2b1355a286c9d30296ccf271519fe54bc57d8e300a` |
| Profile contract | `37d2f589a90ac29dcafea3fc238217022597acc3ea084d1b1f8eb6273a35a33b` |
| Closed profile receipt | `326b09839a7c57b22652305d98fc24a998c256ff65c8239e38c2db1dbeb5c448` |
| Verified archive | `57a49300c596e4f8b63a1617bff7132d52cfe2535ab24f742c02a960dc81f092` |
| Cache metadata | `8da0ac4ac98b7b0e9e3613b02b58c3cb076adb9456c74ef29c90f6a70ee1dfd2` |
| Canonical proposal split | `852b36d9bfee693c3be175ab5365d160a3e9d5c6da08e9fa1a2a0add552abefa` |

All 482 frozen application/test hashes remain unchanged from the passing
4,362-test software gate /two explicit Kali-CUDA skips /97.220138% coverage.
This execution adds evidence and documentation; it does not require another
long software gate. The independent physical audit is separate verification.
Private generated states, pixel packets, raw notebooks, screenshots and account
identifiers remain private. Public receipts contain hashes and bounded results.

The old cache's unavailable final-memory acceptance, all historical attempts,
source holds/exclusions, closed reserved data and exhausted acquisition ledgers
remain unchanged. Native fitting, qualification and deployment are not approved
by this completed profile. The user's preference against local GPU load remains.

The later [cumulative v2 preparation](PNG_BMP_COLAB_CUMULATIVE_AMENDMENT_20261107.md)
implements the separately proposed resource schema. This measurement and its
failed operator target stay unchanged; the successor needs explicit fit approval.
