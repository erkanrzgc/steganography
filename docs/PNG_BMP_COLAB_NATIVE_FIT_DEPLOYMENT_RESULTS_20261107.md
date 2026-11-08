# Colab approved native-fit deployment — 2026-11-07

Status: **unavailable_prelaunch_setup_reservation_exhausted**.
The user explicitly approved the linked 2,700-second cumulative contract,
two seeds /twenty epochs /640 updates. That approval was recorded and verified.
One T4 allocation was requested; the 150-second prelaunch reservation proved
too short for provisioning and the ordinary-UI source/data uploads. The second
setup guard rejected execution. This is an unavailable deployment, not a
learning-gate failure or successful fitting run.

See the [independent result receipt](../benchmarks/png-bmp-colab-native-fit-deployment-results-20261107.json)
and [original approved resource amendment](PNG_BMP_COLAB_CUMULATIVE_AMENDMENT_20261107.md).

## Observed execution

The actual fit contract remains
`7a45cab6774388e9eb156af90eb4c10c962f522a6a2ad6678853fb88e65d7b94`;
the source packet remains
`a465948b02d85321f802535143ddccb0a0d781f7bb8f029b8b99427013b951f1`.
All 522 inputs still match the passed 4,396-test software gate. No service,
model, optimizer, schedule or source-bound fitting limit was changed.

Runtime Version 2026.07 /T4 was selected through the ordinary UI on the intended
existing account. No NVIDIA query, fresh capacity profile, package installation,
local GPU use or original-media acquisition occurred. The fitting worker never
ran, so its complete CUDA environment, allocator peaks and final RAM peak are
unavailable. There are zero native model/tensor operations requested by the
notebook, zero optimizer updates, zero new numeric checkpoints and no native
accuracy result.

The first bootstrap's conservatively charged allocation age was 81.929555
seconds, including a one-second clock allowance. The source upload and proposal
reconstruction completed, followed by verification of the already-approved
private permission. Only then were the existing 64-source train packet and the
new, permission-gated twenty-source development packet uploaded. Their hashes
and all 84 raw cache entries were checked before the second time guard.
No original cache was regenerated; the working cache was an ephemeral verified
copy. Actual roles stay unassigned, with reserved 21 RGB and all 13 L pixels
closed.

| Transport | Bytes | SHA-256 |
| --- | ---: | --- |
| Existing train packet, 64 sources | 16,891,346 | `1773c2991478e40718ac6de37e6705141e7a795d02a8fb80b029dc8173d58327` |
| Approved development packet, 20 sources | 5,246,282 | `f359a4cb4460b17fb75e52a74da566e2b884d406f4ad8b3efe9a4f651b7c5ae6` |

The guard's exact error is `prospective GPU setup reservation exhausted; close
runtime`. It occurs after both install functions, before the launch-timing
print, output-directory assignment and controller call. Its exact elapsed
argument was not printed and remains unavailable. The notebook export contains
no launch marker. The first four cells have counts 1–4; export/closing cells
were never executed. No worker intent/closing receipt or training output exists
for this deployment.

## Closure, debit and preservation

The GPU runtime was disconnected/deleted, confirmed within 252.836224 seconds
of the independent allocation request; a conservative closing upper bound is
254 seconds. The account session manager subsequently reports **No active
sessions**. Preserve the setup-target failure even though deletion completed
within the separate total 300-second external reservation.

Charge that entire 300-second reservation. The cumulative profile/external
debit therefore remains **1,329.297540 seconds**, with the existing
**305,485,000-byte** output reservation unchanged. No old debit is reduced,
reclassified as unused, or reset. The nominal 1,370.702460 seconds left for the
unentered command does not authorize a replacement allocation or retry.
This deployment attempt is consumed. Another intervention needs a separately
frozen deployment/resource contract and user approval, preserving this debit.

The ordinary-UI exported notebook was independently compared to the prepared
private execution copy. Colab's **Show/hide all code** command added one
`# @title` comment to each code cell. The initial literal comparison failed
on that presentation annotation; removing only that first-line comment gives
an exact source match, and all six cell ASTs match independently. Preserve the
first mismatch rather than claiming literal identity. Neither the service
packet nor the approved contract changed.

The deployment code used the same source/data/optimizer services and introduced
only operational inputs: independent allocation time, an already-approved
permission, fixed train/development hashes and read-only closing export. Its
first three cells were also checked in an isolated local process without Torch,
pixels, a model or an attempt marker. That check is software QA, not a cloud
execution pass. No long suite was repeated after this document/evidence-only
closeout; all 522 tested hashes and `git diff --check` verify independently.

Keep screenshots, notebook account/Drive identifiers, actual permission and
native pixel packets private. The public receipt contains hashes and bounded
observations. Preserve every earlier trial, data hold/exclusion, source licence,
cache acceptance and exhausted transfer/metadata ledger.

## Correction needed before another allocation

The operational reservation was underestimated. Future preparation must account
for provisioning, actual browser upload, evidence export and runtime deletion
before fitting permission is exercised. Reusing the same 150-second window or
excluding elapsed provisioning/upload time would repeat or hide this failure.
A revised plan must also charge this consumed deployment; the original
2,700-second proposal is not a fresh allowance. No larger budget, replacement
allocation, reduced recipe or model change is authorized by this closeout.
