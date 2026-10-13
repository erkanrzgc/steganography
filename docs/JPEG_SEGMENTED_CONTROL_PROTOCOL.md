# Bounded JPEG control execution v1

`bounded-half-v1` is an explicit new execution policy, not a reinterpretation
of the failed monolithic trials. Its configuration and independent audit bind
the complete current source closure. Historical receipts and snapshots remain
unchanged; importing external resumes into this policy is rejected.

## Unchanged learning experiment

Run all four arms across both frozen seeds, using the same original identities,
source-balanced pairs, five-view cycle, D4 orientations, initialization,
optimizer equations, L2, FP32 IEEE policy and 32,768 presented image-view rows.
Keep the existing independent audit, singleton evaluation, bootstrap and
learning gates. No reserved-source evaluation, automatic export or deployment.

## Explicit execution budget

The former monolithic default remains four hours for controls, 1,800 seconds
per trial. The new controls phase is capped at six hours, inside the originally
approved eight-hour overall study envelope: it does not additionally authorize
the former two-hour baseline and two-hour adaptation stages. Those later stages
are not implemented or started by this command. No paid compute is authorized.

Each logical trial is two isolated 16,384-row halves. Each worker retains CPU
limits 1,800/1,801 seconds, internal wall deadline 1,800 seconds, parent timeout
at most 1,830 seconds, core dumps disabled and file-size cap 96 MiB. The entire
service retains an 8 GiB kernel host-memory cap, two compute threads and a
4 GiB CUDA allocator cap. The service watchdog is at most 21,720 seconds,
allowing 120 seconds for bounded shutdown; useful controls stop at 21,600.
Record both the logical aggregate and each worker's actual duration/exit.

## Safe checkpoint continuation

The first process stops at row 16,384, verifies its fit cache and writes one
fresh numeric-only snapshot and a `checkpointed` receipt. It never opens
development pixels or emits evaluation scores. The second resumes only this
exact owned snapshot, with SHA-256, recipe/source/input/Torch binding and
validated model, BatchNorm and optimizer arrays. Its schedule is the unchanged
full deterministic schedule, starting at row 16,384, not a fresh permutation.
Noncapturable PyTorch Adamax step scalars stay on CPU as in uninterrupted
training; reference-Adamax steps stay with parameters. No optimizer reset,
skipped updates, lowered exposure or looser numerical tolerances.

Only a complete second half plus original development evaluation may yield a
completed logical trial. Parent checks recipe/source/initialization/binding,
exact half/full row and update counts and each bounded snapshot checksum.
Aggregate cross entropy is the equal-update weighted mean of the two halves.
Timeout, cancellation or resource failure keeps partial evidence but leaves
the logical trial unavailable/cancelled and ineligible for learning gates.
No checkpoint overwrite, symlink following, pickle or arbitrary code loading.

## Verification gate before fresh real fitting

Require CPU prefix-schedule, continuation, rejection and cancellation tests;
complete lint/type/regression checks; actual CUDA eight-update bitwise parity
against the archived validator; and measured whole-update timing. A half's
conservative estimate is `seconds/update * (16384 / rows) * 1.25 + 60` and must
be below 1,800 seconds. Generated probes establish execution only, not learning.
The measured monolithic reference estimates above 1,800 stay ineligible under
their old policy; do not change the safety factor to turn those into passes.
Re-audit every cached view against the new source closure, then use a fresh
own-GPU job directory and preserve all earlier incomplete matrices.

Also verify physical CUDA numeric continuation through a generated full-shape
snapshot: all model/BN/optimizer arrays, step placement and four subsequent
updates must match the unsaved control bitwise in every arm. This fixture uses
explicitly generated counters/slots to exercise the snapshot endpoint schema;
it is not evidence of 16,384 real training rows or a completed learned model.
