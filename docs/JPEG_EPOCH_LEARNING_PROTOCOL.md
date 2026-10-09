# First complete multi-source GPU epoch-chain pilot — preregistered

Explicit user authorization: train on the already privately imported audited
ALASKA2 and BOSS corpus, using the actual laptop GPU. This pilot is separate
from all prior failed single-job/one-epoch/tiny-subset experiments. Preserve
those results. Do not change architecture, preprocessing, thresholds, schedule,
epochs or optimizer after inspecting pilot results.

Bind index 89f4c726922eb22e3c2edefeb24bf7d2a90473c4f9695a419be642df05fe76fd
and independent audit
00c2a19067efa78fd9d2b214a6452d9e606c8a169eb54f9ae2c6521f807b40b8.
Use all 7,380 train rows / 1,638 train lineages, both complete declared sources,
unchanged unrounded Y/256/phase-zero preprocessing and per-epoch balanced
source/quality/method four-row schedules, seed 20261008, exactly five epochs,
3,288 updates each / 16,440 total. No validation pixels during fit. No WIFD
origin or other reserved identities. Full plan binds all schedules, runtime,
decoder, 16 sources, this protocol and fresh physical resume-probe checksums.

Use explicit cuda:0, two CPU threads, unchanged Adamax learning rate 0.001,
weight decay 0, IEEE float32, deterministic execution, no TF32/mixed precision.
Require the frozen generated resume-parity gate first. Each epoch is a fresh
isolated job capped at 1800s / resident host 8 GiB / CUDA allocator 4 GiB.
Verify all train cache bytes before and after fitting. Publish an epoch card
only after complete updates, immutable source checks and deadline checks.
Save model plus exact numeric optimizer/RNG checkpoint, never weights alone
for continuation. Subsequent jobs bind the previous completed card and state.
Stop on any failed epoch; no automatic retry, truncation or partial publication.

Report every complete epoch's loss and duration. Loss decline is optimization
evidence, not detection accuracy. After epoch five, evaluate stored-BN inference
without normalization refresh, paired batch context or test-time adaptation.
Use all 1,611 existing lineage-disjoint development validation rows once,
fixed probability threshold 0.5, per-source/quality/method ROC-AUC, balanced
accuracy, recall and FPR. Do not choose thresholds/epoch from these results.
These same-source development results are not cross-source qualification;
the 1,000-cover + 1,000-stego / two-independent-source support gates remain.
Any absent final evaluation must be explicitly pending/unavailable.

The pilot may fail to learn: publish failure without claiming improvement.
Five epochs are a bounded initial learning diagnosis, not promised convergence.
Further exposure or architecture changes need a new train-only protocol, not
validation-guided tuning. Publish portable aggregate evidence only; raw corpus,
restricted data and checkpoint/model weights remain private and out of wheels.
