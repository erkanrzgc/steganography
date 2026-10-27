# Native SRNet training draft — 2026-10-23

**Draft, not executable.** The current continuation completes source review
and a small admission pilot. It does not turn the consumed capacity profile
into training permission. There is no native deep-model fit adapter or
validated learning recipe yet. No optimizer update, new GPU measurement or
additional acquisition runs from this draft.

The [actual T4 capacity run](PNG_BMP_COLAB_CAPACITY_RESULTS_20261023.md)
established that both existing model skeletons complete synthetic 256×256
FP32 forward/backward at batch 8. It did not establish learnability,
optimizer memory, data throughput or complete training time. The
[native pilot](PNG_BMP_NATIVE_ADMISSION_RESULTS_20261023.md) now contains
**three L and three RGB originals**, in six conservative author groups.
Those six examples establish pipeline functionality only.

## Proposed data and exposure

For a small exploratory training comparison, target **128 originals per
domain**, 256 total, with at least sixteen reviewed ancestry groups per
domain and eight originals per selected group. The pilot supplies three
originals per domain; **125 L and 125 RGB originals are still missing** from
that inventory target. This arithmetic does not qualify the pilot examples
for fitting: two fixed L crops contain little texture, so their eventual
eligibility and the final acquisition requirement remain unresolved.
This is a proposal for an initial learning experiment, not a claim that 128
images suffice for representative detection or deployment.

| Proposed partition per domain | Ancestry groups | Originals | Cover/control image instances |
|---|---|---|---|
| Train | 10 | 80 | 160 |
| Development | 3 | 24 | 48 |
| Future reserved | 3 | 24 | 48 |

All actual roles remain unassigned. Freeze exact source hashes, common-origin
links and a globally consistent group assignment before implementation.
An author represented in both domains must have the same partition; a
shared upstream source or uncertain near duplicate must merge, never cross
partitions. Existing reserved datasets remain closed. PNG/BMP encodings of
one crop are paired transport variants, not extra independent samples.

Retain native channel count, the fixed top-left 256×256 crop and the 0–255
FP32 scale. Do not make L sources from RGB, remove alpha, resize, or import
BOWS2/DIV2K under this draft. Require per-original license attribution and
verified SHA-1/SHA-256/extent, native depth, whole-origin review and exact
independent payload recovery. Capture/processing history and cover
cleanliness require explicit limitations; a Commons license does not prove
a pristine camera source.

The current 32-byte sequential controls remain admission fixtures. A future
learning recipe must separately bind payload rate, positions, random payload
generation and an independent decoder. A candidate high-payload positive
control is 40% of available plane-0 samples, rounded to complete bytes and
scattered with a fixed recorded permutation. That generator is not implemented
or approved here; no learning result follows from the admission fixtures.

Proposed exposure is two models × seeds 2026102301/2026102302, batch eight
images (four original/control pairs), twenty epochs without early stopping,
sweeps, automatic retry or silent batch fallback. With eighty training
originals, this is twenty steps per epoch, **400 updates per fit /1,600 total**
and **12,800 training image presentations**. Learning/evaluation gates and all
evaluation exposures must be frozen before an executable contract.

AdamW at learning rate 0.0001, betas 0.9/0.999, epsilon 1e-8, zero weight decay,
`foreach=False` and `fused=False` is an explicit **proposal**, not a validated
SRNet recipe or a reproduction of the original paper. Its interface is
documented in [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
The original [SRNet publication](https://ws.binghamton.edu/fridrich/Research/SRNet.pdf)
provides architectural context; its published training/accuracy claims are
not transferred to these skeletons or this small corpus. Require a separately
bound generated learning control and numeric resume validation before any
native fit proposal becomes executable. Keep AMP/TF32 disabled.

## Cost and proposed ceilings

Multiplying each observed synthetic batch-8 median by 400 gives only the
following arithmetic projection:

| Model | Seed | Observed forward/backward (ms) | 400-pass arithmetic (s) |
|---|---|---|---|
| Gray | 2026102301 | 192.763 | 77.105 |
| Gray | 2026102302 | 192.439 | 76.976 |
| RGB | 2026102301 | 191.379 | 76.551 |
| RGB | 2026102302 | 192.886 | 77.154 |

The sum is **307.787 seconds of projected forward/backward only**. Actual
training inputs, optimizer steps/states, evaluation, decoding, validation,
checkpoint I/O, startup and Colab compute units are unmeasured. This sum is
neither a full-training estimate nor a guaranteed lower bound. The measured
1,608-MiB allocated VRAM peak also excludes optimizer states; it cannot
certify that a complete AdamW batch-8 step fits.

Proposed future hard ceilings are 1,800 seconds for the complete experiment,
512 MiB for numeric outputs and a separately source-bound 512-MiB maximum
for any next raw acquisition. These are **unapproved resource ceilings**,
not measured cost estimates or permission to spend the remaining old
allowance. Preserve the current 55,307,394-byte conservative transfer debit
and incomplete historical total. No historical JPEG/WAV budget is reused.

Before any real fit, complete the target corpus and group review; implement
and verify the payload/training/evaluation/resume adapter; freeze its whole
source/data/exposure closure; and measure full-step/evaluation/I/O costs
under a distinct generated contract. Then freeze an actual total budget and
review the concrete executable experiment. Keep the completed zero-update
capacity profile and every failed acquisition unchanged.

The [machine-readable draft](../benchmarks/png-bmp-native-training-draft-20261023.json)
retains quantities, arithmetic and missing costs explicitly.
