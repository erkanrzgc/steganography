# Native RGB: separate data size from update budget — 2026-11-08

Status: **metadata preparation verified; proposed experiment is not executable**.
This continuation prepares the next comparison after the
[completed diagnosis](PNG_BMP_NATIVE_RGB_DIAGNOSIS_20261107.md).
It does not run another fit or reuse an old execution permission.

The [source-bound proposal](../benchmarks/png-bmp-native-data-update-study-proposal-20261108.json)
specifies every source, ancestry component, model seed, order, endpoint and
exposure count. Its [independent metadata verification](../benchmarks/png-bmp-native-data-update-study-verification-20261108.json)
replays the plan without Torch, pixels, GPU or network access.

## Comparison

| Training originals | Whole ancestry components | Update endpoints, each seed | Complete corpus passes at endpoints |
| --- | ---: | --- | --- |
| 8 | 6 | 320 and 1,280 | 160 and 640 |
| 64 | 38 | 320 and 1,280 | 20 and 80 |

Both historical model seeds, 2026102301 and 2026102302, initialize fresh models.
For a given seed and training size, the 320-update endpoint is saved inside
the same fresh 1,280-update trajectory. It is not a second fit. There are
**four fresh trajectories /eight endpoints /5,120 total updates /40,960
training presentations** in the proposed comparison.

Architecture, RGB 256×256 raw FP32 inputs, batch eight /four cover-stego pairs,
scalar AdamW at 1e-4, payload seed 2026102805 and disabled AMP/TF32 stay fixed.
The fresh control changes data size and update horizon, rather than tuning the
optimizer or switching architectures at the same time. None of the historical
numeric states initializes or resumes a proposed model.

The eight-original subset is nested in the existing 64-original train overlay.
Whole ancestry components are selected deterministically by metadata-only
ranking and exact subset sum, using selection seed 2026110801. Pixel values,
losses, old score arrays and checkpoint replay do not select the subset. Its
six components comprise two two-image groups and four singleton groups.
Actual source records remain unassigned.

Within each complete corpus pass, IDs are sorted by a model-seed/pass/ID hash.
Shared sources have the same relative order in both sizes. Each four-pair batch
contains distinct originals, and every original is visited once per pass.
The proposal binds all four order hashes; the complete prospective schedules
remain private. Initial parameter hashes must agree between sizes for each
seed in the eventual runner; this preparation does not create initial models.

## Measurements and interpretation

At initialization and updates 320/1,280, preserve full train and existing
twenty-original development scores. On small-subset models, also score the
other 56 originals from the historical training pool. Those 56 are disjoint
in known ancestry from the eight selected originals. They are an additional
fresh-model generalization diagnostic, not a new independent test set: the
previous completed experiment already fitted them.

The proposal totals **2,016 inference rows**: 1,344 train/development rows and
672 training-pool holdout rows. Eight numeric states bind the two endpoints
per trajectory. Record every update loss/count, finite gradient norms, actual
resource/output debits and all scalar scores for independent AUC replay.
Matched-pair ordering remains secondary and cannot replace global AUC.

The tiny-set memorization target is train AUC ≥.95 in both seeds at update
1,280. If it fails, data shortage alone does not explain the failure; investigate
representation/optimization before collecting more images. If the tiny set
is learned but common development remains weak, investigate generalization.
Compare 320 versus 1,280 within each size to assess longer optimization, and
compare eight versus 64 originals at the same update endpoint to assess the
larger-data intervention. Preserve both seeds and every endpoint regardless
of the result; do not select a winning seed, subset or checkpoint afterward.

The smaller subset repeats each original eight times more at equal update
counts and has different source composition/diversity. This one nested subset
cannot establish a universal pure sample-count effect. The repeatedly observed
development set also remains exploratory. No result from this comparison alone
would qualify a production detector.

## Cost and execution gap

Scaling the old measured 569.256510-second /640-update whole command by eight
and an illustrative factor 1.5 gives **6,831.078114 seconds /113.85 minutes**.
That old command includes different scoring/checkpoint overhead. New code,
integrated I/O, setup/transfer/export and the currently available Colab GPU
remain unmeasured. This is a planning proxy, not an accepted cost measurement,
runtime promise, credit estimate, time limit or spending permission.

The existing native adapter caps a plan at twenty epochs. The proposed small
trajectory needs 640 passes and the large one eighty. Do not monkey-patch its
plan, replace its schedule after validation or weaken the old contract. A
distinct step-bounded runner and supervisor/auditor are still missing. Before
any execution they need generated-only QA, the full software gate, a new source/
environment freeze, integrated cost measurement and a separately concrete
execution/resource contract. This document creates none of those permissions.
The user's Colab backend preference and no-local-GPU preference remain active.

## Preserved boundaries and validation

All 118 source metadata records and their transitive ancestry graph are replayed
independently with graph traversal rather than the service's disjoint-set
implementation. The independent check verifies the nested 8/64 sets, closed
partitions, all four 5,120-pair schedules, paired-batch uniqueness, matched
relative order, eight endpoints and every proposed count. It opens no pixels
and creates no optimizer or attempt marker.

The twenty-one reserved RGB /thirteen L originals stay closed, and the metadata
proposal does not open development pixels. No acquisition, cache regeneration,
new source role, model deployment or historical debit reset occurs. All 522
historical software inputs remain unchanged. The old fit's failed gate and its
closed GPU runtime stay preserved. The separate checkpoint-trajectory replay
is CPU inference on the already fitted 64 train originals only; its actual
results are reported independently from this non-executable proposal.
