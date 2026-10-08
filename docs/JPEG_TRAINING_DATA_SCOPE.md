# Current prepared JPEG training scope — 2026-10-07

Subsequent 2026-10-08 preparation completed a **separate new** block index:
7,380 train rows (2,448 ALASKA / 4,932 BOSS) over 1,638 original train lineages,
plus 1,611 validation rows. See [complete results](JPEG_SCALE_PREPARATION_RESULTS.md).
These are not merged into the old manifest below, and no new fit occurred.
The new index requires a separately reviewed streaming trainer; the capped
legacy whole-array interface is unchanged. Counts below remain historical.

Manifest SHA-256
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`;
train float-cache descriptor SHA-256
`828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028`.
Counts are a read-only metadata audit, not new detection results.

| Declared source | Train files | Cover/quality groups | Declared original scene lineages |
| --- | ---: | ---: | ---: |
| ALASKA2 | 2,367 | 789 | 789 |
| BOSSbase-1.01 | 618 | 206 | 103 |
| Total | 2,985 | 995 | 892 |

ALASKA has 789 cover, 789 JUNIWARD and 789 UERD training rows; quality and
payload rate are undeclared. BOSS has 103 cover and 103 rows per stego method
at each Q75/Q95. These two JPEG qualities and the two stego derivatives of
one scene are correlated, not new independent scenes. Declared lineage/source
metadata does not establish camera/device or perceptual independence.

The **24-row sanity subset** used by signal/context/accumulation controls has
eight cover and 16 stego rows, six original scenes and eight quality groups.
It checks whether the training pipeline learns its own inputs; it is not the
entire available train corpus, held-out accuracy, or model qualification.
The earlier full two-source fit used all 2,985 train rows for one epoch and
failed its detector gates (`SRNET_MULTIPAIR_REAL_RESULTS.md`). Thus tiny-subset
failure alone cannot identify insufficient dataset size as the sole cause.

More distinct, licensed training scenes and training exposure remain important
unresolved work. Training counts are not qualification evidence: the reported
held-out cells remain below the project's minimum 1,000 cover + 1,000 stego
per format/method/payload cell over at least two independent source groups.
Keep cover derivatives grouped in splits, do not count repeated presentations
as new data, and do not tune on previously inspected development validation.
No new data was downloaded or distributed.

## Subsequent acquisition — 2026-10-08

The statement above describes the 2026-10-07 audit. Additional 4,000 ALASKA2
JPEGs and 1,000 BOSS PGM covers were subsequently acquired and independently
audited under `DATA_EXPANSION_PROTOCOL.md`; they have **not** entered that
prepared manifest/cache or a new training run. Three unchanged ALASKA
cover/stego lineages are excluded from future preparation. Eligible new
training originals are 816 ALASKA and 822 BOSS; separately reserved development
validation originals are 181 ALASKA and 178 BOSS. These are existing sources,
not independently new camera/scene populations or a measured accuracy gain.

Next prepare a versioned, bounded training dataset with whole-lineage splits
and explicit old/new identities, never overwrite the old caches. The current
BOSS JPEG preparer caps one job at 128 originals and the trainer at 4,000 rows;
do not bypass those bounds by silently truncating the expanded corpus or
allocating an unbounded all-image tensor. A larger fit needs a separately
reviewed streaming/preprocessing contract, exposure schedule, compute budget
and training-only learning gate before inspecting new validation scores.
Observed host: CPU-only PyTorch, eight logical CPUs and about 15 GiB RAM;
GPU/cloud access remains an unanswered user choice, not provisioned resources.
