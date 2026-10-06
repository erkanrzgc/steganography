# Frozen two-source four-row SRNet control — 2026-10-07

Freeze before real batch accounting or fitting. Motivated by the earlier
train-only normalization diagnosis, but not an isolated proof that BatchNorm
was the sole cause. Original models/protocols/failed scores remain unchanged.
This experiment changes both training sources and minibatch context relative
to the BOSS-only pilot; optimizer step count changes too. No causal superiority
or generalization claim is justified without further matched controls.

## Fixed data and recipe

Use unchanged 2,985 training rows and the same manifest/cache:
manifest SHA `0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`,
train cache SHA `828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028`,
train data SHA `fd27c6da9869f4bea8bf7b97edeffd8ebae519824cd23d15b029aba25ceaf392`.
Both ALASKA and BOSS-derived sources are selected; no validation pixels/scores
enter plan preparation or fitting. Keep unrounded float32 component-Y/256
input units/crop, same architecture and no normalization/augmentation.

Explicit opt-in `batch_recipe: "two-source-two-lineage-pairs-v1"` in plan AND
fit configs. One epoch, seed 20261012. Reuse the existing unchanged hierarchical
balanced pair order (equal sources, then quality/method contexts within source).
Group each consecutive two pairs without reshuffling: [coverA, stegoA, coverB,
stegoB], labels [0,1,0,1]. Each batch requires two different declared sources
and source/lineage identities, not a shared image duplicated four times.
Source/camera/perceptual independence is not proven just by distinct metadata.

Complete epoch: 3,160 pairs / 6,320 presented rows / 1,580 optimizer updates.
Each source contributes 1,580 pairs. ALASKA has 790 draws per method; BOSS has
395 draws per quality/method. All original stegos appear at least once; reuse
and quality variants are not new independent samples. Odd/incomplete or
single-source input fails; never drop a tail, quietly truncate or pad a batch.
Publish exact pair/batch order SHA-256 and independent accounting evidence.

## Fit and inference contract

Opt-in plan/card v2 stores full pair schedule plus four-row batch schedule.
Existing v1 plans/cards and default two-row fitting remain unchanged. Bound
cache/scope/plan/card identity must reconstruct the exact recipe before updates.
BN counters equal optimizer updates (1,580), not number of individual pairs.
CPU float32, two threads, Adamax lr .001, decay .0001, betas [.9,.999], epsilon
1e-8, foreach false, constant lr. Start from the seeded initial network, not
the old pilot checkpoint; no best-validation selection, resume or BN repair.

Isolated fitting retains optimizer deadline 1,800 seconds, wall 1,920 seconds,
8 GiB address limit, CPU 3,660/3,661 seconds, per-file 32 MiB and no core dumps.
Timeout is a failed complete-epoch experiment, not permission to use a partial
fit or silently extend the limit. A verified batch plan is not an actual fit.

Completed models use ordinary all-stage eval with persisted statistics and
unchanged single-file semantics. Never enable the paired batch-stat diagnostic
as inference. Evaluate all 765 reused validation rows with fixed threshold .5,
unchanged independent NumPy logit/score/decision gates, then all six source/
quality/method metric cells and lineage intervals. Optional full ONNX remains
separate. Missing/failing gates stay unavailable/failed; no model deployment.

## Remaining evidence

Account for all real batches before fitting. Generated gradient/worker tests
only establish implementation, not real accuracy. After a complete real fit,
publish all failures/point estimates and declare reused validation, source
metadata limitations and sample-size shortfalls. A longer or matched-source
normalization control needs a separate frozen protocol; do not rewrite this one.
