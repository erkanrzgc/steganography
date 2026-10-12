# Fixed five-view JPEG learning study v1

This explicitly approved representation revision moves fixed views into the
first controlled experiment. The historical center-only study and its retained
failures remain evidence, not successful training. No deployed analyzer changes.

## Frozen representation and identities

Use `research jpeg-study --config FILE --out FRESH_DIR`, preparation config
`jpeg-study-config-v1` with the explicit additive field
`"representation": "center-four-corners256"`. Omitting the field retains the
original center-only contract. New data uses
`jpeg-learning-study-five-view-data-v1`; no old cache is reinterpreted.

Keep the same content-identity permutation, seed 20261012, reservation manifests,
256 fit + 64 development historical-train originals from each of BOSSbase-1.01
and BOWS2, Q75/JUNIWARD .4 simulation and entropy bounds. Do not change, filter
or replace originals in response to signal, errors, predictions or labels.
Simulation is unchanged, including its versioned bounded probability refinement.
Native grayscale is not advertised as RGB/color source diversity.

Every 512x512 image has the same five block-phase-zero, unrounded Y views, in
this fixed order (left, top, width, height):

| View | Region |
| --- | --- |
| Center | 128, 128, 256, 256 |
| Top left | 0, 0, 256, 256 |
| Top right | 256, 0, 256, 256 |
| Bottom left | 0, 256, 256, 256 |
| Bottom right | 256, 256, 256, 256 |

The four corners tile the complete image. Identical individual cover/stego
views are retained; only a completely identical five-view pair fails the
whole preparation. A complete independent SciPy IDCT audit checks all 6,400
cached views, native +/-1 coefficient changes and unchanged quantization.
Per-view identical-pair counts are diagnostics, never a sampling filter.
Fit and development caches hold 1,024 and 256 **images**, not 6,400 independent
samples. Their total uncompressed float32 size is 1,677,721,600 bytes.

## Matched training and evaluation

All arms use this same representation. The legacy arm remains an optimizer
control, not a comparison to the old full center-only pipeline. Source-balanced
original-pair order and D4 are unchanged. For each pair, the SHA-256 of
`jpeg-five-view:{seed}:{lineage}` provides an offset modulo five. View index is
`(offset + prior pair exposure count) % 5`. Cover and stego use the same view;
every five exposures visit all five views. It is independent of content,
labels, deltas, scores, arm and batch size. No best-crop selection occurs.

One presented row is one view of one image. Each trial still presents 32,768
rows, not 32,768 images multiplied by five. A genuine batch remains 4/8/16/32
rows in one forward/update: no accumulation or hidden five-fold memory increase.
The schedule hash binds image order, paired D4 and view order. Initialization
and schedule hashes must match across arms of the same seed. The new recipe,
manifest/source hashes and Torch version bind snapshots, so a center-only
snapshot cannot resume a five-view trial.

Inference always evaluates every view with stored BN statistics, independently
as a singleton; a separate batch comparison checks numerical parity only.
Average the five singleton **stego probabilities** arithmetically, equally,
before computing image/pair/source metrics. Do not average logits, choose a
positive crop, use a cover partner or change aggregation after seeing scores.

## Unchanged limits and gates

The four arms/two seeds, 32,768-row exposure, GPU allocator <=4 GiB, enforced
host cgroup <=8 GiB, deterministic FP32, no CPU training fallback, 30-minute
trial / four-hour controls / eight-hour overall budget remain unchanged.
This new cache plus run outputs must stay within the existing 5 GiB ceiling;
no paid compute, model download, automatic deployment or inference promotion.
Preparation and complete audit can run without CUDA; regression CPU kernels
are test-only. CUDA absence leaves real learning unavailable.

The exploratory gates and production qualification in `JPEG_LEARNING_STUDY.md`
are unchanged. Complete preparation is not measured accuracy. BOSS/BOWS camera
independence is not verified, development is not blind, and WIFD remains reserved
and unscored. Pretrained/color comparison, .2 adaptation and reserved-source
evaluation still require separate implementation and frozen decisions.
