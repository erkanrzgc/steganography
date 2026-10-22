# Generated WAV runner evidence amendment — 2026-10-21

The prepared v1 runner never started a full-exposure GPU attempt. Its sources,
manifest preparation, source-only deployment check and zero-update CUDA QA
remain preserved. A source review found that it wrote numerical seed results
only after both seeds finished. If seed two failed, seed one's completed fit
and cell metrics would not have been retained. Fix this before execution.

Version two writes an exclusive `fit-<seed>.json` immediately after the shared
trainer returns, before inference. After that seed's scoring, it writes an
exclusive `seed-<seed>.json` with both train/development cells, unrounded AUC,
fit timing, loss and score digests. These files carry the runner protocol hash.
They survive a later failure but cannot turn one completed seed into a combined
pass. The atomic confirmed-update lower bound and final worker/result/decision
files are unchanged. Neither file contains weights or native media.

The corpus seed, exact frame schedule, all generated WAV/PCM hashes, v3 model,
optimizer, row order, exposure, CUDA/runtime policy, wall/memory/storage bounds,
learning rule and one-attempt/no-retry rule are unchanged. The new source
closure includes this amendment. Reproduce the same frozen corpus manifest
on the execution host, verify the new bundle, and pass the full software gate
before the sole generated attempt. Real fitting remains unauthorized.

The earlier software check was interrupted for this review after 1,747 passed
tests /one explicit CUDA skip at 482.21 seconds, with its original log retained.
It is an incomplete verification attempt, not a full-gate pass or a failed
learning experiment. A regression simulates failure at seed two and requires
seed one's fit, all eight train/development cells and confirmed 1,140-update
lower bound to remain present, with no final complete-worker artifact.

V1's generated CUDA forward/backward QA used zero optimizer updates, unchanged
parameters and no saved checkpoint. The full-exposure attempt count is still
zero before v2 is launched. Nothing resets historical transfer accounting or
consumes additional JPEG experiment time.
