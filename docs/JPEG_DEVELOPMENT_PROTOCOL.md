# JPEG development experiment v1 — 2026-10-04

This is development/validation, not a replacement of the frozen ALASKA2 baseline
or a cross-source support claim. No trained artifact is automatically deployed.

Acquire 1,000 additional complete ALASKA2 lineages with seed 20261005 and the
existing bounded downloader, explicitly using `--purpose development`. Exclude
all original basenames in the prior ALASKA2 manifest before random selection;
also reject hash overlap with every reserved BOSSbase/Kodak/ALASKA2 manifest.
Keep the 1 GiB invocation/output ceilings. No full archive or new license terms.
Save immutable selection before downloading image bodies. Assign whole groups
to train/validation using SHA256(seed:basename), first eight bytes as an unsigned
fraction: below 0.8 is train, otherwise validation. Camera ancestry is unknown;
within-source validation cannot establish cross-camera or cross-source accuracy.

Retain original four-way source data. For training, quarantine any entire
lineage containing conflicting labels on identical bytes; record exclusions
before feature extraction, never choose samples using scores. Shared covers
are not independent copies. Any other duplicate identity or frozen overlap is
a hard failure, not a silently repaired input. All variants stay in one split.

Implement a versioned JPEG feature contract through the research service:
bounded luminance DCT magnitude histograms per AC frequency, low-frequency
block-neighbor co-occurrences, and quantization context. Original JPEG bytes;
no evaluator resizing, conversion to PNG or held-out feature extraction.
This is an experimental summary model, not SRNet, DCTR or a calibrated detector.
Native DCT parsing runs in a bounded subprocess. Verify ordering/dimensions,
finite ranges, provenance, and independent feature fixtures before training.

First model: deterministic CPU linear logistic baseline, seed 20261005,
class-balanced loss, train-only mean/standard-deviation normalization,
Adam learning rate 0.01 for 300 epochs. No architecture/epoch search on the
old test set. Record training configuration and feature/model hashes before
validation prediction. If inadequate, publish the failure and do not deploy.

Validation report: each method compared to the same validation covers; ROC-AUC,
balanced accuracy, recall and FPR at sigmoid threshold 0.5. Scores are not
calibrated probabilities. Validation findings can guide future development,
but are not an unbiased final-test accuracy claim. No test-set threshold
selection or averaging with earlier detector results. Any future model change
needs explicit provenance and a newly untouched final source for qualification.

Resources: CPU-only environment, at most four feature workers, 2 MiB JPEG input,
4 million pixels, 15-second DCT worker deadline, 1,000-lineage acquisition and
bounded feature artifacts. Keep original image data, trained checkpoints and
full feature artifacts outside Git. Export includes the exact preprocessing
and normalization contract; publishing model packs still requires the separate
signature, license and held-out accuracy gates.
