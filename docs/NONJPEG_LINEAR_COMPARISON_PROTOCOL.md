# Frozen native linear comparison — separate CPU budget

This is the first fitting slice of the earlier two-seed draft. It compares
linear models only. Gray/RGB SRNet and PCM CNN training remains a separate
future experiment with its own exposure and resource protocol. The running
JPEG experiment and all 35 of its source bindings remain immutable.

Bind all 14 existing feature artifacts and eight source/split manifests using
`nonjpeg-comparison-readiness-20261015.json`: BOWS2 baseline/local parity,
DIV2K baseline/local/RGB parity and ESC-50/FSDD native PCM residual features.
Use every historical 32-train/8-validation original per source. No new Commons
sample, test media, download, source replacement, resampling or regeneration.
Verify feature/document hashes, exact ordered exposure tuples, feature names,
complete carrier cells and disjoint original/speaker/recording roles before fit.

Use seeds 20261016 and 20261017 and precisely 20 complete training epochs.
The model is one float64 linear sigmoid, initialized with zero bias and normal
weights of standard deviation 0.01/sqrt(feature count). A separate RNG stream
generates each epoch's row permutation, identical across matched feature arms.
Mini-batches contain at most 64 rows; plain SGD uses constant LR .01 and
weight-only L2 .0001. No optimizer/epoch/model selection, dropout, deep models,
early stopping, clipping of inputs or validation-dependent adaptation.
Sigmoid logits alone are clipped to [-40,40] for bounded exponentials.

Each original contributes total weight 1/N; its covers and stegos each receive
half of that weight. Share cover weight across formats and positive weight
across all stego cells. PNG/BMP, payload/channel/plane/rate derivatives never
become independent originals. Fit weighted mean/variance on the fitting subset
only, with a fixed scale floor 1e-6. The mini-batch data gradient is
N/batch_rows times the subset's globally normalized weighted gradient.

Thresholds use grouped training-only out-of-fold scores. Hash-sort independent
group IDs with the fixed nonjpeg-group-folds-20261018 namespace and assign them
round-robin to min(4,group count) folds. Thus FSDD uses two speaker folds;
ESC-50 uses four recording-group folds (32 originals /31 groups), images four
original folds. All derivatives stay together. Fit normalization and model
again within every training fold, never on its held-out group. Train folds and
the final full fit each consume their own 20 epochs; record both exposures,
all optimizer updates and the extra CV work. Select a threshold on the .00..1.00
grid maximizing original/class-balanced OOF accuracy; round ties to 12 decimals
and prefer the higher threshold. Refit once on all train originals.

Spatial reference: the same-data 684-value model for each source/seed, with its
own train-only threshold. Also report its self-reference explicitly. WAV
reference: rerun the unchanged native AudioWav analyzer on checksum-verified
copies of the same validation media, at the fixed historical threshold 70/100.
Require analyzer coverage status `ok`; never infer new baseline scores from
historical numbers. Native heuristic scores divided by 100 are uncalibrated
diagnostic scores, not probabilities. No threshold tuning for this reference.

Report every source/seed/format/order/channel/plane/rate separately, using one
same-format cover and one stego per original. Include AUC (half credit for
ties), balanced accuracy, recall, FPR, confusion and ten-bin ECE. Bootstrap
2,000 matched cluster draws with seed 20261018, clustering by original for
images, speaker for FSDD and whole-recording key for ESC-50. Use identical
draws for all metric and paired-reference differences. Every interval is an
underpowered development interval: validation has eight originals per source
and only two FSDD speakers. Do not pool away a failed source, seed or cell.
This slice defines no production or overall learning success gate and deploys
nothing. Natural-source, blind, encoder-specific and deep-model gates remain open.

The CPU experiment has its own 1,800-second wall/CPU allowance, <=2-GiB kernel
resident memory, zero swap, one CPU thread/quota and 32-MiB total retained
outputs, enforced under an owned service with at most 60 seconds for shutdown.
It consumes none of JPEG's six-hour budget. All outputs/temporaries still count
toward the evening's +10-GiB cumulative ceiling and four-GiB non-JPEG share,
leaving at least six GiB for JPEG. Run no automatic retry. Preserve partial
models, failed receipts and actual work including CV/uncheckpointed updates.

Only numeric float64 parameter NPZs and bounded JSON are written. No pickle,
export, model-pack installation or automatic promotion. `CANCEL` is an owned
regular marker. Reject symlinks, overwrite, wrong hash/domain/split, missing
cells, altered speaker roles, invalid vectors and output/time limit violations.
Use generated scalar arithmetic and adversarial tests before real updates,
then the unchanged >=90% full coverage gate, >=95% new-code coverage, Ruff,
mypy, source and diff checks. Generated tests and software CI are preparation,
not real-data learning evidence.
