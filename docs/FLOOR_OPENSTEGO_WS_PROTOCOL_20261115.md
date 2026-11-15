# OpenStego weighted-stego feature intervention (2026-11-15)

Under the user's current request, this distinct intervention adds pixel-level
weighted-stego/parity features for the unchanged OpenStego .1-bpp corpus.
The preceding kernel-only image study remains closed, including its 79.75%
miss; no old model/worker is refitted or restarted. Existing 512x512 RGB PNGs
are read under bounded decoding. No new media/model download, password/key
lookup, extraction, GPU, ALASKA2 or native Commons role change occurs.

## Features, roles and fixed models

Use the same manifest-bound 2,560 fit /640 calibration crop pairs, keeping
four crops per DIV2K training parent together; 200 test crops /100 validation
parents stay outside fit/calibration. Hash-bound test bytes may be inspected
for integrity, but no test features are constructed before choice freeze.
Reject symlinks, files >2 MiB, dimensions other than 512x512 RGB, and nonfinite
or out-of-bound derived values. No original file or payload is rewritten.

Extract 114 scalar features from each file independently. Per channel:
LSB occupancy, low-two/three/four-bit zero fractions, horizontal/vertical
LSB agreements, mean absolute and RMS first differences (30 total). On a
fixed stride-two interior grid, use cardinal-neighbor mean, eight-neighbor
mean and KB predictor (.5 cardinal sum -.25 diagonal sum). Each contributes
per-channel unweighted/neighbor-variance-weighted signed parity residual
estimates, residual RMS/MAD, weighted RMS, near-zero residual fractions, and
one pooled weighted estimate (22 each). Neighbor-variance log mean/median and
fractions below 1/4/16/64 add eighteen. Weight is 1/(5+neighbor variance),
parity sign +1 for odd /-1 for even, WS estimate twice weighted mean(sign*error).
Each feature uses only that analyzed image, never its paired clean image.

Fit three Fisher ridges .03/.3/3, four numeric-export HistGradientBoosting
models (300 iterations, learning rate .05, leaves 7/15 crossed with l2 1/10,
seed 0, no early stopping/categories), then three standardized-WS Gaussian
ridge .01 classifiers with fit-cover-median bandwidth factors .25/1/4
(256 fixed fit covers, seed 2026111518; positive direct-distance median,
<1e-12 zero rule). Thus ten fresh models. Kernel Cholesky residual <=1e-6;
exported score parity absolute/relative 1e-7; tree probability parity 1e-10.

Select the WS model by calibration BA, AUC, original order. Compare the
existing source-frozen SRM kernel reference first, the selected WS model,
and fixed WS/reference score mixtures with WS weight .25/.5/.75. Mixture
scalar standardization uses calibration covers only. Recalibrate one BA
threshold per mixture with the exact closest-to-zero/lower tie rule; select
by calibration BA/AUC/order. Freeze state/scalars/choice before test features.
If the reference wins, reuse its saved outcome without another inference.
Otherwise one selected test inference may reuse saved reference scores for
a fixed mixture. Report all misses, TPR/FPR and paired-parent BA interval
(2,000 draws, seed 2026111519). Repeated historical testing remains adaptive,
not untouched qualification, and no test-based adjustment follows.

## Execution and evidence

One fresh exclusive parent, **1,500 seconds** including input binding,
feature extraction, model comparison, freeze, inference and independent
closing. One BLAS/OpenMP thread, sampled <=4-GiB RSS plus wait4 peak and
<=8-MiB private output. New compact feature matrices may persist privately
within that cap; no raw pixels, copied media or Gram matrix persists.
All learned states stay private and no model is automatically deployed.

Application/test sources are unchanged. Verify the passed full gate's 570
bindings (4,646 passes /two CUDA skips /97.42% coverage, Ruff /270-file mypy).
Generated scalar controls independently verify WS features and numeric
Fisher/kernel/tree exports before actual fitting. Closing replay verifies
all calibration thresholds/counts/AUCs, global choice, test metrics/interval,
source hashes and sampled numeric predictions. Preserve every failure and
old outcome; no allowance reset or automatic old retry.
