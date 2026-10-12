# JPEG causal learning study v1

This opt-in study does not change deployed analyzers, old source closures or
legacy checkpoints. Missing GPU/optional dependencies are unavailable; no CPU
fallback, cloud spending, model export or automatic dataset download occurs.

## Evidence and hypotheses

The five-epoch full-corpus model improved training loss but retained approximately
chance development AUC. Stored-BN versus paired-batch statistics gave 46 versus
76 correct predictions on the same 96 training rows. Population-BN refresh and
matched GroupNorm trials failed. Two real pairs can be memorized; this is not
generalization. The 80-epoch curriculum used only 48 distinct originals.

The [author implementation](https://dde.binghamton.edu/download/feature_extractors/)
uses an Adamax rule without first-moment bias correction and puts epsilon on the
decayed infinity accumulator. PyTorch Adamax differs in both places. At gradient
.01, LR .001, zero slots, its first update is approximately ten times larger.
This is an optimizer difference, not evidence that PyTorch is incorrect or that
the difference alone caused our failure. Reference equations are independently
implemented; upstream research-only code is not redistributed.

Convolution-only L2 is `2e-4 * sum(W**2) / 2`; biases, BN affine parameters and
the classifier are excluded. Historical trainers instead use all-parameter
coupled weight decay .0001. Initialization remains identical legacy Kaiming in
all controls, not full upstream TensorFlow parity. SAME pooling and BN epsilon
.001 are common to all four arms, so their effects are not retested here.

The [JPEG EfficientNet study](https://ws2.binghamton.edu/fridrich/Research/11_SURGERIES_v8_preprint.pdf)
documents the faulty boundary-concentrated ALASKA2 UERD implementation. A read-only
240-fit-pair native-coefficient audit found three of 24 ALASKA2 UERD center Y
crops unchanged although full Y coefficients changed. Do not discard inconvenient
test pairs or attribute every method's failure to this issue.

## Preparation

Explicit `steganography research jpeg-study --config FILE --out FRESH_DIR` uses
configuration schema `jpeg-study-config-v1`. For `action: prepare`, `inputs`
contains exactly BOSSbase-1.01 and BOWS2 objects with `root`, original `source.json`
`sha256`, and `source_group`. `reserved_manifests` contains `path`/`sha256` objects
covering each acquisition's complete reservation list. These private paths are
never copied to portable reports.

Select 256 fit and 64 development original **historical-train** groups per source
by a fixed content-identity permutation (seed 20261012), never image content,
stego deltas or prediction scores. Former validation/test images cannot enter.
Prepare Q75/JUNIWARD .4 bpnzAC with pinned conseal 2025.11. Covers and stegos share
JPEG tables/compressor; stegos round-trip native coefficients without recompression.
Keep upstream adjusted costs, coefficient layout and seeded ternary RNG.
Independently calculate Shannon entropy of all three outcomes and
require residual <=0.1% of `round(.4 * nzAC)`. Nonfinite/out-of-range probabilities,
unknown numerical warnings or unresolvable residual fail preparation. A solver's
iteration warning alone does not prove non-convergence: the retained first
preparation stopped at original `c76246d8930c74e8e0b9b277c12218d290698df1749e9cd2515908fa157039d5`,
where the independently measured residual was 0.07937%, inside the fixed bound.
The corrected preparation uses the same identities and retains that failed run.
The second stopped at original `b5f54b1039044c2d1f9078ad2c724d68c18d4336233664a8a80d6c9e598ccff9`
on an actual excess residual. A versioned independent Gibbs/Shannon bisection
now refines excess-residual maps, using the same costs/seed/originals; at most
64 bracketing and 80 bisection evaluations, target residual <=1e-6. Capacity or
convergence failure still stops the whole job. No threshold is relaxed and no
original is discarded. Upstream-converged maps retain exact simulation parity;
refined maps are declared a different numerical simulation contract. Both
earlier incomplete runs remain local; this does not explain chance model AUC.
Requested payload is not an extracted message or exact stochastic payload length.
Decoded-original duplicates fail the preparation before output creation. Identical
selected cover/stego inputs fail the whole job, never select a better replacement.

1280 JPEGs have separate 1024-row fit and 256-row development float32 caches.
An isolated complete SciPy IDCT audit independently checks every cache row,
coefficient differences and unchanged quantization. Ground truth is embedding
simulation, not actual message extraction. The fresh development role is not a
blind test; BOSS/BOWS are not verified independent camera populations.

## Controlled fitting

`action: controls` requires `root`, `manifest_sha256`, `audit`, `audit_sha256`.
The independent audit and current source closure must match. CUDA-only preflight
must pass before data is opened. Run existing generated 8/16/32 capacity checks
separately. The largest completed size is only a candidate; re-test during real
fitting. OOM is unavailable, never increase caps or retry automatically.

Two seeds (20261012, 20261013), serial arms in this order:

1. Four rows, existing PyTorch Adamax and all-parameter decay.
2. Four rows, author-rule Adamax; all-parameter decay unchanged.
3. Four rows, author-rule Adamax and convolution-only L2.
4. Largest feasible genuine batch, otherwise same as arm 3.

Every trial presents 32768 rows at LR .001. Original/source exposure, paired D4
and initial learned parameters match. Different batch sizes necessarily change
update counts; a batch effect is not an update-matched causal proof. No gradient
accumulation, AMP, TF32, cover-partner inference or test-time batch statistics.
Separate numeric snapshots at 16384/32768 rows bind recipe, source, input, Torch
version and optimizer slots. New snapshots are never loaded as old checkpoints.
An optional `resume` list in the controls config contains `arm`, `seed`, private
`path` and `sha256`. Only a matching 16384-row snapshot with remaining work is
accepted. Order/D4 are deterministic and this architecture contains no dropout;
resumption regenerates the frozen schedule. Tail and total update counts are
reported separately; this is not a general stochastic-trainer checkpoint format.

GPU allocator <=4 GiB, enforced host cgroup <=8 GiB, two CPU threads, trial
deadline 30 minutes, control phase <=4 hours. The approved overall budget is
8 hours: controls 4h, optional comparison 2h, successful recipe continuation 2h.
New outputs/cache <=5 GiB, optional weights <=250 MiB, no paid compute. Create a
regular `CANCEL` file in the run directory or interrupt the parent to stop;
owned subprocess groups are killed and reaped on timeout/cancellation.

Worker exit evidence is numeric and additive: return code, signal/name and
stdout byte count; never raw stderr, command or failed-worker stdout. CPU/file
size limits, other signals, nonzero exits, empty output and output overflow
are unavailable, not successful trials. Capacity and fit loops retain these
failures and continue within existing budgets. Malformed responses, changed
source/audit/data contracts and cancellation remain fail-closed. SIGKILL alone
does not prove OOM or CPU exhaustion. A completed numeric snapshot and later
successful read-only evaluation cannot complete a CPU-budget-exceeded trial.
The shared worker owner is part of the execution-source closure; a runner
revision requires a new matching audit/run, never silent rebinding of old
checkpoints or retroactive qualification of a failed run.

## Gates and remaining phases

Only frozen-statistics singleton predictions count. Every source cell reports
ROC-AUC, balanced accuracy, FPR, recall, ECE and 1000-resample original-pair
bootstrap 95% intervals. A candidate passes only when both seeds in both sources
have AUC >=.70, AUC CI lower bound >.50, and paired AUC improvement CI >0 over
the corresponding unchanged control. This is a development learning milestone,
not support qualification. Publish all unavailable and failed controls.

The optional EfficientNet-B0 comparison, label-independent center/corner color
representation, .2 adaptation and one-shot reserved-source evaluation are later
gated phases: do not start merely because preparation or regression tests passed.
They remain explicitly `not_started` in reports. No signed-model installation
policy, old support gates or public accuracy claims are relaxed. WIFD stays
reserved; do not score it to choose a recipe. Source/threshold calibration must
be frozen before any final reserved-source evaluation.
