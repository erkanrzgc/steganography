# Next PCM pair intervention — non-executable draft

**No training, retry, download or new dataset role is authorized here.**
The [read-only diagnosis](WAV_PAIR_DIAGNOSIS_RESULTS_20261022.md) supports
investigating feature-scale imbalance, but does not establish its causal role
during the completed fit. Preserve all failed experiments and their bindings.

The proposed next comparison changes **one recipe factor**: raw pair features
versus training-only standardization before the same 418-parameter FP32 head.
Do not simultaneously change architecture, optimizer, loss weights, epochs,
batch sizes, target cells, seeds or data distribution. Instrumentation is
common to both arms and cannot adapt their schedules.

Before execution a separate protocol must freeze:

1. A fresh generated train/development manifest from the unchanged generator
   family, with a new fixed seed and disjoint raw/PCM/source identities. Both
   arms use identical rows, split order, model initializations and minibatches.
   Both original optimizer seeds and full 20-epoch exposures apply **per arm**:
   four complete fits, 4,560 total updates /288,000 presentations, without early
   selection. These are proposed counts, not an allowed execution budget.
2. The exact training-only mean/std definition, FP32 conversion order, variance
   floor, immutable parameter storage and a distinct transformed-input domain.
   Never estimate normalization from development, labels or oracle values.
   The existing raw ten-feature [0,1] contract remains unchanged. Standardized
   model inputs have a separate contract; do not quietly relabel them as raw.
3. Complete source and runner bindings, zero-update checks, a measured resource
   projection and an independent CPU wall/storage allowance. No GPU or old
   JPEG allowance is assumed. Preserve the cumulative raw-transfer ledger.
4. Numeric telemetry for each arm/seed: initial/final parameter digests and
   parameter-change norms, actual per-epoch overall and per-cell training
   losses/AUCs, nonfinite checks, actual update counts, and preregistered
   gradient/step norms at fixed updates. Freeze the loss sampling points and
   keep any diagnostic derivative from changing `.grad` or optimizer state.
   Retain bounded final score arrays for independent metric replay. No weights
   or optimizer checkpoint unless a new contract explicitly authorizes them.
5. One final evaluation after each full fit, all eight train/development cells,
   the unchanged fixed global-p0 comparator and unchanged learning thresholds.
   Training AUC >=0.60 and each 40%-p0 development AUC >=0.70 must pass in each
   complete seed. Report both arms whether either fails. Do not select a seed,
   feature, threshold or epoch from development and call it confirmation.

Possible outcomes are informative: a matched improvement would support the
scaling intervention within this generator; failure would leave it unsupported
and should not trigger another automatic recipe change. Neither outcome proves
natural-speech performance. Another intervention requires another frozen
contract, and real AudioMNIST fitting still needs explicit approval plus an
accepted complete generated control. No raw media was opened for this draft.
