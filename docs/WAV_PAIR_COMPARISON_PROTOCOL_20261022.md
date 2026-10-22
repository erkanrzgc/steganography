# Fresh matched PCM pair scaling comparison — 2026-10-22

This separate intervention implements the user's request to continue from the
read-only diagnosis and matched-comparison draft. It permits **one CPU profile
and, only if all gates pass, one four-fit generated comparison**. No real
AudioMNIST fit, native/reserved-media access, download, role reassignment,
checkpoint, old-run restart, extra JPEG time or automatic retry is authorized.
Every earlier source binding, failed outcome and non-executable draft remains
immutable. This is a new experiment, not a repair of a historical failure.

## Single intervention and new data

Freeze the exact specification and complete **42-file source closure** in
`benchmarks/wav-pair-comparison-protocol-20261022.json` after focused tests,
Ruff and provisioned-environment mypy pass. Before either execution phase,
require the complete unchanged pytest gate >=90% and each new module >=95%.
The separate coordinator's source/commands/environment and QA hashes must
also be frozen before launch; record independent systemd phase durations.

Generator seed **2026102204** feeds the unchanged speech-like construction and
the original metadata-only length schedule: 600 mono PCM16 WAVs at 48 kHz,
400 train /200 development originals in 20/10 disjoint generated groups.
Raw and decoded-PCM hashes must be unique across splits and disjoint from
both the old v3 corpus and the completed fresh pair-control corpus. Group
labels use `pair-scale-<role>-<group>`. They are generated bookkeeping groups,
not natural speakers or parent-recording evidence. No original WAV is opened.

Extract the existing ten pair-energy FP32 values for all nine rows per source:
cover then the original eight fixed sequential/scattered, 5/40%, p0/p1 recipes.
All **4,800** independent payload recoveries must pass. Raw cache matrices are
finite [0,1], exactly 3,600×10 /1,800×10; filenames, IDs, labels and oracle data
are excluded from inputs. Manifest, raw cache and row-order hashes bind both
arms identically before any metric.

The sole recipe factor is model-input scaling:

- **raw:** copy the unchanged FP32 raw cache. Domain remains
  `audio-pcm16-pair-energy-mlp-draft-v1`.
- **train_zscore:** compute per-column mean and population std (`ddof=0`)
  over **all 3,600 train rows only**, converting raw FP32 entries to float64.
  Store exact float64 means and `max(std, 1e-6)` in immutable JSON. Transform
  each role in float64 as `(x-mean)/std`, then cast once to FP32. No clipping
  or development/label/oracle-dependent estimation. Transformed values must
  be finite and have a separate, potentially unbounded domain:
  `audio-pcm16-pair-zscore-mlp-v2`. The original raw feature contract is intact.

The profile estimates these unsupervised train statistics; **zero optimizer
updates does not mean no preprocessing statistics are estimated**. The fit
recomputes and compares the exact train-only state before constructing an
optimizer. Changed cache/manifest/order/domain/normalizer/protocol fails first.

## Zero-update profile

Use CPU Torch 2.14.0+cpu /NumPy 2.4.6, two threads, deterministic algorithms,
FP32 heads. For each arm construct the same 418-parameter 10→32→2 ReLU head
with smoke seed **2026102205**, run 20 forward/backward batches of 64 rows,
and require finite gradients and an unchanged parameter digest. No optimizer
is constructed; both initial digests must match. Generation and all extraction
use the complete native length schedule, not shortened throughput examples.

The profile has 300 seconds including imports, reserving five seconds for
evidence. Require the fixed conservative projection
`profile_elapsed + max(arm_smoke_seconds)/20 * 4560 * 10 + 60 <= 1800`.
This is a projection only. On success freeze a separate four-key binding:
protocol SHA, exact profile SHA, `fit_attempts_allowed=1`, and profile debit
`ceil(independent systemd monotonic duration)+1` seconds. The debit must cover
worker elapsed time and remain <=300. Missing evidence/profile failure means
unavailable, no fit and no automatic retry.

## Matched full learning recipe

Order is fixed: raw seed 20261016, train_zscore seed 20261016, raw seed
20261017, train_zscore seed 20261017. Each fit starts from a new head; matching
seed initial-parameter hashes must be identical across arms. There is no
warm start, transferred state or learned-input-statistic update.

Each fit gets **20 epochs /1,140 updates /72,000 row presentations**:
four fits total **4,560 updates /288,000 presentations**. NumPy row permutation
seed is model_seed XOR 0x4E4A5047, sequential across epochs. Batches contain
64 actual rows, final batch 16. Save the complete order digest and require it
to match between arms for each seed. No early stop, extra epochs or exposure.

Both arms use identical AdamW: lr=0.001, betas=(0.9,0.999), eps=1e-8,
amsgrad/foreach/fused=false, matrix weight decay 0.0001 and bias decay zero.
Loss is summed unreduced cross entropy times weights cover=4.5 /stego=0.5625,
divided by actual batch rows. Finite gradients precede every optimizer step;
finite parameters follow each step. No scheduler, dropout, BatchNorm, AMP,
architecture change, class/recipe reweighting, normalization update or search.

## Numeric telemetry and independent replay

Instrumentation is identical in both arms and cannot adapt training:

- Keep atomic confirmed update counts throughout. Save every completed epoch's
  numeric train telemetry immediately and each complete fit before scoring;
  preserve a scored seed before beginning another fit. Partial work cannot
  substitute for a complete arm/seed, and failure must never restart it.
- After each of 20 complete epochs, score all train rows in inference mode:
  overall/all-eight-cell train AUC, overall unweighted CE, exact full-cache
  weighted objective, cover CE, each cell's stego CE and balanced binary CE.
  Restore training mode and require telemetry to leave parameters unchanged.
  This measures training behavior; no development metric is computed per epoch.
- At actual optimizer updates **1,57,570,1140**, retain pre-step objective and
  gradient L2 norm plus post-step parameter-change L2 norm. Use the already
  computed gradient; no additional diagnostic backward pass or altered `.grad`.
- Retain initial/final parameter digests and final change norm, but **no model
  values, weights or optimizer checkpoint**. A valid complete numeric result
  requires changed finite parameters, all 20 epoch records and four step
  records. Numeric validation requires finite nonnegative losses/norms <=1e30.
- After each complete fit, save train/development stego probabilities as
  pickle-free float64 NPY, bounded <=1 MiB each, shape 3,600/1,800 and [0,1].
  Retain file hashes, source/split/order lineage and all eight scalar AUC cells.
  Independently replay every saved score file, including the bound integer
  half-tie scalar metric and separate all-pairs half-tie counts. Missing or
  mismatched score evidence makes acceptance unavailable.

Retain the unchanged untrained global-p0, higher=stego comparator for all cells
and both splits. It cannot rescue a failed head. Both arms and all seeds/cells
are reported, including regressions; there is no best-seed/feature/epoch,
threshold/direction selection or new development-based adjustment.

## Resources and acceptance fixed before results

This independent CPU intervention has **1,800 combined profile+fit wall
seconds**, including imports, generation, extraction/oracles, telemetry,
learning, inference and worker receipt assembly. The fit subtracts the
independent conservative profile debit and reserves five final seconds.
Each phase uses a separate user systemd unit: MemoryMax=2G, MemorySwapMax=0,
CPUQuota=200%, Restart=no, KillMode=control-group, TimeoutStopSec=5, UMask=0077.
Outer watchdogs are profile=310 seconds, fit=ceil(1800-debit)+10 seconds;
termination grace never extends eligible exposure. Coordinator acceptance
uses profile debit plus the full independent fit-unit monotonic duration.

Owned temporary generated WAVs total <=48 MiB, individual outputs/logs <=1 MiB,
and all new experiment output/temporary storage <=64 MiB. Generated WAVs are
confined to the exclusive owned profile directory and normally removed. No
historical dataset/checkpoint is deleted. Record verification scratch separately;
no JPEG time or historical raw-transfer allowance is silently reset.

Each arm independently requires both full seeds, train AUC >=0.60 and **each**
sequential/scattered 40%-p0 development AUC >=0.70 in each seed. Full source,
normalizer/cache/score bindings, matched initialization/order, exact exposure,
telemetry and in-bound resources are mandatory. Missing/incomplete numeric or
resource evidence is **unavailable**; a complete below-target arm is
**failed_learning_gate**; a passing arm is **passed_generated_only**.

A scaled pass with a failed matched raw arm would support this scaling
intervention within this generator. Both passing or both failing do not prove
scaling necessary or sufficient. Two seeds and a single synthetic family do
not establish statistical significance or natural-speech generalization.
Neither a pass nor a failure grants real fitting or production qualification.
Any further fitting intervention requires another separately frozen contract;
real AudioMNIST also requires explicit approval and accepted generated control.
