# Native AudioMNIST standardized pair pilot — 2026-10-22

This is a separate source/data/domain/exposure/grouping contract for the next
native experiment. The user's continuation authorizes implementation, software
verification and **one cached train/development data profile with zero optimizer
updates**. It does not authorize real fitting. Freeze the JSON source closure
before that profile; complete the unchanged software gate before opening media.
Every older protocol, result, failed receipt and model remains immutable.

## Source and grouping

Reuse only the already acquired 600 files bound by
[`wav-audiomnist-preparation-20261017.json`](../benchmarks/wav-audiomnist-preparation-20261017.json).
The source is AudioMNIST commit `630d7dab4c040882834de6fa21baf9a60372accd`.
The previous pinned metadata records repository MIT licensing, the license
digest and the paper's written-consent statement. This preparation reuses
that evidence; it grants no new rights or independent consent certification.
No media, model, metadata or license download is part of this intervention.
The incomplete historical cumulative transfer ledger remains incomplete.

Keep the exact hash-sorted speaker assignment, namespace
`wav-audiomnist-pilot-20261016`, and receipt order: 20 train speakers /400 files,
ten development speakers /200 files, digits 0–9 and takes 0/1 per speaker.
The remaining 30 speakers stay reserved and closed. Do not scan media trees:
open only the 600 allowlisted paths. Match both raw and decoded PCM hashes;
reject duplicates, crossed roles, altered paths and symlinks. Parent-recording
ancestry is unavailable, so cuts cannot be independent original recordings;
all cuts and controlled descendants stay grouped by speaker. These are reused
development groups, not a fresh external test set or certified clean speech.

Require native mono, uncompressed PCM16 at 48 kHz, 160–220,500 frames and
<=512 KiB per source. Check exact receipt frame/byte counts and decoded hashes.
No resampling, padding, truncation, audio normalization or native derivatives
on disk. One cover plus the unchanged sequential/scattered ×5/40% ×planes0/1
recipes yields 3,600 train and 1,800 development rows. All **4,800** independent
opaque-payload recoveries must pass. Filenames, source IDs, speakers, digit
labels and oracle values never enter model inputs.

## Model, exposure and measurements frozen before native feature inspection

Use only `audio-pcm16-pair-zscore-mlp-v2`, the standardized domain that passed
the [full generated comparison](WAV_PAIR_COMPARISON_RESULTS_20261022.md).
That pass does not qualify the earlier raw CNN or its native pilot protocol.
Reuse the unchanged ten global/quarter plane-0/1 pair-energy features and
418-parameter FP32 10→32→2 ReLU head. The histogram summary is lossy; some
one-LSB changes collide. This is a small feature head, not a raw-audio CNN.

Compute normalization using **all 3,600 train rows only**, converting raw FP32
values to float64: population mean/std (`ddof=0`), std floor 1e-6. Apply
`(x-mean)/std` in float64 then cast once to FP32, with no clipping. Freeze
the state/cache/row-order hashes after preparation and recompute exact train
statistics before any optimizer exists. The profile estimates preprocessing
statistics; zero optimizer updates does not mean no statistics are estimated.
No generated statistics, model values or optimizer state are transferred.

Two seeds, in order **20261016, 20261017**, each get 20 epochs /1,140 updates
/72,000 row presentations: **2,280 updates /144,000 presentations** total.
Reuse the successful generated arm's AdamW exactly: lr .001, betas (.9,.999),
eps 1e-8, matrix decay .0001, bias decay zero, amsgrad/foreach/fused false.
Use cover weight 4.5 and stego weight .5625 in unreduced CE, sum divided by
actual batch size. Batch 64, last batch 16; NumPy permutations use model seed
XOR 0x4E4A5047 across epochs. CPU, two threads, deterministic algorithms,
Torch 2.14.0+cpu /NumPy 2.4.6; no GPU or SSH dependency. No scheduler, AMP,
early stop, threshold/direction selection, CV, OOF refitting or hyperparameter
search. No raw-reference refit or old 760-feature/CNN fit is included.

Keep all 20 epochs of train-only loss/AUC telemetry and actual step/gradient
records at updates 1,57,570,1140. Save each completed seed before the next,
atomic update counts and bounded pickle-free train/development score arrays;
retain no weights, optimizer state or checkpoints. Development is scored only
after each complete fit, never for intermediate decisions. Preserve both seeds
and all eight cells, including failures and low-payload regressions.

For each development cell report ROC-AUC (half ties), balanced accuracy,
recall, false-positive rate and ten-bin ECE at fixed threshold **0.5**.
Use 2,000 bootstrap draws over the ten whole speakers, RNG seed 20261018;
all 20 cuts and their paired cover/stego descendants share each speaker's
draw count. Report percentile 95% intervals and paired differences against
the unchanged, untrained higher-is-stego global plane-0 statistic at .5.
This comparator has no claim to probability calibration. Its ECE is a score
diagnostic. Intervals are descriptive and underpowered, not corrected for
multiple cells. They do not turn reused development data into qualification.
Independently replay every saved score-AUC with all-pairs half-tie counts
before accepting a result. A missing score/resource/source/exposure record is
unavailable. Both complete seeds must reach train AUC >=.60 and each
sequential/scattered 40%-p0 development AUC >=.70 to meet the exploratory
learning target; a complete miss is a failed target. Neither outcome deploys
or qualifies a detector, establishes unseen-source accuracy or authorizes tuning.

## One preparation, separately approved fitting

The zero-update profile checks all 600 originals, all 4,800 payloads, both raw
feature caches, train-only normalization and 20 forward/backward batches of 64
with smoke seed 2026102206. It constructs **no optimizer**, measures no native
detection accuracy and requires unchanged parameter hashes. It has 300 seconds
including imports, reserving five final seconds. Require the conservative
projection `elapsed + smoke_seconds/20 * 2280 * 10 + 120 <= 1800`.

The software gate requires full Ruff, provisioned mypy, unchanged pytest >=90%
total coverage, each new service/adapter >=95% and whitespace checks. Its
receipt is bound to the new protocol. Generated fixtures may exercise the
native adapter; they cannot be reported as native learning evidence. Missing
Torch is unavailable, never a pass. Python 3.12–3.14 remain CI requirements.

Preparation may complete under this contract. **Real fitting requires the
user's later explicit approval**, recorded separately in a small approval
receipt bound to the protocol and successful profile digests. It must contain
a traceable approval reference, `explicit_user_approval=true`,
`fit_attempts_allowed=1` and the conservative profile time debit. Do not create
this receipt from a synthetic pass, generic earlier continuation, or timeout.
It changes no frozen source. Before the one fit, require unchanged QA/source,
cache/normalization/order bindings, profile proof and resource deployment.

Independent allowance: **1,800 combined profile+fit wall seconds**, including
imports, cache extraction, all telemetry, scoring, intervals and receipt
assembly; **2 GiB host memory, zero swap, <=64 MiB new experiment outputs**
including temporaries, individual file/log <=1 MiB. Existing raw sources are
not new storage; account verification scratch separately. No JPEG time/storage
or raw-transfer ledger is reset or extended. Conservative profile debit is
`ceil(independent systemd unit duration)+1`; it must cover worker elapsed and
remain <=300. Fit subtracts that debit and reserves five final seconds.

Use separate user systemd units: MemoryMax=2G, MemorySwapMax=0,
CPUQuota=200%, Restart=no, KillMode=control-group, TimeoutStopSec=5,
UMask=0077. Outer deadlines are 310 seconds for profile and
`ceil(1800-debit)+10` for fit. Grace time never extends eligible exposure.
Acceptance requires independent unit monotonic durations, completed exit
status and storage/resource evidence. A worker receipt alone cannot pass.
Existing exclusive output directories prevent automatic retry/overwrite;
failures retain their receipts and any partial confirmed update count.
No automatic retry, extra seed, download, reserved-media access or production
change follows. Any later fitting intervention needs another frozen contract.
