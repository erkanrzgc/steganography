# Native AudioMNIST pair-head preparation — 2026-10-22

**Prepared; real fitting pending explicit approval.** This is the standardized
pair head that passed the generated control, not the earlier raw CNN.
The separate [native protocol](WAV_PAIR_NATIVE_PROTOCOL_20261022.md) and its
49-file source binding are frozen. No real-data detection accuracy was measured.

- Rechecked **600 cached native WAVs**, all raw and decoded PCM hashes,
  exact mono PCM16 /48-kHz headers, frame counts and source byte counts.
  The source receipt remains AudioMNIST commit
  `630d7dab4c040882834de6fa21baf9a60372accd`, with recorded MIT license evidence.
  **400 train /200 development** originals stay in **20/10 speaker groups**;
  all 30 reserved groups remain closed. Parent-recording ancestry is unavailable.
- All **4,800 payload recoveries** pass. The bound raw caches are
  3,600×10 /1,800×10 finite FP32 values in [0,1]. Only train rows estimate
  population mean/std; the frozen transform uses float64 then one FP32 cast.
  No labels, IDs or oracle values enter the head. These reused development
  sources are not certified clean files or a fresh external test set.
- **20 forward/backward smoke batches; zero optimizer updates.** Parameters
  are unchanged; no optimizer, weights/checkpoint, download or fit was produced.
  No native AUC, accuracy, threshold sweep or reserved-source evaluation occurred.

The actual bounded CPU preparation took **22.983947 s**
by independent systemd monotonic evidence; conservative charged debit is
**24 /1,800 s**, leaving **1776 s** for the
separately approved two-seed fit. The worker projection is
149.975934 s; it is not measured fit time.
Peak cgroup memory was **283,750,400 bytes /2 GiB**,
with zero swap and CPU quota 200%. Conservative new experiment storage including
closeout reserve is **1,343,586 /67,108,864 bytes**. The private verification workspace, including the small attempt,
currently allocates **6,959,845,376 bytes**; it is reported separately from the
experiment-output figure. Existing cached raw media remain **36,335,714 bytes**;
new raw-transfer bytes are **zero** and the incomplete historical ledger is unchanged.
The deployment probe passed separately with no media or optimizer access.

The full unchanged software gate passes **3,729 tests /2 CUDA skips**,
**97.00455% total coverage**; new service/adapter coverage is
**98.31933% /
96.77419%**.
Ruff, 225-file provisioned mypy and whitespace checks pass. The 67 new tests
use generated fixtures; adapter fit records are mocked plumbing evidence,
not native learning results. Fourteen supervisor failure fixtures also pass.
Only Python 3.11.9 ran locally; Python 3.12–3.14 remain CI requirements.
All **14 historical numeric models** and the older source closures are unchanged.
Initial Ruff style failures and the supervisor fixture's initial erroneous
expectation are preserved; neither was an actual native profile/fit attempt.

The next reviewable action is exactly **two CPU seeds /20 epochs /2,280 updates
/144,000 presentations**, using the generated arm's unchanged AdamW recipe,
fixed threshold .5 and all eight development cells with 2,000 speaker-cluster
bootstrap draws. No GPU/SSH connection is needed. Later explicit user approval
must bind this protocol and the successful profile digest before a fit receipt
can be created. No approval receipt currently exists for this real protocol.
A failure cannot restart automatically; no training, tuning or production
qualification follows from preparation. JPEG budgets remain unchanged.

Evidence: [software QA](../benchmarks/wav-pair-native-qa-20261022.json),
[profile](../benchmarks/wav-pair-native-profile-20261022.json),
[normalization](../benchmarks/wav-pair-native-normalization-20261022.json) and
[readiness](../benchmarks/wav-pair-native-readiness-20261022.json).
