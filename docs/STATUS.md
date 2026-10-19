# Implementation status — 2026-10-20

AudioMNIST generated positive control: **v1 and draft v3 CNN both fail**. On
quiet speech-like generated PCM (120 train /60 development files, 170 updates,
about 15% of one real seed), v1 training AUC is 0.495. The draft v3 recipe
(`core/wav_pilot_v3.py`: truncated residuals, bit planes, GroupNorm, AdamW)
removes the train/eval normalization mismatch but reaches only 0.53/0.54 on the
40% plane-0 cells against a preregistered 0.70; a classical pairs-of-values
statistic reaches 0.73-0.76. Reduced exposure makes this non-decisive; a
full-exposure GPU control is next. No v3 protocol is frozen and no real fit is
authorized. See the
[control receipt](../benchmarks/wav-audiomnist-positive-control-20261020.json).

AudioMNIST generated CUDA preflight (receipt amendment v2): **passed,
generated only**. All 19 amended bindings verified. Both CNN seeds completed
72,000 presentations /1,140 updates each in 2,705.9 s of the shared 3,600-s
ceiling; both linear seeds completed 1,140 updates each in 42.2 s of 600 s.
Peak CUDA allocation was 182 MiB (four-GiB fraction); host peak 2.2 GiB, zero
swap. No real or reserved media, checkpoint or retry. This establishes
generated execution capacity only: no accuracy is claimed and a real
AudioMNIST fit still needs separate authorization. See the
[amendment](WAV_PILOT_RECEIPT_AMENDMENT_20261019.md) and
[v2 receipt](../benchmarks/wav-audiomnist-cuda-preflight-v2-20261019.json).
Amended full gate: **3,421 passed /two CUDA skips, 96.87%** coverage; Ruff and
213-file mypy pass.

The first attempt below remains the preserved `unavailable` record.

AudioMNIST generated CUDA preflight: **unavailable** (receipt defect). The
single bounded attempt on the owned RTX 5060 verified all 16 bound sources,
then ran both generated CNN and linear pairs without reaching their 3,600/600-
second ceilings (3,081 s wall, 2.3-GiB host peak, zero swap). Receipt assembly
then failed with `KeyError: 'seed'`: `generated_cnn_preflight` reads a seed
key that `core.nonjpeg_linear.fit` never returns. Per-arm timings, update
counts and CUDA peak were therefore not recorded. No retry, real media,
checkpoint or AudioMNIST fit occurred. Fixing the key changes a bound source
and needs a new protocol version before one new generated preflight. See
[the attempt receipt](../benchmarks/wav-audiomnist-cuda-preflight-20261019.json).

New generated-only tests raise focused coverage to 100% (`wav_pilot`), 98%
(`wav_pilot_training`) and 98% (`research_wav_prepare`) without editing bound
files; three strict xfails document the receipt defect. Full gate: **3,419
passed /two CUDA skips, 96.87%** coverage, Ruff and 213-file mypy pass. Its
first run failed only the README 900-word limit (908); that log is preserved
and the README sentence was shortened.

Earlier 2026-10-17 status follows.

Training readiness audit: owned RTX 5060/PyTorch CUDA access and five generated
gray/RGB/PCM skeleton forward/backward cases pass; that earlier check made no
optimizer updates or checkpoints. The prior full gate passed **3,321 tests /two
local CUDA skips, 96.81% coverage**; those results predate the newest WAV slice.

New WAV pilot preparation acquired and verified **600 AudioMNIST files** (400
train /200 development, 30 speaker groups) under a separate 384-MiB raw transfer
ceiling; the historical cumulative ledger remains incomplete and unchanged.
The PCM controlled-row generator, payload oracle, two-seed trainer and clustered
metrics are implemented. The synthetic linear preflight passed 2,280 updates,
all eight cells, and both seeds in 44.82 seconds, with no checkpoint or accuracy
claim. No real AudioMNIST fit or reserved-speaker access occurred. Fourteen WAV
focused tests pass. After the missing cryptography/piexif packages were installed,
a fresh full pytest run reached 119 passed /23 optional skips without reported
failures, then was interrupted at 194 seconds; full coverage remains unverified
and focused WAV coverage is 38%. Full mypy reports 25 dependency/NumPy typing
errors in 14 files; Ruff and targeted new-module mypy pass.

Execution is still **not ready** until the complete trainer passes its
source-bound generated CUDA preflight. The pinned SSH attempt was unavailable
because port 22240 refused the connection; no remote unit started, and this
interpreter has no Torch/CUDA. See [reconnect instructions](WSL_GPU_RECONNECT.md),
[WAV pilot preparation](WAV_PILOT_PREPARATION_20261017.md), and the
[machine-readable readiness receipt](../benchmarks/wav-audiomnist-readiness-20261017.json).

All-format data follow-up: [23 built-in extensions](ALL_FORMAT_DATA_STATUS.md)
are inventoried individually. Local PDF/TXT/MD/ZIP preparation is **completed**:
16 documents /four package groups, source/license hashes and 48 controlled
artifacts. All 16 payload oracles and on-disk checksums pass; outputs remain
unassigned. **3,321 tests pass /two CUDA skips**, **96.81%** total /100% new-code
coverage; Ruff and 210-file mypy pass. Source closures and all 14 old model hashes
remain unchanged. No download or fitting occurred; MP3/TIFF/AU/mail/natural
archive and CSV/LOG gaps remain. See [results](ALL_FORMAT_LOCAL_RESULTS.md).
The first README-length failure and its explicit repair remain separately
preserved; the complete unchanged full gate passed after the repair.

WAV read-only diagnosis: **completed**; 4 audio models and 32 train cells, zero optimizer updates. All 14 snapshot digests remain unchanged; validation artifacts and test media stayed closed. See [results](WAV_READONLY_RESULTS.md).

Full gate: **3,273 passed /2 actual-CUDA skips**, **96.78%** total coverage and **100%** for both diagnosis modules. Ruff and mypy pass.

Training AUC ranges from 0.513–0.915 across sources/seeds/cells, while cached development AUC ranges from 0.359–0.656. This is evidence of limited generalization, not a demonstrated root cause. The FSDD seed with threshold 1.00 also predicts no positives on training data; missing OOF snapshots prevent threshold replay. No threshold was retuned.

The next [speaker-grouped WAV draft](WAV_NEXT_PROTOCOL_DRAFT.md) specifies two seeds and matched exposure, but is not executable. AudioMNIST has only pinned metadata/code review; no media or accepted native pilot was added. The prior cumulative raw-transfer ledger remains incomplete. See [source gaps](../benchmarks/native-source-gaps-20261016.json).

Native process preflight amendment: **completed**; 14 models and 480 cells replayed. Production qualification remains unavailable; no deployment or native deep fitting. See [results](NONJPEG_PROCESS_PREFLIGHT_RESULTS.md).

Full gate: **3,233 passed /two actual-CUDA skips**, **96.75%** total coverage and **100%** for both amended modules. Ruff and 206-file mypy pass. All 14 numeric models and 480 cells replay exactly, with 20,200 real optimizer updates including CV; replay adds zero updates. Fitting/replay takes 19.80 seconds, or 40.80 seconds with prior debit. Original sources/failed receipts and JPEG results remain unchanged.

WAV balanced accuracy remains 0.50 in all cells. Spatial extensions have mixed paired AUC changes, with no consistently better source/seed arm. These are underpowered development pilots, not qualified detectors. Resources and every cell remain in the [dated result report](NONJPEG_PROCESS_PREFLIGHT_RESULTS.md). Earlier records below are historical.

JPEG refinement is **completed**: 28,672 verified updates, 56 segments and six evaluation endpoints. No arm passes the combined source/seed learning gate; no model was selected or deployed. See [results](JPEG_REFINE_NUMERICS_RESULTS.md).

Native software verification passes **3,211 tests /two explicit CUDA skips**, **96.74%** total coverage and **100%** for all four new comparison modules; Ruff and 204-file mypy pass. The CPU comparison is **unavailable**: its resource gate rejects the trusted kernel self-process symlink before any media or optimizer update. A numeric own-PID read-only probe passes the same memory/CPU/swap checks. The failed attempt is retained; a separately bound preflight correction and kernel integration check are required before any new fitting. See [diagnosis](../benchmarks/nonjpeg-linear-preflight-diagnosis-20261015.json). No automatic retry occurred.

Earlier execution checkpoints below are historical.

Native linear follow-up: **unavailable**; 0 numeric models and 0 cells replayed read-only. Production qualification remains **unavailable**; no deployment. Deep native models remain untrained. See [the execution receipt](../benchmarks/nonjpeg-linear-handoff-20261015.json) and [the frozen protocol](NONJPEG_LINEAR_COMPARISON_PROTOCOL.md).

The amended GPU controller started after a fully observed unchanged gate:
3,113 tests passed /two explicit CPU-host CUDA skips, 96.67% total coverage,
>99% for both new modules, Ruff and 200-file mypy pass. Independent child exit
status is zero. Resource observer retry passes with a 157,889,989-byte peak;
the first observer failure remains unavailable and its original log is retained.
The owned service enforces 8-GiB resident memory, zero swap and a 5h 50m 56s
watchdog after the conservative prior debit. Final learning results remain
unavailable while the matrix runs. A one-shot supervisor collects its final
portable report without restarting training. See the
[verification](../benchmarks/jpeg-refine-numerics-verification-20261015.json) and
[launch](../benchmarks/jpeg-refine-numerics-launch-20261015.json) receipts.
The first real GPU segment completed 512 updates and its numeric checkpoint
checksum was independently replayed. Its 174.55-second segment and both loss
records remain evidence of execution only; final learning is still pending.
See [the first checkpoint](../benchmarks/jpeg-refine-numerics-first-checkpoint-20261015.json).

Numerical preflight diagnosis: CPU/CUDA logits differ by at most 2.7567e-7;
independently rounded gradients produce post-Adamax weight differences near
2e-4, failing the original state comparison. The separately bound amendment
checks forward/derivative/state parity before optimization, independent scalar
optimizer equations and exact GPU save/resume. Actual owned-GPU validation
passes at 0.320273 seconds maximum generated update time. Forty focused tests
pass with >99% coverage for both new modules. The first full log reports 3,113
passed /two CUDA skips and 96.67% coverage, but its resource observer crashed
on a concurrently removed pytest directory before recording the child exit.
That observer attempt is unavailable, with its log retained. The unchanged
gate was rerun with a race-tolerant walker and independent atomic exit record,
as recorded above. No real-data update occurred before that completed gate.
The original 31 source hashes stay identical,
and 604.309597279 seconds /256 MiB are conservatively debited before an amended
physical attempt. See [the amendment](JPEG_REFINE_NUMERICS_PROTOCOL.md) and
[generated evidence](../benchmarks/jpeg-refine-numerical-evidence-20261015.json).

Latest physical attempt: the existing SSH tunnel is restored and the physical
RTX 5060/CUDA identity is verified. A fresh isolated runtime ran under the
8-GiB resident /zero-swap bound and six-hour-plus-shutdown watchdog. Its fresh
6,400-view audit passed; the generated CUDA preflight worker then exited 2.
Execution is **unavailable**, with zero real-data updates, training segments
or checkpoints. The controller spent 34.309597279 seconds; 21,565.690402721
seconds remain in the cumulative budget. No automatic retry occurred. The
frozen supervisor discarded worker stderr and did not preserve failed stdout,
so the underlying exception is unavailable. Further execution needs separately
bound diagnostic error reporting, preserving this attempt and its spent budget.
See [the physical-attempt receipt](../benchmarks/jpeg-refine-physical-attempt-20261015.json).

Earlier follow-up launch readiness: the owned GPU reverse SSH tunnel had no listener
and connection attempts return refused. JPEG execution remains **unavailable**;
the controller has not started. A fresh local audit passes all 6,400 views and
31 frozen source hashes. The complete 203-file runtime bundle passes isolated
imports but has not been transferred. CUDA identity, bounded preflight and
the in-budget GPU audit remain required; the six-hour/ten-GiB budget is unchanged.
See [the launch receipt](../benchmarks/jpeg-refine-launch-readiness-20261015.json).

Non-JPEG comparison inputs now include four exact 684-value reference prefixes,
giving 14 validated feature artifacts across eight source/split groups. Ordered
rows match across feature arms. FSDD has only two training speakers; ESC-50's
32 training originals represent 31 recording groups. Grouped fitting/threshold,
optimizer and separate budget contracts remain to be frozen before training.
No fitting occurred. See [the input receipt](../benchmarks/nonjpeg-comparison-readiness-20261015.json).

Native non-JPEG preparation is available: 1,548/1,332-value spatial contracts,
760-value PCM descriptors, ten source/split-bound artifacts and generated-only
gray/RGB/PCM model skeleton checks. The 160-original/4,080-artifact corpus passes
fresh checksum and 3,840 scalar payload-oracle checks; test media stayed closed.
Unicode 17 controls cover six categories. A metadata-only hash recovery binds
13 native GIFs to file-level licenses; the complete new-media pilot and
independent original ancestry remain unavailable after original-file HTTP 429.

Final gate: 3,073 passed /two actual-CUDA skips, 96.64% total coverage and
98.37% scoped new-core coverage (features/models each 100%). Ruff, 198-file
mypy and diff checks pass. The first full attempt is unavailable after exceeding
the non-JPEG temporary-output allocation; its log/resource receipt is retained.
The unchanged gate passes with pytest's normal failed-only temporary retention,
with a 793,377,955-byte peak task footprint. Historical data/checkpoints were
not removed. No real-data training, new model checkpoint or deployment occurred;
all 31 frozen JPEG source hashes and its training recipe/budget stay unchanged.
See [the readiness report](NONJPEG_EVENING_RESULTS.md) and
[portable evidence](../benchmarks/nonjpeg-evening-readiness-20261015.json).

The following JPEG-specific verification records describe the earlier slice.

The separately frozen second-round `research jpeg-refine` implementation is
available: shared paired warm-ups, paired/unpaired continuations, numeric v2
branch ancestry, bounded round-robin execution and explicit cumulative-budget
resume. See [the protocol](JPEG_REFINE_PROTOCOL.md).

At that earlier verification, physical execution was **unavailable**: the existing host-key-verified SSH
connection is reset by its peer and local Torch is CPU-only. No second-round
training updates or models have been produced; no learning or production
qualification is claimed. All historical snapshots and receipts are preserved.

The final-source read-only data audit passes all 6,400 cached views (5,120 fit,
1,280 development) and native JPEG checks. The unchanged full gate passes
3,012 tests /two actual-CUDA skips in 1,156.63 s, with 96.60% total coverage
and 99.18% new-code coverage. Ruff, 192-file mypy, identity and diff
checks pass. Optional Torch absence is checked separately: 27 tests pass and
115 skip explicitly. Python 3.11 is locally verified; the configured CI matrix
remains 3.11–3.14 and has not been executed for this working-tree change.
Earlier sandbox-stalled/interrupted verification logs are retained. Owned
generated test scratch is cleaned; data and historical checkpoints are not.

Portable evidence: [frozen protocol](../benchmarks/jpeg-refine-protocol-20261015.json)
and [implementation/data readiness](../benchmarks/jpeg-refine-readiness-20261015.json).
Readiness is not physical training or learning success.

## Historical status — 2026-10-14

Final publication verification: 2,871 passed /one actual-CUDA skip, 96.48%
total coverage, 852.69 s; all ten new portable-evidence checks pass. Ruff,
185-file mypy and identity/diff checks pass. The first publication invocation
failed when the 7.8 GiB `/tmp` tmpfs filled; its log is preserved, not reported
as passed. A fresh disk-backed basetemp plus worker `TMPDIR` completed the full
unchanged coverage gate. Contributing/project-memory guidance now covers both
temporary locations. No datasets or GPU checkpoints were removed to fix it.

COMPLETE: all eight matched JPEG controls and 16 numeric snapshots verified.
Two complete controls were inherited, six newly fitted; all source closures,
initialization/schedules, parent evidence and cumulative accounting preserved.
No arm passes both sources/both seeds; production qualification unavailable,
no deployed model. Genuine-16 development accuracy is 67.97%/63.28% in the
first seed, 57.81%/57.03% in the second; both reproduce the old genuine scores
exactly. Execution repair is not an algorithmic accuracy increase.

Both read-only diagnoses complete for all eight models. Full fit/development
AUC gaps are small (-.034317 to +.051026), weakening a pure-overfitting
explanation. A fixed 16-original fit-only mode contrast establishes substantial
stored/paired-BN loss sensitivity in all six small-batch controls, not a safe
fix or standalone accuracy. All model/buffer digests unchanged, zero new
optimizer updates, no threshold or reserved-source use. Total cumulative
controls plus diagnoses 15,666.88 s within the original 21,600 s. Own-GPU
services finished, snapshots retained. Post-diagnostic checks confirm all 16
snapshot checksums, original parent and 52+3 source files remain unchanged;
four services stopped successfully and CUDA compute inventory empty. See
`JPEG_CONTROL_RESULTS.md` and its
four portable records. Earlier progress/pause entries below are historical.

Paired-score diagnostic slice verified: 2,861 full tests /one actual-CUDA skip,
96.48% total coverage, 3,752.50 seconds on the CPU regression host. New core
diagnostic coverage 100%; lint/type/diff checks pass. The active 52 fitting and
three controller source files stay unchanged. The first two preserved controls
show stronger within-pair score direction than cross-image discrimination:
reference scores increase in all 64 matched pairs per source, but standalone
AUC stays .594482/.536377 and learning gates fail. This is NOT 100% standalone
accuracy or confirmed evidence of a causal content/BatchNorm defect. See
`JPEG_PAIRED_SCORE_DIAGNOSTICS.md` and
`benchmarks/jpeg-paired-score-diagnosis-20261014.json`.

Live matrix progress: first seed fully completed; second-seed legacy complete,
five controls total. Convolution-L2 first-seed AUC .528564/.518311 does not
improve baseline. Genuine-16 first-seed AUC .770508/.679199 matches all 256
development probabilities and standalone metrics of the historical genuine
control exactly; the historical incomplete matrix remains unavailable with its
distinct binding. Current matched baseline now permits paired improvements,
but BOWS still fails the .7 research learning gate. Remaining three controls
are running/pending, no production qualification or new algorithmic gain yet.

USER-REQUESTED RESUME at 08:58:01 UTC (11:58 local Oct 14): fresh own-GPU
attempt at the unchanged `c218e5e` fitting/controller implementation, following
numeric validation of all four saved snapshots and verified success of all
eight CI jobs in run 38471203584 (including Python 3.11–3.14). Replayed only
the completed legacy/reference first-seed controls; no repeat GPU fitting.
Convolution-L2 first half is running at 08:59:45 UTC; six full controls remain.
Fresh capacity 8/16 passes; 32 remains allocator-OOM. Prior cancelled evidence
and all source/recipe/Torch bindings stay unchanged. Remaining useful budget
starts at 16,196.48 s, watchdog 16,317 s; no reset or later-stage extension.
Separate bounded own-host observation/final verification survives temporary
SSH expiry without automatic restart. User requested unattended completion;
no paid compute, model promotion or source changes to the live fit authorized.
See `benchmarks/jpeg-owned-resume-start-20261014.json`. Earlier pause/running
records below are historical, not current activity or final results.

USER-REQUESTED PAUSE, 2026-10-13 21:56:09 UTC (00:56 local Oct 14): training
and its own-host observer are stopped. The runner wrote `cancelled:user_cancelled`;
service exit 2 is expected cancellation. CUDA compute-process inventory is empty.
Two full first-seed controls are preserved: legacy and reference-Adamax, each
32,768 rows, with four checksum-bound numeric snapshots. The third control's
first half had not reached its checkpoint and must restart; never claim that
partial work saved. No artifacts/data/checkpoints were deleted or overwritten.
The cumulative useful controls time, including discarded work, is 5,403.52 s;
remaining budget 16,196.48 s. Preserve all 52 fitting-source hashes and the
separate controller closure. Parent cancellation SHA-256:
`da85544091ba0ff84d8074fc3b05e4576e2c9d2051735c7fed8e623bdbfc23e9`.
Next private explicit prefix plan: `next-resume-plan.json` in the owned
continuation handoff directory; use a fresh next attempt, no automatic restart.

New reference development results for first seed only: BOSS accuracy 60.94%,
AUC .594482; BOWS accuracy 51.56%, AUC .536377. Both learning gates fail;
BOWS AUC is not above legacy, and the second seed/six controls remain pending.
Do not claim general learning success or promote/deploy this snapshot. Earlier
running observations below are historical, not current activity.

Actual continuation started 2026-10-13 21:27:35 UTC (00:27:35 local Oct 14),
execution `c218e5e`, after the complete unchanged regression gate: 2,830 passed,
one actual-CUDA skip, total coverage 96.47%, 1,039.29 seconds. New controller
coverage is 97.71% /CLI 97.56%; lint/type/diff checks pass. The own-GPU bounded
precheck validates all three original numeric snapshots with the unchanged
52 fitting-source files and Torch binding. The cancelled parent remains intact.
Fresh capacity 8/16 passes; 32 remains allocator-OOM under the same 4 GiB cap.
At 21:28:05 UTC legacy is inherited-complete, not newly fitted; reference-Adamax
seed 20261012 is running its missing second half from the verified 16,384-row
snapshot. Seven controls remain; no new complete trial or learning gain yet.
Remaining useful budget starts at 17,911.88 seconds; whole-service watchdog
18,032 seconds. The bounded own-host observer independently watches/finally
validates results without restarting jobs if SSH expires. New CI remains
pending, not substituted for the actual complete local regression. See
`benchmarks/jpeg-owned-continuation-start-20261014.json`.

User explicitly resumed work and SSH is reconnected. Owned-job continuation
is implemented separately from the unchanged 52 fitting-source files; see
`JPEG_OWNED_CONTINUATION.md`. It retains the cancelled parent, replays only
verified prefix receipts/snapshots into a fresh attempt, and deducts the entire
3,688.12-second prior attempt from the same six-hour controls budget. Completed
legacy fitting and the reference first half need no repeated GPU updates.
80 independent CPU orchestration/rejection tests, including a full-shape
numeric snapshot parser integration, pass; the complete regression and physical
inheritance/startup are verified above. Final fitting/learning gates remain
pending. No accuracy gain claimed.
The earlier stop and running observations below remain historical records.

Historical USER-REQUESTED STOP at 18:40 UTC: do not restart anything until the user asks
to continue. Training received SIGINT and wrote `cancelled: user_cancelled`;
the service's exit code 2 is expected cancellation, not a CPU/GPU failure.
Both remote observer/final-check services and the local SSH watcher were
stopped. CUDA compute-process inventory is empty, utilization 0%. No data,
reports or checkpoints were deleted. The latest publication CI run 38446034109
was also requested cancelled; older verified successful CI results remain.

Resume handoff: legacy seed 20261012 is fully complete; reference-Adamax seed
20261012 has its owned 16,384-row /4,096-update first-half snapshot. Its SHA-256
is `71b89cc56c30b4b745c89724134127fe1d32884b961fb54679e034ffd76af4ed`, binding
`2562e5ed5dfa6c7cd12fb034e178f14a4cb508155e2ad421b15dabc52ab8de8a`.
The incomplete second half must be repeated from that snapshot, not claimed
as saved/completed. All three real snapshots remain on the own-GPU host.
Preserve execution `16a2c69` /52-source mapping
`c440553cbe100023d79d666564ee19910ac758e547e2114e1587f6d8b4e06c17`
and the existing bound audit/config. Do not overwrite the cancelled job or
rebind/import historical failed snapshots. The existing segmented launcher is
fresh-only: safe owned-job continuation orchestration still needs explicit
implementation/verification before resuming; never silently reset budgets.
Useful elapsed controls time before cancellation was 3,688.12 seconds.
The cancelled report is retained privately with SHA-256
`5785d91f1769ee577fc5506665f3507b69c314e28e0c61274a6adf67f0c2f477`.
No new tests, training, automatic restart or CI-triggering push is authorized
during this pause. Earlier running observations below are historical.

At 18:11 UTC the first actual segmented logical control completed: legacy,
seed 20261012, all 32,768 rows /8,192 updates. Kernel durations 830.67 and
1,073.06 seconds stay below unchanged worker limits, aggregate 1,903.73
seconds. Both real snapshots pass shape/dtype/finite/BN/optimizer-counter
checks. Read-only comparison with the preserved earlier uninterrupted legacy
fit matches all 498 numeric arrays bitwise and every development probability
and metric. Different original bindings are retained; no historical snapshot
was resumed/rebound/reclassified. This verifies real continuation, not an
accuracy increase: legacy remains BOSS 53.91%/.547241 AUC and BOWS
50.78%/.538818 AUC, with high FPR and failed learning gates. The next reference
arm is running; seven required controls remain. See
`benchmarks/jpeg-segmented-first-trial-20261013.json`.

The separate startup-publication revision `77549b6` also passed all eight CI
jobs: 2,749 full tests /one actual-CUDA skip, coverage 96.40%, 1,231.84 seconds
(run 38441084959). It changed no active execution source or training binding.

Bounded-segment own-GPU startup observed at 17:39:41 UTC, service started
17:37:24 UTC at verified `16a2c69`. All eight actual CI jobs passed, including
the complete unchanged gate: 2,743 passed /one actual-CUDA skip, 654.35 seconds,
total coverage 96.40%, new segment core 98.77%. All 12 native CUDA generated
proofs passed: archived-optimizer and disk-continuation bitwise parity across
four arms, plus half-budget estimates 877.22 /1,112.45 /1,224.52 /662.10 seconds.
The disk fixtures use explicitly generated counters, not real previous fits.

Fresh independent 52-source audit verifies all 6,400 existing views; original
identities and data remain unchanged. Capacity 8/16 passed, 32 retains allocator
OOM under the same 4 GiB policy. The new eight-trial `bounded-half-v1` job is
running its first actual GPU half at this observation; no completed trial,
learning gain or qualified model is claimed. Per-worker CPU/wall caps stay
unchanged; the six-hour controls budget is explicit. Remote bounded observation
and separate final snapshot/metric verification run independently of temporary
SSH availability. They do not restart jobs or alter sources/budgets. Local SSH
watching also retains changed receipts; it stops on three failed connections.

The first private preflight controller failed on a receipt-schema KeyError
(`real_model_trained` is absent in the archived parity receipt). All native
probes had completed; a separate strict schema-specific gate verified their
actual fields and checksums before audit/start. Preserve the failed controller
unit and original receipts, not a silently successful rewrite. See the dated
portable `benchmarks/jpeg-segmented-gpu-start-20261013.json`; startup is not
completion. Earlier implementation observations below remain historical.

Actual generated CUDA eight-update parity at `b37793d` passed all four arms:
losses and all model/BatchNorm/optimizer arrays are bitwise equal to the pinned
old validator, maximum difference zero. Whole-update means (seconds) are legacy
.159203, reference .208184 and convolution-L2 .210910. Conservative full-trial
estimates are 1,690.23 /2,191.80 /2,219.72 seconds: reference arms remain
ineligible for the historical 1,800-second monolithic policy. All eight actual
CI jobs at `b37793d` passed (run 38433316288), including unchanged coverage.

The new opt-in `bounded-half-v1` policy executes the same complete exposure in
two checkpoint-linked halves. Worker limits/learning gates stay unchanged;
six-hour controls budget is explicitly separate, within the approved original
eight-hour study envelope, with no additional later-stage authorization.
Implementation verification, new source-bound audit and actual fitting are
pending at this observation; no accuracy or completion claim. Old receipts,
snapshots and failed matrices are retained. See `JPEG_SEGMENTED_CONTROL_PROTOCOL.md`.

Segment implementation checks: 291 focused regression tests passed; 58
continuation/guard tests independently measured the new core module at 98.77%
coverage. Provisioned lint and type checks pass across 182 source files.
One invocation through the unrelated system Python failed on old cryptography
and absent Torch; it is not a verification pass. The correct provisioned
`venv/bin/python` supplied the results above. Complete CI remains pending for
this revision; unchanged full coverage gate is required before fresh fitting.

Final receipt of the frozen eight-trial job was retrieved after tunnel renewal:
`unavailable: incomplete_required_controls`, 13,766.60 monotonic seconds;
five CPU-limit outcomes /one parent timeout and two completed genuine-16 fits.
Each completed fit presented all 32,768 rows /2,048 updates. Seed 20261012
development accuracy/AUC: BOSS 67.97%/.770508, BOWS 63.28%/.679199; seed
20261013: BOSS 57.81%/.613037, BOWS 57.03%/.574951. Both remain experimental,
all learning gates false; missing matched baselines are not silently imported.
Final report SHA-256 `ef83fab7f89451c7bef353945dc9a1d2cd4d06992e0dd656fb0bbe88651d92a0`
matches the remote receipt and all 50 archived `fb649d6` source blobs. Models,
partial checkpoints and prior successful reports are retained, nothing deployed.

Physical generated profiling of verified `9cdd31f` locates a remaining reference
optimizer bottleneck: 0.44516 of 0.66157 seconds/update in the optimizer. The
next candidate preserves all dense/FP32/finite/moment/norm/integral-step checks
but consolidates their CUDA scalar transfers per device before any parameter
update. 67 focused oracle/guard tests passed in 4.85 seconds; full regression,
physical exact CUDA parity and whole-update timing remain pending. No budgets,
exposure or accuracy gates were increased to call a failed trial successful.
See `JPEG_EXECUTION_OPTIMIZATION.md`. New sources are not used for real fitting
until those gates and a fresh complete audit pass.

Unattended monitoring (user explicitly requested continued work): a bounded
read-only watcher polls the current job and preserves numeric states/report
hashes privately in `.benchmark/jpeg-own-gpu-session-20261013/monitor.jsonl`.
It stops after terminal state, three failed connections, or 15:00 UTC; it never
changes remote sources/budgets or restarts a worker. The first new legacy trial
became `unavailable: worker_timeout`; a kernel SIGXCPU event was also observed.
The parent stayed active and started reference-Adamax, unlike the old failed
parent. No new completed model or accuracy increase. `cc66511` passed all eight
actual CI jobs, including Python 3.11–3.14 and the full research coverage gate
(run 38387100939); this is regression, not GPU learning qualification.

An execution-only next candidate consolidates JPEG gradient and full-state
finite checks using the already-tested shared gradient helper and a complete
parameter/buffer boolean reduction. Every tensor is still checked; CPU four-arm
comparisons retain identical losses, model state and optimizer slots. 29 focused
tests passed in 46.28 seconds, including CPU-tensor CUDA-branch emulation; Ruff
and 181-file type checking pass. Physical CUDA speed/equivalence and complete
regression for this candidate are pending. It is not installed into the live
WSL job; do not transfer changed sources until that frozen job stops. No model
arithmetic/exposure, tolerance, precision, budget or accuracy gate is relaxed.
Nine additional kernel/training/resume tests passed in 31.43 seconds. The new
candidate is queued separately for the full CI gate, avoiding another heavy
local full-suite workload on the same Windows host during GPU controls.

Checkpoint `fb649d6` is now fully verified on the provisioned Kali CPU runtime:
2,635 passed /one explicit actual-CUDA skip, 32 existing ONNX warnings,
1,035.38 seconds, total coverage 96.44%. New core worker coverage is 98.59%;
JPEG orchestration 96.49%. Ruff, 181-file type checking and whitespace pass.
The 512,304-byte wheel includes the worker service, no model/cache/media.
Only this completed invocation's regenerable fixtures were removed
(6,048,128,019 apparent bytes); its log/coverage and all private corpora remain.
These CPU checks do not qualify a GPU model or the remote Python-version matrix.

The own-GPU tunnel was restored, and the clean WSL checkout fast-forwarded to
`fb649d6` without changing historical runs. Existing five-view carriers received
a fresh complete independent audit bound to all 50 current execution sources;
all hashes also match the archived Git blobs. All 1,280 image rows /6,400 views
passed unchanged 1e-4 IDCT checks; old unchanged-view counts and original
identities are preserved. A fresh bounded generated GPU/CPU probe passed.
Generated batches 8/16 completed; 32 remains allocator-OOM/unavailable, no retry
or larger cap. No old audit/checkpoint was rebound and no corpus regenerated.

The new service `steg-jpeg-five-controls-20261013` started at 10:53:00 UTC.
At 10:54:30 UTC its first actual four-row `legacy` fit (seed 20261012) was
running. No completed trial or snapshot was observed, and update counts are
not inferred from utilization. Eight freshly initialized controls retain
32,768 presented view rows each, 1800/1801 CPU-second limits, 30-minute trial,
four-hour control, 4 GiB allocator and 8 GiB host caps. Failed worker exits now
retain portable unavailable evidence while later controls can continue within
the existing budget. Private files: `.benchmark/jpeg-own-gpu-session-20261013`.
Portable startup evidence: `benchmarks/jpeg-five-view-gpu-restart-20261013.json`.
This is a dated running observation, not completion or accuracy improvement.
Keep the laptop awake, ventilated and plugged in; the SSH listener expires
four hours after its creation, independently of the training service.
WIFD/reserved sources remain unscored and nothing is deployed.
The subsequent three startup/diagnosis evidence guards pass separately
(0.55 seconds); the new portable source closure is self-contained for shallow
CI clones/source archives. No executable training source changed after the
complete regression, audit or GPU startup.

The 2026-10-12 own-GPU controls stopped at 18:35:07 UTC: two of eight trial
reports completed, and the third (`convolution-l2`, seed 20261012) retained its
32,768-row /8,192-update numeric snapshot but no completed report. Kernel
records show soft CPU timeout /SIGXCPU at 18:34:54 UTC and hard timeout at
18:35:06 UTC. This was the unchanged 1800/1801 CPU-second worker limit, not
evidence of VRAM exhaustion. A bounded read-only replay of the final snapshot
evaluated all 256 development image rows /320 batch-view comparisons in
23.18 seconds, with unchanged weights, 49-file source closure and 1e-4 parity
tolerances. No optimizer update or accuracy qualification resulted.
Portable diagnostic evidence: `benchmarks/jpeg-study-limit-diagnosis-20261013.json`.

The runner now separates CPU/file-size limits, other signals, nonzero exits,
empty output and oversized output into numeric portable `unavailable` records.
Capacity/fit loops retain failures and continue; malformed results and altered
source/data contracts still fail closed. The shared core worker owner preserves
group cleanup, timeouts, cancellation and discarded secret-bearing stderr.
CPU/RAM/GPU/output budgets and learning gates are unchanged. The new execution
closure includes this service and differs from the archived 49-file closure;
old audits and snapshots are never silently rebound. Before another GPU run,
produce a fresh matching audit and new outputs. The last connection check
returned SSH connection reset; no training restart or model deployment.

Handoff checkpoint at the user's stop request (2026-10-13): 182 focused tests
passed in 102.58 seconds before the final preparation/audit failure guards;
the subsequent targeted 19-test run passed in 27.58 seconds. Ruff and type
checking passed. The complete regression invocation collected 2,636 tests but
was interrupted before completion; it is not a pass and coverage is unmeasured
for this revision. Its private log/scratch remains under
`.benchmark/jpeg-runner-verification-20261013.bDTXwngF`; no owned pytest process
remained at handoff. Next: rerun complete regression in fresh disk-backed
scratch, verify new worker coverage >=95%, then create a newly source-bound
audit before any new GPU run. Do not reuse the old audit/snapshot binding or
claim recovery of the historical failed trial. Unrelated user files are intact.


Own-GPU connection restored on 2026-10-12. The clean WSL checkout was
fast-forwarded to verified `cc82608`; all 49 JPEG study sources match the
completed five-view preparation/audit. A new generated CUDA preflight passed
with native RTX 5060 Laptop /Torch 2.14.0+cu130, 4 GiB allocator and kernel
8 GiB host bound. All 1,285 private media/cache/metadata files transferred to
the own laptop with no symlinks or recursive checksum differences; no cloud.
Genuine generated batches 8/16 completed; 32 hit the unchanged allocator cap
and remains unavailable, without retry or memory increase. Batch 16 is only
the later genuine-batch arm's candidate, not a guarantee of real-data success.

The systemd user service `steg-jpeg-five-controls-20261012` started at
17:09:20 UTC. At the 17:11:25 observation, the actual four-row `legacy` fit
worker for seed 20261012 was running on GPU. No completed trial, numeric
snapshot or final scores had been observed; optimizer-update count is not
inferred from utilization. Four arms/two seeds retain 32,768 presented view
rows per trial, 30-minute trial and four-hour control budgets. Do not mutate
the remote execution source closure during this job. Active remote files are
under `.benchmark/jpeg-own-gpu-session-20261012/controls-run`; the existing
regular `CANCEL` marker contract remains available. Portable startup evidence:
`benchmarks/jpeg-five-view-gpu-start-20261012.json`. This is a dated startup
observation, not a claim the process is still running when read later or that
it completed/learned. GPU was 86 C against its reported 87 C target; keep the
laptop ventilated, plugged in and awake. WIFD/reserved sources remain unscored;
accuracy qualification/deployment are unavailable. Earlier SSH-unavailable
attempts below are retained historical outcomes, not overwritten successes.

GPU-deferred non-JPEG preparation completed on actual media: 160 existing
originals (80 images /80 recordings), four declared origins, each 32 train /8
validation, produced 4,080 artifacts (240 covers /3,840 marker-free positives).
Independent scalar audit checked every saved payload and every unchanged bit,
fixed source/role/cell membership, original ancestry and complete WAV metadata.
All 192 source-role-format-method-rate-channel-plane cells retained their 32/8
positive counts. Test media remained unopened; BOWS stayed native gray, DIV2K
is one RGB origin, audio stayed native mono PCM16. Total artifact bytes
613,635,514; preparation/audit 278.87 seconds, audit alone 236.94 seconds,
observed concurrently with regression, not inference latency qualification.
Compatible spatial/audio research manifests independently pass the existing
validator. Portable evidence: `benchmarks/nonjpeg-preparation-20261012.json`,
six code/protocol bindings at `4271395`. Initial focused checks passed 109 tests;
eight final binding/cancellation/adapter/subprocess checks also passed.
The first complete invocation failed when the separate 7.8 GiB `/tmp` tmpfs
filled: errors accumulated and even its log stopped at 16,384 bytes; it is not
a regression pass. An isolated 39-test SRNet training rerun passed. Only this
invocation's regenerable scratch (5,741,727,223 apparent bytes) was removed,
restoring 5.5 GiB tmpfs availability; failed log/wheel, raw corpora and audited
derivatives are retained. The unchanged complete suite reran with an exclusive
disk-backed fixture directory, not a larger parser/tool limit.
No GPU, fitting, scoring, download, paid compute or detector improvement.
GIF/text/container/stereo/tool families retain explicit native data/oracle gaps.
See `NONJPEG_PREPARATION_PROTOCOL.md` and `NONJPEG_METHOD_READINESS.md`.

Non-JPEG verification: complete provisioned CPU suite 2,604 passed /one actual
CUDA-unavailable skip, 32 existing ONNX deprecation warnings, 963.18 seconds;
total coverage 96.34%. Its first independent-auditor coverage was only 91.85%,
so ten additional source/identity/decoded-leakage/selection/integrity/PCM/symlink
guard tests were added and passed separately in 25.11 seconds. Appended coverage
over the complete run plus these ten tests is 96.41% total; each new executable
module exceeds 95% (preparation 98.25%, independent audit 97.00%, orchestration
98.92%; aggregate 97.83%). Production code/protocol hashes are unchanged since
the audited preparation. Ruff, 180-file type checking, whitespace, identity
checks and a 511,177-byte model/data-free wheel pass. Actual Python-version
matrix/GPU/full-Docker execution is not inferred from local CPU checks.
Only the two completed invocations' owned regenerable fixtures were removed
(6,047,857,095 apparent bytes); both logs, original/full and appended coverage
JSON, the first failed invocation log, raw corpora and audited carriers remain.
`CONTRIBUTING.md` now documents safe disk-backed regression scratch.

Approved five-view revision (2026-10-12): preparation can explicitly select
`center-four-corners256` using a separate manifest/representation contract.
The same original identities and simulation remain fixed; unchanged individual
views are retained. Matched training cycles all views per pair and stored-BN
evaluation equally averages five singleton probabilities. A complete SciPy
audit checks every view. This implements the grayscale representation stage,
not a pretrained/color comparison or a measured accuracy increase. The own-GPU
SSH tunnel is still unavailable. See `JPEG_FIVE_VIEW_STUDY.md`.

Actual preparation completed: the unchanged 640-original identity selection
produced 1,280 JPEGs, 1,024 fit /256 development image rows and 6,400 views.
All views passed independent SciPy reconstruction (maximum absolute difference
5.684341886080802e-14, unchanged 1e-4 tolerance), native coefficient/quantization
checks and lineage-role separation. The historical blocking original remains
with exactly the same cover/stego JPEG hashes. Identical coefficient-view pairs
are retained: fit [1,4,11,1,4], development [0,0,1,0,1] in the fixed view order.
Total cache/JPEG bytes 1,724,097,458, preparation+audit 1,291.42 seconds.
No optimizer update occurred, no model was trained/deployed, camera independence
is unverified and WIFD remains unscored. The explicit local controls preflight
returned `unavailable: cuda_not_visible_or_cpu_torch`, with no CPU fallback.
See [portable preparation evidence](../benchmarks/jpeg-five-view-preparation-20261012.json),
bound to 49 source files at `fd2b0a4` and verified against its exact Git blobs.
Only this turn's regenerable test scratch (5.84 GiB) was removed; datasets,
failed preparations, coverage JSON and the model-free wheel are retained.

Five-view implementation verification: 165 focused tests passed; full provisioned
CPU suite 2,490 passed /one actual-CUDA unavailable skip, 32 existing ONNX
deprecation warnings, 694.88 seconds. Total coverage 96.37%; the eight study
modules aggregate 97.03%, each >=95%, new view decoder 99.04%. Ruff, 177-file
type checking, whitespace, identity checks and a 498,021-byte model/data-free
wheel pass. The all-optional-absent profile passed 119 with 46 skips, but it
does not represent partially provisioned environments. The preceding `b329fc9`
CI research/quality/package/benchmark jobs passed; all four lightweight Python
jobs failed on two missing-Torch test guards. Keep this red historical outcome.
These are regression checks, not real-GPU execution or learning qualification.

The two missing-Torch guards are now explicit in cancellation/kernel-resume
tests. This does not skip native JPEG/SciPy checks or reduce the complete
research coverage gate. The complete lightweight import-absence profile,
with native JPEG/SciPy present, now passes 1,836 tests with 234 explicit skips
in 103.56 seconds. Both repaired tests also pass with Torch present. This is
a local dependency profile, not actual Python 3.11–3.14 matrix qualification;
the all-optional-absent profile alone concealed the preceding CI failure.
Remote `4e36149` subsequently passed all eight CI jobs: the actual Python
3.11–3.14 matrix, quality, package, benchmark and complete CPU research gate
([run 38239679695](https://github.com/erkanrzgc/steganography/actions/runs/38239679695)).
Later documentation/evidence-only commits have their own CI runs; their success
is not inferred from this code checkpoint.

Causal JPEG controls (2026-10-12): `research jpeg-study --config FILE --out DIR`
implements bounded original-disjoint preparation, complete independent SciPy
IDCT audit, four source/initialization/order-matched optimizer/L2/batch arms,
two seeds, numeric resume snapshots and paired bootstrap learning gates.
Legacy `srnet_reference` and old trainers/checkpoint formats are unchanged.
An initial actual preparation stopped on an upstream iteration warning; direct
Shannon measurement found 0.07937% residual, inside the unchanged 0.1% entropy
bound. Both that run and a later actual excess-residual failure are retained.
An independently implemented bounded Gibbs/Shannon refinement now corrects
excess-residual probabilities without changing costs, seeds, tolerances or
selected originals. Native audit input is a verified private byte copy.
CUDA/SSH are unavailable and no new GPU fitting or accuracy
gain is claimed. Pretrained/color comparison, .2 adaptation and reserved-source
evaluation are not implemented yet; WIFD remains unscored. See
`JPEG_LEARNING_STUDY.md` for the complete first-slice contract and remaining work.

Actual center-only data gate: after 153 complete fit pairs, the next fixed BOSS
original failed unchanged-center detection. Independent SciPy reconstruction
confirms 646 full-image coefficient changes but **zero** in the center crop;
cover and stego inputs there are identical. The four fixed corners change
576/0/7488/5504 pixels. The incomplete corpus was retained, not fitted or filtered.
See [the portable single-original audit](../benchmarks/jpeg-study-center-signal-20261012.json).
This one case proves representation loss, not the cause of chance AUC on the
whole previous corpus. The label-independent five-view revision above was
explicitly approved; the center-only run stays incomplete and no new GPU trial
has occurred. Like the other
source-bound research runners, this command requires a source checkout with
protocol docs and explicit optional research dependencies, not only the wheel.

Verification of this slice: 127 focused tests; full provisioned CPU suite
2,452 passed /one actual-CUDA unavailable skip, 32 existing ONNX deprecation
warnings, 597.48 seconds. Total coverage 96.34%; the seven new executable study
modules have 96.59% aggregate coverage and each exceeds 95%. Ruff, 176-file
type checking, diff whitespace and a 493,882-byte model/data-free wheel pass.
An explicit import-absence profile passed 88 tests with 39 optional-dependency
skips; this is a regression profile, not actual GPU or Python-version qualification.
Malformed worker input no longer imports Torch to classify non-CUDA failures.
These checks do not qualify real-GPU execution or learning. Only this turn's
generated test scratch directories are removed after verification; datasets,
partial failed preparations, caller files and public evidence are retained.

Report parity (2026-10-11): HTML now shows CTF routes, required/optional
coverage and tool execution status. Escaped complete normalized JSON and SARIF
run properties preserve additive fields, limits and incomplete jobs; bundle
views round-trip the same evidence. No score, verdict or recovery change.
The prior `c077797` checkpoint passed both normal and full-image CI.

Report parity verification: seven new tests cover complete/failed/cancelled
jobs, empty/legacy views, additive fields, HTML injection escaping, bundle
round-trips and real service output without host paths. The focused integration
run passed 77 tests. Full suite: 2,325 passed, one actual-CUDA unavailable skip,
96.32% total coverage in 548.58 seconds; all seven changed executable lines in
`report.v2` were covered. Lint, 169-file type checking and a model-free wheel
build passed. These are content round-trip checks, not formal SARIF schema or
consumer interoperability qualification; detection/recovery scores are unchanged.

CTF inspection routes (2026-10-11): successful allowlisted native findings now
rank specific next steps without changing verdicts or extraction. Unknown,
informational and failed findings are excluded; raw plug-in/tool details are
not echoed. Required native coverage gaps are not disguised as optional tools.
The prior `0b2f2ce` checkpoint passed both normal and full-image CI; its local
full suite passed 2,282 tests with one explicit CUDA skip and 96.31% coverage.
Blind method discrimination and real-dataset accuracy gates remain open.

Triage verification: 36 new recommendation tests; 86 focused CTF/evaluator/matrix
tests passed. Full provisioned suite: 2,318 passed, one actual-CUDA unavailable
skip, 96.32% total coverage (789.07 seconds); all 20 changed executable lines
in `core.ctf` covered. Lint, type checking and model-free wheel build passed.
Fresh hardened full-image smoke and independent matrix passed (17 native plus
four upstream-tool fixtures). These counts do not qualify real-world accuracy.

Paired development recovery (2026-10-11): frozen native fixtures move 14/17 ->
17/17 completed exact payload recovery, with unchanged inputs/budgets. Fixes:
BMP trailer carving and per-channel stereo WAV LSB extraction, with invalid
BMP/PCM/read-limit tests. One generated group, no real-world/blind claim.
`CTF_MATRIX_RESULTS.md` preserves both runs. The preceding `1b54f0c` checkpoint
passed complete GitHub CI and full Docker.

CTF evaluation checkpoint (2026-10-11): the frozen-manifest runner verifies
exact non-root artifact bytes after solving, retains unavailable/failed cases,
and separates complete recovery from incomplete recovery and timing. Its two
encoding examples remain development evidence. Blind qualification is always
unavailable pending independent challenge custody and audit. See `CTF_EVALUATION.md`.

The preceding `b3ba459` checkpoint passed GitHub CI (Python 3.11–3.14, quality,
package, benchmark and full CPU research) and hardened full-image recovery.
Local complete suite: 2,207 passed, one actual-CUDA-unavailable skip, 96.17% coverage.

Latest checkpoint (2026-10-11): full-image direct tools and three base-image
digests are pinned. The checked 12-tool/four-gem inventory retains upstream
license identities and verifies executable/JAR/gem-tree hashes at runtime.
Local non-root/read-only/network-disabled smoke passed native, Steghide,
explicit-wordlist Stegseek and OpenStego exact recovery, cancellation/timeout,
symlink/overwrite/output limits and recovered-passphrase redaction. See
`FULL_IMAGE_VERIFICATION.md`; these are controlled fixtures, not blind accuracy.
Focused inventory/runner coverage is >=95%; complete regression/remote Docker
checks passed as recorded above. Old failed CI is not
rewritten: seven jobs passed, but its complete CPU suite had one strict
readiness-test assumption failure. The production numerical gate is unchanged.

Source version is 0.6.0, unreleased and classified beta. Baseline 0.5.0 was
validated and committed as `1aaa747`. The complete v0.6–v1.0 roadmap has not
been delivered or validated by this slice.

## Implemented

- Frozen train-only SRNet BN diagnosis completed: 32 selected training pairs,
  26 layers, original model unchanged. Native scores saturate in 56/64 inputs;
  the batch-stat contrast in 0/64. This reveals normalization-mode sensitivity,
  not held-out accuracy; diagnostic BN inference is never enabled for primary
  verdicts. See `SRNET_TRAIN_DIAGNOSTIC_RESULTS.md`; previous failures stand.

- First frozen real-data SRNet pilot completed: one BOSS-only epoch, 412
  updates, 229.29 seconds, all 765 validation rows evaluated and nine independent
  NumPy comparisons passed. Detection failed all six cells (AUC 0.502–0.510,
  balanced accuracy 50%, FPR 24–28.78%, ECE 0.491–0.500). Nothing deployed;
  source/sample-size qualification and full ONNX replay remain unavailable.
  See `SRNET_REAL_PILOT_RESULTS.md` and portable per-cell evidence.

- `research srnet-evaluate` binds complete validation inference to verified
  training/cache/schedule/card/model provenance and independent NumPy gates.
  Real pilot results are now recorded above. Full ONNX replay/qualification
  remain unavailable. Direct evaluation deadlines are cooperative; the pilot
  invocation additionally uses an external hard wall timeout.

- SRNet fitting CLI now uses hard-limit isolated jobs and artifact verification;
  independent NumPy forward and optional dynamic-batch ONNX have generated
  readiness checks. ONNX preflight rejects unsafe allocation/topology before
  runtime. Real NumPy replay is recorded above; full ONNX replay remains pending.

- Shared SRNet paired CPU training engine and `research srnet-fit`, bound to
  complete float-cache/plan provenance with finite learning/BN checks and
  pickle-free model cards. Real one-epoch fitting/NumPy evidence is now recorded;
  trained-real-model ONNX export evidence remains unavailable. Hard-limit jobs and generated
  numerical readiness are implemented as described above.

- SRNet train-only paired schedule service and shared `research srnet-plan` CLI:
  source-first quality/method balance, complete original pair coverage, bounded
  oversampling, independent RNG, per-epoch order hashes and source exclusions.
  Frozen `SRNET_SAMPLING_PROTOCOL.md`; all 30 real scope/epoch accounting checks
  pass, complete original pair coverage and 765 validation hashes excluded.
  See `SRNET_SAMPLING_RESULTS.md`; no fitting or detector score changes.

- Full unrounded float256 preparation completed: 3,750 original hashes verified,
  18 preregistered independent mathematical examples exact; no accuracy claim.
  Separate SRNet numeric persistence validates all 183 weights/BN arrays and
  hostile NPY headers before allocation, preserves CPU RNG on reload.

- Explicit unrounded phase-aligned component-Y/256 float research caches and
  bounded SRNet fractional inference; all legacy pixel/cache defaults preserved.
  `JPEG_FLOAT256_PROTOCOL.md` freezes preparation/audit, not accuracy/training.

- Separate optional 4.78M-parameter SRNet-style structural block and explicit
  `research srnet-preflight` fixed resource-limited worker. Finite gradient/
  parameter-update and singleton/batch eval readiness, nested BN eval guards.
  No real fit, trained model or accuracy improvement; `SRNET_PREPARATION.md`
  records uint8/crop/paper differences and remaining preparation gates.

- Completed controlled residual-domain trial: three fixed fresh fits, all
  twelve cells still fail, AUC .495–.509, zero recall. Source/crop/optimizer
  controls and all previous failures retained; native/NumPy/ONNX replay passes.
  No deployment; see `PIXEL_CNN_SCALE_RESULTS.md`.

- Additive architecture-bound pixel-unit residual stem for a fixed controlled
  research follow-up; unchanged legacy default and learned layers. Numeric
  filters reject cross-domain cards/weights. `PIXEL_CNN_SCALE_PROTOCOL.md`
  freezes the experiment before real fitting; no detector improvement assumed.

- First real pixel-CNN trial completed: three frozen fits, all 12 cells fail
  detection (AUC .502–.509, zero recall). Full-source integrity, independent
  NumPy forward/metrics and ONNX batch 1/17/64 replay pass. No deployment or
  qualification; see `PIXEL_CNN_TRAINING_RESULTS.md` for every regression.

- Explicit CPU minibatch pixel-CNN train/predict services and shared CLI;
  source/class weighting on selected train rows only, fixed-scope provenance,
  bounded numeric-only model storage with array-header/finite/filter guards.
  First real fitting protocol frozen in `PIXEL_CNN_TRAINING_PROTOCOL.md`;
  no automatic model installation or primary verdict change.

- Explicit bounded JPEG uint8 center-crop caches, pinned decoder/crop provenance,
  framed isolated worker and shared `research pixel-cache` CLI. Custom optional
  residual CNN building block has independent filter/gradient smoke tests, but
  preparation alone is not training or deployed detector evidence.
  Actual preparation completed: all 3,750 original files rehashed, 18 bounded
  independent crop examples exact; see `PIXEL_RESIDUAL_PREPARATION_RESULTS.md`.

- Additive JRM training-scope provenance and a shared two-way source-exclusion
  diagnostic through the existing research CLI. Cache/split identities and both
  complete families validated before fitting; protocol in `JRM_TRANSFER_PROTOCOL.md`.
  Legacy all-source cards remain compatible; no deployed detector change.
  Completed `JRM_TRANSFER_RESULTS.md`: all eight cells fail, cross-origin AUC
  .52–.55; exact independent scalar votes for both models and all metric cells.

- Optional pinned sealwatch JRM/FLD research reference, bounded framed workers,
  separate checksum-bound float32 caches and immutable numeric inference without
  pickle. Explicit `research jrm-reference` CLI; protocol frozen in
  `JRM_REFERENCE_PROTOCOL.md`. Not deployed, not an installed ONNX model.
  Numeric model headers are checked before allocation, including adversarial
  tiny-archive/giant-array tests; score arithmetic remains unchanged.
  Full 3,750-file reference completed and independently audited: ALASKA/UERD
  AUC .698, FPR 31.71%, but JUNIWARD recall and BOSS ECE regress. All measurable
  numeric detection gates fail; see `JRM_REFERENCE_RESULTS.md`. No deployment.

- Explicit research-only residual-feature MLP64 with a linear skip, bounded
  architecture/domain validation, regularization provenance and shared export/
  inference. Legacy linear models remain compatible. The fixed comparison is
  preregistered in `JPEG_NONLINEAR_PROTOCOL.md`; not a deployed CNN.
  The completed run improves ALASKA AUC to .609/.603 and FPR to 38.54%, but
  BOSS/UERD and ALASKA calibration regress. Detection still fails; all cells
  and independent numerical audits are in `JPEG_NONLINEAR_RESULTS.md`.

- Research-only 2,066-feature JPEG block-DCT residual/parity contract and shared
  bounded worker dispatch. Two-origin weighting and old contracts preserved;
  fixed comparison is in `JPEG_RESIDUAL_PROTOCOL.md`. The actual 3,750-file run
  is complete: FPR 46.34% ALASKA / 40% BOSS, recall regressions retained,
  numerical export passes but detection fails. See `JPEG_RESIDUAL_RESULTS.md`.

- Versioned minimum native coverage by content-detected format: missing required
  components force low-score `inconclusive`, while independent positive findings
  stay visible. JSON v2 revision 2, HTML, SARIF and bundle views preserve the
  assessment. Failed analyzer signals cannot confirm. See `COVERAGE_POLICY.md`.

- Versioned measured JPEG context features, bounded local BOSS JPEG simulation,
  provenance-bound matched-family multi-origin preparation and train-only
  source/class-balanced objective. Fixed next experiment: `JPEG_CONTEXT_PROTOCOL.md`.
  The completed 3,750-file experiment and independent audit are published in
  `JPEG_CONTEXT_RESULTS.md`: source-specific FPR 50.24% / 48%, ALASKA AUC
  regression, stable ONNX parity, zero unseen validation sources. Not deployed.

- Shared cached-score source/context diagnostics with a `research diagnose`
  CLI: contract binding, source overlap, method/rate × format/declared-quality
  metrics, paired uncertainty, unknown metadata and single-label cell handling.
  Actual-artifact audits completed: 24 spatial / 9 JPEG cells independently
  checked. Both models have zero unseen validation sources and missing quality/
  camera/device metadata. See `GENERALIZATION_RESULTS.md`; no score/deployment changes.

- Explicit train-only controlled-lineage low-payload weighting and a
  checksum-bound cached-vector comparison runner. The fixed experiment is
  complete: sequential/scattered-5% recall rises to 67.93%/29.89%, but shared
  FPR doubles to 3.26%, higher-rate ranking regresses and calibration fails.
  No deployment; see `SPATIAL_WEIGHTING_RESULTS.md` for all cells and gates.

- Shared analysis context with legacy analyzer compatibility; additional
  spatial-image, JPEG structure, GIF, WAV and text signals.
- Recursive CTF service, bounded decoder/archive candidates, artifact hashes
  and provenance, CLI, API jobs/SSE/cancel/download and guided TUI.
- Additive JSON v2, HTML, SARIF and portable evidence bundles.
- Explicit signed-model catalog installation and research train/calibrate/
  export commands; lineage-aware manifests and benchmark metrics/gates.
- Full-image upstream tool installation with checksummed Stegseek, zsteg and
  dependencies, and OpenStego; Debian package licenses recorded.
- Research partitioning with reserved-pilot exclusions, connected lineage/
  camera/device groups, explicit held-out sources and exclusive output creation.
- AI triage cannot change primary scores/findings or act as detector coverage.
  See `DEVELOPMENT_DATA.md` for next-experiment prerequisites and limitations.
- Manifest-bound, bounded PNG/BMP feature extraction for train/validation;
  training input validation rejects unbound NPZ and non-train or altered rows.
  The initial spatial-summary features are experimental, not a validated detector.
- Corrected duplicate SPA/RS adjacency evidence fusion; added PDF ASCII decoder
  preflight limits, complete Flate-stream checks and shared decode budgets.
- CTF candidate hygiene now excludes crash dumps and failed/unnamed tool output,
  validates BMP carving headers, avoids lossy binary URL decoding, and preserves
  native budget cancellation and recursion depth. Per-job pilot reports are saved.
- Explicit ALASKA2 holdout acquisition: fixed-seed complete lineage selection,
  bounded authenticated ZIP64 range reads, credential isolation, CRC/JPEG/SHA
  checks and exclusive, provenance-bound resume. See `ALASKA2_ACQUISITION.md`.
- Preregistered ALASKA2 native-score evaluator: unchanged shared service,
  manifest integrity, complete paired lineages, failure-aware coverage, no
  held-out threshold search and per-method bootstrap intervals. Its protocol
  is frozen in `ALASKA2_PROTOCOL.md`; results are in `ALASKA2_RESULTS.md`.
- Separate ALASKA2 development import, bounded 968-feature JPEG extraction,
  class-balanced CPU training, train-only normalization, validation and ONNX
  export. The first fixed experiment completed, but is not deployed; see
  `JPEG_DEVELOPMENT_RESULTS.md` for inadequate scores and the original export failure.
- Explicit pinned FSDD acquisition: 3,000 real WAV covers, audited source/
  license evidence, SHA/CRC and PCM data. The preregistered 7,000-file baseline
  completed with failed native detection; see `WAV_RESULTS.md`. Public
  source/usage inventory and method-level evidence are in `DATASET_CATALOG.md`
  and `BENCHMARK_RESULTS.md`; raw corpora remain local.
- Separate spatial development acquisition completed: 1,000 BOSSbase originals
  (816 train / 184 validation), all independently rehashed/CRC/decode checked,
  with no reserved hash/member overlap. The fixed 7,000-file controlled PNG/BMP
  development comparison now completed: richer spatial features improve AUC
  but low-rate scattered recall/FPR/calibration fail. The original export
  failure remains recorded; derived stable inference now passes numerical
  gates. Neither model is deployed; see `SPATIAL_DEVELOPMENT_RESULTS.md`.
- Explicit numerical precision repair now passes for both spatial models and
  the JPEG model at unchanged 1e-6 tolerance and unchanged decision outputs.
  A separated-role threshold experiment reduces assessment false alarms by
  one cover but worsens low-payload recall; it is not installed/calibrated.
  See `INFERENCE_PRECISION_RESULTS.md` for numerical and statistical boundaries.
- Fixed 684-feature parity/residual comparison completed on the same corpus:
  scattered-5% AUC 0.716801 → 0.906486, shared FPR 4.89% → 1.63%, but fixed-.5
  low-rate recall/ECE regress. Stable export passes; no model is deployed.
  All cells and failures are in `SPATIAL_PARITY_RESULTS.md`.

## Verification

- Train-only SRNet diagnosis: Python 3.11.14, 1,139 tests pass, 94.86%
  total coverage; new core/service 125/125 statements covered. Ruff, mypy
  (115 files), whitespace and model/data-free wheel/sdist checks pass. Real
  32-pair probes verify unchanged model state/disk hash and no validation pixel
  loading. This is diagnosis, not better held-out accuracy. Python 3.12–3.14
  and fresh full Docker remain unverified locally.

- Frozen real SRNet pilot/results service: Python 3.11.14, 1,120 tests pass,
  94.80% total coverage; new diagnostic service 67/67 statements covered.
  Ruff, mypy (113 files), whitespace and model/data-free wheel/sdist checks
  pass. Both complete real evaluation runs reproduce all predictions/audits/
  metrics/intervals. All six detection cells fail; numerical success is not
  accuracy. Python 3.12–3.14 and fresh full Docker remain unverified locally.

- SRNet complete-validation service: Python 3.11.14, 1,093 tests pass,
  94.77% total coverage; evaluation/shared fit 149/149 statements covered.
  Ruff, mypy (112 files), whitespace and model/data-free wheel/sdist checks
  pass. Generated actual paired-fit/native/NumPy integration and distinct-row
  ordering are verified; no real-corpus fit/accuracy qualification. Python
  3.12–3.14, full Docker and hard-kill evaluation isolation remain unverified.

- SRNet isolated-job/numerical readiness: Python 3.11.14, 1,068 tests pass,
  94.72% total coverage; new runner/reference/ONNX/preflight 241/242 statements
  covered (99.59%). Ruff, mypy (111 files), whitespace and model/data-free
  wheel/sdist checks pass. Actual limited child fits generated fractional data;
  independent NumPy and dynamic batches 1/2/4 pass fixed logit/score/decision
  guards. No real-corpus training/replay or accuracy qualification. Python
  3.12–3.14 and fresh full Docker remain unverified here.

- SRNet fitting implementation: Python 3.11.14, 1,034 tests pass, 94.62%
  total coverage; training/fit/plan services 172/172 statements covered.
  Final strengthened source-index tests: 39 focused tests pass. Ruff, mypy
  (107 source files), whitespace and model/data-free wheel/sdist checks pass.
  Fitting uses generated fractional fixtures; forged plans, invalid gradients/
  weights and cooperative timeout cannot publish completed models. No actual
  real-corpus fit or accuracy audit. Other Python versions/full Docker unverified.

- Published SRNet schedule evidence: Python 3.11.14, 995 tests pass, 94.55%
  total coverage; sampler/plan service 116/116 statements covered. Ruff, mypy
  (105 source files), whitespace and model-free wheel/sdist checks pass.
  All 30 real schedule accounting checks pass; checkout-bootstrapped replay
  produces byte-identical records. No real fitting, accuracy or deployment.
  Python 3.12–3.14 and fresh full Docker not rerun locally.

- SRNet schedule implementation freeze: Python 3.11.14, 993 tests pass,
  94.55% total coverage; sampler/plan service 116/116 statements covered.
  Ruff, mypy (105 source files), whitespace and model-free wheel/sdist checks
  pass. Independent real schedule accounting remains pending at this freeze;
  no real fitting or accuracy evidence. Other Python versions not rerun here.

- Float256 evidence and numeric persistence: Python 3.11.14, 955 tests pass,
  94.48% total coverage; decoder/network/cache/persistence 336/337 statements
  covered (99.70%). Ruff, mypy (103 source files), whitespace checks and
  wheel/sdist builds pass. Both packages contain the services but no models or
  corpora; Torch/jpeglib remain optional. Python 3.12–3.14 not rerun locally.

- Float256 implementation freeze: Python 3.11.14, 938 tests pass, 94.44%
  total coverage; decoder/SRNet/cache code 247/248 statements covered. Ruff,
  mypy (102 files) and diff check pass. Fixed real preparation/audit is still
  pending at freeze; no accuracy or actual SRNet training implied.

- SRNet-style preparation: Python 3.11.14, 907 tests pass, 94.38% total
  coverage; new architecture/readiness code 132/133 statements covered.
  Ruff, mypy (101 files), diff check and model-free wheel/sdist inspection pass.
  Real fixed-worker generated readiness passes; not a trained detector or
  accuracy test. Old CTF Pilot race fixed by waiting for the queued UI result.
  Python 3.12–3.14/fresh full Docker remain unverified.

- Residual-domain results: Python 3.11.14, 877 tests pass, 94.31% total coverage;
  CNN/model/training code 244/244 statements covered. Ruff, mypy (99 files),
  diff check and model-free wheel/sdist build/inspection pass. Complete real
  corpus and both previous references verified; all three NumPy/native/ONNX
  replays pass, all twelve detection cells fail. Python 3.12–3.14 and fresh
  full Docker remain unverified here.

- Residual-domain implementation freeze: Python 3.11.14, 876 tests pass,
  94.31% total coverage; new CNN/model/training code 244/244 statements covered.
  Ruff, mypy (99 files), diff check and fresh wheel/sdist build pass. Other
  Python versions/full Docker remain unverified; no real new scores at freeze.

- Pixel-CNN result publication: Python 3.11.14, 869 tests pass, 94.30% total
  coverage; new network/numeric model/training code 228/228 statements covered.
  Ruff, mypy (99 files), diff check and model-free wheel/sdist build/inspection
  pass. Real 3,750-file integrity, three complete NumPy/native replay audits and
  ONNX batches 1/17/64 pass; detection fails in all twelve cells. Python
  3.12–3.14 and fresh full Docker remain unverified in this environment.

- Pixel-CNN training implementation: Python 3.11, 861 tests at freeze verification,
  94.29% total coverage; network/numeric model/training code 227/228 statements
  covered. Ruff, mypy (99 files) and diff checks pass. No real trial results or
  ONNX qualification implied; other Python versions/full Docker remain unverified.

- Latest pixel-preparation slice: Python 3.11, 824 tests, 94.19% total coverage;
  new pixel/CNN/cache code 190/191 statements covered. Ruff, mypy (97 files),
  diff checks and fresh wheel/sdist pass; Torch optional, no raw cache/models
  bundled. Real-corpus CNN training/inference, Python 3.12–3.14 and fresh Docker
  remain unverified. Earlier counts below are historical slice evidence.

- Latest source-transfer slice: Python 3.11, 779 tests, 94.08% total coverage,
  JRM services 255/255 statements covered. Ruff, mypy (94 files), diff checks
  and fresh wheel/sdist build pass. Corpus rehashed; both 765-row scalar vote
  audits exact, numeric batches 1/17/765 exact. Other Python versions/full Docker
  unverified. Earlier test counts below remain historical slice evidence.

- Latest JRM slice: Python 3.11, 757 tests, total coverage 94.03%; new reference
  services 289/290 statements covered (99.66%). Ruff, mypy (93 application files),
  diff checks and fresh wheel/sdist build pass. All 3,750 corpus files rehashed;
  independent scalar FLD votes match all 765 predictions exactly. All 12 metric
  cells checked independently; numeric batches 1/17/765 exact. Eighteen upstream
  JRM boundary examples match, not independent algorithm qualification. Earlier
  counts below are historical; Python 3.12–3.14/fresh Docker remain unverified.

- Local checks use the provisioned `venv/bin/python` environment. The host's
  system Python has an older cryptography package without Argon2id and cannot
  collect the full suite; activate the environment before running check commands.
- Python 3.11: 692 tests pass; total coverage 93.84%. New JPEG residual descriptor
  coverage is 96.30%, shared JPEG worker 98.89%. Required-coverage assessment,
  shared pipeline and v2 report renderer coverage are 100%; 52 focused coverage
  cases include a real JPEG without optional DCT and recursive CTF gap handling.
  New JPEG context features
  are 100%, corpus service 95.60%,
  multi-origin preparation 98.10%, train weighting 100%. PDF analyzer coverage is
  97.69% and image-bitplane analyzer coverage is 98.68%. The two warnings concern
  deprecated ONNX export APIs (twenty-four occurrences), not test failures.
- New spatial descriptor coverage is 100%; the explicit development runner is
  96.60%; shared feature/checkpoint code is 99.40%, reconstruction is 100%,
  precision audit is 96.39% and operating-point service is 97.40%. Independent scalar filter and
  extraction oracles, source mutations, symlinks, overwrite and budget tests
  are included. New parity descriptor coverage is 100%, comparison runner
  97.33%. New weighting helper coverage is 100%, cached comparison runner
  98.55%; source/context diagnostic coverage is 100% with 36 dedicated tests.
  Ruff, mypy (90 application files) and diff checks pass.
- The standalone ALASKA2 downloader has 59 dedicated tests and 99.09% statement
  coverage in a separate script-coverage run. Tests cover credential/redirect
  isolation, ignored HTTP ranges, ZIP64 bounds, decompression limits, symlinks,
  overwrite/resume integrity, redacted errors and reserved-corpus overlap.
- ALASKA2 evaluation has seven new tests (including the real shared-service
  JPEG path), with 98.34% statement coverage of the evaluator. Missing required
  native coverage invalidates a cell rather than turning failures into negatives.
- Extended gradient corpus and spatial proxy tests are regression evidence,
  not calibration or real-world accuracy evidence. The two legacy SPA/RS names
  represent a single adjacency observation and are aggregated only once.
- New JPEG feature/model/research-feature/research-JPEG files have 98.82%,
  100%, 100% and 98.19% statement coverage respectively. Thirty standalone FSDD
  tests cover archive/PCM bounds, license evidence, symlinks, no overwrite,
  redirects and redacted failures; separate downloader coverage is 98.18%.
- Thirty-two WAV benchmark tests verify independent payload recovery, RIFF/PCM
  bounds, speaker reservations, immutable inputs, coverage failures and fixed
  scoring. New benchmark code has 97.67% statement coverage. Twelve BOSSbase
  development acquisition tests pass; standalone code coverage is 95.54%.
- The earlier acquisition slice passed Ruff, mypy (78 then-current files) and
  diff whitespace checks; current full application verification is recorded above.
- PyTorch, ONNX, and ONNX Runtime are provisioned in the research environment.
  CPU optimizer execution, determinism/repeatability, nonfinite loss guards, and
  ONNX export parity against onnxruntime were verified end-to-end.
  Small-fixture parity passes; the actual JPEG validation artifact exceeds a
  strict absolute 1e-6 score tolerance by 1.92e-7, with zero threshold-decision
  changes. That stricter artifact check is recorded as failed, not waived.
- Version 0.6.0 wheel and sdist build successfully; the current build was verified
  to include new residual/context/coverage modules, no models or datasets, and simulation/
  DCT dependencies only behind optional extras. No package publication occurred.
- Web: one test passes; TypeScript/Vite build passes.
- Full Docker build and non-root/read-only/network-disabled smoke have passed
  for native payload and independently generated Steghide and OpenStego exact recovery.
  `scripts/docker-ctf-smoke.py` reproduces these limited integration checks.
  OpenStego's writable-preferences initialization was fixed; tool completion
  alone is not evidence of a hidden payload or successful extraction.
- The 2026-10-02 CTF regression completed and exactly recovered all 30 frozen
  Kodak challenges in each of two final runs, without increasing budgets.
  OpenStego's native allocation/thread overhead was bounded; ten cold-start
  extractions passed the hardened Docker smoke. Failed intermediate runs remain
  recorded in `CTF_REGRESSION_20261002.md`. This used read-only current-source/
  launcher mounts on the existing image, not a freshly rebuilt full image.
- Python 3.12–3.14 are configured in CI but were not executed locally.

## Open acceptance gates and limitations

- BOSSbase single-source pilot: 1,000 covers + 1,000 independently embedded
  stegos, all stegos verified through upstream extraction. Base detection
  failed: recall 0% at threshold 70, ROC-AUC 0.499021. Controlled CTF recovery
  passed 30/30, including known-password image cases and simple decoders.
  See `PILOT_RESULTS.md`; these are different capabilities, not overall accuracy.
- ALASKA2 baseline completed on all 4,000 files, without analysis errors.
  At the frozen threshold 70, JMiPOD recall is 0.4% (4/1,000), JUNIWARD and
  UERD recall 0.1% each (1/1,000); AUC is 0.525617 / 0.505157 / 0.507181.
  The same 1,000 covers give FPR 0.2%. All three methods fail the detection
  targets. The preregistered identical-pair sensitivity does not change that
  conclusion. See `ALASKA2_RESULTS.md`; this is not a trained JPEG model,
  full-Docker verdict test or cross-source support qualification.
- StegoAppDB results and adequate cross-source evaluation remain **unavailable**.
  No support cell has demonstrated the requested cross-source accuracy gates.
- The separately trained JPEG development model has validation AUC
  0.648804 / 0.592481 / 0.585306 and FPR 41.46% on 205 shared covers. It is
  experimental, uncalibrated and local; this different subset is not evidence
  of improved old-test performance. Existing primary verdicts are unchanged.
- FSDD native WAV baseline: all six sequential/scattered LSB cells at 0.05,
  0.20 and 0.40 bits/sample have AUC 0.50, balanced accuracy 50%, recall/FPR 0%.
  All 7,000 files completed with required native coverage. Independent oracle
  extraction passed for all 6,000 generated payloads; this is ground truth,
  not a native recovery score. Training/validation speakers remain separate.
- Small external-source check on all 24 Kodak images: 48 method-specific pairs,
  all stegos independently recovered. Base detection still failed (0/48 recall,
  AUC 0.49349). Controlled CTF payload recovery was 30/30, but only 25 jobs
  completed; five reported cancelled. See `KODAK_RESULTS.md`. This is a second
  source check, not the adequately sized cross-source release evaluation above.
  The subsequent published-challenge regression fixed these CTF cancellations
  (`CTF_REGRESSION_20261002.md`); it does not change the failed detection result.
- Blind 120-challenge recovery, top-three recommendations and latency: **unavailable**.
  The smoke examples are not evidence of 90% CTF recovery.
- Published, trained spatial/JPEG model packs: absent; model catalog is empty.
  Training/export plumbing is not an independently evaluated detector.
- Spatial RS/sample-pair/weighted signals are approximations. Full calibrated
  algorithms and JPEG recompression/family discrimination remain research work.
- The shared pipeline now requires minimum native per-format execution; even
  coverage-complete `no_indicators` is not a clean-file certificate or qualified
  method support. Unknown/TIFF/generic formats lack complete format coverage.
  The historical ALASKA2 evaluator explicitly requires native JPEG components
  and marks failed/incomplete cells unavailable instead of counting clean files.
- New CTF code has not yet reached the 95% coverage target across every new file.
- Per-tool OS filesystem isolation, hard native-stage deadlines, exhaustive
  fuzzing and archive-bomb/process-escape gates are incomplete. A subprocess
  working directory does not restrict its filesystem permissions.
- Debian and Python transitive dependencies are not fully locked. Installation
  of every executable does not demonstrate successful recovery with every tool.
- CTF API jobs are process-local; durable restart/resume and a full live-output
  workflow need additional work.

Use the current release as a local exploratory/CTF assistant. Validate findings
with independent tools. Statistical scores are heuristic and uncalibrated;
they are not probabilities or forensic proof.
