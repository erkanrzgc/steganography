# Project memory

This file is the provider-independent handoff contract for maintainers and
automation working on this repository. Read it before changing code.

## Architecture boundaries

- `core/` owns reusable application services and immutable domain types.
- `modules/` contains carrier/analyzer plug-ins. Third-party analyzers using the
  existing `Analyzer.analyze(Path)` contract must continue to work.
- CLI, API, TUI, and reports call the shared service layer; they do not contain
  independent detection or extraction logic.
- Untrusted files, archives, decoder output, and external-tool output are always
  bounded. Never execute an extracted artifact or follow an artifact symlink.
- Cloud AI providers and uploads have been removed. Do not reintroduce them
  through old compatibility options. Historical triage cannot affect primary
  verdicts, calibration, or benchmark gates; local ML research stays explicit.
- The base wheel remains model-free and does not require external executables.
  Model and dataset downloads require an explicit user command.

## Compatibility rules

- Preserve payload v2/v3 extraction and the public `Carrier`/`Analyzer` plug-in
  contracts.
- Preserve existing `/v1` and `/v2` routes. New report fields are additive.
- `confirmed` requires a verified project marker or successful extraction.
- Missing required coverage produces `inconclusive`, never a clean verdict.
- Reports never contain absolute host paths, passwords, steg keys, environment
  secrets, or an unredacted secret-bearing command.
- Do not write project memory beneath `.codex`, `.agents`, `.claude`, or another
  provider-specific directory.

## Verification

Run from the repository root using a fully provisioned environment:

```text
ruff check .
mypy core modules report api steganography ui registry.py config.py cli.py
pytest
```

The PR gate is Python 3.11–3.14 and total coverage >= 90%; new analysis, CTF,
and report code targets >= 95%. Real-dataset jobs may be skipped only as
`skipped`/`unavailable`, never passed.

CI's lightweight version matrix uses `pytest --no-cov`; explicit optional
dependency skips are expected. The separate CPU `research` job provisions
`requirements-research-ci.txt` and runs the complete unchanged coverage gate.
Quality checks also need Torch/ONNX to check the source-bound export contracts.

If `/tmp` is a small tmpfs, use a fresh disk-backed pytest `--basetemp` and
`TMPDIR`. Preserve the first failed run's log; filesystem exhaustion is an
unavailable verification attempt, not a code pass. Do not delete datasets or
checkpoints to fix a full temporary filesystem.

## Active milestone

PNG/BMP Colab preparation is separate from the frozen JPEG/WAV experiments.
See `docs/PNG_BMP_COLAB_PROTOCOL_20261023.md` and its preparation status.
The portable notebook binds the existing gray/RGB SRNet skeletons, Runtime
Version 2026.07 and a generated 600-second zero-optimizer profile. Native
Commons candidates remain unassigned with pending ancestry/native review.
No new media was downloaded and no real fitting is authorized. Numeric
checkpoint QA uses generated states only; it does not authorize a recipe.
The one actual Pro/T4 capacity profile completed all twelve cells in
16.034616 seconds with zero updates; see
`docs/PNG_BMP_COLAB_CAPACITY_RESULTS_20261023.md`. This attempt is consumed;
do not rerun it automatically. Optimizer/evaluation/I/O and full-training
budgets remain unmeasured; no real fit follows from this capacity result.
Keep all historical budgets, reserved data and consumed attempts unchanged.

A later explicit continuation permits the separate native Commons admission
pilot, bound by `docs/PNG_BMP_NATIVE_ADMISSION_PROTOCOL_20261023.md` and its
unattempted-source amendment. Six originals (three L /three RGB), six author
groups and 24 PNG/BMP artifacts pass twelve independent payload checks;
all roles remain unassigned. Preserve the first partial unavailable acquisition
and later HTTP 429, with no retries. Conservative cumulative raw-transfer
debit is 55,307,394 /67,108,864 bytes; the older historical total stays null.
See `docs/PNG_BMP_NATIVE_ADMISSION_RESULTS_20261023.md`. No further download,
GPU attempt or real fit follows from the completed pilot. The 128-originals
per-domain training plan is a non-executable draft with missing data, adapter,
learning control and optimizer/evaluation/I/O cost; it does not grant fitting.
The native admission full software gate passes 3,888 tests /two CUDA skips,
97.05% total coverage; the new admission service has 100% coverage.

A later continuation permits data-only expansion under
`docs/PNG_BMP_NATIVE_EXPANSION_PROTOCOL_20261023.md`: a separate 512-MiB
reservation ceiling, native photographic L/RGB PNG sources, fixed crop
texture gates and at most eight originals per author group per mode.
The private source-bound paced schedule preserves every intent/reservation,
respects server cooldowns and never retries a previously requested SHA-1.
Keep actual roles unassigned, microscopy excluded pending the user's
preference, existing reserved data closed and both historical ledgers intact.
This is collection/preparation only; no new GPU attempt or real fitting.

The original expansion parent was interrupted after batch-1057. Batch-1058
has an attempt marker but no source intent/reservation; keep it unavailable
and skip that source. The user's 2026-10-24 continuation permits the bound
unattempted-only parent in
`docs/PNG_BMP_NATIVE_UNATTEMPTED_CONTINUATION_20261024.md`, not a retry.
Its user service `png-bmp-native-unattempted-20261024.service` has a four-hour
deadline, 2-GiB memory /zero swap and no restart. Preserve all 84,115,935
previously reserved expansion bytes and the old 55,307,394-byte prior debit.
All eighty metadata requests are consumed. Original source code, the frozen
catalogue, source holds and all fitting restrictions remain unchanged.
The user subsequently requested a stop. The continuation service is inactive
with no automatic resume. Preserve the stop receipt in
`benchmarks/png-bmp-native-expansion-user-stop-20261024.json`. The last fully
reviewed subset is snapshot 007: 13 L /79 RGB originals, 368 PNG/BMP artifacts
and 184 scalar payload checks. Later acquisitions remain unreviewed and
unassigned; do not admit them or restart collection automatically.

The user's 2026-10-26 explicit continuation supersedes that stop for data-only
preparation. Offline snapshot 008 reviews the backlog: 13 L /87 RGB originals,
400 PNG/BMP artifacts and 200 independent payload checks. Preserve the three
holds and both montage/illustration exclusions. The fresh source-bound parent
and service are documented in `docs/PNG_BMP_NATIVE_USER_RESUME_20261026.md`.
They skip every prior attempt/log and SHA-1 intent/result, keeping all
248,412,639 expansion bytes and the separate 55,307,394-byte debit at resume.
Debit 4,143 seconds from the old continuation: the new service allows at most
10,257 seconds of the original four hours, with 2-GiB memory /zero swap and
no restart. The original 512-MiB ceiling and exhausted metadata budget remain.
Sixteen generated resume checks pass with 100% coverage. New acquisitions
still require independent review; no role assignment, GPU attempt or fitting.

This resumed collection/preparation is now complete at its transfer ceiling.
The service is inactive /PID zero with no restart. Snapshot 016 has 118
reviewed originals (13 L /105 RGB), 472 PNG/BMP artifacts and 236 independent
scalar payload checks; no unreviewed candidate remains. Preserve all three
holds and both exclusions. Final replay closes 192 requests /189 raw files
with the complete reservation chain. Expansion debit is 536,421,895 of
536,870,912 bytes, plus the separate old 55,307,394 bytes; the older total
stays null. The unused 449,017 bytes cannot fit the next bound source.
Conservative old-plus-new continuation time is 7,035 /14,400 seconds; none
of the remaining time grants a retry or new download. Exact post-exit peak
memory is unavailable after transient-unit unloading; preserve the failed
observer and earlier kernel samples instead of rerunning collection.
See `benchmarks/png-bmp-native-expansion-closeout-20261026.json` and the
updated results. Fresh full QA passes 3,888 tests /two CUDA skips, 97.05%
coverage; all 59 additional generated checks pass. The corpus still lacks
115 L /23 RGB originals and ten L groups for the draft target. Keep roles
unassigned, reserved data closed, metadata exhausted and fitting unauthorized.
This attempt is consumed; no automatic collection, GPU attempt or real fit.

The existing local GPU host is Windows 11 /Ubuntu-24.04 WSL2 with the user's
RTX 5060 Laptop /8-GB VRAM. `docs/WSL_GPU_READINESS_RESULTS.md` already records
physical CUDA verification with Python 3.12.3 /Torch 2.14.0+cu130 on 2026-10-09;
later local probes and real JPEG fits are complete historical experiments.
Read that record, `docs/WINDOWS_GPU_TRAINING.md` and `docs/WSL_GPU_RECONNECT.md`
before proposing GPU/WSL installation. This Kali workspace is the separate
CPU-only VMware guest; its missing CUDA device says nothing about existing
host installation. The dedicated `steganography-wsl` SSH alias/key/known-host
files remain; the initial 2026-10-26 check found the port-22240 listener absent.
Restore the existing connection rather than reinstalling Ubuntu, replacing
keys or rerunning a consumed experiment. Current host driver/package state
needs read-only revalidation after reconnect. See the corrected
`docs/PNG_BMP_LOCAL_READINESS_20261026.md`. No new PNG/BMP GPU profile, media
acquisition or fitting was executed by this hardware assessment. Preserve
the CPU CI, frozen Colab environment and every previous local result/budget.

On 2026-10-28 the user restored the existing WSL tunnel. Read-only inventory
confirms RTX 5060 Laptop /capability 12.0, driver 617.42, Python 3.12.3 and
Torch 2.14.0+cu130; see `benchmarks/png-bmp-wsl-connection-20261028.json`.
This is not a new GPU profile. New generated-only software lives in
`core/spatial_training_data.py` and `core/spatial_generated_training.py`:
byte-rounded forty-percent scattered plane-0 controls, paired FP32 batches,
bounded AdamW step/scoring and numeric checkpoint continuation. Fifty-seven
focused software tests pass, including actual small CPU optimizer steps and
exact gray/RGB interrupted-state parity. These are QA, not a full learning
control or native accuracy result. See the dated preparation document.
The new generated-control and data-gap documents are non-executable drafts;
the actual source-bound supervisor/acceptance/independent debit contract is
still missing. Do not run a GPU profile, full generated control or native fit
from these drafts or rerun any old job. Existing real media stays unassigned.
The largest ten RGB groups hold only 41 originals; 23 more images cannot
produce the old ten-group /80-original training partition. A future split
needs independent group-feasibility review, without changing historical data
or assigning real roles. All prior transfer/metadata ceilings remain consumed.
The new preparation full gate passes 3,945 tests /two explicit Kali-CUDA
skips, 97.07% total coverage; Ruff and 233-file mypy pass. New data/training
modules have 100% /99.34% statement coverage. The proposed generated manifest
contains 220 sources /220 payload checks with disjoint source/group namespaces;
no image/model was persisted by manifest generation. See
`benchmarks/png-bmp-generated-preparation-verification-20261028.json` and
`benchmarks/png-bmp-generated-corpus-20261028.json`. Preserve the first focused
fixture failure. This is generated CPU software QA, not a full learning gate,
physical CUDA execution or permission to fit or acquire real media.

A later user continuation permits the separate generated-only intervention in
`docs/PNG_BMP_GENERATED_CONTROL_PROTOCOL_20261028.md`, conditional on its full
software gate, source/environment freeze and deployment checks. The new
`spatial_control` /`spatial_control_io` /`spatial_control_supervisor` services
implement four fresh profile cells (sixteen generated updates), conservative
cost projection and, only if eligible, four fresh fits (1,600 updates).
Its separate allowance is 1,800 seconds /8-GiB RAM /zero swap /4-GiB allocator
and 512-MiB output, with one exclusive attempt marker and no retry. All actual
learning inputs remain generated in memory; PNG/BMP transport is separately
measured. The earlier draft and preparation results remain historical snapshots.
This conditional continuation never authorizes native fitting or acquisition.
Record the actual outcome before any later action; never relaunch a consumed
attempt or treat generated-only accuracy as native qualification.

That one generated intervention is now complete: **passed_generated_only**.
All four cells completed 1,600 learning updates /12,800 presentations, plus
sixteen profile updates; 1,616 commits and eight numeric checkpoints verify.
Final generated development AUCs are .723958 /.706597 (L) and .763889 /.769097
(RGB), with train AUC 1.0 in every cell. All sixteen AUCs also replay through
an independent pairwise calculation; 1,533 parent observations and all 1,528
nonnull progress anchors verify.
Kernel execution is 1,356.435690 seconds; conservative time through independent
closeout is 1,598.898706 /1,800 seconds. Peak host memory is 3,182,301,184 bytes;
allocator allocated/reserved peaks are 3,849,946,112 /4,089,446,400 bytes.
The complete private WSL package is 469,630,247 logical bytes, below 512 MiB.
Preserve the failed direct post-exit cgroup observer: the released empty
cgroup is unavailable, while retained systemd and parent memory peaks agree.
All eight checkpoint hashes and 243 staged files pass a separate closing
audit. Numeric models remain private in WSL; the unit is inactive /PID zero,
with no restart. This attempt is consumed; no remaining time grants a retry.
See `docs/PNG_BMP_GENERATED_CONTROL_RESULTS_20261028.md` and its full receipts.
Full software QA passes 4,008 tests /two explicit CUDA skips, 97.13% coverage;
63 focused tests pass, and all four new modules exceed 95%. No native media,
download, role assignment, real fit or deployment follows from this pass.
The corpus deficit, RGB group feasibility and native cache/training adapter
remain unresolved. Native integrated I/O and its full-training allowance are
unavailable; any native fit needs a separately frozen contract and approval.

`v0.6`: shared analysis context, PNG/BMP native analysis, bounded CTF playbook,
portable JSON/HTML/SARIF/bundle reporting, and initial CLI/API/TUI surfaces.
See `docs/ROADMAP.md` for current status and later milestones.

The dated JPEG four-arm/two-seed matrix and both read-only diagnoses are
complete; see `docs/JPEG_CONTROL_RESULTS.md`. No candidate passed the combined
learning rule; production qualification is unavailable and none was deployed.
Preserve the 16 private snapshots and historical cancelled receipts; any next
training intervention needs a
separately frozen source/data/exposure protocol, not a blind restart.

The separate second-round JPEG refinement protocol is frozen in
`docs/JPEG_REFINE_PROTOCOL.md`. Use its shared warm-up/branch ancestry and
cumulative six-hour/ten-GiB limits; unavailable GPU/preflight is not learning
evidence. Historical source bindings and snapshots remain immutable.

The separately frozen numerical preflight amendment is documented in
`docs/JPEG_REFINE_NUMERICS_PROTOCOL.md`. Preserve the original failed gate;
use the complete amended source closure and explicit prior time/storage debits.
It changes validation only, never the training recipe or learning rule.

Native linear fitting is separately frozen in
`docs/NONJPEG_LINEAR_COMPARISON_PROTOCOL.md`: two seeds, grouped train-only
CV/thresholds, 20 epochs, independent 30-minute CPU /32-MiB output allowance.
It does not authorize native deep-model training or consume extra JPEG time.

The separately bound native process-path correction is specified in
`docs/NONJPEG_PROCESS_PREFLIGHT_PROTOCOL.md`. Keep the failed zero-update
attempt and original source bindings immutable; debit its time/storage before
one amended attempt, including read-only replay. Never relax carrier symlink
checks or automatically retry a failed amended fit.

The amended native pilot is complete: 14 numeric models and 480 cells replayed,
with 20,200 real optimizer updates including CV. WAV balanced accuracy remains
0.50 and spatial changes are mixed; no detector is qualified or deployed.
The amended JPEG refinement also completed all 28,672 updates without passing
its combined source/seed rule. Preserve both completed experiments and every
failed receipt; any further fitting needs a separately frozen intervention.

The separate WAV read-only diagnosis is specified in
`docs/WAV_READONLY_PROTOCOL.md`: train feature artifacts only, existing model
inference and cached development metrics at their original thresholds. Never
refit missing OOF folds, open reserved media or sweep validation thresholds.
Preserve all 14 model digests; diagnosis does not authorize another fit.
This diagnosis completed four audio models /32 train cells with zero updates;
all 14 digests are unchanged. See `docs/WAV_READONLY_RESULTS.md`.
`docs/WAV_NEXT_PROTOCOL_DRAFT.md` is not executable: AudioMNIST review is
metadata-only, parent-recording ancestry is unavailable, and its cuts must stay
grouped by speaker. The original cumulative raw-transfer ledger is incomplete;
do not reset it or treat a draft allowance as permission to download or fit.
The bound AudioMNIST pilot (`docs/WAV_PILOT_TRAINING_PROTOCOL_20261017.md`)
first failed its generated CUDA preflight on a missing linear `seed` receipt
key (keep that record `unavailable`). The receipt amendment
(`docs/WAV_PILOT_RECEIPT_AMENDMENT_20261019.md`) passed one generated preflight
within all limits. Real fitting stays unauthorized until explicitly approved.
A generated positive control then showed the v1 CNN does not learn; draft v3
(`core/wav_pilot_v3.py`) also failed its preregistered gate at reduced exposure.
Do not fit real AudioMNIST until a recipe passes a full-exposure control.
`docs/WAV_FULL_CONTROL_PROTOCOL_20261021.md` defines the next source-bound,
read-only acceptance contract. Its verifier requires both full seed schedules,
disjoint generated splits, all eight cells and resource evidence. At the
acceptance-contract freeze, the full generated runner/supervisor was missing
and needed its own frozen binding;
the acceptance contract alone authorizes neither a GPU attempt nor real fitting.
A separate full runner is now implemented in `core/wav_control_runner.py` and
bound by `docs/WAV_CONTROL_RUNNER_PROTOCOL_20261021.md`. Its generated corpus
manifest is frozen with zero updates. It permits one bounded generated control
only after the full software gate and deployment checks; it never authorizes
real fitting. Its v2 evidence amendment preserves completed seed evidence.
The full software gate passed 3,528 tests /two CUDA skips, 96.93% coverage;
new modules have 100% coverage. The one full generated attempt is unavailable:
1,937 confirmed updates as a lower bound, one complete scored seed and
797/1,140 confirmed second-seed updates before the frozen deadline. The full
first seed also fails the learning rule: train AUC 0.585101 and 40%-plane-0
development AUCs 0.557425/0.549725. Independent systemd monotonic time agrees
with the supervisor; the human wall-clock discrepancy is not a timer fix or
retry authorization. See `docs/WAV_FULL_CONTROL_RESULTS_20261021.md`.
Never relaunch it automatically; any further fit needs a separately frozen
intervention. Real fitting is unauthorized. Keep all prior acceptance, QA and
failed-control records intact.

The next untrained representation is `core/wav_pair_candidate.py`: ten
full-alphabet pair-energy statistics, with an optional 418-parameter FP32
head. Its 26 generated scalar/gradient tests pass with 100% module coverage;
the full gate passes 3,554 tests /two CUDA skips, 96.94% coverage. `docs/WAV_PAIR_CANDIDATE_DRAFT_20261022.md`
is a non-executable design, not a new training recipe or retry authorization.
There is no optimizer, cache/training adapter or candidate accuracy result.
Pair-count swaps can merge a one-LSB change; universal raw-PCM equivalence
is not claimed. See `docs/WAV_PAIR_CANDIDATE_VERIFICATION_20261022.md` for
this limit and the adaptive-development-data reuse caveat.
Do not replace the old recipe-bound gate with this domain; any fit needs a
separately frozen contract and real AudioMNIST remains unauthorized.

A separate pair-control service now exists in `core/wav_pair_control.py`,
bound by `docs/WAV_PAIR_CONTROL_PROTOCOL_20261022.md` and its 34-file protocol.
This later contract permits one CPU zero-update profile followed only on
success by one fresh generated two-seed fit. Preserve the earlier untrained
design snapshot, every old failed control and all historical source bindings.
Its independent 30-minute /2-GiB /zero-swap /64-MiB allowance adds no JPEG time
and resets no transfer ledger. A profile failure forbids fitting; any attempt
failure forbids automatic retry. Real media, downloads, checkpointing and real
AudioMNIST fitting remain unauthorized by this separate generated contract.
That one intervention is now complete and **failed_learning_gate**: both seeds
completed 1,140 updates /72,000 presentations, with target development AUCs
0.541750/0.520375 and 0.546025/0.495800 against 0.70. All 4,800 payload checks
passed; independent charged time is 30.556292 seconds. Full QA passes 3,598
tests /two CUDA skips, 96.97% coverage; both new modules exceed 95% coverage.
See `docs/WAV_PAIR_CONTROL_RESULTS_20261022.md`. Keep this complete failure;
do not rerun or tune it automatically. Any next fit needs a newly frozen
intervention; real AudioMNIST fitting remains unauthorized.

The pair-control read-only diagnosis is complete; see
`docs/WAV_PAIR_DIAGNOSIS_RESULTS_20261022.md` and its separately frozen protocol.
Only the 3,600-row train cache and metadata were opened, with zero updates.
All 80 feature/cell AUCs agree between independent methods; both initial
gradient identities and unchanged-parameter checks pass. Q0 plane-0 train
AUCs 0.909031/0.753063 show signal survives caching. Plane-1 dominates the
initial signal-gradient projection (93.67%/89.57%), supporting a scale/
competition hypothesis, not a proven full-training cause. No trained weights,
loss trajectory or actual gradient history exist; do not refit to reconstruct
them. The next raw-versus-standardized matched comparison in
`docs/WAV_PAIR_NEXT_INTERVENTION_DRAFT_20261022.md` is non-executable. It grants
no new fit, download, retry, native-media access or real-data permission.

A later, separate matched scaling comparison is implemented in
`core/wav_pair_comparison.py`, bound by the 42-file contract in
`docs/WAV_PAIR_COMPARISON_PROTOCOL_20261022.md`. The user's continuation
authorizes one conditional generated CPU intervention under that new contract:
fresh generator seed 2026102204, raw versus train-only z-score input, both
original model seeds and four full fits /4,560 updates. The old draft remains
non-executable by itself. Require the full software gate, successful zero-update
profile and frozen normalization/cache/debit binding before fitting. Preserve
the matched initialization/order, all epoch/step telemetry and score arrays
for independent AUC replay. Its separate 30-minute /2-GiB /zero-swap /64-MiB
allowance adds no JPEG time and resets no transfer ledger. No retry, checkpoint,
native-media opening, download or real AudioMNIST fit is authorized.
This one comparison is complete: all four fits /4,560 updates /288,000
presentations. Raw failed; train_zscore **passed_generated_only** with target
development AUCs 0.984600/0.950725 and 0.979150/0.945625. The old raw CNN
attempt remains unavailable and the earlier pair run remains failed.
All 4,800 payload recoveries, 72 independent score-AUC replays and resource
limits pass; independent charged time is 31.407780 seconds. Full QA passes
3,662 tests /two CUDA skips, 96.99% coverage; new modules exceed 95%.
See `docs/WAV_PAIR_COMPARISON_RESULTS_20261022.md`, including weak/regressing
scattered 5%-p1 cells. Do not rerun or tune the completed comparison. This
pass belongs to the standardized pair-feature domain, not the old CNN-bound
native pilot. Any native fit needs a separately frozen source/data/domain/
exposure/grouping protocol and explicit approval; no model was deployed.

The separate native standardized pair-head contract is now implemented in
`core/wav_pair_native.py` and `docs/WAV_PAIR_NATIVE_PROTOCOL_20261022.md`.
The current continuation permits software QA and one cached train/development
zero-update preparation profile, not real fitting. Preserve the 20/10 speaker
roles and all 30 reserved groups; no download or historical budget reset.
Its two-seed /2,280-update CPU fit adapter requires later explicit approval
bound to the protocol/profile and independent profile time debit. Do not
create an approval receipt from this preparation request or a generated pass.

That one native preparation profile completed: all 600 originals /4,800
payload checks, 20 backward batches and unchanged parameters, zero optimizer
updates. Full QA passes 3,729 tests /two CUDA skips, 97.00% coverage;
new modules exceed 95%. Independent profile debit is 24 seconds of the
separate 1,800-second allowance. See `docs/WAV_PAIR_NATIVE_PREPARATION_20261022.md`.
Do not rerun the profile automatically. Actual native fitting is still
pending explicit user approval; no approval receipt exists for this protocol.

On 2026-10-23 the user explicitly approved the one native two-seed CPU fit.
The bound approval and original supervisor completed both full seeds:
2,280 actual optimizer updates /144,000 presentations, with 40 train epochs,
eight actual gradient/step records and all saved score arrays. The exploratory
target failed: required development 40%-p0 AUCs are 0.478800/0.486625 and
0.466650/0.468700 against .70; pooled AUCs are .616641/.622978, with .55/.42
cover FPR. All 36 AUCs and 80 metric/cluster/paired-interval checks replay;
all 49 source bindings and 14 old model digests remain unchanged. Charged
profile+fit time is 30.059356/1,800 seconds. No checkpoint or deployment;
reserved speakers remain closed. See `docs/WAV_PAIR_NATIVE_RESULTS_20261023.md`.
The earlier pending-approval records are historical snapshots. This one fit
is consumed; never relaunch it automatically or use remaining time for tuning.
Any further fit requires a distinct frozen intervention and explicit approval.

All built-in extension data states are recorded in
`docs/ALL_FORMAT_DATA_STATUS.md`. The local PDF/TXT/MD/ZIP preparation protocol
is frozen in `docs/ALL_FORMAT_DATA_PROTOCOL.md`; source/license-bound local
documents and controlled variants stay `development_unassigned`, grouped by
whole upstream package. They are not representative detection benchmarks or
certified clean files. No download, role assignment or fitting is authorized
by the local preparation contract.
The local pilot completed 16 documents /48 controlled artifacts across four
package groups; all payload/hash checks pass and all roles remain unassigned.
The complete inventory has 23 suffixes, including CSV/LOG. See
`docs/ALL_FORMAT_LOCAL_RESULTS.md`; retain the original README-gate failure
and explicit repair receipts. No representative all-format dataset or new
training qualification follows from this preparation.

Full-image tool changes must update `tools/full-image.lock.json`, preserve
upstream license notices and run the hardened checks in
`docs/FULL_IMAGE_VERIFICATION.md`. Inventory presence is not recovery evidence.
Never publish Stegseek wordlist cracking stdout (it can expose recovered keys).

## Handoff checklist

1. Preserve unrelated working-tree changes and avoid history rewrites.
2. Update architecture, roadmap/status, benchmark protocol, and changelog when
   a slice changes their contract.
3. Add independent fixtures and adversarial limit/security tests.
4. Record optional dependency/tool absence as coverage, not success.
5. Run focused tests, the full suite, lint, type checking, and `git diff --check`.
6. Keep commits small and state any unverified environment-dependent checks.

## Interaction and response rules

* Yanıtlarını kısa, net ve doğrudan ver.
* Yaptığın rutin işlemleri adım adım anlatma.
* Açıklama yapmadan önce mümkünse doğrudan işlemi gerçekleştir.
* Ben istemedikçe bariz veya basit şeyleri açıklama.
* Terminal komutlarının çıktısını önemli bir hata yoksa tekrar etme.
* Araç ve terminal kullanımını gereksiz yere anlatma.
* Kod değişikliklerinden sonra yalnızca önemli değişiklikleri belirt.
* Bir görev tamamlandığında sonucu en fazla 3 kısa maddeyle özetle.
* Uzun giriş, sonuç ve gereksiz bağlam yazma.
* İşlem başarılıysa uzun açıklama yapma.
* Sorun yoksa final cevabını mümkünse 5 satırdan kısa tut.
* Bir hata varsa yalnızca hatanın nedenini ve gerekli çözümü açıkla.
* Kod yazarken veya düzenlerken açıklama yerine uygulamaya öncelik ver.
