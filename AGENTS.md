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
