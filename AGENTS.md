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
