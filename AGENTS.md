# Project memory

Provider-independent handoff for maintainers and assistants (Claude, Codex,
ChatGPT or others). Read it before changing code. The full historical record
of every past attempt, budget and failure is archived in
[`docs/HANDOFF_HISTORY.md`](docs/HANDOFF_HISTORY.md); consult it only when an
old result or source binding matters. Do not grow this file into a log again:
put study details in a results document and keep this file short.

## Interaction rules

* Answer the user in short, direct Turkish.
* Do the work first; do not narrate routine steps or repeat clean tool output.
* After a task, summarise in at most three short bullets.
* If something fails, state only the cause and the fix.

## Architecture boundaries

- `core/` owns reusable application services and immutable domain types.
- `modules/` contains carrier/analyzer plug-ins. Third-party analyzers using
  `Analyzer.analyze(Path)` must keep working.
- CLI, API, TUI and reports call the shared service layer; they hold no
  independent detection or extraction logic.
- Untrusted files, archives, decoder and external-tool output are always
  bounded. Never execute an extracted artifact or follow an artifact symlink.
- No cloud AI providers or uploads. The base wheel stays model-free and needs
  no external executables. Model/dataset downloads need an explicit command.
- `research/` holds experiments (type-checked); `modules/` holds analyzers.
- **Frozen closure:** do not add or change `core/` or `steganography/`
  sources. The consumed Colab budget tests reject it. Lifting this needs the
  user's approval and a change to those tests.

## Compatibility rules

- Preserve payload v2/v3 extraction and the `Carrier`/`Analyzer` contracts.
- Preserve `/v1` and `/v2` routes; new report fields are additive.
- `confirmed` requires a verified project marker or successful extraction.
- Missing required coverage gives `inconclusive`, never a clean verdict.
- Reports never contain absolute host paths, passwords, steg keys, secrets or
  an unredacted secret-bearing command. Never publish Stegseek cracking stdout.

## Environment and verification

- Interpreter: `venv/bin/python` (Python 3.11, with conseal, jpeglib,
  sealwatch 2024.12, scikit-learn, soundfile). System `python3` lacks them.
- This Kali VM is CPU-only. No local GPU work (the user's laptop becomes
  unusable); GPU runs go through Google Colab, operated by the user.
- The user runs `sudo` in their own terminal; never ask for a password.
- `git push` over HTTPS has no credentials here; push with
  `git push git@github.com:erkanrzgc/steganography.git main` (SSH key works).
- Full gate (about 35 minutes; use a disk-backed `--basetemp` and `TMPDIR`):

  ```text
  venv/bin/ruff check .
  venv/bin/mypy core modules report api steganography ui registry.py config.py cli.py research
  venv/bin/python -m pytest -q -p no:cacheprovider --basetemp <fresh dir>
  ```

  Last pass (2026-11-16): 4,651 passed, two CUDA skips, 97.42% coverage.
  Coverage gate is 90%; new analysis code targets 95%.
- Doc-only changes: run `tests/test_cloud_removal.py` (README must stay
  400–900 words with four badges and valid local links), ruff and
  `git diff --check`. Re-running the full gate is unnecessary when no
  application or test source changed.

## Commits

- One commit per file or folder. Subject: `update <path>: <what changed>`,
  all lowercase (acronyms too), a real sentence, not a few words.
- Never add co-author, "Generated with" or any assistant trailer.
- Push only after the relevant checks pass. Never rewrite pushed history
  without explicit confirmation. Never commit `nano.383070.save`.

## Data and privacy

- DIV2K, BOSSbase, ALASKA2 and derived weights/features stay private under
  `.benchmark/` (gitignored); never redistribute them.
- The ALASKA2 holdout (1,000 lineages) stays locked unless the user approves.
- Keep EVREN/Colab screenshots and account identifiers private.
- Do not delete datasets, checkpoints or failed-run evidence.

## Experiment discipline (lightweight)

The old per-step protocol/amendment chain is retired. Keep what makes results
trustworthy and skip the bureaucracy:

1. Choose models, features and thresholds on fit/calibration data only.
2. Look at the historical test once per frozen choice; never tune on it.
   Repeated looks make it adaptive evidence, so say so when reporting.
3. Report every miss with its payload; never round a miss into a pass.
4. One results document plus one `benchmarks/*.json` receipt per study is
   enough. A private `.benchmark/<study>/` folder holds code, logs and models.
5. Known source traps: BOSS training row 3,044 duplicates test row 503 by
   pixels (use the clean 999-pair test); ESC-50 roles group whole recordings;
   FSDD speaker nicolas is upscaled 8-bit; `research/dctr_features.py` needs
   `rounding="legacy"` to reproduce old caches.

## Current status against the 80% balanced-accuracy target

The overall target is not met; calibration-only gains do not replace test misses.

**Target scope (user decision, 2026-11-18):** nsF5 is targeted at 0.1 bpnzac,
and FSDD is excluded from the 80% target. Their results stay reported as
out-of-scope misses, never as passes. The only remaining in-scope gap is
AudioMNIST matching 5/10.

| Area | Passing | Still below 80% |
|---|---|---|
| Gray LSB (BOSS) | all payload-specific rows; unknown-payload mixed detector 85.5–88.8% | — |
| Spatial adaptive, 0.4 bpp | HILL 80.03, S-UNIWARD 80.73, MiPOD 80.68 (clean 999); WOW 80.28–80.38 bounds | — (all marginal, adaptive test) |
| Real tools (DIV2K) | steghide, OutGuess, stegano, F5 (all), OpenStego 0.4 and 0.1 (87.75) | — |
| JPEG (BOSS Q75) | J-UNIWARD 83.80, UERD 82.65, EBS 93.30 (corrected nzAC payload, 2026-11-18), nsF5 0.2 94.25, nsF5 0.1 80.85 (target payload) | out of scope: nsF5 0.05 (best test 60.70; calibration 66.17%) |
| WAV | AudioMNIST replacement and matching 20/40; ESC replacement all (5/10: 95.27/96.77) and ESC matching all (82.59–88.68; matching 5 interval crosses 80) — adaptive test, 2026-11-18 | AudioMNIST matching 5/10 (Colab CNN v3, all 60 speakers, 2026-11-21: 10% fresh test 79.51%, LOSO calibration 83.49%; 5% LOSO calibration 70.79%, not tested; earlier 78.43/56.5/59.5 test); out of scope: FSDD all eight (~50; cover-limited) |
| Network, video | covert channels, containers | — |

Details: `docs/DETECTION_EXPANSION_20261111.md` (main table and sections),
`docs/JPEG_REFERENCE_FEATURES_RESULTS_20261116.md`,
`docs/WAV_CONDITIONAL_MOM_RESULTS_20261117.md`,
`docs/LOW_PAYLOAD_CONTINUATION_RESULTS_20261117.md` and
`docs/EBS_RESULTS_20261118.md`. Current overview and
next-step priorities: `docs/STATUS.md` and `docs/ROADMAP.md`; older entries
there are historical, not permission to restart consumed studies.

## Private data locations (`.benchmark/`)

- BOSS gray features/splits: `boss-gray-rich-20261111`, academic
  `boss-academic-20261112`, JPEG `boss-jpeg-academic-repair-20261113`.
- WAV corpora and layout: `wav-lsb-20261112`; corrected roles in
  `wav-modular-calibration-chunks-20261115/freeze.json`.
- Real tools: `real-tools-20261111`, binaries `tools-20261111` (F5,
  OpenStego may be run). DIV2K/BOSS raw: `external-datasets-20261109`.

## Active work

The user asked (2026-11-17) to diagnose and raise the remaining 50–70% cells.
Private continuation: `.benchmark/low-payload-continuation-20261117/`.
JPEG recovery completed and passed its numeric audit: best calibration BA
66.17% at nsF5 .05; the .80 gate prevented test inference (closed).

WAV stopped at its 1,800-second cap during ESC test features because the frozen
runner re-hashed all 6,183 bound sources about twelve times per test row. On
2026-11-18 the user approved a resume: `wav/resume.py` (verify once before and
after, exact replay of the 32-row partial checkpoint, 370 new rows, no refit)
completed in 160.53 s and the unchanged `audit.py wav` passed via
`wav/resume_audit.py`. All six ESC cells now pass on the adaptive test (see the
status table). The stopped `wav/closing.json` stays preserved; WAV is closed.

Diagnosis of the remaining misses (fit covers, 2026-11-18):
- FSDD: cover-limited, not model-limited. Loud, silence-trimmed 8 kHz speech
  with ~4,000 samples; only 0.2% of samples have |x| ≤ 2. The noise floor
  (~643) dwarfs ±1 changes (0.025, about −44 dB). More model capacity will not
  reach 80%. The user excluded FSDD from the target on 2026-11-18.
- AudioMNIST matching 5/10: signal-limited, not data-limited. The approved
  2026-11-18 download (`docs/AUDIOMNIST_FULL_RESULTS_20261118.md`) gives 7,500
  fit/2,500 calibration pairs from the 30 opened speakers; the best calibration
  is 63.88/75.54%, flat from 1,500 pairs on. No test features were built. The
  user opened the 30 reserved speakers on 2026-11-21 (CNN v3, below).
- nsF5 .05: more BOSS data will not reach 80%. A fit/calibration learning
  curve (2026-11-18) gives 60.1/64.6/66.2% at 600/1,200/2,400 pairs, about
  +1.6 points per doubling, so all ~7,500 BOSS fit sources would give ~69%.
  The rich-feature extraction was not launched. The user set nsF5's target
  payload at .1 bpnzac (80.85%) on 2026-11-18; .05 stays a reported miss.
- EBS (resolved 2026-11-18): conseal 2025.11 `ebs.simulate_single_channel`
  sizes the message by all DCT coefficients, not nonzero AC, so "0.4" changed
  ~71% of nzAC. Private `.benchmark/ebs-20261118/` keeps the conseal cost and
  sets n = nzAC; test BA 93.30% (92.20–94.30%), first EBS test look.

AudioMNIST CNN (2026-11-18, `docs/AUDIOMNIST_CNN_RESULTS_20261118.md`): the
Colab T4 run of `research/wav_cnn_audiomnist.py` (notebook pinned to
`e441704`; Drive folder `steg-audiomnist-cnn`) passed its 40% control
(99.26%). 10%: calibration 83.36%, single test look 78.43% (AUC .895; the
frozen threshold shifts across speakers: TPR 64.8/FPR 7.9%). 5%: calibration
69.52%, not tested. 2026-11-19: a speaker-robust threshold rule chosen by
leave-one-calibration-speaker-out gave 78.00% on a second (adaptive) test
look; the 5 calibration speakers do not capture the shift. v2 (2026-11-19, 12 fit / 8
calibration speakers, notebook `audiomnist_wav_cnn_v2_20261119.ipynb`, Drive
`steg-audiomnist-cnn-v2`): held-out-speaker calibration BA 79.08% at 10%,
66.05% at 5%; both below the gate, test not opened. More calibration
speakers do not help; the model generalises at ~79%. v3 (2026-11-21, user
approved opening the reserved 30; 34 fit / 16 calibration / 10 fresh test
speakers; layout commit `277b479`, Drive `steg-audiomnist-cnn-v3`): 10%
LOSO 83.49% passed the gate, first fresh test look 79.51% (pair 95%
78.86–80.21%), a 0.49-point miss; 5% LOSO 70.79%, not tested. All AudioMNIST
speakers are now used and the v3 test speakers are looked at once; a new
attempt needs a different model chosen on fit/calibration only, and any
further test look is adaptive.

Strict private-script Ruff still has 13 recorded style findings; do not alter
hash-frozen scripts to clear them.

## Session handoff — 2026-11-17

The final request is documentation-only: no new experiment/development, commit
or push. At handoff `main` is at `809d9ed`, two commits ahead of the locally
cached `origin/main` (no remote refresh). The continuation results document and
`benchmarks/low-payload-continuation-20261117.json` were already untracked;
`AGENTS.md` already had uncommitted continuation notes. Preserve those changes
and unrelated `nano.383070.save`. This closeout updates documentation only;
the recorded 38 study checks and 4,651-test full gate are earlier evidence,
not newly rerun full-suite verification. See the results document's final
handoff section for exact checks, private evidence and continuation boundaries.
