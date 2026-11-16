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

## Current status: at least 80% balanced accuracy per method

| Area | Passing | Still below 80% |
|---|---|---|
| Gray LSB (BOSS) | all payload-specific rows; unknown-payload mixed detector 85.5–88.8% | — |
| Spatial adaptive, 0.4 bpp | HILL 80.03, S-UNIWARD 80.73, MiPOD 80.68 (clean 999); WOW 80.28–80.38 bounds | — (all marginal, adaptive test) |
| Real tools (DIV2K) | steghide, OutGuess, stegano, F5 (all), OpenStego 0.4 and 0.1 (87.75) | — |
| JPEG (BOSS Q75) | J-UNIWARD 83.80, UERD 82.65, nsF5 0.2 94.25, nsF5 0.1 80.85 | nsF5 0.05 (best test 60.70; calibration ≤66%); EBS unavailable |
| WAV | AudioMNIST replacement and matching 20/40; ESC replacement 20/40 | ESC replacement 5/10 (73.1/78.4), ESC matching all (65–69), AudioMNIST matching 5/10 (56.5/59.5), FSDD all eight (~50) |
| Network, video | covert channels, containers | — |

Details: `docs/DETECTION_EXPANSION_20261111.md` (main table and sections),
latest studies `docs/JPEG_REFERENCE_FEATURES_RESULTS_20261116.md` and
`docs/WAV_CONDITIONAL_MOM_RESULTS_20261117.md`.

## Private data locations (`.benchmark/`)

- BOSS gray features/splits: `boss-gray-rich-20261111`, academic
  `boss-academic-20261112`, JPEG `boss-jpeg-academic-repair-20261113`.
- WAV corpora and layout: `wav-lsb-20261112`; corrected roles in
  `wav-modular-calibration-chunks-20261115/freeze.json`.
- Real tools: `real-tools-20261111`, binaries `tools-20261111` (F5,
  OpenStego may be run). DIV2K/BOSS raw: `external-datasets-20261109`.

## Active work

The user asked (2026-11-17) to find why the remaining cells are stuck at
50–70% and raise them, and to keep the process efficient. Next steps:
diagnose WAV detectability per dataset (signal residual vs payload, clip
length), try targeted detectors (e.g. weighted-stego estimators for
replacement), and add calibrated JPEG features (CC-JRM) for nsF5 0.05.
