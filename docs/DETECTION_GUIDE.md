# Detection guide

## Profiles

| Profile | Signal floor | Use |
|---|---:|---|
| Sensitive | 10 | Broad triage where missed indicators cost more than review time |
| Balanced | 25 | Recommended interactive default |
| Strict | 50 | High-volume review where weak heuristics are too noisy |

Independent evidence categories are combined with a noisy-OR score so several
correlated signals from one detector do not count as independent proof.
Verified evidence raises the deterministic score to at least 95. Optional AI
is explanation/triage only and cannot change that score or satisfy native coverage.

The [minimum native coverage policy](COVERAGE_POLICY.md) is independent of profile:
absent/failed required format components make low-score results inconclusive.
Complete execution and `no_indicators` are not proof that a file is clean.

## Verdicts and evidence

- `confirmed`: verified known marker or successful extraction.
- `likely`: score at least 70 without verified proof.
- `suspicious`: score from 30 through 69.
- `no_indicators`: usable analyzers ran and the combined score stayed below 30.
- `inconclusive`: no usable analyzer could decide because inputs or tools were
  unsupported, unavailable, or failed.

Findings carry `heuristic`, `strong`, `verified`, or `informational` evidence
strength. Reports retain analyzer status and errors. Missing optional tools are
visible and do not become negative evidence.

## Analyzer matrix

| Family | Typical formats | Signals and limits |
|---|---|---|
| Image LSB / bit plane | PNG, BMP | Bit distribution, planes, and versioned markers; noisy/generated images can resemble embedded data |
| JPEG structure / DCT | JPEG | Markers, EXIF, appended bytes, optional coefficient analysis; encoders legitimately vary |
| Audio LSB | PCM16 WAV | Sample-bit patterns and markers; processed audio may be statistically unusual |
| Text | TXT, MD | Whitespace and zero-width anomalies; typography, copy/paste, and localization can trigger them |
| File structure | PNG, JPEG, GIF, PDF, TIFF | Trailer and metadata inconsistencies; signatures and application data may be legitimate |
| Archive safety | ZIP and recognized archives | Path traversal, depth, count, ratio, and aggregate-size indicators without extraction |
| External tools | Format-dependent | Bounded `zsteg`, Stegseek, and ExifTool adapters; availability is recorded |
| Models | Declared model domain | Signed local ONNX inference; out-of-domain or uncalibrated scores remain diagnostic |

## False positives and false negatives

Steganalysis is probabilistic outside verified markers. High entropy, image
noise, transcoding, metadata editors, digital signatures, append-only formats,
Unicode typography, and deliberately adversarial input can produce false
positives. Unknown schemes, low-density embedding, encryption, lossy
transformation, unsupported formats, and unavailable tools can produce false
negatives.

Recommended validation sequence:

1. Preserve and hash the original.
2. Inspect each finding’s analyzer, evidence strength, and location.
3. Re-run with the same profile and recorded dependency versions.
4. Attempt non-destructive recovery with an appropriate key/password.
5. Compare with an independent tool or controlled clean/stego baseline.
6. Report uncertainty and unavailable coverage.

Benchmark methodology and model release gates are documented in
[MODELS.md](MODELS.md).
