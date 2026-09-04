# User guide

## Guided tasks

### Scan suspicious files

Quick Scan accepts files or directories and supports multiple semicolon-separated
paths. It is deliberately temporary: it computes hashes and analyzes source
files but creates no case, evidence, database analysis, or vault object. The
default **Balanced** profile filters weak signals. **Sensitive** increases
recall and false positives; **Strict** suppresses weaker heuristics.

The result view explains the verdict and lists analyzer, status, evidence,
offset or region when supplied, unavailable dependencies, false-positive
limits, and a suggested next action. `confirmed` is reserved for a verified
project marker or a successful extraction—not merely a high score.

### Hide data

Guided Hide performs a capacity check, uses payload v3, chooses a compatible
method automatically, and refuses to overwrite an output. Compression is
enabled by default; Reed–Solomon symbols may be added for damage recovery.
Recompression, resizing, re-encoding, metadata stripping, and text
normalization can destroy a payload. They are not interchangeable with ECC.

Passwords encrypt content with Argon2id-derived AES-256-GCM. A placement key
changes where supported methods store bits but is not a substitute for a
password. A passwordless payload requires explicit acknowledgement because it
is hidden but not confidential.

### Recover hidden data

Automatic recovery tries compatible extractable carriers and rejects ambiguous
results. Metadata preview reports method, payload version, encryption,
compression, ECC, original name, and original size when present. Recovery uses
an atomic temporary file and refuses an existing target. A wrong password
produces an authentication error and no output.

### Manage evidence

Initialize the vault once with a strong password. The derived key exists only
in process memory while unlocked. Cases contain metadata; evidence bytes are
encrypted in a content-addressed vault and deduplicated by SHA-256. Lock the
vault when custody work is complete.

Persistent case scans materialize one evidence object at a time, analyze it,
remove the plaintext staging file, update progress, and retain normalized
findings. Cancellation is honored between files. Reports can be exported as
portable HTML, JSON v2, or SARIF without exposing staging paths.

## Advanced workspace

Advanced tabs expose Cases/Evidence/Scans, raw findings, Studio configuration,
installed Models, Research status, and Doctor/Settings. Studio controls map to
core method, channel, compression, ECC, payload-version, and placement-key
options. Dataset import and held-out benchmark are supported. Training and ONNX
export are explicitly shown as requiring an experiment configuration; the TUI
does not pretend to run them.

## Keyboard and terminal behavior

- `Tab` / `Shift+Tab`: move focus
- `Enter` or `Space`: activate the focused control
- `Esc`: return to the task center
- `?`: open help
- `Ctrl+C`: exit

The interface supports 80×24. Smaller terminals display a warning and may need
scrolling. Password inputs are masked and cleared after use.

## Operational guidance

Preserve originals before destructive experiments. Record external-tool and
model availability with the report. Treat `no_indicators` as “nothing found by
this configured toolchain,” not “clean.” Validate meaningful findings with an
independent technique and document transformations applied to carriers.

See [THREAT_MODEL.md](THREAT_MODEL.md), [MODELS.md](MODELS.md), and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
