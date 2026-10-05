# CLI reference

Research source/context audit: `steganography research diagnose --manifest FILE
--predictions FILE --predictions-sha256 HASH --model-card FILE
--model-card-sha256 HASH --out NEW_FILE [--threshold 0.5]` evaluates bound cached
validation scores without retraining or opening images. See
`GENERALIZATION_PROTOCOL.md`; this never qualifies a model for deployment.

All commands support `--help`; `steganography --version` prints the version.
The banner is suppressed by `--quiet` and for machine-oriented commands.

## Terminal and web interfaces

```text
steganography tui [--state-dir PATH]
steganography ui [--state-dir PATH] [--host HOST] [--port PORT]
steganography serve [--host HOST] [--port PORT] [--workers N]
```

With no arguments, an interactive terminal opens the TUI. Pipes and CI receive
help text. `ui` starts the bundled browser workspace and API; `tui` talks to
services directly and starts no localhost server.

## Hide and recover

```text
steganography embed --in PAYLOAD --carrier CARRIER --out OUTPUT
  [--method NAME] [--steg-key KEY] [--channels rgb]
  [--payload-version {2,3}] [--compress] [--ecc-symbols 0..255]
  [--no-clobber] [--password TEXT|--password-file FILE|--password-stdin]

steganography extract --in CARRIER --out OUTPUT [--method NAME]
  [--steg-key KEY] [--no-clobber]
  [--password TEXT|--password-file FILE|--password-stdin]
```

`STEGANO_PASSWORD` is the last password fallback. Prefer stdin or a
permission-restricted file to avoid shell history. Compatible CLI embedding
defaults to payload v2; Guided TUI embedding defaults to v3.

## Analyze and scan

```text
steganography analyze --in FILE [--profile sensitive|balanced|strict]
  [--format text|json-v1|json-v2] [--json] [--max-file-size MIB]
  [--fail-on medium|high] [--ai] [--allow-ai-file-upload]

steganography scan --dir DIRECTORY --out REPORT
  [--report json|html|json-v1|json-v2|ndjson|sarif] [--jobs N]
  [--follow-symlinks] [analysis options]
```

Directory scan skips its own output and does not follow symlinks unless asked.
Module failures become structured results. `--fail-on` returns 1 when a result
meets the threshold; user errors return 2 and interruption returns 130.

## CTF recovery playbook

```text
steganography ctf INPUT --out NEW_DIRECTORY
  [--mode quick|balanced|deep] [--wordlist FILE] [--password-stdin]
  [--max-depth 3] [--max-artifacts 256] [--max-bytes 1GiB]
  [--timeout 180] [--report json|html|sarif|bundle]
```

`balanced` is the default. The total job and every optional process are bounded;
decoder/carving output remains a candidate until a marker or successful
extraction verifies it. The output directory must not already exist. Reports
contain sandbox-relative names and redacted commands, never passwords or host
paths. Wordlists are user-supplied and are not packaged.

## Cases and reports

```text
steganography case --state-dir PATH create --name NAME
  [--description TEXT] [--retention-days DAYS]
steganography case --state-dir PATH add CASE_ID --file FILE [password options]
steganography case --state-dir PATH scan CASE_ID
  [--profile sensitive|balanced|strict] [password options]
steganography case --state-dir PATH export SCAN_ID
  [--format json|html|sarif] --out FILE
```

## Models, research, and diagnostics

```text
steganography models --state-dir PATH list
steganography models --state-dir PATH catalog
steganography models --state-dir PATH install MODEL@VERSION --accept-license
steganography models --state-dir PATH install --manifest FILE --public-key KEY
steganography models --state-dir PATH verify MODEL_ID MODEL_VERSION
steganography research import --source DIR --out FILE [--seed N]
  [--license NAME] [--source-url URL]
steganography research benchmark-suite --manifest FILE --predictions FILE --out FILE
  [--min-roc-auc .90] [--min-balanced-accuracy .85] [--min-recall .80]
  [--max-fpr .03] [--max-ece .05] [--bootstrap-samples 200]
steganography research calibrate --manifest FILE --predictions FILE --out FILE
steganography research train --config EXPERIMENT.json --out CHECKPOINT.pt
steganography research export --checkpoint CHECKPOINT.pt --out MODEL.onnx
steganography doctor --state-dir PATH
steganography list-modules
```

Training consumes an explicit NumPy-feature experiment configuration and
requires the `research` extra. Export emits ONNX plus a preprocessing/model-card
contract. Calibration is dependency-free scalar temperature fitting on the
declared train or validation split; test data is rejected for calibration.

## Reproducible synthetic benchmarks

```text
steganography corpus --out DIR [--seed N] [--force] [--exclude-dct]
steganography benchmark --corpus DIR --out JSON [--html HTML]
  [--profile PROFILE ...] [--threshold N] [--jobs N]
  [--min-recall FLOAT] [--max-fpr FLOAT] [--baseline FILE]
```
