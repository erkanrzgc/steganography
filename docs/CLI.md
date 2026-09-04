# CLI reference

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
steganography models --state-dir PATH install --manifest FILE --public-key KEY
steganography models --state-dir PATH verify MODEL_ID MODEL_VERSION
steganography research import --source DIR --out FILE [--seed N]
steganography research benchmark --manifest FILE --predictions FILE --out FILE
steganography doctor --state-dir PATH
steganography list-modules
```

`research train` and `research export-onnx` fail clearly unless an explicit
experiment workflow is supplied; no placeholder output is created.

## Reproducible synthetic benchmarks

```text
steganography corpus --out DIR [--seed N] [--force] [--exclude-dct]
steganography benchmark --corpus DIR --out JSON [--html HTML]
  [--profile PROFILE ...] [--threshold N] [--jobs N]
  [--min-recall FLOAT] [--max-fpr FLOAT] [--baseline FILE]
```
