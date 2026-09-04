# Getting started

## Install

Use Python 3.11 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install steganography-dfir
```

The installed command and import are both `steganography`.

## Start the guided terminal workbench

```bash
steganography tui
```

Running `steganography` with no arguments does the same thing in an interactive
terminal. In a pipe or CI job it prints help and exits instead of attempting to
draw a terminal application. Use `--state-dir` to keep case data somewhere
specific:

```bash
steganography tui --state-dir ./workbench-state
```

Choose **Scan suspicious files**, enter one or more paths separated by
semicolons, leave **Balanced** selected, and start the scan. Quick Scan reads
the files in place and does not copy them. Review the verdict, individual
analyzer findings, unavailable tools, location details, and next-step note.

If the file should become evidence, enter a case name and vault password under
**Preserve in a case**. That explicit action initializes or unlocks the vault,
encrypts the source into content-addressed custody, and creates the case record.

## Hide and recover a test payload

In **Hide data**, select a supported carrier, a payload, and a new output path.
Run the capacity check. Supply a password for confidentiality; without one the
TUI requires an acknowledgement. Guided mode uses payload v3 and automatic
method selection.

In **Recover hidden data**, select the resulting carrier and a new output path.
Previewing metadata reveals envelope fields, not encrypted content. Supply the
password or placement key when required, then recover. Existing output files
are never overwritten.

## Start the browser workspace

```bash
python -m pip install 'steganography-dfir[api]'
steganography ui --state-dir ./workbench-state
```

Open `http://127.0.0.1:8000`. The API token is written with restrictive
permissions inside the state directory. See [API.md](API.md) before changing
the bind address.

Next: [User guide](USER_GUIDE.md), [Detection guide](DETECTION_GUIDE.md), and
[CLI reference](CLI.md).
