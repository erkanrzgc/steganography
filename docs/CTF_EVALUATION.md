# Frozen CTF evaluation

This runner measures exact **payload** recovery, not detector accuracy. It does
not certify blind custody. The two checked-in encoding examples are development
fixtures, not an independent 120-challenge package.

The explicit `generate-ctf-matrix.py` creates 17 independent-encoder development
fixtures, or four upstream-tool fixtures with `--external`. See
`CTF_MATRIX_RESULTS.md` for paired failures/fixes. Generated groups must not be
relabeled as independent real-world sources.

## Run

```bash
python -m pip install -e .
sha256sum examples/ctf-evaluation/manifest.json
python scripts/evaluate-ctf.py \
  --manifest examples/ctf-evaluation/manifest.json \
  --manifest-sha256 HASH_FROM_ABOVE \
  --out .benchmark/ctf-example-first-run \
  --hardware cpu-reference
```

Use a fresh output directory for each run. `evaluation.json` contains the frozen
manifest identity, evaluator/playbook/runner source hashes, application version,
hardware label, mode/limits, per-case status, exact recovery and top-three method
recommendation match, source-group count and timing. Per-method summaries expose
unavailable and unrecovered cases instead of combining methods into one detection
score. Hardware labels must be
non-secret identifiers; document the actual hardware alongside published results.
Record the repository revision too: the three source hashes are not a complete
dependency or analyzer implementation lock. CLI exit 1 means at least one job
was unavailable/failed/cancelled; exit 0 means jobs completed, not that payload
recovery or a qualification gate passed. Inspect the metrics.

## Import a challenge package

Follow the example manifest, with at most 120 cases. Freeze and record its SHA-256
**before** evaluation. Every case needs a unique opaque ID, source-group label,
method family, relative input path, input SHA-256, exact expected payload SHA-256
and explicit required tool list. Use neutral input filenames. Expected payload
bytes, method labels and hashes are never passed to the solver. Optional user
wordlists need both `wordlist` and `wordlist_sha256`; they must be <=1 MiB. No
wordlist or passwords are supplied automatically, and manifest passwords are
rejected. Required tools are checked, not installed.

The runner rejects duplicate inputs, an input used as its own answer, escaping
paths, symlink files/parents, changed checksums and limits above the production
default budgets. Inputs must be <=10 MiB. All input identities are checked before
creating output, then checked again before each job. The manifest explicitly
freezes mode and all five limits; rerunning with larger budgets is a different
experiment. Jobs use the shared CTF service and retain its extraction limits.
This is a bounded local evaluation, not a replacement for the hardened full
Docker sandbox or protection against concurrent privileged filesystem tampering.

Exact recovery is verified by hashing a **non-root artifact** inside its job
directory; copied inputs and solver verdicts alone do not count. A cancelled or
failed job can recover bytes, but cannot count as completed exact recovery. Both
counts are reported. Missing tools and failures remain in the overall denominator.
Timing includes every job returning a duration, including incomplete jobs; absent
timings are counted explicitly, not treated as zero. Suite reports omit raw tool
output, commands, oracle bytes, host paths and exception details.

## Qualification remains separate

Every run is `development_only`, with `blind_gate: unavailable`. Even 120/120
does not automatically qualify the tool. A separate custodian must establish
independent provenance, untouched challenges, appropriate method/source coverage,
no prior tuning, reference hardware and a predeclared audit. Once inspected,
challenges become regression evidence. Current generic playbook recommendations
are not comprehensive method-family recommendations; top-three misses remain
visible. Flag substring recovery and a certified blind package are not implemented
by this payload-hash runner. Do not publish fixture success as real-world accuracy.
