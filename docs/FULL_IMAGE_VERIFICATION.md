# Full image verification

The optional full image is an integration environment, not a stronger accuracy
claim. The base wheel and core image still need no forensic executables.

## Reproduce

```bash
docker build -f Dockerfile.full -t steganography-full-check .
docker run --rm --read-only --network none --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --mount type=bind,src="$PWD/scripts/docker-ctf-smoke.py",dst=/smoke.py,readonly \
  --entrypoint python steganography-full-check /smoke.py
```

Build inputs use three digest-pinned base images, exact direct Debian tool
versions and SHA-256-checked Stegseek/OpenStego/gem distributions. See
[the lock](../tools/full-image.lock.json) and [notices](../THIRD_PARTY_NOTICES.md).
APT retains its signed Release/package checksum verification. Missing pinned
versions fail the build: update pins and reverify deliberately when repositories
retire them. These pins opt out of automatic tool/security upgrades; they do
not mean the image is vulnerability-free. Review and rebuild regularly.

The build verifies 12 tools and four Ruby dependencies and records executable,
OpenStego JAR, installed gem-tree and license-document hashes. Runtime smoke
checks the installed inventory against that baked baseline. Gem license labels
are upstream package declarations; this is not a legal certification. Debian
copyright notices and Stegseek's COPYING are retained in the image.

Inventory verification uses bounded probes through the shared ToolRunner.
Unavailability or version/license/identity differences fail the gate. Reports
exclude host paths, credentials and environment contents. An inventory without
an actual recovery test is not proof that a tool works.

## What the smoke checks

The workflow also runs `scripts/docker-ctf-matrix.py` under the same hardened
settings: 17 independent native encoder fixtures and four actual upstream-tool
fixtures (Steghide BMP/JPEG, OutGuess JPEG, OpenStego PNG). Its separate
`ctf-development-matrix` artifact is controlled recovery evidence, not blind
validation or detector accuracy. See `CTF_MATRIX_RESULTS.md`.

- Non-root operation on a read-only root with no network and dropped privileges.
- Exact independent payload recovery through native, Steghide, Stegseek and
  OpenStego paths; Stegseek wordlist recovery through the shared CTF playbook,
  and repeated cold JVM extraction within existing limits.
- Successful artifact provenance plus password and sandbox-path redaction.
- Output-directory overwrite and input symlink rejection, job/live-process
  cancellation, real process timeout and bounded tool-output capture.

The Stegseek wordlist is a two-line generated fixture, not a bundled password
dictionary. Extracted files are never executed. No real corpus/model download,
real-world detection score or blind CTF success is implied.

The full-image workflow runs manually and on relevant push/PR changes. It
publishes the verified inventory only after recovery passes. Python/OS
transitive dependencies are not fully locked, and this limited smoke is not
exhaustive parser fuzzing, all tool-specific recovery or the 120-case blind gate.
Those acceptance gates remain open.

An encrypted Steghide fixture can fail the playbook's seed-only Stegseek attempt
under its existing budget. This remains a failed tool execution, not a recovery
claim. Known-password Steghide and explicit two-line wordlist Stegseek recovery
are separate checks; do not suppress the failed attempt or increase its budget.
Stegseek wordlist cracking stdout is omitted from ToolRunner results because it
can disclose a recovered passphrase unknown before extraction. Status, version,
duration, redacted command and artifact provenance remain available.
