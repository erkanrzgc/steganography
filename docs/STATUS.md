# Implementation status — 2026-09-17

Source version is 0.6.0, unreleased and classified beta. Baseline 0.5.0 was
validated and committed as `1aaa747`. The complete v0.6–v1.0 roadmap has not
been delivered or validated by this slice.

## Implemented

- Shared analysis context with legacy analyzer compatibility; additional
  spatial-image, JPEG structure, GIF, WAV and text signals.
- Recursive CTF service, bounded decoder/archive candidates, artifact hashes
  and provenance, CLI, API jobs/SSE/cancel/download and guided TUI.
- Additive JSON v2, HTML, SARIF and portable evidence bundles.
- Explicit signed-model catalog installation and research train/calibrate/
  export commands; lineage-aware manifests and benchmark metrics/gates.
- Full-image upstream tool installation with checksummed Stegseek, zsteg and
  dependencies, and OpenStego; Debian package licenses recorded.

## Verification

- Python 3.11: 145 tests pass; total coverage 90.50% in the final full run.
- Ruff, mypy (65 source files), diff whitespace and package identity checks pass.
- Version 0.6.0 wheel and sdist build successfully; the wheel contains no models.
- Web: one test passes; TypeScript/Vite build passes.
- Full Docker build and non-root/read-only/network-disabled smoke have passed
  for native payload and independently generated Steghide exact recovery.
  `scripts/docker-ctf-smoke.py` reproduces these limited integration checks.
  In the final BMP run, zsteg, OpenStego, Stegseek, Steghide and ExifTool
  completed. Only native and Steghide payload recovery were asserted; tool
  completion alone is not evidence of a hidden payload.
- Python 3.12–3.14 are configured in CI but were not executed locally.

## Open acceptance gates and limitations

- Real BOSSBase/ALASKA2/StegoAppDB evaluation: **unavailable**, no supplied corpus.
  No support cell has demonstrated the requested cross-source accuracy gates.
- Blind 120-challenge recovery, top-three recommendations and latency: **unavailable**.
  The smoke examples are not evidence of 90% CTF recovery.
- Published, trained spatial/JPEG model packs: absent; model catalog is empty.
  Training/export plumbing is not an independently evaluated detector.
- Spatial RS/sample-pair/weighted signals are approximations. Full calibrated
  algorithms and JPEG recompression/family discrimination remain research work.
- New CTF code has not yet reached the 95% coverage target across every new file.
- Per-tool OS filesystem isolation, hard native-stage deadlines, exhaustive
  fuzzing and archive-bomb/process-escape gates are incomplete. A subprocess
  working directory does not restrict its filesystem permissions.
- Debian and Python transitive dependencies are not fully locked. Installation
  of every executable does not demonstrate successful recovery with every tool.
- CTF API jobs are process-local; durable restart/resume and a full live-output
  workflow need additional work.

Use the current release as a local exploratory/CTF assistant. Validate findings
with independent tools. Statistical scores are heuristic and uncalibrated;
they are not probabilities or forensic proof.
