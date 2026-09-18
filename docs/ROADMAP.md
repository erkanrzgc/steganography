# Roadmap and status

Synthetic corpus results are smoke/regression evidence only. They are not a
real-world accuracy claim. A cell is `supported` only after the gates in
`BENCHMARK_PROTOCOL.md` pass; otherwise it is `experimental` or `unsupported`.

## v0.6 — in progress

Source version: 0.6.0 (beta classification; not published). See
`STATUS.md` for verification evidence and remaining release gates.

- [x] Validated and checkpointed the existing 0.5.0 platform.
- [x] Provider-independent project memory and architecture/benchmark docs.
- [x] Shared `AnalysisContext` and expanded PNG/BMP native analysis.
- [x] Bounded CTF service with CLI and common portable reports.
- [x] CTF API jobs, SSE/cancellation, artifact download, and guided TUI entry.
- [ ] Full-image tool inventory, pinned versions/checksums/licenses, and E2E.

## v0.7

The frozen BOSSbase pilot (`PILOT_RESULTS.md`) found chance-level discrimination
for the tested Steghide/OpenStego families despite 30/30 controlled CTF recovery.
Next detector work needs separate development/validation data and an untouched
second source; tuning on this published baseline is not a new held-out result.

JPEG segment/DQT and calibrated DCT analyzers, signed spatial/JPEG ONNX model
packs, and tool-specific JSteg/F5/OutGuess/Steghide recovery.

## v0.8

GIF, WAV, text, and generic-container depth; complete API/TUI CTF controls and
cross-format recursive recovery.

## v1.0 gate

Release only when every advertised support cell passes the real-dataset gates,
the blind CTF suite meets recovery/latency targets, and all sandbox/limit tests
pass. Until then the package remains beta and claims are per-cell.
