# Unattended research execution — 2026-10-15

The existing verification and collection processes were moved out of their
terminal scope without restarting them. The portable
[independence receipt](../benchmarks/research-terminal-independence-20261015.json)
records the verified migration. Terminal closure no longer stops these jobs;
user logout and power-loss survival are not established.

The native full-suite gate and its one-shot supervisor share an owned,
two-hour scope. Only a passed unchanged coverage, lint, type and source gate
can start the separately bounded 30-minute CPU comparison. Its numeric models,
predictions and all 480 cell measurements must replay before completion is
published. See [the frozen protocol](NONJPEG_LINEAR_COMPARISON_PROTOCOL.md).

A separate bounded publication service waits for the execution receipt or an
observed failure. It records completion or `unavailable` and updates the current
status when that file has no concurrent edits. It never starts or retries
training. Existing first-failure logs, partial artifacts and historical results
remain intact.

The JPEG controller already runs as an owned remote service. Its local result
collector now has a separate six-hour scope; that collection watchdog does not
extend the controller's frozen cumulative six-hour training budget. The
previously validated, bounded transport handover remains in place. See
[the amendment](JPEG_REFINE_NUMERICS_PROTOCOL.md).

Both supervisors publish portable JSON and dated result documents. Current
state is recorded in [STATUS.md](STATUS.md); neither a running process nor a
passed software gate is evidence of statistical detection success. Production
qualification remains unavailable and no model is deployed.
