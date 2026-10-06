# Two-source SRNet real run

Status: fitting and independent evaluation completed on 2026-10-07; all six
detection cells failed. See `SRNET_MULTIPAIR_REAL_RESULTS.md`; no model installed.
This run follows unchanged
`SRNET_MULTIPAIR_PROTOCOL.md` (commit `837cb79`) and the verified complete
batch plan in `SRNET_MULTIPAIR_RESULTS.md`. The execution source base is
`1b0c80c`; full real results are published separately without corpus/weights.

## Fixed execution and audit

The isolated shared fitting service uses both declared sources, one epoch,
1,580 four-row updates, seed 20261012, two CPU math threads, Adamax .001 and
decay .0001. Optimizer deadline 1,800 seconds; hard parent wall 1,920 seconds,
address 8 GiB, CPU 3,660/3,661 seconds, model file 32 MiB, no core dumps.
Timeout is a failed complete run, never a partial fit or an extended budget.

The completed audit used the following external hard wall. For an optional
replay select a different fresh output directory; the original already exists:

```sh
timeout --signal=TERM --kill-after=5s 1800s venv/bin/python \
  scripts/audit-srnet-multipair-real.py \
  --job .benchmark/srnet-multipair-real-20261007 \
  --out .benchmark/srnet-multipair-real-20261007/evaluation
```

The script verifies the frozen plan SHA, recipe, optimizer, selected rows and
update count before calling the same shared complete-validation service.
All 765 rows, nine metadata-selected independent NumPy replays and unchanged
fixed numerical/decision gates are required. Shared summary retains all six
context cells, fixed .5 threshold and 200 cover-lineage bootstrap intervals.
A failed numerical gate retains audit evidence without usable predictions or
accuracy comparisons. The underlying eval service has no hard worker yet;
the external timeout is a wall bound, not a filesystem/network sandbox.

Comparison with the original BOSS-only pilot is descriptive only: training
sources, BN context and number of optimizer updates all change. It is not
proof of an isolated BN cause, increased blind accuracy or qualified support.
Small reused development validation, uncertain camera/device independence and
unknown ALASKA quality/payload still prevent generalization claims.

## Safe handoff

The bounded fit waiter and subsequent audit both completed; no training/audit
process remains. A failed/missing fit card cannot produce usable evaluation.
Fit and evaluation terminal logs stay local; portable evidence was reviewed
before publication. The waiter had its own 1,920-second bound, not an endless loop.

The local fit config lives under the job directory; the existing verified plan
remains in the accounting directory. Outputs must be fresh/non-symlink. Do not
rerun over an existing output, execute artifacts or change the frozen recipe.
If the fit failed or is still running, accuracy stays unavailable. If it
completed, audit the checksum-bound model/card/plan and publish only portable
aggregate evidence. Keep all original failed pilot results unchanged.

This VM exposes no NVIDIA training device; sustained longer training/independent
source qualification remains separate work. Completion of one short epoch or
test coverage alone does not make a useful trained detector.

Verification for the audit slice: Python 3.11, 1,155 tests passed, total coverage
94.89%; Ruff, mypy (116 files), diff checks and wheel/sdist builds passed. Audit
tests cover fixed-setting rejection, incomplete/duplicate context comparisons,
failed numerical gate publication, overwrite/symlink guards and CLI exit status.
Other Python versions/full Docker and actual trained-model ONNX remain unverified.
