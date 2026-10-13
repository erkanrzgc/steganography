# Owned JPEG study continuation v1

`python -m steganography.research_jpeg_continue --plan PRIVATE.json --out FRESH`
is an explicit user-requested continuation of one cancelled `bounded-half-v1`
job, not an external model import. Plan metadata is bounded to 64 KiB. It pins
the parent report and original config by SHA-256 and enumerates checksum-bound
segment receipts in the exact seed/arm/half prefix order. Paths to receipts and
snapshots are derived inside that one parent; no free-form checkpoint paths.

Require the parent to say `cancelled: user_cancelled`, unchanged fitting-source
closure/config/manifest/independent audit/Torch version, and the same recipes,
initialization, exposure, numeric snapshot headers, arrays, finite values and
optimizer counters. Inherited first halves cannot contain evaluation. Complete
halves must have a parent-owned completed trial receipt. Revalidate inherited
snapshots before copying with exclusive creation and checksum checks; reject
symlink inputs/outputs. Never alter the parent job, report or checkpoint.

The new orchestrator and this protocol have a separate SHA-256 source closure.
They do not change or rebind the 52 fitting-source files of the cancelled job.
Workers run exactly the old isolated source-bound implementation with its
existing CPU/wall/file/GPU limits. No optimizer reset, fake updates, lower
exposure or automatic model deployment. Replaying an inherited receipt is
explicitly marked with parent/receipt/checkpoint provenance and inherited
segment indices; its original computation time remains the logical fit time,
while the new replay wall time is recorded separately. No new GPU fitting is
claimed for inherited segments. A partial interrupted half must run again.

The six-hour controls budget includes the cancelled parent's entire elapsed
time, including discarded work. New verification, capacity probes, copying
and workers consume only the remainder. The new service watchdog must be at
most that remainder plus 120 seconds for shutdown; pauses do not consume GPU
time. Do not silently reset the budget. Fresh capacity probes precede new
fitting; batch changes relative to inherited state are rejected. Preserve the
same matched baseline/development/bootstrap learning gates; this development
study does not establish production accuracy or independent source quality.

The continuation report is portable: parent is a checksum, artifacts use
relative generated names, no absolute private paths or commands are included.
Cancellation/source mutation/limits retain evidence and do not qualify a
model. Before physical continuation run rejection, replay, budget, cancellation
and frozen-source tests plus lint/type checks and the full regression gate.
