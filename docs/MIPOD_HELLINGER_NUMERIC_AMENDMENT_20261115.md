# Hellinger generated-control numeric correction (2026-11-15)

The first Hellinger preparation exits 1 during generated-only bandwidth QA,
before source-freeze completion or any real-data fitting/test inference.
Retain `.benchmark/mipod-hellinger-20261115/` unchanged, including all source,
failed log/exit and the separate read-only distance diagnosis. Do not relaunch
its parent. No failed command is relabelled as passing.

The dot-product identity for squared distance produces nonzero cancellation
for two exactly identical generated vectors: 1.194e-12 for square roots and
3.092e-11 after RC8. Counting that pair as positive changes the sample median.
The original generated parity guard correctly rejects it. This is an
implementation defect in the sample-distance helper, not evidence that the
histogram recipe or classifier failed on actual media.

A source-corrected completion uses SciPy's direct-difference `pdist` for the
small 256-cover bandwidth sample, rather than the cancelling dot identity.
The prescribed median of positive distances, <1e-12 zero rule, fixed indices,
bandwidth factors, classifiers, selection order and all numeric guards remain
unchanged. Large Gram matrices retain their existing bounded kernel routine;
independent direct-distance prediction replay still uses 1e-7. No guard is
relaxed and no failed actual model is refitted; zero real fits existed.

Before any actual fit, freeze the corrected runner/parent/auditor, this
amendment, original protocol and the failed source/exit/diagnostic hashes.
Rerun generated controls on corrected source, retaining the old failure.
The exclusive completion directory is
`.benchmark/mipod-hellinger-completion-20261115/`. Charge a conservative ten
seconds for the failed 2.504-second child, parent overhead and .410-second
read-only diagnosis. The original 900-second allowance is not reset: this
completion has at most 890 seconds, including all preparation and closing.
The same sampled 10-GiB RSS /eight-MiB output and no-download/GPU/media/holdout
restrictions remain. Preserve all observations and record the actual outcome.
