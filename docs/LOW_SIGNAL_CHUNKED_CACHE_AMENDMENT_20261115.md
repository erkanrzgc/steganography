# Bounded chunked-cache source amendment (2026-11-15)

The common legacy image helper sets an eight-MiB hard per-file limit at import.
The corrected nsF5 preparation passes and computes all training features,
then export fails with `file too large` after 254.107 seconds. No actual
classifier was fitted. Preserve that source/freeze/log and truncated eight-MiB
archive as unavailable; never read it as a usable cache or delete it.
The first WAV parent is explicitly stopped with SIGTERM during data-only ESC
preparation after 126.696 parent seconds to prevent the same known export
failure. Its external-stop receipt and actual -15 exit remain, not relabelled
as a watchdog result or successful preparation. No actual model was fitted.

Both separate source completions keep the eight-MiB per-file guard. Persist
ordered chunks of at most 100 sources, with per-chunk hashes and one complete
manifest. Even uncompressed arrays stay below eight MiB. Loaders verify every
part and concatenate in the original source order. Descriptor bytes, floating
point arithmetic, payloads, source roles, models, thresholds, calibration gate
and all numeric guards remain exact. No corrupted archive prefix is reused.
A generated chunk/manifest roundtrip must pass before any real fit.

For nsF5, preserve the original 2,400-second /six-GiB summed sampled RSS /
256-MiB output caps. Conservatively debit 280 seconds for the earlier ten-
second charge, successful preparation, failed export and source/closing costs;
at most 2,120 seconds remain, not a fresh allowance. Four file workers stay.
Count both earlier parents and the new completion against the output cap.
The exclusive new directory is `.benchmark/nsf5-calibrated-relations-chunks-20261115/`.

For WAV, preserve the original 1,800-second /six-GiB summed sampled RSS /
256-MiB output caps. Charge 150 seconds for the old five-second preparation
allowance, stopped parent, independent fit-only feature-cost probe and source
overhead; at most 1,650 seconds remain. The one cost probe takes .091548 seconds
for a 220,500-sample fit clip, with no optimizer/model/test work. Use at most
four independent-file workers for the identical deterministic descriptors;
one BLAS/OpenMP thread each. Count the stopped and new directories together.
The exclusive new directory is `.benchmark/wav-modular-calibration-chunks-20261115/`.

The source-corrected parents never relaunch an old attempt. Preserve every
old partial file, log, source and consumed debit; closing must record actual
combined output and cumulative wall debit. No source media/roles are rewritten,
old model refitted, new media downloaded, GPU used or research model deployed.
