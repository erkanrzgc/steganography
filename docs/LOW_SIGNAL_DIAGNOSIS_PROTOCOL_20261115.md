# Low-payload signal diagnosis (2026-11-15)

The user's new explicit research/fix request permits this distinct bounded
CPU diagnosis on already-admitted data. It never relaunches a consumed fit,
changes payloads, downloads media/models, accesses locked ALASKA2 or reserved
native data, uses a GPU, or deploys a detector. Preserve all old outcomes.

Use the corrected source roles from the closed remaining-floor studies.
Only fit/calibration raw JPEG/WAV sources may be decoded in this diagnosis;
no historical test source is opened. WAV reads require regular files <=4 MiB,
PCM16, <=500,000 frames and <=two channels, then the same first channel as the
old cache. JPEG requires regular files <=2 MiB and 512x512 single-channel DCT.
Hash every selected raw file and all service/cache/role inputs before decode.
No source bytes, roles, cache or payload is rewritten.

WAV: two fixed evenly spaced clips per fit/calibration speaker for FSDD and
AudioMNIST; eight evenly spaced clips per ESC role. Measure clip lengths,
low-bit occupancies, first-to-fourth difference saturation under the existing
quantizers, quiet-frame noise, local prediction RMS and effective weighted
sample size. Recreate unchanged .05/.4 replacement/matching seeds for change
counts and compare representative feature rows exactly against the existing
v2 cache. Report source-group variation, with no classifier or threshold fit.

JPEG: twelve evenly spaced fit and twelve calibration sources among the
original first 3,000. Recreate unchanged nsF5 .05/.1 seeds. Check DC/zero
invariance, absolute-one decrements, actual changed counts against the old
cache, and theoretical binary-entropy change rates. Replay representative
DCTR/calibrated feature rows. Measure histogram directions/scale, without
training a new classifier or inferring a new test score.

Generated controls check PCM16 embedding/boundaries and coefficient changes
before real decoding. One exclusive attempt, 240 seconds including source
binding and closing, sampled three-GiB RSS /four-MiB output, one BLAS/OpenMP
thread; disk-backed TMPDIR. Store aggregate diagnostics and hashes only,
never raw samples, copied media or large matrices. Preserve any failed run.
All 570 application/test sources must match the passed 4,646-test CPU gate;
this diagnosis changes no application or test code. New model intervention
requires a separate prospective feature/model freeze after these findings.
