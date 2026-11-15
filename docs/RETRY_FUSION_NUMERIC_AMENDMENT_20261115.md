# Fusion generated-control dtype amendment (2026-11-15)

The first generated preflight accidentally constructs composite kernels from
float32 arrays, while the actual model pipeline explicitly casts both feature
blocks to float64 first. Its strict Cholesky residual check fails. A separate
generated-only diagnosis preserves all twelve dtype/kernel/weight residuals;
float32 kernel asymmetry explains the mismatch, while float64 satisfies the
unchanged guard. This is a control-source correction, not a relaxed tolerance
or an actual-data recipe change.

The original parent was incorrectly launched after that failed preflight;
it then exits 1 after 1.503 seconds on the missing success receipt, before
binding actual inputs, comparing real candidates or decoding any new test.
Preserve the failed control source, generated diagnosis and parent log/exit.
No actual classifier exists in this failed attempt.

A separately sourced completion in `.benchmark/retry-fusion-completion-20261115/`
casts only generated kernel operands to float64, matching the already-frozen
actual pipeline. It must pass every control and write its success receipt
before launch; failed preflight prevents launch. Model families, parameters,
source roles, seeds, payloads, numeric guards and global choice remain exact.
Conservatively charge ten seconds for prior preparation, failed parent and
generated diagnosis, replacing the old five-second charge inside the same
1,800-second cap. At most 1,790 seconds remain; six-GiB summed RSS /96-MiB
output and the inherited eight-MiB file guard remain. Count both parents'
output. Never relaunch the failed parent or refit any consumed actual model.
