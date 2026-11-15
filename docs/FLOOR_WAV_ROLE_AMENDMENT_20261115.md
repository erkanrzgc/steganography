# WAV source-receipt role correction (2026-11-15)

The first fresh WAV preparation exits 1 at the source-group guard, before
source-freeze completion, generated model checks or any actual fitting.
Keep `.benchmark/floor-wav-20261115/` source/log/exit unchanged. The older
v2 loader used the fold digit in file names. The already-admitted ESC-50
`source.json` instead binds `esc-original-group-max-fold-v1`: original
recording groups move together to their maximum fold. Its upstream CSV
retains four source IDs spanning filename folds; clips are different files,
so this is not a claim of byte-identical waveform copies.

Replay the existing source receipt rather than assigning new media roles.
ESC-50 fitting remains 1,200 clips, calibration is 398 and testing is 402,
with all source groups disjoint. Two former filename-fold-4 rows belong to
source groups already in fold 5; two further fold-2 rows move within the
fit pool to fold 3. No source, cache, old score, role receipt or payload is
rewritten. Existing 400-test results remain historical rows with their old
cohort; the new 402-test rows are not a controlled ablation against them.
AudioMNIST/FSDD roles, all candidates and all numeric guards are unchanged.

Bind the source receipt, upstream CSV, original failed source/log/exit and
read-only audit before fitting. Public aggregate evidence is
`benchmarks/esc50-role-audit-20261115.json`. The exclusive corrected-source
completion is `.benchmark/floor-wav-completion-20261115/`. Conservatively
debit fifteen seconds for the old 5.519-second command, metadata/byte/feature
diagnosis and parent overhead; the original 2,400-second allowance is not
reset, leaving 2,385 seconds. Same single-thread /sampled 10-GiB RSS /24-MiB
output bounds. Preserve failures without relabelling; no original worker is
restarted or actual fitted model retried.
