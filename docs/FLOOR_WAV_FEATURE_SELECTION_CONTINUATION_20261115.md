# WAV fit-only feature-selection continuation (2026-11-15)

The completed-state continuation retains all sixty ESC dedicated models and
completes twenty AudioMNIST dedicated models, then its full-feature mixed
AudioMNIST model reaches the 240-second phase limit (exit -9 /wall_limit).
No global choice or test inference exists. Preserve both partial parents,
all completed states, source freezes and failed mixed-model exits. Neither
unfinished full-feature mixed model is retried or reported passing.

This separately source-bound continuation reuses all eighty completed
dedicated states byte-for-byte. FSDD has not been attempted. Its nine kernel
candidates per cell retain the original recipe. For the still-unattempted
FSDD gradient-boosting family and three mixed alternatives, reduce inputs to
256 features selected using fit data only. All existing calibration/test
roles and 300-iteration /15-leaf /lr .05 /l2=1 /seed-zero tree settings stay.
This is a distinct model family, not reconstruction of either lost state.
Original full-5,433-feature mixed references remain explicitly unavailable.

For each dedicated condition, rank feature j by squared difference of cover
and stego fit means divided by the sum of their population variances plus
1e-12. For mixed models, the stego class pools all eight conditions equally.
Stable descending scores break ties by original feature index; take 256 and
store their indices in ascending order. No calibration/test values, speaker
names or source labels select columns. Numeric tree exports bind the selected
indices and traversal first selects those input columns. Class-balanced
mixed weights and minimax calibration thresholds remain unchanged.

Freeze this new feature-selection recipe before FSDD or mixed fitting.
Generated controls must check column-selection replay and exported prediction
parity. Independent closing replay verifies all retained/new calibration
records, source roles, choices, test counts/AUC/intervals and numeric states;
no old observed score is substituted for a new selected outcome.

The exclusive continuation is `.benchmark/floor-wav-f256-20261115/`.
Conservatively debit 980 seconds for the previous 720-second charge plus
4.552-second preparation, 240.549-second child and source/transport overhead.
The original total remains 2,400 seconds, leaving at most 1,420 seconds.
Original sampled ten-GiB RSS /24-MiB combined logical output /one-thread
bounds remain. Combined output counts all partial and current directories,
including hard-linked copies conservatively. No original bytes, data roles,
payload, numeric guards, media, GPU or software-gate sources change.
