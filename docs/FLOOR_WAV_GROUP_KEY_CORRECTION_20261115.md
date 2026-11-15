# WAV recording-key implementation correction (2026-11-15)

The first corrected-source preparation also exits 1 before generated checks,
a source freeze or any real fitting. Its implementation mistakenly uses
`source_group`, the constant dataset label `ESC-50`, as a recording identifier.
The admitted receipt's actual identifier is `group_key` (`ESC-50:<recording>`),
which binds `upstream_recording_id`. The source-role guard correctly rejects
the mistaken all-one-group implementation. Preserve its source/log/exit in
`.benchmark/floor-wav-completion-20261115/` without relabelling.

The separately bound source-corrected completion in
`.benchmark/floor-wav-groupkey-20261115/` uses `group_key`. Existing role
counts remain 1,200 fit /398 calibration /402 test; AudioMNIST/FSDD, all
candidates and all numeric/resource guards are unchanged. A metadata-only
role preflight must verify actual nonempty group counts before launch.
No dataset is modified, role invented, fitted model retried or guard relaxed.

Charge twenty-five seconds cumulatively for both failed preparations
(5.519 and 3.508 seconds), source/metadata diagnosis and overhead. The
original 2,400-second allowance is not reset; the completion has at most
2,375 seconds including independent closing. Preserve both earlier attempts,
the original protocols/amendment and all source bindings.
