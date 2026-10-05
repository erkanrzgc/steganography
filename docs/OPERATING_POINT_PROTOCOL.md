# Spatial operating-point development protocol — 2026-10-05

Freeze before selecting any threshold. Use the precise spatial co-occurrence
checkpoint from the numerical repair audit, not a retrained detector. All
validation data were previously inspected; neither cohort is blind/new test.

- SHA-256 of `operating-point:20261005:<original lineage>` determines whole
  validation lineage roles: first 64 bits / 2^64 < .5 is threshold fitting,
  otherwise development assessment. All six derivatives follow their cover.
  Format, method, labels and scores cannot influence role assignment.
- Require complete, unique seven-file lineages: cover plus sequential/scattered
  LSB at 5%, 20%, 40%. At least two lineages in each role; no test rows accessed.
- Choose threshold using fitting COVERS ONLY. Empirical false alarms must be
  <= floor(.03 * fitting cover count). Sort cover probabilities descending and
  take the next representable float above the (allowed+1)th score; ties are
  rejected together. Saturated scores are clipped to [1e-12, 1-1e-12] solely
  for a finite threshold in [0,1]. No assessment labels/scores select it.
- Compare original .5 against that one threshold on exactly the same assessment
  pairs. Report all six cells, false positives, recall, balanced accuracy/AUC
  and paired-lineage 200-replicate 95% intervals (seed 20261005).
- Do NOT claim probability calibration: scores and ECE are unchanged. Operating
  point selection cannot improve ranking or cure low-payload sensitivity.
  False alarms and missed stegos must both remain visible; no deployment,
  threshold recommendation for general files or cross-source support claim.
- Bind original manifest, features, precise checkpoint and role/score records.
  No image bytes accessed, no downloads, no overwrite or symlink output.

Future work needs stronger low-payload features, proper calibration development
and a frozen model/operating point on a genuinely untouched independent source.
