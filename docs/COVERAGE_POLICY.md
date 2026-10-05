# Minimum native coverage policy

`native-format-minimum-v1` uses the content-detected type, not the extension,
to assess required execution. A low score with any required component absent,
unavailable, unsupported, failed or duplicated produces **inconclusive** in
`AnalysisPipeline`. Generic heuristics, optional external tools, cloud AI and
models cannot substitute for a missing native format component.

Every format requires `file_structure` and `signatures`, plus:

| Detected type | Required native components |
| --- | --- |
| PNG / BMP | `image_lsb`, `image_lsb_scatter`, `image_bitplane` |
| JPEG | `image_jpeg`, `image_jpeg_dct` |
| GIF | `image_gif` |
| WAV | `audio_wav` |
| MP3 | `audio_mp3` |
| PDF | `file_pdf` |
| Text | `text_whitespace`, `text_zerowidth`, `text_anomalies` |

Unknown, TIFF and otherwise unlisted formats have an explicit unavailable
format-specific policy; generic inspection can still run but is not complete
format coverage. Archive recursion/extraction is separate from certifying a
container as clean. Formats remain experimental regardless of completion.

Independent positive evidence is preserved: verified marker findings can still
confirm and threshold-level heuristic signals can still be suspicious/likely,
while incomplete coverage stays visible. Signals from failed/unavailable
analyzers are inconclusive, never verified proof. No suspicion score, numeric
threshold, calibration, payload extraction or historical benchmark changes.

JSON v2 adds `coverage_policy` and advances additive `schema_revision` to 2.
Required gaps produce informational findings and a restore/rerun recommendation.
HTML renders the assessment; SARIF retains it under `runs[].properties.analyses`
without inventing a positive security result. Portable bundles contain all views.
The CTF wrapper keeps its own revision; nested pipeline reports carry revision 2.
CTF uses the same content-based requirements for every recursively analyzed
artifact; successful coverage on another artifact cannot erase a required gap.
CLI/API/TUI reuse the shared pipeline. `/v1` raw service contracts are unchanged;
third-party `Analyzer.analyze(Path)` still executes, but cannot implicitly erase
missing named native requirements. Requiring DCT does not install/download it.

**Complete execution is not validated method support or a clean-file certificate.**
These components are experimental approximations, not exhaustive calibrated
detectors. `no_indicators` only means that the minimum executed components found
no threshold-level signals. Real-data false negatives, source limitations and
the independent qualification gates in `BENCHMARK_PROTOCOL.md` still apply.
