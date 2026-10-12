# Non-JPEG method readiness

Prepared controlled carriers, tested parsers and qualified detectors are
different states. Nothing here changes inference or qualifies a method.
Existing failed measurements remain in `WAV_RESULTS.md` and
`SPATIAL_PARITY_RESULTS.md`. No scores or models are produced here.

| Family | CPU preparation now | Required next evidence |
| --- | --- | --- |
| PNG/BMP gray replacement | BOWS native L; sequential/scattered, 5/40%, planes 0/1, both formats | Matched descriptors, original-paired source-specific intervals and train-only calibration |
| PNG/BMP RGB/channel replacement | DIV2K native RGB; r/g/b/all with the same rate/order/plane matrix | A second independently licensed native RGB source; gray replication and train/validation are not independent RGB origins |
| WAV mono sample planes | FSDD speech +ESC environmental PCM, historical speaker/recording roles, scalar recovery oracle | Versioned residual/temporal descriptors vs failed detector; source-stratified paired comparisons and train-only calibration |
| WAV stereo/channel difference | Existing independent encoder/regression fixtures only | Licensed native stereo, genuine left/right/both embeddings and channel byte oracles; not duplicated mono |
| WAV spectrogram/frequency | Native inspection only | Specified frequency-domain encoder, inverse-transform distortion/rate oracle and benign tonal/music negatives |
| GIF comments/extensions | Existing independent CTF recovery fixtures only | Licensed native single/multi-frame GIFs; benign comments/loop/application metadata negatives; opaque bytes, not flag-pattern smoke |
| GIF palette/frame timing | Existing structural/frame fixture checks only | Freeze palette/delay encoder, visual/duration bounds, extraction oracle and benign animations; RGB conversion does not qualify native GIF |
| Text zero-width/whitespace | Existing decoder/CTF and round-trip fixtures only | Licensed multilingual documents with whole-document/source roles; legitimate ZWJ/ZWNJ, emoji, indentation, line endings and minification negatives; marker-free positives |
| Text bidi/homoglyph/normalization | Native heuristic signals, not a learned model | Legitimate Arabic/Hebrew/Persian and mixed-script identifiers; canonical-equivalent/control negatives and an independently specified positive decoder |
| Container appended/carving/polyglot | Bounded CTF evaluator and independent development recipes | Original containers and opaque-message oracles, benign overlays/metadata; independent blind custody |
| Archive/decoder recursion | Existing safe recursion/decoder regression matrix | Depth/ratio/member/byte/time adversarial cases plus benign archives; exact non-root hashes and incomplete denominators |
| MP3/compressed audio / upstream tools | Existing metadata/tool fixtures only | Native licensed data and actual upstream-method positives, pinned versions/licenses and verified extraction; no WAV-to-MP3 accuracy inference |

## Next experiments — freeze before fitting

1. Prepare/audit the explicit matrices with `research nonjpeg-prepare --config
   FILE --out FRESH_DIR`. Local manifests/checksums are mandatory, no implicit
   acquisition. See `NONJPEG_PREPARATION_PROTOCOL.md`.
2. Successful audit exports `spatial-research.json` and `audio-research.json`,
   compatible with the existing manifest validator. Spatial extraction can use
   `--source FRESH_DIR`; audio descriptors need a new versioned contract. Never
   send WAVs into spatial features or expose marker/oracle metadata to models.
   Neither manifest contains test rows.
3. Preserve original-pair and speaker/recording grouping for uncertainty;
   shared PNG/BMP/channels are not independent observations. Freeze two seeds,
   matched input order/exposure, per-cell metrics and one calibration policy
   before CPU/GPU fitting. These small 32/8 preparations are learning pilots,
   not the production 1000+1000/two-independent-source qualification gate.
4. Freeze separate native-source/license/negative/encoder protocols for each
   remaining family. Fixture success is not a completed dataset or a reason
   to mark missing statistical detection support as available.

Raw media/derivatives/weights remain private under original terms. Publish only
portable hashes/counts, code/protocols and honest audit/failure records. No new
download, GPU, paid compute or training is required for this preparation slice.
Blind test origins remain unopened.
