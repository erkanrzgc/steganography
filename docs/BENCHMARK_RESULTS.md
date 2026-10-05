# Measured results, not a global accuracy score

As of 2026-10-05, **no method has passed the cross-source support gates**.
The application is an exploratory inspection/extraction tool, not a reliable
certificate that a file is clean. More formats, datasets or passing unit tests
do not imply better detection.

| Scope | Evidence | Outcome |
| --- | --- | --- |
| PNG/OpenStego and BMP/Steghide native detection | [Frozen BOSSbase pilot](PILOT_RESULTS.md): 1,000 pairs, 500 per family | Combined AUC 0.499021; recall 0%, FPR 0% at threshold 70; failed |
| PNG/BMP small second-source check | [Kodak](KODAK_RESULTS.md): both methods on 24 originals | AUC 0.49349; recall 0%; insufficient independent sample size and failed detection |
| JPEG JMiPOD / JUNIWARD / UERD native detection | [Frozen ALASKA2 baseline](ALASKA2_RESULTS.md): 1,000 pairs per family, shared covers | AUC 0.525617 / 0.505157 / 0.507181; recall 0.4% / 0.1% / 0.1%; failed |
| Experimental trained JPEG model | [Separate development validation](JPEG_DEVELOPMENT_RESULTS.md): 205 pairs per family, shared covers | AUC 0.648804 / 0.592481 / 0.585306; FPR 41.46%; not deployed; not an independent test |
| Experimental trained spatial model | [Controlled LSB development](SPATIAL_DEVELOPMENT_RESULTS.md): 184 pairs/cell, six method/rate cells | AUC 0.716801–0.997460 versus same-data reference 0.516422–0.648423; FPR 4.89%, low-rate scattered recall 19.57%; not deployed; support/export gates failed |
| WAV sample-LSB replacement | [FSDD baseline](WAV_RESULTS.md): 1,000 test covers, 6,000 stegos; sequential/scattered at three rates | All six cells: AUC 0.50, recall 0%, FPR 0%; failed. Generation oracle verified every payload; native CTF recovery unmeasured |
| GIF, text, PDF, containers, MP3/TIFF-specific claims | Unit/integration and generated examples | Representative labeled corpus evaluation **unavailable**; no real-world score |
| Controlled CTF extraction | [Kodak regression](CTF_REGRESSION_20261002.md): two final 30/30 completed exact recoveries | Published-challenge regression, not blind recovery or automatic detector accuracy |
| Blind CTF and release qualification | Planned 120 challenges and independent-source gates | **Unavailable** |

Native baselines and the trained JPEG validation use different samples/models/
thresholds. Their numbers are **not** a before/after accuracy comparison. Refer
to each report for environment, confusion counts, uncertainty where computed,
coverage and limitations. Scores are not calibrated probabilities.

## Next measured development slices

1. PNG/BMP: the [first richer-feature comparison](SPATIAL_DEVELOPMENT_RESULTS.md)
   completed on 7,000 controlled files. Next improve low-rate scattered recall,
   false positives and calibration, resolve export parity and evaluate a newly
   untouched independent source. Named upstream methods require separate tests;
   do not train on the published pilot or claim this generic LSB result for them.
2. JPEG: use the new development split for stronger residual/co-occurrence
   features or a suitable trained architecture. Freeze calibration and model
   before a newly untouched source; keep this failed linear baseline visible.
3. WAV: use the separately reserved training/validation speakers for richer
   features/models; preserve the failed six-cell baseline and inspect false
   positives on validation. Acquire another untouched audio source for testing.
4. GIF/text/container: first establish reproducible independent method fixtures
   and representative benign corpora, then method-specific exact recovery and
   false-positive measurements. A decoder solving base64 is not generic detection.

These are remaining gates, not deployed improvements. The same rules apply
to each new method: publish failures, preserve frozen tests, and promote support
only after the [acceptance gates](BENCHMARK_PROTOCOL.md) actually pass.

## Stegseek availability is environment-specific

The host has no `stegseek` executable, which explains the frozen native baseline's
`unavailable` coverage. Existing full Docker image `e2c255ae0709` contains
StegSeek 0.6. A 2026-10-04 network-disabled, non-root, read-only-root check used
upstream Steghide on a generated BMP, a single known fixture-password wordlist,
and Stegseek cracking; exit status was zero and payload recovery was exact.

That verifies one controlled Steghide integration, not blind cracking or all
JPEG methods. [Stegseek's upstream scope](https://github.com/RickdeJager/stegseek)
is Steghide; installing it does not repair the JMiPOD/JUNIWARD/UERD classifier.
No real wordlist was bundled, no host installation occurred, and the full image
was not rebuilt in this slice.
