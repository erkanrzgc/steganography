# Dataset preparation for every built-in extension — 2026-10-17

Format support, test fixtures, prepared controlled data, training and qualified
detection are separate states. This inventory covers the **23 explicitly handled
built-in suffixes**, plus three compound TAR aliases; third-party plug-ins can
add others. Generic byte inspection is not format-specific dataset coverage.

| Extension(s) | Available data / preparation | Training and remaining gap |
| --- | --- | --- |
| `.jpg`, `.jpeg` | BOSS/BOWS/ALASKA controlled JPEG experiments; both seeds and failures retained | JPEG fits completed; no qualified detector; not every JPEG embedding method evaluated |
| `.png`, `.bmp` | Controlled original-bound spatial pilot, BOWS gray /DIV2K RGB; PNG/BMP derivatives share ancestry | Linear fits completed; deep skeletons untrained; missing additional licensed native RGB origin |
| `.wav` | FSDD speech /ESC environmental mono PCM16; scalar payload oracles and source roles | Linear fits completed, 50% development balanced accuracy; stereo/frequency-specific data missing |
| `.wav` AudioMNIST follow-up | 600 source-bound mono PCM16 files (400 train /200 development); eight controlled bit-plane recipes and model trainer prepared | No fit or accuracy result; shared owned-GPU capacity preflight remains |
| `.wave` | Explicit WAV trailing-data dispatch; existing `.wav` corpus is one format family | No separately measured suffix corpus; renaming WAVs does not add originals |
| `.gif` | 13 source-bound licensed native files, unassigned; existing structural/recovery fixtures | Incomplete 20-file target; independent ancestry and palette/timing encoder/oracle missing |
| `.txt` | Unicode 17 controls; nine prepared local document candidates from two package groups | Controlled payload/oracles pass; roles unassigned; technical English is not representative multilingual text |
| `.md` | Two prepared local README candidates from two package groups | Controlled payload/oracles pass; roles unassigned; realistic benign Unicode/formatting coverage incomplete |
| `.csv`, `.log` | Existing text-anomaly dispatch and generated regression fixtures | No accepted representative natural corpus or format-specific trained detector |
| `.pdf` | Four prepared local NetworkX PDF plot-document candidates, one package group | Append-payload oracles pass; roles unassigned; header/end checks are not full parser validation |
| `.zip` | One prepared local NetworkX example-source ZIP candidate, shared package ancestry | Append-payload oracle passes; roles unassigned; members are not extracted or executed |
| `.tif`, `.tiff` | EXIF/structure implementation and generated regression fixtures | No accepted representative native TIFF dataset or trained TIFF detector |
| `.mp3` | MPEG/ID3/trailer implementation and generated fixtures | No accepted licensed native audio pilot or real embedding/extraction corpus |
| `.au` | Optional external Stegseek dispatch | No native AU corpus, native audio model or measured detection result |
| `.tar`, `.tgz` | Archive inspection and generated path/resource fixtures | Representative labeled natural corpus and opaque payload/recovery matrix unavailable |
| `.gz`, `.bz2`, `.xz` | TAR-container dispatch when the bytes are a valid compressed TAR | No representative labeled corpus; ordinary single-stream compression is not automatically a TAR |
| `.eml` | MIME-part count and generated normal/excessive-part fixtures | No accepted natural mail corpus or method-specific measured detection |

`.tar.gz`, `.tar.bz2` and `.tar.xz` use the corresponding compressed TAR path;
these aliases are not independent datasets. ZIP uses content-based dispatch.
RAR/7z/PE/ELF header recognition does not imply full format support or a trained
detector. PGM and reserved MPO files are source data, not additional advertised
carrier coverage; reserved WIFD originals remain closed.

## Current local preparation

The [frozen local protocol](ALL_FORMAT_DATA_PROTOCOL.md) prepared **16 hash-unique
natural files** from **four declared package groups**: four PDF, nine TXT, two
MD and one ZIP. PDF plots, documentation and the example-source ZIP from
NetworkX stay in the same group across extensions. Package version, original
SHA-256, exact package-relative file identity and applicable installed copyright
hash are pinned. Package archive authenticity and independent authorship are
not established. These are narrow local candidates, not certified clean media.

Each file receives an opaque-payload derivative and an equal-byte-length
whitespace control, for **48 artifacts /16 originals**, not 48 independent
originals. An independent scalar oracle verifies payload bytes using the known
original extent. Every output remains `development_unassigned`; no training,
calibration, autonomous-recovery score or detector accuracy is produced.
Generated fixtures do not fill missing natural-source targets.

The owned one-shot preparation is **completed**: all 16 payloads and every
output checksum pass, with **519,675 bytes** including license evidence and
manifest. Execution takes 0.1045 seconds including imports, with 40,404 KiB
peak resident memory. No download or optimizer update occurred.

The unchanged full gate passes **3,321 tests /two actual-CUDA skips**, **96.81%**
total coverage and **100%** for both new modules; Ruff and 210-file mypy pass.
Focused verification passes 48 tests. First scoped harness failures are retained
and not counted as successful coverage runs. See [results](ALL_FORMAT_LOCAL_RESULTS.md)
and the [23-extension machine-readable inventory](../benchmarks/all-format-dataset-readiness-20261016.json).
The first full gate passed 3,320 tests but failed the unchanged README
400..900-word contract (one failure /two actual-CUDA skips, 96.81% coverage).
No real preparation ran in that failed attempt. The repaired README and all
63 combined documentation/preparation checks passed before the full gate retry;
the failed [receipt](../benchmarks/all-format-local-results-20261016.json) and
[explicit repair binding](../benchmarks/all-format-local-readme-repair-20261016.json)
remain separate and immutable.
The cumulative raw-transfer ledger remains unavailable; new media downloads and
Commons retries have not started. A new date does not reset that allowance.

## Reviewed candidates, not acquired media

The official [Open Preservation format corpus](https://github.com/openpreserve/format-corpus/blob/366f068cec399d0cdfd61fa473de3ab6dc858098/README.md)
is pinned at commit `366f068cec399d0cdfd61fa473de3ab6dc858098`; its README
declares CC0 unless otherwise stated. The complete tree metadata identifies one
TIFF under two MiB. Its per-file exceptions/header and natural-source suitability
remain unverified; a format example does not fill the representative TIFF gap.
No media was downloaded. The bounded direct metadata review retained 241,728
bytes, including commit/tree/README records.

The [Commons bicycle-bell source page](https://commons.wikimedia.org/wiki/File:Bsumusictech_bike-bell.wav)
declares a 312,632-byte stereo recording by bsumusictech under CC0. Native PCM
headers have not been checked. Its MP3 transcode is the same recording and
cannot count as an independently acquired native MP3 source. An
[MDN MP3 example](https://github.com/mdn/interactive-examples/blob/main/live-examples/media/cc0-audio/t-rex-roar.mp3)
is also only a candidate: origin/license linkage, a pinned revision and actual
headers remain unverified. AU, natural EML and representative TAR sources remain
unavailable. See [the candidate receipt](../benchmarks/all-format-source-candidates-20261016.json).

A bounded, filename-only system-resource scan also found 153 local MP3
candidates across two application installations. Their shared bytes, recording
ancestry and file-level license linkage have not been accepted; no sound body
was read or counted as a new corpus. Its sole EML candidate is an exploit
regression example and is excluded from natural benign email data. No TIFF/AU
candidate under the two-MiB scan ceiling was found. This is a local scan result,
not proof that those formats lack usable public datasets.

Further primary-source review found [Hamlet](https://github.com/mirage/hamlet/blob/e5bf9e661020667f7c0174061155baae0dc7c6e0/README.md),
which declares CC0 **randomly generated** emails, and
[Audio Inspector](https://github.com/Tliens/audio-inspector/blob/7ddfaac6a795d23fef09529076bc21b07fb493d5/README.md),
which declares CC0 synthesized audio demos. These are candidates for format
regression, not natural mail/field recordings. Hamlet's bounded tree request
failed, so no complete file count is claimed; the pinned Audio Inspector tree
contains no separately listed AU/SND files. Neither corpus was acquired.
The [extra metadata receipt](../benchmarks/all-format-extra-source-candidates-20261016.json)
preserves both those limits and the retained metadata hashes.

See [native family readiness](NONJPEG_METHOD_READINESS.md),
[dataset sources](DATASET_CATALOG.md), [measured results](BENCHMARK_RESULTS.md)
and [remaining source gaps](../benchmarks/native-source-gaps-20261016.json).
