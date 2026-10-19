# Native non-JPEG preparation — 2026-10-15

Explicit preparation only: no fitting, optimizer steps, checkpoints, model
downloads, production integration or reserved-media access. The frozen JPEG
refinement recipe, its 31 source bindings, historical snapshots, six-hour
budget and ten-GiB cumulative ceiling remain unchanged. This work reserves
six GiB for JPEG and permits at most four GiB of additional non-JPEG outputs
and temporary files. Downloads are an explicit command, never import behavior.

## Reused data and input custody

Re-run `core.nonjpeg_audit.audit` against the existing configuration and original
source bindings; do not regenerate or overwrite the corpus. Audit all 160
original lineages, 4,080 artifacts and 3,840 marker-free scalar payload oracles.
Preserve FSDD speaker roles, ESC whole-recording groups, licenses and hashes.
Read test metadata for isolation checks only; never open test media. Shared
PNG/BMP, channel, plane, order and rate derivatives are not independent originals.
The original failed spatial/audio measurements remain historical evidence.

`core.nonjpeg_readiness.prepare_features(root, audit, fresh_output)` verifies
the new audit's corpus/source/count bindings and exports a separate manifest
for each original source and train/validation split. Manifests contain no host
paths. All sample bytes are checked again against their size and SHA-256 before
feature computation. Source group, filenames, lineage and payload/oracle fields
are provenance only; model matrices contain descriptor values alone. Each
artifact retains the existing 64-MiB cap, two workers, complete split membership,
exclusive outputs and symlink rejection. The source/split preparation has an
1,800-second deadline and a 512-MiB feature-output ceiling. A failure retains
partial outputs and never emits a completed report.

## Frozen descriptor definitions

Spatial inputs must already be native single-frame 256x256 PNG/BMP in L or RGB.
The existing preparation crops before embedding; extraction never resizes,
crops, converts palettes/alpha or reconstructs an original from a JPEG.

| Version | Definition | Dimensions |
| --- | --- | --- |
| `spatial-local-parity-v2` | Unchanged 684-value v1 prefix, then TL/TR/BL/BR 128x128 quarters | 684 +4x216 =1,548 |
| `spatial-rgb-parity-v2` | Same prefix, then separate native R/G/B channels | 684 +3x216 =1,332 |
| `audio-pcm16-parity-residual-v1` | Whole recording and four quarters; two lags, two bit planes, residual and transition histograms | 5x2x2x38 =760 |

For each spatial region/channel, direction order is horizontal, vertical,
down-right, down-left; bit planes 0/1/2. Each histogram has joint center-parity
0/1 and neighbor-minus-center bins clipped to [-4,4], normalized by all eligible
pairs. There are 12 histograms of 18 bins per region. Native grayscale pools
one channel; RGB pools all three for the prefix and local version. RGB-specific
features reject grayscale input modes. Replicated grayscale is not a native
RGB source; ancestry must be established by acquisition, not pixel appearance.

Audio accepts RIFF/WAVE native mono PCM16, 160..220,500 samples, 8..192 kHz,
at most 30 seconds, one fmt and one data chunk and at most 128 chunks.
No resampling, normalization, stereo mixdown or padding. Full recording comes
first; quarter boundaries are floor(N/4), 2floor(N/4), 3floor(N/4), N, keeping
the remainder in the final quarter. Lags are 1/2 and bit planes 0/1. Differences
use signed int32, including transitions between -32768 and 32767. Each cell
has 34 joint parity/residual bins clipped to [-8,8] followed by four joint bit
transition bins (00,01,10,11), each histogram normalized by eligible pairs.
All descriptors are rounded to eight decimals, finite and in [0,1].

The public entry point remains `research features --feature-version VERSION`.
Every new version has its own linear-model domain; old versions and deployed
analyzers are unchanged. No checkpoint from another domain is interchangeable.

## Optional model skeletons

`core.nonjpeg_models.network(domain)` constructs FP32 CPU research models.
`spatial-native-gray-srnet-v1` and `spatial-native-rgb-srnet-v1` reuse the existing
SRNet block layout without editing any JPEG-bound source. RGB replaces only
the first convolution's input width with three channels and the same Kaiming
rule; both receive new architecture identities. Native L/RGB inputs become
float32 without intensity normalization, preserving single image LSB changes.

`audio-native-pcm16-resnet1d-v1` receives raw PCM, first differences and second
differences. Differences are computed in int32, left boundary delta is zero,
then all three channels are divided by 32768 in FP32. A single LSB remains
1/32768. The network has a 32-channel stem, two 32-channel residual blocks,
stride-two residual blocks at 64/128 channels, global average pooling and a
two-logit classifier. It has no spectrogram conversion.

`generated_smoke()` checks generated forward/backward passes, finite nonzero
input gradients, finite parameter gradients and unchanged parameters. RNG and
thread policy are restored. It performs zero optimizer updates and writes no
weights. Torch absence is explicitly `unavailable`; generated gradients are
neither training nor detection evidence.

MP3 spectrogram research requires a separate decoder/STFT/window/rate protocol,
native compressed sources and actual encoder/extraction pairs. The
[Spec-ResNet paper](https://arxiv.org/abs/1901.06838) evaluates AAC/MP3 settings;
its results do not establish PCM-LSB detection success.

## Native pilot acquisition and Unicode controls

```text
python -m steganography.research_media_pilot --out FRESH_PRIVATE_DIR
python -m steganography.research_media_pilot --unicode-only --out FRESH_PRIVATE_DIR
```

Targets: 40 RGB PNG, 20 GIF, 20 genuine stereo PCM16 WAV, 20 native MP3.
The [Commons imageinfo metadata API](https://commons.wikimedia.org/wiki/Help:Machine-readable_data)
provides file-level author/license/source identities. Acquire original bytes,
not thumbnails or transcodes. Preserve page ID, original SHA-256, metadata
SHA-256, license URL, credit, known author and recording group when known.
Unknown ancestry/grouping is unavailable, never inferred independent. New
samples remain `development_unassigned`, entirely separate from JPEG and the
historical corpus. Deduplicate decoded pixels/PCM where supported; MP3 native
frame validation is inventory evidence, with full decoder validation unavailable.
At most two files per known author (or shared unknown group) per family; this
does not establish independence between excerpts or undocumented derivatives.

One acquisition permits at most 1 GiB including metadata and rejected media,
32 MiB per original, eight 50-candidate metadata pages per family, 1,800 seconds,
15-second network timeouts, fixed HTTPS hosts and validated redirects. Accepted
files receive immediate source sidecars; byte counters are journaled around
every request. HTTP 403/429/503 stops Commons requests with no retry, while
unrelated Unicode acquisition may proceed. Shortages remain incomplete or
unavailable, never a passed readiness claim. Preserve interrupted-attempt logs.

Unicode controls use the official, version-pinned
[17.0.0 auxiliary tests](https://www.unicode.org/Public/17.0.0/ucd/auxiliary/),
NormalizationTest, BidiCharacterTest and 17.0.0 emoji-test. Raw files retain
Unicode copyright/license headers. Record hashes, file names, version and line
numbers. `core.nonjpeg_unicode.controls` selects at most 32 rows per category
(ZWJ/ZWNJ/whitespace/normalization/bidi/emoji), retaining literal codepoints and
all five normative normalization columns. Hash mismatch fails. Missing categories
are incomplete. These are benign algorithm-conformance controls, not a corpus
of naturally occurring multilingual documents or measured false-positive rates.

## Draft two-seed comparisons — freeze separately before fitting

Seeds 20261016/20261017 are proposed for BOTH families, not authorized training.
Freeze code, environment, complete manifests, exposure tuple hashes and recipe
before fitting. Each arm/seed must use the same originals, derivative cells,
shuffle order and cumulative presentations. Do not count a view, PNG/BMP format,
payload order, channel or plane as another original. Missing arms/seeds are
unavailable, never pooled away. No early validation-dependent stopping or source
replacement. Declare compute/time/checkpoint budgets separately from JPEG.

Spatial: compare the 684-value baseline against local parity in native gray;
in native RGB compare baseline/local/channel parity. Compare linear and native
gray/RGB SRNet arms separately, using the same input preparations. WAV: freeze
the existing failed detector as reference, compare the 760-value linear model
and raw/delta PCM CNN at equal original exposure. Keep native mono research
separate from stereo and compressed-audio extensions.

Use complete 32-train/8-validation originals per historical source, holding all
derivatives and FSDD speaker/ESC recording groups in their original roles.
Fit normalization, class weights and thresholds using grouped TRAIN-only
cross-validation. Use matched 1:1 original cover/stego weighting per cell;
the many stego derivatives cannot inflate the cover denominator. Freeze 20
complete train epochs as the initial exposure proposal, report actual original
and image/sample presentations, and do not tune this on validation scores.

Report per source/format/order/channel/plane/rate: ROC-AUC, balanced accuracy,
recall, FPR, ECE and paired differences versus reference at frozen thresholds.
Use 2,000 paired bootstrap resamples, seed 20261018, clustered by original for
images, speaker for FSDD and whole-recording key for ESC; report tiny-group
intervals as underpowered. No pooled success hides a failed source or seed.
These development pilots cannot meet independent blind production qualification;
test and unseen sources stay closed. Freeze an independent native-source and
encoder/extraction protocol for each additional family before evaluation.
