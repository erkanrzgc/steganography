# Detection expansion toward 80% per method (CPU, 2026-11-11–13)

Goal set by the user: every steganography method should reach at least 80%
balanced accuracy on data never used in training, at a stated payload.
Balanced accuracy is the mean of the true-positive and true-negative rates
over paired covers and stegos. Statistical image rows use a zero threshold
except spatial v3, which selects its threshold from out-of-fold training
scores. Audio, video and rules use the operating points stated in their
sections. Repeated test access is flagged below.

## Results against the 80% target

| Method | Carrier / data | Payload | Test accuracy | AUC | Detector |
|---|---|---|---:|---:|---|
| LSB replacement | gray BOSSbase | 5% | **85.7%** | .946 | SRM gray, payload-specific |
| LSB replacement | gray BOSSbase | 10/20/40% | **90.2 / 94.1 / 97.8%** | .976–.998 | SRM gray, payload-specific |
| ±1 LSB matching | gray BOSSbase | 5% | **86.3%** | .947 | SRM gray, payload-specific |
| ±1 LSB matching | gray BOSSbase | 10/20/40% | **91.1 / 94.2 / 96.6%** | .976–.996 | SRM gray, payload-specific |
| LSB, unknown payload | gray BOSSbase | 10/20/40% | **88.7–94.5%** | .961–.994 | SRM gray, one mixed detector |
| LSB, unknown payload | gray BOSSbase | 5% | 72.1% (miss) | .877–.879 | SRM gray, one mixed detector |
| stegano (Python) | RGB PNG | 0.1 / 0.4 bpp | **100%**, 0/2,000 false alarms | — | `modules/stegano_lsb.py` rule |
| stegano | RGB PNG | 0.1 / 0.4 bpp | **84.8 / 98.0%** | .918 / .994 | rich RGB, statistical |
| OutGuess 0.4 | JPEG q75 | 0.05 bpnzac | **95.5%** | .987 | calibrated DCTR |
| OutGuess 0.4 | JPEG q75 | 0.1 / 0.2 bpnzac | **82.0 / 94.6%** | .897 / .987 | DCTR + ensemble |
| steghide 0.5.1 | JPEG q75 | 0.05 bpnzac | **98.0%** | .994 | calibrated DCTR |
| steghide 0.5.1 | JPEG q75 | 0.1 / 0.2 bpnzac | **82.7 / 93.8%** | .901 / .986 | DCTR + ensemble |
| steghide 0.5.1 | RGB BMP | 0.1 bpp | **84.0%** | .908 | colour SRM, 4 crops per train image |
| steghide 0.5.1 | RGB BMP | 0.4 bpp | **89.0%** | .931 | rich RGB |
| F5 (Westfeld) | JPEG q75 | 0.05 bpnzac | **80.8%** | .891 | calibrated DCTR, 4 crops per train image |
| F5 (Westfeld) | JPEG q75 | 0.1 / 0.2 bpnzac | **93.8 / 99.8%** | .986 / 1.000 | calibrated DCTR |
| OpenStego 0.8.6 | RGB PNG | 0.1 bpp | 79.8% (miss) | .870 | colour SRM, 4 crops per train image |
| OpenStego 0.8.6 | RGB PNG | 0.4 bpp | **88.5%** | .947 | colour SRM |
| HILL | gray BOSSbase | 0.4 bpp | 76.90% (miss) | .865 | SRM + weighted, v3; train OOF threshold |
| S-UNIWARD | gray BOSSbase | 0.4 bpp | 79.95% (miss) | .892 | SRM + weighted, v3; train OOF threshold |
| WOW | gray BOSSbase | 0.4 bpp | **80.30%** | .901 | SRM + weighted, v2 |
| MiPOD | gray BOSSbase | 0.4 bpp | 76.00% (miss) | .858 | SRM + weighted, v3; train OOF threshold |
| HILL | gray BOSSbase | 0.6 bpp | **83.20%** | .920 | SRM + weighted, zero threshold |
| S-UNIWARD | gray BOSSbase | 0.6 bpp | **86.45%** | .948 | SRM + weighted, zero threshold |
| MiPOD | gray BOSSbase | 0.6 bpp | **82.20%** | .915 | SRM + weighted, zero threshold |
| J-UNIWARD | gray BOSSbase JPEG q75 | 0.4 bpnzac | **83.80%** | .900 | DCTR, train-selected |
| UERD | gray BOSSbase JPEG q75 | 0.4 bpnzac | **82.65%** | .901 | DCTR, train-selected |
| nsF5 | gray BOSSbase JPEG q75 | 0.05 bpnzac | 60.70% (miss) | .658 | DCTR, train-selected |
| nsF5 | gray BOSSbase JPEG q75 | 0.1 bpnzac | 75.15% (miss) | .834 | DCTR, train-selected |
| nsF5 | gray BOSSbase JPEG q75 | 0.2 bpnzac | **94.25%** | .987 | DCTR, train-selected |
| EBS | gray BOSSbase JPEG q75 | 0.4 bpnzac | unavailable (invalid simulator) | — | skipped |

WAV, rule, network and video results follow in their sections; FSDD WAV is a
miss at every payload. Academic spatial results include three remaining
0.4 bpp misses and separate 0.6 bpp passes; their reused test set limits
independent confirmation of the target. JPEG nsF5 misses at 0.05 and
0.1 bpnzac; EBS remains unavailable. The overall 80%-per-method goal is
not met at all reported payloads.

## Gray LSB on BOSSbase (`research/srm_features.py`)

- **Data**:
  - BOSSbase 1.01, 10,000 gray 512×512 images.
  - Every image used by earlier BOSS experiments (pilot, development, scale,
    preview, JPEG context) was removed by file and pixel SHA-256. That removed
    3,003 images, and all 128 earlier JPEG lineages matched.
  - The remaining 6,997 were shuffled with seed `2026111102`: 5,497 train,
    500 validation, 1,000 test.
- **Stegos**:
  - Eight per cover: replacement and ±1 matching at 5, 10, 20 and 40% of
    pixels.
  - Train, validation and test stegos use separate seeds (`2026111103/4/5`).
- **Features**: 28 symmetrised SRM-style submodels, 4,732 dims:
  - first/second/third-order residuals at several quantisation steps, with
    parallel and perpendicular scans
  - two- and four-direction min/max residuals
  - KB, KV and edge kernels
  - Sign/reversal symmetry merges 625 patterns into 169, and pooling makes the
    features transpose-invariant.
- **Classifier**:
  - Ridge Fisher discriminant; the ridge is chosen on a held-out fifth of
    train images.
  - One mixed detector (all eight conditions) and eight payload-specific
    detectors.
  - The protocol was frozen before the first test access. Validation and test
    were scored once in the same run.
- **Ablation** (3,000 train images, validation only): symmetrised first/second
  order alone gives 78.9% at 5% matching. Adding perpendicular scans gives
  84.8%; the full set gives 85.3%.
- **Earlier features**: the RGB rich-model features (gray block, 1,375 dims)
  reached 78.7–79.2% at 5% and 84–86% at 10% on validation.
- **Run history**:
  - The first full run was stopped after a slow co-occurrence implementation
    was found. Its replacement gives bit-identical features about four times
    faster.
  - The rerun was OOM-killed in the mixed fit, because float64 copies of
    88k × 4,732 rows needed ~14 GB.
  - Fisher fitting now uses chunked moments, matching the closed form to 1e-15.
  - Both logs are kept.

The mixed detector's false-positive rate is 9.1% at every payload. Its 5% miss
comes from low recall (53%) at a threshold shared with the stronger payloads.

## Academic adaptive spatial methods (`.benchmark/boss-academic-20261112`, private)

- **Embedding**: conseal 2025.11 simulators (`simulate_single_channel`) of
  HILL, S-UNIWARD, WOW and MiPOD at 0.4 bpp (payload-limited sender, optimal
  coding simulated). They change 7.4–9.0% of the pixels.
- **Data**: the gray BOSSbase split above; covers are the same images and the
  cover SRM features are reused. One stego per cover and method.
- **v1** (frozen before any academic result): plain SRM features, the first
  3,000 train images, ridge chosen on a held-out fifth, one test score.

| Method (0.4 bpp) | v1 accuracy | v1 AUC |
|---|---:|---:|
| HILL | 70.5% (miss) | .792 |
| S-UNIWARD | 75.6% (miss) | .850 |
| WOW | 73.9% (miss) | .825 |
| MiPOD | 71.9% (miss) | .806 |

All four miss 80% with v1. The features are a 4,732-dim subset of the
34,671-dim spatial rich model, and adaptive methods put their changes in
texture, where these residual statistics barely move.

- **v2** (`run_maxsrm.py`, frozen before its results): adds a
  selection-channel-aware block (`weighted_gray_features`, in the spirit of
  maxSRM). Each co-occurrence adds the largest change probability of the four
  pixels its residuals sit at, instead of one.
  - The probability map is recomputed from the analysed image itself (cover or
    stego) at the known method and payload, never from the stego's cover.
  - All 5,497 train images are used. Plain SRM, weighted, and both
    concatenated (9,464 dims) compete on the held-out fifth of train images;
    "both" won for every method.
  - Stegos are bit-identical to v1 (same seeds; checked).
  - **Adaptivity**: v2 was designed after v1's test misses, so this is a
    second look at the same test set.

| Method (0.4 bpp) | Plain SRM | Weighted | **Both (selected)** | AUC | TPR / FPR |
|---|---:|---:|---:|---:|---|
| HILL | 72.4% | 74.5% | 76.4% (miss) | .859 | 80.0 / 27.2% |
| S-UNIWARD | 76.7% | 76.4% | 78.9% (miss) | .888 | 81.0 / 23.2% |
| WOW | 75.3% | 79.7% | **80.3%** | .901 | 83.1 / 22.5% |
| MiPOD | 73.6% | 74.6% | 75.2% (miss) | .853 | 78.7 / 28.3% |

- The extra train images alone lift plain SRM by 1.0–1.9 points over v1.
  Adding the weighted block lifts it by a further 1.6–5.0 points.
- The fixed zero threshold leaves 22–28% false alarms, so accuracy trails
  what the AUC allows.

- **v3** (`run_v3.py`, HILL / S-UNIWARD / MiPOD at 0.4 bpp):
  - Uses 9,000 train images: the original 5,497, the 500 former validation
    images and 3,003 images used by earlier BOSS experiments. The extra
    images were pixel-hash checked against the unchanged 1,000 test images;
    train and test names are also disjoint. The 500 images are now training
    data, not a validation set.
  - Retains concatenated SRM + weighted features (9,464 dims), the original
    simulators and the cached v2 stegos for the original train/test images.
    Ridge selection uses a held-out fifth of training images.
  - Five-fold out-of-fold training scores choose the threshold that maximizes
    balanced accuracy; each cover/stego pair stays in one fold. The final
    detector uses all 9,000 training pairs.
  - **Adaptivity**: v3 was designed after v1 and v2 results. This is a third
    look at the same test set, with no test-based threshold selection.
    WOW was not rerun; its 80.30% result remains the v2 result.

| Method (0.4 bpp) | v3 accuracy | Zero-threshold accuracy | AUC | TPR / FPR |
|---|---:|---:|---:|---|
| HILL | 76.90% (miss) | 77.05% | .865 | 80.6 / 26.8% |
| S-UNIWARD | 79.95% (miss) | 79.80% | .892 | 85.5 / 25.6% |
| MiPOD | 76.00% (miss) | 75.95% | .858 | 78.3 / 26.3% |

All three v3 rows miss 80% at 0.4 bpp. S-UNIWARD has 1,599 correct decisions
out of 2,000, one short of 80%; **79.95% is not rounded into a pass**.
The 25.6–26.8% false-positive rates also remain substantial.

**Higher payload, 0.6 bpp** (`run_rate.py`):

- Runs only the three v3 misses, with 5,497 train / 1,000 test images and new
  stego seeds. Features remain SRM + weighted; ridge is chosen on a held-out
  fifth of training images, and the operating threshold is zero.
- Weighted maps use the analysed image at the known method and payload.
  These are method/payload-specific detectors.
- This condition changes payload, training size and threshold policy from
  v3. It does not isolate the causal effect of payload. The same test images
  have already been viewed, and the choice to run 0.6 follows the 0.4 misses.

| Method (0.6 bpp) | Accuracy | AUC | TPR / FPR | Pixels changed |
|---|---:|---:|---|---:|
| HILL | **83.20%** | .920 | 86.1 / 19.7% | 14.22% |
| S-UNIWARD | **86.45%** | .948 | 88.1 / 15.2% | 12.32% |
| MiPOD | **82.20%** | .915 | 85.4 / 21.0% | 14.69% |

All three pass at the stated 0.6 bpp payload; their 0.4 bpp misses above
remain reported. These adaptive results do not establish a qualified or
deployed detector.

## Academic JPEG methods (`.benchmark/boss-jpeg-academic-20261112`, private)

The detached queue's first JPEG attempt is **unavailable**, exit 1: it fails
during training-feature extraction on 15 empty or zero-filled derived JPEG
caches dating from the earlier run. The completed audit finds 2,634 valid
and 15 invalid cached covers, with no test covers or result JSON produced.
All original covers, corrupt files, source code and the failed log are kept.

An isolated cache repair uses `.benchmark/boss-jpeg-academic-repair-20261113`.
It reuses the 2,634 bounded, fully decoded JPEGs by hard link and recreates
missing/invalid derived covers from the unchanged BOSSbase originals. The
worker's only AST change is its output directory; embedding methods, payloads,
seeds, features, split and selection policy are unchanged. This separate
completion exits 0. A closing hash audit verifies all 4,000 raw sources,
2,649 old cached file contents, four source/split bindings and all five
balanced-accuracy calculations. Hard linking changes inode link counts;
no old media bytes are rewritten. The original failed attempt stays unavailable.

- **Data**: first 3,000 train / all 1,000 test images from the gray split,
  compressed with libjpeg through jpeglib at QF75. Each condition has 1,000
  cover/stego test pairs; none of those sources is in this detector's training
  set. The same BOSS test images have been used in the spatial experiments.
- **Embedding**: conseal 2025.11 J-UNIWARD, UERD and nsF5 simulators, at bits
  per nonzero AC coefficient (bpnzac). These are simulated coding results;
  nsF5 here is distinct from the real F5 encoder evaluated on DIV2K above.
- **Features/selection**: DCTR (8,000 dims) and Cartesian-calibrated DCTR
  (16,000 dims) compete by AUC on a held-out fifth of training pairs. Plain
  DCTR with ridge 0.01 wins all five cells. The final ridge Fisher detector
  fits all 3,000 training pairs and uses a fixed zero threshold.
- Methods, payloads, seeds and selection policy were fixed before the first
  JPEG test result; the failed cache attempt produced no JPEG test metrics.

| Method | Payload (bpnzac) | Accuracy | AUC | TPR / FPR |
|---|---:|---:|---:|---|
| J-UNIWARD | 0.4 | **83.80%** | .900 | 83.3 / 15.7% |
| UERD | 0.4 | **82.65%** | .901 | 82.3 / 17.0% |
| nsF5 | 0.05 | 60.70% (miss) | .658 | 60.1 / 38.7% |
| nsF5 | 0.1 | 75.15% (miss) | .834 | 75.6 / 25.3% |
| nsF5 | 0.2 | **94.25%** | .987 | 94.2 / 5.7% |

**Remaining JPEG misses**: nsF5 at 0.05 and 0.1 bpnzac. Its mean changed-AC
fractions are 0.56 /1.30 /3.12% at 0.05 /0.1 /0.2 bpnzac; J-UNIWARD and UERD
at 0.4 change 8.36 /7.94%. These changed fractions are separate from payload.
No further payload or threshold tuning follows from these test scores.

**EBS is skipped/unavailable**: the recorded conseal 2025.11 simulator check
changes about 71% of nonzero AC coefficients at 0.4 bpnzac, with a block-entropy
division-by-zero problem. That embedding is invalid; no EBS accuracy or pass
is claimed.

The requested `.benchmark/boss-jpeg-academic-20261112/result.json` is a
byte-identical copy of the isolated completion's `result.json`, SHA-256
`6f0f7ec431249227ada47cf3d757b2d8e4a9dcdd0d75ed7a102c7738cf7adace`.
The successful log and closing audit remain in the repair directory; the
original `run.log` still records exit-1 evidence.

## WAV 16-bit LSB (`research/wav_lsb_features.py`)

- **Features**:
  - Symmetrised co-occurrences of first- to fourth-order sample differences,
    on the whole clip and on its quietest quarter of 1,024-sample frames.
  - Small-amplitude histograms and isolated-value counts inside exact digital
    silence.
  - Weighted-stego estimates with three polynomial predictors.
- **Classifier**:
  - Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`,
    300 iterations, fixed before any test access), with threshold 0.5.
  - One detector per kind/rate and one mixed detector.
- **Design**: on ESC-50 folds 1–3 vs 4.
  - A ridge Fisher discriminant reached only 68–72% at 40%.
  - Clips with digital silence (25%) were detected at ~99%. Noisy clips were
    near chance, because their quietest frames still vary by ≥6 LSB.
  - Trees combine both cues.
- **Splits** (group-disjoint):
  - ESC-50: folds 1–4 train, fold 5 test (400 clips; folds keep one source
    recording together).
  - AudioMNIST pilot: train speakers vs development speakers (200 cuts; no
    reserved speaker is stored locally).
  - FSDD: four speakers train, theo/yweweler test (1,000 clips).

| Test set | Replacement 5 / 10 / 20 / 40% | ±1 matching 5 / 10 / 20 / 40% |
|---|---|---|
| ESC-50 fold 5 (v1) | 73.9 / 77.0 / **85.6 / 89.1%** | 64.6 / 65.6 / 68.6 / 69.8% |
| ESC-50 fold 5 (v2) | 74.0 / 77.1 / **84.5 / 89.5%** | 64.0 / 65.6 / 66.2 / 69.8% |
| AudioMNIST dev (v1) | **83.7 / 95.2 / 99.2 / 99.8%** | 57.5 / 65.7 / 79.7 / **92.3%** |
| AudioMNIST dev (v2) | **84.0 / 95.5 / 99.2 / 99.8%** | 57.8 / 66.5 / **80.8 / 92.5%** |
| FSDD theo/yweweler (v1) | 49.5–56.7% (chance) | 48.2–50.8% (chance) |
| FSDD theo/yweweler (v2) | 49.5 / 50.9 / 52.1 / 58.0% | 49.2–51.6% (chance) |

**Payload-specific results**:
- AUC is .85–.97 (ESC-50 replacement) and .91–1.0 (AudioMNIST replacement).
- The mixed detectors rank well (AudioMNIST AUC up to .999), but their 0.5
  threshold is poorly placed: 57–78% accuracy (v1 and v2).
- **ESC-50 ±1 matching misses 80%.** About three quarters of these field
  recordings never fall to within a few LSB of silence, so ±1 changes are
  masked by recording noise.
- AudioMNIST ±1 matching at 20% passes only in v2 (80.8%, AUC .905), by
  0.8 points. v2 was designed after the v1 test result was seen, so treat
  this row as borderline.

**v2 low-bit profile and the FSDD miss**:
- v2 adds the fraction of samples whose lowest k bits are zero (k = 1..8). It
  was added after the v1 FSDD result, with splits and classifier unchanged.
- An earlier note in this file said every FSDD sample is a multiple of 256.
  **That was wrong.** Only speaker nicolas (train) is 8-bit audio up-scaled to
  16 bits. The other five speakers, including both test speakers, are genuine
  16-bit PCM: about half of their samples are even.
- FSDD stays near chance in v2, and **all FSDD rows miss 80%**. This looks like
  a limit of the data, not a feature gap:
  - The test clips are short: 2,950–3,240 samples at 8 kHz (median), against
    28,000 for AudioMNIST cuts.
  - Their quietest 256-sample frame still carries 4.9–8.0 LSB of noise
    (median second-difference RMS / √6). AudioMNIST is at 1.1 LSB.
  - A few thousand noisy samples carry too little LSB evidence.

## New structural rules (deterministic, no training)

All analyzers below are auto-discovered, bounded, and report only lengths and
hashes of recovered content.

- **`container_hiding`**: walks the structure of four formats.
  - **PNG chunks**: unregistered chunks; tEXt/zTXt/iTXt values that are large or
    encoded; CRC mismatches.
  - **JPEG**: APPn segments with unknown identifiers; encoded COM text.
  - **ZIP**: archive comments, member comments, unregistered extra fields.
  - **MP4**: free/skip/unknown top-level boxes and trailing data.
  - ICC, XMP, EXIF and ImageMagick raw profiles are exempt.
  - On this machine's 3,000 PNG, 219 JPEG and 790 ZIP/JAR/wheel files it raised
    **no** signal.
- **`text_invisible_unicode`**: recovers two kinds of hidden content as
  verified extraction.
  - ASCII smuggled in Unicode tag characters. Subdivision-flag emoji sequences
    are excluded.
  - Bytes carried by runs of variation selectors.
- **`image_jpeg`**: flags the fixed COM text of the encoder bundled with F5
  (strong, score 70). It identifies the tool's encoder, not a payload.

## Network traffic (`modules/network_covert.py`)

A pure-Python, bounded reader for pcap and pcapng.
- It understands Ethernet/VLAN, Linux SLL and raw IP.
- It decodes DNS questions, ICMP(v6) echo and TCP header fields.

Signals:
- **DNS tunnelling** under one base domain: at least 20 queries with mean
  subdomain length ≥ 24, ≥ 80% unique, ≥ 3.5 bits/char.
- **NULL/private query types.**
- **ICMP tunnelling**: at least 5 echo payloads of 16+ bytes and ≥ 4 bits/byte
  that match no ping stack (byte ramps, cyclic alphabets, zeros), with ≥ 50%
  distinct.
- **Header channels**: IPv4 reserved bit, TCP reserved bits, and an urgent
  pointer without URG.

Evaluation on the public tcpreplay **bigFlows** capture (791,615 real packets,
2,221 DNS queries, 2,801 echo packets):
- The capture was cut into 100 slices. Each slice was scored clean, then with
  one injected channel.
- The injected channels follow the tools' encodings: iodine-like base32 labels
  with NULL queries, dnscat2-like hex labels, random 64-byte ICMP echo, and
  TCP urgent pointers.

| Channel | Detected slices | Clean slices flagged | Accuracy |
|---|---:|---:|---:|
| iodine-like DNS (30 queries) | 100/100 | 0/100 | **100%** |
| dnscat2-like DNS (30 queries) | 100/100 | 0/100 | **100%** |
| ICMP tunnel (10 echo) | 100/100 | 0/100 | **100%** |
| TCP urgent pointer (10 packets) | 100/100 | 0/100 | **100%** |

**Caveats**:
- The first ICMP rule averaged over every echo payload and detected only 34/100.
  It was changed after inspecting bigFlows' benign echo payloads (zeros and two
  alphabet patterns). The ICMP row is therefore not an independent result.
- smallFlows (14,261 packets) was not used for design and is also clean.
- Real iodine/dnscat2 traffic could not be captured here: both need root (tun
  device or capture).
- Hashed-name services (CDN, telemetry, DNSBL) remain a known false-positive
  risk for any DNS-length rule.

## Video

**Container level** (`modules/video_container.py`):
- **MP4/MOV**: rebuilds every sample's byte range from `stco/co64`, `stsc` and
  `stsz`. It flags `mdat` bytes that no sample references (TCSteg-style hidden
  volumes that still play).
- **Matroska**: attachments other than fonts or cover art, binary tag values,
  and non-zero `Void`.
- **AVI**: non-zero `JUNK`, unregistered chunks, and data after RIFF.

Benchmark:
- 35 ffmpeg-made clean files: 6 synthetic sources plus DIV2K slideshows, in
  MP4/MOV/MKV (H.264, FFV1) and AVI (MJPEG), with audio.
- 49 hidden variants. Every hidden file was decoded by ffmpeg without error.

| Hiding | Files | Detected | Still plays |
|---|---:|---:|---:|
| TCSteg-style volume in `mdat` (20% of file) | 14 | **14** | 14 |
| Random `free` box | 14 | **14** | 14 |
| Binary MKV attachment | 14 | **14** | 14 |
| AVI `JUNK` payload | 7 | **7** | 7 |
| Clean files flagged | 35 | **0** | — |

**Frame level** (`modules/video_frames_lsb.py`, optional ffmpeg):
- Applies only to lossless codecs (FFV1, HuffYUV, UtVideo, PNG/QTRLE, raw).
- A single-threaded, resource-limited ffmpeg child decodes up to six evenly
  spaced frames. Weighted-Stego scores the median frame.
- Test data: 60 clean clips (20 per codec) of eight resized DIV2K validation
  frames, plus LSB replacement in every frame.

| Payload | Accuracy at WS > 0.10 (score > 0) | Accuracy at score ≥ 45 (heuristic evidence) |
|---|---:|---:|
| 5% | 51.7% | 50.0% |
| 10% | **98.3%** | 50.0% |
| 20% | **100%** | 53.3% |
| 40% | **100%** | **100%** |

- No clean clip scored above zero.
- The 45-point evidence threshold is the conservative still-image mapping
  (WS ≈ 0.26). For multi-frame video, the median estimate is precise enough
  for a WS > 0.10 operating point.
- H.264/VP9 and other lossy codecs are out of scope for pixel LSB.

## Real tools on DIV2K (`.benchmark/real-tools-20261111`, private)

- **Covers**: one random 512×512 crop of each of the 800 DIV2K train images
  (train), and two crops of each of the 100 validation images (test: 200
  covers). Train and test are source-disjoint.
- **Cover formats**: PNG, BMP 24-bit and JPEG q75 (Pillow, 4:2:0).
- **Payloads**:
  - Random bytes (stegano: random base64 text), with a random password per
    stego.
  - Spatial rates are bits per pixel; JPEG rates are bits per nonzero AC
    coefficient (all components).
- **Tools**: steghide 0.5.1 (BMP and JPEG), OutGuess 0.4, stegano 3.0.0.
  - 10,000 stegos were made.
  - 15 OutGuess embeddings at 0.2 bpnzac exceeded the tool's correctable
    capacity; they are recorded and excluded.

**Statistical detectors** (fitted per tool and payload):
- **v1**:
  - Spatial: rich RGB features with ridge-FLD.
  - JPEG: DCTR luminance (8,000 dims) with a random-subspace Fisher ensemble.
- **v2**:
  - Spatial: colour SRM (6,104 dims).
  - JPEG: Cartesian-calibrated DCTR (16,000 dims; the reference is a
    4×4-cropped, same-table recompression).
  - The classifier is chosen by a held-out fifth of train images.
- v2 was run only on the three weakest v1 rows. Calibration lifts both JPEG
  tools at 0.05 bpnzac from 67.5–70% to 95.5–98%.
- v2 colour SRM moved steghide BMP 0.1 only from 80.2% to 81.2%. **v3** adds
  three more train crops per DIV2K image (3,200 pairs; 29 of 24,000 extra
  embeds failed on capacity), with the held-out fifth chosen by source image.
  It reaches 84.0% (TPR 83%, FPR 15%).
- **Adaptivity**: the v2 design was motivated by v1 test rows, so the test
  split has been seen twice. Selection inside v2 uses train data only.

**Rules**:
- **stegano**:
  - `modules/stegano_lsb.py` reads stegano's `"<n>:"` length prefix from
    R, G, B LSBs. It handles row-major and prime (`eratosthenes`) pixel walks.
  - It recovers the full message, which must decode strictly in UTF-32LE or
    UTF-8.
  - Result: 2,000/2,000 stegos recovered with the right length; 0 of 1,000 PNG
    and 0 of 1,000 BMP covers flagged.
  - `image_bitplane` reports it as verified extraction. Only the length,
    encoding and SHA-256 are reported, never the content.
- **zsteg adapter fix**:
  - Once zsteg was installed, the adapter scored every clean image 70. zsteg
    always prints short random "text" and libmagic guesses, and smooth
    high-bit planes read as long printable text.
  - This also failed two existing accuracy tests.
  - Now only b1/b2 planes, appended data, or text-chunk flags count. Content
    must look real: long varied text, known file types, or `word{...}` flags.
  - `zsteg --all` also walks bit planes 3–8, which the filter discards. On a
    loaded machine it took 15–24 s per 512×512 image, over the adapter's 20 s
    limit; six PNG/BMP samples of the full test run then failed as timeouts.
  - The adapter now runs `zsteg FILE` (metadata, chunks, appended data) and
    `zsteg --all --bits 1,2 FILE` (every order, prime and alignment variant of
    the low planes). The rows the filter reads stay the same, and both runs
    take about 5 s together.
  - `meta <key>` rows (PNG text) never matched the row pattern, because the
    name has a space. They now count when they carry a flag.
- **zsteg on the test split** (both adapter runs, then the adapter's filter):
  - Clean covers: 2 of 200 PNG and 2 of 200 BMP flagged (1% false alarms).
  - stegano 0.1 and 0.4 bpp: 200/200 and 200/200 flagged.
  - steghide BMP 0.1 and 0.4 bpp: 3/200 and 0/200. steghide scatters its bits
    with a password-seeded walk, which a sequential bit-plane reader cannot
    see; the statistical detector above covers it.

**F5 and OpenStego** (Java tools, run with the user's approval):
- **Tools**:
  - F5: Westfeld's encoder (matthewgao build d08b5f8). It re-encodes the BMP
    cover with its own JPEG encoder at q75. The cover is the same encoder with
    no message, so both share the encoder and its COM marker. Rates are bits
    per nonzero AC coefficient.
  - OpenStego 0.8.6 RandomLSB on the PNG cover, at bits per pixel.
  - Every stego was extracted again with its password and compared
    byte-for-byte.
- **v1/v2 recipe** (one train crop per DIV2K image): calibrated DCTR for F5,
  colour SRM for OpenStego, classifier selected on the held-out fifth.

| Tool | Payload | One crop (v2) | Four crops (v3) |
|---|---|---:|---:|
| F5 | 0.05 bpnzac | 77.8% (AUC .855) | **80.8%** (AUC .891) |
| F5 | 0.1 / 0.2 bpnzac | **93.8 / 99.8%** (AUC .986 / 1.000) | — |
| OpenStego | 0.1 bpp | 74.5% (AUC .826) | 79.8% (AUC .870), miss |
| OpenStego | 0.4 bpp | **88.5%** (AUC .947) | — |

- v3 adds three more train crops per DIV2K image, as for steghide BMP; 11 of
  12,000 extra embeds failed on capacity. The held-out fifth is chosen by
  source image.
- Only the two rows that missed 80% were rerun, after their test results had
  been seen. F5 at 0.05 passes by 0.8 points; OpenStego at 0.1 misses by
  0.2 points.
- The F5 encoder comment is also a deterministic tool fingerprint
  (`image_jpeg`, see above).

## Verification

Fresh repaired CPU gate on **2026-11-14**: **4,588 passed**, two explicit
CUDA skips, 32 warnings and **97.39%** total statement coverage, in 1,891.03 s.
Ruff, mypy over 268 files (including `research`) and `git diff --check` pass.
Both `research/srm_features.py` and `research/rich_color_detector.py` have
100% statement coverage. The focused scanner regression passes one test and
the README/cloud-removal checks pass all fifteen tests. All 570 bound input
hashes match at full-gate close; wrapper time is 1,899.503044 s.

All 4,590 collected cases use the unchanged 90% coverage gate. The run uses
one BLAS/OpenMP thread, fresh disk-backed TMPDIR/basetemp and
`-o tmp_path_retention_policy=failed` to clean only new passed-test temporary
directories. Historical QA trees, datasets, checkpoints and failed logs remain
untouched. Only the rebuildable pip package-download cache was purged for
space; installed dependencies are unchanged. Complete commands, logs, input
bindings and coverage JSON are in
`.benchmark/academic-closeout-qa-20261114-q65ywhk3/`.

The first new full gate (2026-11-13/14) records **4,588 passed**, two explicit
CUDA skips, 32 warnings and **one teardown error**, with 97.39% coverage in
1,803.54 s. It is a failed command, not a pass. The progress-rename unit
test left its fake global `os.scandir` active during pytest's new passed-test
temporary-directory cleanup. Its assertions passed; cleanup then encountered
the fake directory entry. A focused run reproduced one pass /one error.
The test now scopes that patch to its own assertions, restoring the real
filesystem function before fixture cleanup. Application code is unchanged.
The original full log, 570-file input binding, pre-fix test source and coverage
data are retained in `.benchmark/academic-closeout-qa-20261113-ppi9ynip/`.

Full CPU gate on 2026-11-12, after the zsteg fix: **4,578 passed**, two
explicit CUDA skips, **97.39%** total coverage, in 1,837 s. Ruff, mypy over
268 files (including `research`) and `git diff --check` pass.

- The first full run of the same day failed two benchmark tests (4,572
  passed): `zsteg --all` exceeded the adapter's 20 s limit on six PNG/BMP
  samples while other benchmarks loaded the machine. Its log is kept.
- Logs: `.benchmark/full-pytest-20261112.log` (failed) and
  `.benchmark/full-pytest-20261112b.log` (passed).

## Evidence (private)

- `.benchmark/boss-gray-rich-20261111/`:
  - `split.py`/`split.json`, `run.py` (rich v1 validation) and `run_srm.py`
  - `srm-result.json` and the detectors
  - stopped and OOM logs
- `.benchmark/real-tools-20261111/`:
  - `make_corpus.py`, `manifest.json`
  - `eval_tools.py` → `eval-result.json`
  - `eval_tools_v2.py` → `eval-v2-result.json`
  - `make_extra.py`/`manifest-extra.json`, `eval_tools_v3.py` → `eval-v3-result.json`
  - `stegano-rule-result.json`
  - `eval_zsteg.py` → `zsteg-result.json`
- `.benchmark/wav-lsb-20261112/`: `run.py` → `result.json` (v1); `run_v2.py` →
  `result-v2.json`
- `.benchmark/boss-academic-20261112/`:
  - `run.py` → `result.json` (v1); `run_maxsrm.py` → `result-maxsrm.json` (v2)
  - `run_v3.py` / `run_v3.log` → `result-v3.json`
  - `run_rate.py` / `run_rate06.log` → `result-rate0.6.json`
- `.benchmark/queue-20261113.log`: v3 exit 0, JPEG exit 1, higher-payload exit 0
- `.benchmark/boss-jpeg-academic-20261112/`: original `run.py`, derived covers
  and failed `run.log`, preserved; `result.json` copies the isolated completion
- `.benchmark/boss-jpeg-academic-repair-20261113/`: `staging.json`, original
  source/log snapshots, output-only amended `run.py`, exclusive `attempt.json`,
  successful `run.log`, `result.json`, `exit.json` and `closing-audit.json`
- `.benchmark/network-20261112/`:
  - `benchmark.py`, `result-v1.json` (first ICMP rule) and `result.json`
  - `downloads/`: smallFlows.pcap (SHA-256 `77d06d3f…`), bigFlows.pcap (`2b630291…`)
- `.benchmark/video-20261112/`: `benchmark.py` → `set/`, `result.json`;
  `lsb_benchmark.py` → `lsb-result.json`
- `.benchmark/real-tools-20261111/`:
  - `make_java_tools.py`/`manifest-java.json`, `eval_java_tools.py` →
    `eval-java-result.json`
  - `make_java_extra.py`/`manifest-java-extra.json`, `eval_java_v3.py` →
    `eval-java3-result.json`
  - F5 and OpenStego binaries under `.benchmark/tools-20261111`
- Datasets and derived detectors are not redistributed.
- Software verification: `.benchmark/academic-closeout-qa-20261113-ppi9ynip/`
  (failed full gate, pre-fix source/coverage and focused reproduction), and
  `.benchmark/academic-closeout-qa-20261114-q65ywhk3/` (clean complete gate).
