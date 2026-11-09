# Native RGB: external-cover training plan — 2026-11-09

The parity-input SRNet passes its native gate (reserved AUC .87), and an
LSB-matching fine-tune lifts low-rate development AUC by 7–11 points. Both are
limited by the 64 native training originals. This step adds external lossless
RGB covers and moves training to a Colab T4 GPU. The user explicitly approved
dataset expansion and downloads on 2026-11-09.

## Data

| Source | Content | Use | Terms |
| --- | --- | --- | --- |
| DIV2K HR (ETH Zurich) | 800 train /100 valid 8-bit RGB PNG, ~2K | RGB covers | Academic research only; not redistributed |
| BOSSbase 1.01 (Binghamton) | 10,000 512×512 grayscale PGM | Later L-mode model | Research benchmark; not redistributed |

Downloads are kept privately in `.benchmark/external-datasets-20261109` with
SHA-256 hashes. The Colab notebook downloads DIV2K directly inside the runtime;
no dataset or native pixels are uploaded. Native reserved originals stay closed.

## Training (`research/external_rgb_training.py`)

- `bank`: a fixed number of random 256×256 crops per source image (`crops.npy`
  memory map plus per-image `groups.npy`); non-RGB or small images are skipped.
  DIV2K train and valid stay image-disjoint.
- `train`: every presentation applies a random D4 transform and a fresh LSB
  replacement or ±1 matching payload; rate curriculum 40% → {20, 40} →
  {10, 20, 40} → {5, 10, 20, 40}% (epochs 2 /4 /8). Parity channels are
  computed on the device; CUDA uses AMP. Validation scores fixed stegos for
  eight conditions (both kinds × 40 /20 /10 /5%) with AUC and accuracy at .5.
  Checkpoint and model are written atomically after every epoch and resumed
  automatically; training stops before an epoch would miss the deadline.
- Notebook: `notebooks/external_rgb_training_div2k_20261109.ipynb` (T4, 16
  crops per train image = 12,800 pairs per epoch, 12 epochs, decay at 9,
  16 pairs per batch, 3.5-hour deadline, checkpoints in Google Drive).

After training, the exported model is scored locally on the native
development set to measure cross-dataset transfer. Generated CPU QA covers
the module at 100% statement coverage.

## Before GPU training: DIV2K validation baseline (CPU)

Local banks: 800 train images × 16 crops = 12,800 crops (2.4 GB), 100 valid
images × 2 crops = 200 crops, no source skipped. Existing models and WS on
the fixed DIV2K validation stegos (cross-dataset for the native-trained CNNs):

| Validation AUC (200 pairs) | WS | Decay40 seed 1 | Matching seed 1 | Matching seed 2 |
| --- | ---: | ---: | ---: | ---: |
| Replacement 40% | .991 | .756 | .728 | .720 |
| Replacement 10% | .908 | .567 | .609 | .624 |
| Matching 40% | .496 | .758 | .728 | .715 |
| Matching 20% | .508 | .638 | .681 | .689 |
| Matching 10% | .526 | .564 | .608 | .625 |
| Matching 5% | .513 | .532 | .559 | .573 |

The CNNs reach accuracy .61–.68 at 40% on this unseen camera/content mix;
WS still owns replacement but stays at chance on matching. These are the
numbers the GPU run must beat. Private result `div2k-baseline-20261109.json`.

## Verification

The first full QA attempt aborted near 72% when the 7.8-GB `/tmp` tmpfs
filled (basetemp was placed there by mistake); its log is preserved privately
as unavailable, not a pass. The disk-backed rerun passes 4,483 tests /two
explicit skips (CUDA hardware, optional dependency) with 97.27% total
coverage; both new research modules have 100% statement coverage. Ruff
(including notebooks) and 257-file mypy pass.
