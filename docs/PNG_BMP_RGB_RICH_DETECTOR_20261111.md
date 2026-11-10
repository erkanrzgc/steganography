# RGB rich-model detector (CPU, 2026-11-11)

## Headline

A classical rich-model detector (`research/rich_color_detector.py`) beats the
parity-input SRNet by a wide margin on DIV2K. Both are scored on the same
fixed validation stegos: 200 crops from the 100 DIV2K validation images, never
seen in training. One detector handles both embedding kinds at mixed rates.
It trains in about two minutes on eight CPU cores, with no GPU.

| Condition | Best CNN AUC / acc | Weighted Stego AUC | Rich detector AUC / acc |
|---|---|---|---|
| matching 40% | .758 / .680 | .496 | **.989 / .930** |
| matching 20% | .689 / .618 | .508 | **.968 / .912** |
| matching 10% | .625 / .598 | .526 | **.901 / .802** |
| matching 5% | .573 / .557 | .513 | **.781 / .613** |
| replacement 40% | .756 / .690 | .991 | .990 / .930 |
| replacement 20% | .690 / .623 | .977 | .970 / .912 |
| replacement 10% | .624 / .598 | .908 | .897 / .797 |
| replacement 5% | .574 / .565 | .829 | .783 / .625 |

- "Best CNN" is the best of the three frozen states for each row, taken from
  `div2k-baseline-20261109.json`.
- Accuracy uses the detector's fixed zero threshold, which is not tuned on
  validation. At 5% the threshold is too high, so accuracy lags AUC.
- On native development originals the gain does not transfer (cover-source
  mismatch). See below.

## Design

- **Per-channel SRM-lite co-occurrences**:
  - first- and second-order residuals: four consecutive values, T = 2 (625 bins each)
  - third-order residuals: three values (125 bins)
  - each pooled over R/G/B and over horizontal and vertical scans
- **Colour co-occurrences**: the joint (R, G, B) residual triplet at each
  pixel, T = 3 (343 bins). Four variants: first-order q=1, second-order q=1
  and q=2, third-order q=2. Independent ±1 changes per channel break the
  inter-channel residual correlation of natural photos.
- 2,747 normalised features in total.
- **Classifier**: ridge-regularised Fisher discriminant on standardised
  features. The ridge (0.03–1) is chosen by AUC on a held-out fifth of
  source-image groups.
- **Training pairs**: one fresh stego per crop, drawn with seed `2026111101`.
  Kind is replacement or matching; rate is 5, 10, 20 or 40%. The embedding is
  the same `spatial_matching_probe.embed` used by the CNN experiments.
- The model is stored as plain `npz` (no pickle) and loaded with
  `allow_pickle=False`.

Command:

```text
python -m research.rich_color_detector BANK OUT --valid VALID_BANK --workers 7
```

DIV2K run details:

- 12,800 training pairs; selected ridge 0.03 (group-holdout AUC .913)
- 125 seconds in total, including validation
- Detector hash: `e50fbba7167006c8…`

## Native development originals (20 RGB, development role)

Stegos are the fixed hard-condition stegos (seed 2026110903), as in the
matching fine-tune. Reserved originals stay closed: they were consumed by the
earlier one-time evaluation.

| Detector | m40% | m20% | m10% | r40% | r20% |
|---|---|---|---|---|---|
| DIV2K only | .667 | .690 | .675 | .650 | .688 |
| DIV2K + native train ×4 (D4), ridge 0.03 by mixed holdout | .725 | .725 | .693 | .713 | .708 |
| DIV2K + native train ×4 (D4), ridge 0.3 by native-train holdout | .750 | .733 | .693 | .755 | .743 |
| Matching-fine-tuned CNN (seeds 1/2, from the matching probe) | – | .78/.74 | .66/.70 | – | – |
| Weighted Stego | .525 | .557 | .505 | 1.0 | .953 |

- The native-train ×4 rows add the 64 native train originals, four times each
  with D4 augmentation.
- The detector responds to embedding: the stego scores above its own cover in
  80–90% of pairs. AUC stays low because image-specific score offsets
  dominate, i.e. cover-source mismatch between DIV2K and the heterogeneous
  Commons PNGs.
- With only 20 sources, one AUC step is .0025 per pair and the confidence
  intervals are wide. Exploratory scratch runs with other stego draws ranged
  from .70 to .80 at 40% matching.

Two variants did not help and were dropped:
- Re-embedding calibration (features minus features after a full-rate
  matching pass) did not improve native development.
- DIV2K-only training does not help native development either.

## Interpretation

- **The CNN was data-limited, not concept-limited.** With 12,800 DIV2K crops,
  hand-designed residual and colour statistics plus a linear classifier
  already reach 90–99% AUC above 10% rates. This is the bar the planned
  DIV2K GPU CNN run must beat.
- **Native accuracy is limited by cover-source mismatch and only 64 native
  training originals.** More native-like covers (similar source, processing
  and resizing) are the next lever, not a bigger model alone.
- **Weighted Stego remains the replacement specialist.** The rich detector is
  the first detector here that works on ±1 matching without native-only
  training.

## Evidence (private, `.benchmark/rich-color-detector-20261111/`)

- `div2k/` holds `detector.npz`, `result.json` and `div2k-train.log`.
- Native runs:
  - `native_mixed.py` with `native-development-result.json` and
    `div2k-native-detector.npz` (`a1bf4bb0…`)
  - `native_domainridge.py` with
    `native-development-domainridge-result.json` and
    `div2k-native-domainridge-detector.npz` (`0ae6bd2f…`)

Verification:
- 13 generated tests; 100% module coverage
- Ruff and mypy pass
- No GPU, network, reserved-pixel access or `core/` change
