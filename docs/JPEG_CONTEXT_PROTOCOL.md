# JPEG measured-context development protocol — 2026-10-05

Freeze this protocol and implementation before generating or scoring the new
128-lineage experiment. This is iterative **development**, not blind testing,
cross-source qualification or permission to install a model. Historical held-out
and development results remain unchanged. No threshold/model search is allowed.

## Bound inputs and roles

- ALASKA2 development manifest SHA-256:
  `a3fd898fecbd50135f15a9f36c990d97e9d1779afb58076a35d8a39e1ce5a16d`.
  Original acquisition SHA-256:
  `f54f1f66c366a5e57c4ab65214a211e807ac4a0d0a5a1a549e736b34c2bbe288`.
  Train/validation 968-feature artifact hashes:
  `43d76a3c613a82d2cb021d43183793efad8105e8dcb8fe9bc246750f5b0f29a7` /
  `28492353142025628a260284d9d622f93eb6df16a2a69b308116ab1fb8a9375b`.
- BOSSbase development original manifest SHA-256:
  `eea495733f0243b420850ca00372877e30c89836738ebc9175ec4b2bbe571d57`.
  These originals were already used for spatial development. Their JPEG
  derivatives are not fresh unseen scenes or a new independent held-out test.
- Preserve original train/validation roles and SHA-based cover lineage across
  every quality/method derivative. Check all originals against frozen BOSS,
  Kodak, pilot and ALASKA2-holdout identity manifests before generation.
- Select 128 BOSS originals by ascending SHA-256 of
  `jpeg-context:20261006:<original-sha256>`, without looking at scores.
  Encode grayscale JPEG covers with Pillow at qualities 75 and 95, no optimize.
- Use optional `conseal==2025.11`, record all numerical/library versions.
  JUNIWARD uses that release's default ORIGINAL implementation; UERD explicitly
  uses `payload_mode="bpnzAC"`. Both simulate 0.2 bpnzAC changes, seed from
  SHA-256 of `20261006:<lineage>:<quality>:<method>` (first four bytes, big-endian).
  These are upstream embedding simulations, **not encoded message/flag recovery**.
  Verify nonzero unit coefficient changes and exact DCT/quantization round trips.
- Restrict **both** origins to covers, JUNIWARD and UERD. Exclude ALASKA2 JMiPOD:
  conseal 2025.11 has no matching simulator. Do not silently substitute a method.
  ALASKA2 payload rate and declared quality remain unknown, not guessed labels.
- Two distinct declared acquisition origins are mandatory. They do not prove
  independent cameras, scenes, preprocessing or absence of perceptual duplicates.
  No raw corpus, simulated JPEGs or trained weights are redistributed.

## Features and objective

`jpeg-context-summary-v1` has 1,098 dimensions: the existing 968 DCT summary
plus 63 magnitude/quantization interactions, 63 entropy/quantization interactions
and four global context summaries. For each AC position, histogram magnitude is
the mean clipped-bin index divided by seven; entropy is normalized by three;
quantization strength is `q / (q + 32)`. Interactions are magnitude × strength
and entropy × (1 − strength). Globals are mean strength, nonzero fraction,
entropy and magnitude. Round this new contract to 1e-8; old contracts unchanged.
These are measured byte-derived interactions, not a deep contextual model or
proof of robustness. No source ID, filename, method, label, reference cover or
declared quality is a feature. Cached legacy summaries derive the same contract.

Fixed CPU linear training: seed 20261006, 300 Adam epochs, learning rate 0.01,
train-only standardization (standard deviation floor 1e-4), float32 training,
explicit float64 normalization/accumulation with float32 input/output inference.
Recipe `jpeg-source-class-balanced-v1` weights each training row by
`N / (2 * source_count * count(source, label))`. Both classes and identical stego
method sets are required per source, with distinct original-manifest hashes and
nonempty source URL/license provenance. Validation never selects weights.
Class balancing is based on weighted masses. Fixed probability cutoff is 0.5.

## Outputs, gates and failure handling

Generate into a fresh exclusive directory via `research_jpeg_corpus.generate_boss`;
merge bound origins with `research_jpeg_multisource.prepare_dataset`. The final
`preparation.json`, not partial files, marks successful preparation. Worker
processes have 90-second timeout, 2 GiB address-space, CPU/file/output bounds;
two workers maximum. These controls are not an OS filesystem sandbox.
Use shared `train_model`, `predict_validation`, `export_onnx`, `onnx_parity` and
`diagnose_validation`, retaining hashes/version/provenance and every failure.

Publish pooled and per-origin/per-method metrics, all available declared-quality
cells (BOSS 75/95; ALASKA unknown), single-label/missing metadata as unavailable,
paired-lineage 95% intervals and fixed-threshold FPR/recall/BA/AUC/ECE. Pooled
development numbers cannot satisfy cross-source held-out support gates. Source
overlap and absent camera/device metadata must stay visible. Independently audit
generated JPEG hashes/coefficients, original integrity, cached feature values,
confusions and CPU/NumPy/ONNX inference (batch sizes 1, 17 and all validation;
absolute tolerance 1e-6, relative tolerance zero). Do not loosen failed gates,
hide regressions, tune on held-out data or install/export a supported model pack.
