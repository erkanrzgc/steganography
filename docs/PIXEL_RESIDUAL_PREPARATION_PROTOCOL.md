# Frozen JPEG pixel-residual preparation — 2026-10-05

Commit implementation and this preparation contract before extracting the real
development corpus. This slice prepares bounded raw tensors and a custom CNN
building block; **no real-corpus training, accuracy improvement, deployment or
source/context-invariance claim**. Training/evaluation still requires a separate
preregistered protocol, safe model persistence and independent-source gates.

## Inputs and preprocessing

Reuse all original 3,750 JPEGs and original split/order of the preceding JRM
experiments: 2,985 train / 765 validation, both ALASKA and BOSS-derived simulations,
JUNIWARD/UERD. No regeneration, download, lineage move, calibration, test access
or exclusions. Manifest SHA-256:
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`.
Previously inspected validation is development, not blind independent evidence.

`jpeg-center128-luma-u8-v1`: decode RGB/grayscale JPEG with Pillow 12.2.0,
reported JPEG codec 6.2; convert the decoded raster with Pillow `convert("L")`,
then center-crop exactly 128×128 at floor((width−128)/2), floor((height−128)/2).
This is Pillow decoded luminance, not raw JPEG Y coefficients. Reject CMYK,
non-JPEG, images smaller than 128, files >2 MiB and rasters >4 million pixels.
No resizing/interpolation, padding, EXIF rotation, augmentation or reference
cover. Only this central region is represented; it can miss noncentral payloads.
Do not imply full-image carrier coverage or richer DCT/context features.

Separate cache: `pixels.u8`, ordered N×1×128×128 uint8, descriptor with manifest,
row identities, split, exact decoder versions, image dimensions/crop regions and
data SHA-256. No labels, source IDs, filenames or declared quality enter tensors.
Loader binds descriptor checksum and exact size/data SHA before reconstruction;
array headers cannot allocate arbitrary dimensions. No NPZ/pickle/raw data in
the base wheel; generic 64 MiB feature-JSON/4,096-dimension model limits unchanged.

At most 4,000 rows and 64 MiB per split; expected train/validation raw bytes
48,906,240 / 12,533,760. Fixed framed worker, eight inputs maximum (≤16 MiB plus
headers), output exactly N×16,384 bytes; POSIX 1 GiB address space, 15/16-second
CPU, 2 MiB file, zero-core limits and 15-second wall timeout. Four workers max,
one batch queued per worker, 1,800-second split deadline. Incomplete output gets
no descriptor; no symlink input/output or overwrite. Optional model absent is
not detection success.

## Custom CNN building block

`jpeg-center128-residual-cnn8-v1`, optional Torch imported only when requested.
Input uint8 crops become float32 /255. Three fixed 3×3 filters: horizontal
centered difference, vertical centered difference and four-neighbor Laplacian;
valid convolution, residual clamp [-1,1]. Learn Conv(3→8,3,pad1)/ReLU/AvgPool2,
Conv(8→16,3,pad1)/ReLU/AvgPool2, Conv(16→16,3,pad1)/ReLU/global-average/Linear(16→1).
No batch normalization, dataset metadata, pretrained weights or adaptive shape
selection. This custom small network is **not SRNet, DCTR or a licensed upstream
model**. Logits are not calibrated probabilities. Fixed filters are buffers, not
trainable parameters. Bounded inference wrapper permits 1–64 exact uint8 crops,
eval mode, finite N×1 output. No model file loader, installed model or detector
integration is claimed in this preparation slice.

## Verification and next gate

Use independent full-Pillow-raster slicing for real crop parity, both splits and
sources/families/quality contexts; rehash every original file and bind every
cache descriptor/data checksum. Store portable audit hashes/versions/timing, not
raw images/tensors. Direct Pillow parity shares the native decoder and is not an
independent JPEG algorithm proof. Test malformed frames, truncation, oversized
headers, native failures, deadline, altered identity/decoder/region, symlink and
overwrite. Verify NumPy sliding-window filter math against Torch and an actual
optimizer/backprop smoke step with unchanged fixed filters, without accuracy
claims from generated fixtures or random weights.

Next training protocol must specify minibatches/CPU limits, seeds/objective,
bounded numeric model serialization, all-source and source-exclusion controls,
same-row comparisons against JRM, ONNX/numeric audits and all failures. Retain
the weak `.52–.55` transfer results. Untouched licensed sources, sufficient
independent scenes, calibration and blind CTF evaluation remain unfulfilled.

Explicit shared-service CLI:

```text
steganography research pixel-cache --manifest MANIFEST --source CORPUS
  --split train|validation --workers 1..4 --out FRESH_DIR
```

No automatic download, training, model installation or primary score change.
