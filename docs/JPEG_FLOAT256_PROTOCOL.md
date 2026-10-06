# Frozen unrounded JPEG preparation — 2026-10-06

Commit this contract and implementation before real corpus extraction/audit.
This is preprocessing readiness, not training or accuracy. Preserve all old
128-byte crops, models, failed results and original JPEG files.

## Decode contract

Feature ID `jpeg-y-idct-center256-phase0-f32-v1`: load quantized Y coefficients
with optional jpeglib under explicitly selected libjpeg 6b. Read the actual
component-0 quantization-table assignment, never assume table index 0.
Only grayscale/YCbCr color-space names accepted; RGB-component JPEG, CMYK/YCCK
or undersampled Y geometry rejected. Names are deliberate: jpeglib 1.0.2 enum
equality compares distinct color spaces as equal. No RGB-to-gray conversion.

Dequantize in float64, apply separable orthonormal 8×8 inverse DCT using cosine
basis, add 128 level shift, then cast to little-endian float32. No rounding,
clipping, EXIF rotation, resampling, recompression or reference-cover input.
Crop 256×256 at `(8*floor((width-256)/16), 8*floor((height-256)/16))`.
This near-center origin retains original JPEG 8-pixel phase, not an arbitrary
center crop. Native Y cropping differs from prior Pillow grayscale/128 and is
not a claim of binary-equivalent published SRNet JPEG preprocessing.

Separate decoder provenance records jpeglib/NumPy version, native selection,
IDCT formula, no rounding/clipping. Version strings cannot guarantee identical
native binaries/BLAS on every host; raw cache SHA binds actual output here.
API details: [jpeglib DCT reference](https://jpeglib.readthedocs.io/en/latest/reference.html).

## Bounds and storage

Input <=2 MiB, decoded area <=4M pixels, minimum side256. Exact Y block grid,
signed16 coefficient bounds, integer quantization [1,65535]. At most four
JPEGs per fixed framed subprocess, 15-second wall / 15–16 CPU deadline,
1 GiB address space, 2 MiB output file limit, core dumps disabled. Only fixed
module is executed; extracted data never code. Frames reject trailing/truncated
bytes; exact output size and finite/magnitude guards before accepting a tensor.
Conservative absolute pixel limit 2^36 allows genuine unrounded overshoot.

Explicit `research srnet-cache --manifest M --source D --split train|validation
--workers 1..4 --out FRESH`. Shared service, new schema `research-float256-cache-v1`,
`pixels.f32`, `<f4`, shape N×1×256×256. Descriptor written only after complete
success, provenance/cache/source identities verified; no symlinks/overwrite.
At most4,000 rows and 1 GiB raw per split, 1,800-second split deadline, bounded
worker queue. Full loader may allocate up to1 GiB plus validation temporaries;
it is an explicit research operation, not a generic API read. Original uint8
64 MiB limit and default schema/128 preprocessing remain unchanged.

SRNet has separate bounded float256 inference in pixel units, preserving
fractional values, requiring every module eval. No trained weights saved here.

## Fixed real preparation/audit

Same manifest SHA
`0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14`:
all 2,985 train and765 validation rows, original order and lineages, no test
access, exclusions, regeneration or download. All original file SHA rechecked.
Retain two BOSS quality variants in each original-scene lineage.

Independent synthetic mathematical oracle: SciPy orthonormal inverse DCT,
nonzero component table, fractional DC/negative overshoot and odd dimensions.
Real audit: bind all row/cache/original hashes; compare all same-source-method
contexts on18 fixed examples (nine per split: three cover/JUNIWARD/UERD from
ALASKA plus three at each of BOSS Q75/Q95), using independent SciPy full-raster
IDCT followed by declared crop. Tolerance 1e-4 absolute / zero relative on
real float32 pixels. Same quantized coefficient parser is shared, not an
independent native-decoder correctness proof. Record failures, never widen
tolerances silently. Preparation cannot pass detection gates or supply an
untouched source; real training/numeric persistence/qualification remain pending.
