# Frozen native-format acquisition retry — 2026-10-08

The JPEG-only run under `DATA_EXPANSION_PROTOCOL.md` failed before producing
`source.json`: Canon EOS M6 SDR `.jpg` files are native two-frame MPOs.
Observed first-file size/Git blob identity matched upstream; this is not
network corruption. Retain the partial directory and original protocol.
No acquired file or accuracy result from that attempt is counted as success.

An explicit `--allow-bounded-mpo` retry uses the identical pinned source,
20-per-camera hash-ordered selection and reserved manifests, in a new directory.
No camera, difficult sample or selection identity is dropped. All original
network/byte/pixel/resource/license/integrity bounds remain unchanged.
JPEG must be one frame. MPO must declare 2–4 frames; decode only the primary
frame (<=32M pixels), preserve the complete original bytes and record native
format, declared frame count and primary-only decode status. Never relabel MPO
as JPEG or claim secondary frames were decoded/validated. This is acquisition,
not model-ready JPEG preparation or detection coverage for MPO.

Reserve the entire WIFD origin from training/calibration. Independently reread
all primary frames and byte/blob identities, and audit native format/frame
metadata, license evidence and exclusions. Completion is not independent scene
proof or a qualified benchmark. No training/evaluation or format conversion in
this retry; a later evaluation protocol must decide preprocessing without
consulting WIFD detection scores.
