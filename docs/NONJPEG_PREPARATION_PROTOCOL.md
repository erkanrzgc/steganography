# Non-JPEG controlled preparation v1 — 2026-10-12

This CPU-only stage prepares real-media, controlled LSB derivatives. It does
not train, score, calibrate, install models or change deployed analyzers.
Existing unsuccessful spatial/audio results remain visible.

## Fixed source roles and selection

Explicit config `nonjpeg-preparation-config-v1` binds each source manifest by
SHA-256 and supplies counts per origin, at most 32 train /8 validation originals.
The first actual run uses those maxima. DIV2K train/validation are one origin;
BOWS2 is native grayscale, not another RGB population. FSDD uses the historical
speaker roles (lucas/nicolas train, theo/yweweler validation, george/jackson
test). ESC preserves the already audited whole-recording max-fold roles.
Test originals are never decoded, embedded or scored. All source rows are
checked for identity, decoded-identity and speaker/recording/camera/device role
conflicts before selection. Whole origins/cameras are not certified independent.

Within each origin/role, sort SHA-256(`20261012:source_group:original_sha256`)
then original SHA; take the fixed requested count. No signal, label, model or
payload outcome selects/replaces originals. Every derivative preserves original
lineage, role, recording/speaker group, manifest hash and license. Duplicate
selected decoded carriers fail the entire job, never silently resample.

## Spatial and audio recipes

For native L/RGB images, make one fixed 256x256 center crop BEFORE embedding,
without resize or color conversion. This defines the controlled carrier; it
does not test payloads already present elsewhere in the original. Save that
same carrier as PNG and BMP. Gray has channel `gray`; RGB has `r/g/b/all`.
Test sequential and scattered replacement, planes 0/1, rates 5%/40% of the
selected channel units, floored to whole bytes. Every format/method/rate/channel/
plane cell is distinct; higher planes are not called ordinary plane-zero LSB.

WAV preserves complete native mono PCM16 (8 kHz FSDD, 44.1 kHz ESC), RIFF chunks,
sample rate and duration: no resampling, normalization, padding, stereo copies
or spectrogram resize. Same two orders, rates and planes; each sample is a unit.

Payload bytes are SHAKE-256 of the public recipe context
`20261012:original_sha256:method:rate:channel:plane`; no project marker or flag.
Scattered order uses Python's local Random(context).sample, sequential uses the
prefix. Formats share the same spatial bits/order. The generator checks changes
only touch the selected bit; a separate audit (no generator imports) checks
every saved carrier with scalar extraction and unchanged nonselected bits.
It rereads original bytes and recreates the label-independent carrier, verifies
complete fixed-cell membership, hashes, rates, grouping and source bindings.
Known-order oracle recovery is not autonomous tool recovery or accuracy.
After complete audit only, export separate schema-1.0 spatial/audio research
manifests with lineage/group roles and audit bindings. No absolute source path
is embedded; pass the explicit output directory as `--source`. These remain
development inputs; WAV descriptors need their own versioned contract.

## Bounds and interpretation

At most five input manifests /15,000 original metadata rows; each manifest
<=8 MiB, input <=32 MiB, decoded image <=8 million pixels, native mono WAV
160..220,500 frames. At most four origins /160 selected originals, 12,000
artifacts, 2 GiB additional bytes, 30 minutes per preparation/audit subprocess.
Workers have a 2 GiB address-space /32 MiB file /1,800 CPU-second limit, owned
process groups and no shell/external tool/network. Exclusive nonsymlink outputs;
failures retain partial artifacts and portable incomplete status. No automatic
retry, overwrite, download, GPU or paid compute. Bind code/protocol hashes.

Native GIF palette/frame methods, legitimate multilingual Unicode negatives,
stereo/spectrogram audio, compressed audio, tool-specific image families and
blind archive/polyglot recovery need separate source/embedding/oracle protocols.
Their current fixture presence is regression evidence, not dataset readiness.
Do not call them prepared by converting these images/audio. Future experiments
must freeze paired baselines, preprocessing, train-only calibration and per-cell
bootstrap metrics before fitting; the existing production gates are unchanged.
