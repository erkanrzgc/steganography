# Private real-data timing preparation

The 2026-10-10 preparation is complete, not a training/accuracy result or a
measured cloud cost. No cloud resource was created and nothing was uploaded.

The frozen protocol is
`benchmarks/jpeg-real-timing-protocol-20261010.json` (SHA-256
`a96560070d08a3e9de34ccac1a8d9e5124d71fd48f1337b255af91ee396bcf96`).
It binds the previous ALASKA/BOSS preparation and independent audit and the
new BOWS original acquisition and independent audit. Thirty-two complete
**training** originals per declared source are selected by deterministic
identity ordering, without model scores or validation pixels.

| Source | Originals | JPEG rows | Provenance |
| --- | ---: | ---: | --- |
| ALASKA2 | 32 | 96 | Native cover/JUNIWARD/UERD, unknown quality/rate retained |
| BOSSbase-1.01 | 32 | 192 | Previously audited Q75/Q95 simulations |
| BOWS2 | 32 | 192 | New Q75/Q95 conseal 2025.11 simulations |

BOWS uses the historical, unchanged 0.2 bpnzAC JUNIWARD/UERD recipe and seed
20261006. Source identity is explicitly BOWS2, not relabeled BOSS. Simulated
embedding does not supply an encoded secret or exact payload recovery evidence.
The legacy BOSS worker's default source and CLI contract are preserved.

The private output occupies approximately 166 MiB including simulation evidence.
Its complete 480-row unrounded Y-component, phase-aligned center-256 float32
cache is 125,829,120 bytes (120 MiB). Manifest SHA-256:
`53af09292d57f4ddba383060452a21979f00028a9938af3a86ccdf4372d136ca`.
The kit's balanced three-source schedule has **192 four-row optimizer updates**;
the prospective full 12,222-training-row corpus would have **4,932**. These are
not interchangeable epoch sizes. The full BOWS-derived corpus and full new
reader/trainer integration are still pending.

`scripts/audit-jpeg-timing.py` imports none of the preparation or sampler
implementation. It rereads all JPEG bytes, all 32 selected BOWS original bytes,
prior metadata bindings, deterministic training selection and whole method
families. All 480 cached crops pass a separate full-raster SciPy IDCT calculation:
maximum absolute difference 5.684341886080802e-14 (fixed tolerance 1e-4, rtol 0).
All 128 BOWS cover/stego coefficient comparisons pass count, magnitude and
quantization checks. Native jpeglib parsing is shared, not independently verified;
camera/scene independence remains unverified. Portable aggregate evidence:
`benchmarks/jpeg-real-timing-independent-audit-20261010.json`.

The initial auditor mistakenly applied the generated-file 256 KiB limit to
native ALASKA JPEGs. Preserve that failed attempt in
`benchmarks/jpeg-real-timing-audit-bound-failure-20261010.json`; the retry uses
the existing native 2 MiB carrier bound. No dataset/production limit was raised.

## Execution and limits

Local preparation is explicit:
`python -m steganography.research_jpeg_timing --help`. It requires all frozen
manifest/audit/protocol checksums and a fresh private output directory. The
shared service verifies previous train caches, reads no validation cache and
never overwrites output. Input symlinks, traversal, duplicate bytes, cross-role
originals and cross-source original overlap fail closed. Isolated BOWS workers
have 2 GiB RAM, 90-second timeouts and bounded stdout/files. JPEG decoding uses
the existing four-row/15-second isolated worker. Preparation has a 600-second
deadline and 512 MiB data budget, including retained simulations and full tensor.
Actual execution also used a 610-second process-group outer timeout.

Corpora, raw JPEGs/PGMs and tensors remain ignored private artifacts. Source
licenses/restrictions are unchanged; do not publish them or upload to a cloud
provider without reviewing permission. Only aggregate evidence is tracked.

## Next cost gate, not a promise

Before a paid full run: integrate a separately bounded, checksum-bound timing
reader/worker, freeze warm-up/measured-update and cost-projection rules, then
measure real GPU optimizer updates with real kit I/O. Include setup/upload,
validation/checkpoint overhead and disk charges separately. Quote a fresh live
hourly rate and a firm spending cap before creating a paid resource. Do not
present old generated CUDA probes, CPU preparation time or illustrative GPU
prices as a measured training duration. The unchanged BN model is a timing
control, not the pending normalization intervention or an accuracy improvement.

No new model was trained or deployed, no real GPU duration/cost is available,
and all previous failed detection cells remain failed.

## Verification

Final application-code regression: 1,932 passed, one actual-CUDA hardware skip
(not a pass), 32 existing ONNX deprecation warnings, 511.78 seconds; total
coverage 95.77%. All 64 focused preparation/worker/legacy compatibility tests
also pass separately. New shared service coverage 99%, explicit frontend 98%.
Three portable-evidence/auditor guard tests added after full-run collection
pass separately. Ruff, mypy, pip dependency checks and whitespace checks pass.
Wheel (160 members) and sdist (324 members) contain the new code and no private
corpus/tensors/weights. Verified on Python 3.11.14; other Python versions,
Docker and a physical cloud GPU are not verified in this slice.
