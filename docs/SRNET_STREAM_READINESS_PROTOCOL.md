# Full-block streaming readiness v1 — frozen before real execution

This is an engineering I/O and exact-exposure gate, not a new accuracy
experiment. No real model fitting, validation pixels, model selection,
calibration, network requests or deployment are authorized by this protocol.
Prior chance-level real detector scores remain unchanged.

## Immutable inputs and schedule

- Complete prepared index SHA-256:
  `89f4c726922eb22e3c2edefeb24bf7d2a90473c4f9695a419be642df05fe76fd`.
- Independent full preparation audit SHA-256:
  `00c2a19067efa78fd9d2b214a6452d9e606c8a169eb54f9ae2c6521f807b40b8`.
- All 7,380 train rows, 1,638 original train lineages; no subset, old-corpus
  merge or truncation. All 1,611 validation rows have metadata checked for
  role/lineage separation, but no validation cache/tensor/JPEG is opened.
- Exactly one source/quality/method-balanced epoch, seed 20261008, new recipe
  `srnet-block-source-quality-method-four-row-v1`, labels `[0,1,0,1]`.
  Expected 6,576 pairs / 3,288 planned updates / 13,152 row presentations;
  each of the 7,380 unique train rows must appear. Counts are exposures, not
  independent scenes, trained updates or accuracy measurements.
- ALASKA declared unknown quality remains unknown; BOSS Q75/Q95 are correlated
  views. JMiPOD stays outside the matched-family recipe. Reserved WIFD and old
  acquisition exclusions remain as bound in the audited input index.

## Execution and failure rules

Use the fixed trusted module `steganography.research_srnet_stream`, operation
`check`, with explicit local configuration. Fresh output only, no resume or
overwrite. Snapshot all 12 execution dependency files before loading the
corpus and verify identical hashes before publishing a complete report.
The plan binds those hashes, decoder, index/audit, seed, optimizer settings
and every ordered pair/batch schedule. File contents are never executable.

The new reader bounds metadata to 16 MiB/document, 32 blocks, 768 rows/block,
12,000 total metadata rows and 4 GiB train tensor bytes. Legacy 4,000-row
sampler/trainer and whole-array cache contracts remain unchanged. Verify
every training cache through <=1 MiB chunks, checking length/SHA/finite values
before scheduling. Keep its regular non-symlink file handle; reject later
inode/length/mtime/ctime changes. Reads allocate at most four 256x256 float32
rows per batch (1,048,576 bytes), not a whole-corpus tensor. This is a tensor
allocation bound, **not** a total-process RAM claim. One reader is owned by
one job thread; its seek/read operations are not shared concurrently.

Every hash chunk, scheduled batch and fit optimizer boundary receives the
same job deadline, starting before corpus reads. Configured limit 1800s,
CPU threads 2; isolated CLI has an 1830s hard parent subprocess timeout,
8 GiB address limit, 3630s CPU bound, 32 MiB per output file, no core dumps.
The fixed worker launches no child processes. Timeout/failure leaves no
complete report; partial outputs are retained but unusable. Missing optional
dependencies mean failure/unavailability, not an engineering pass.

Report complete unique coverage, scheduled exposures/updates, ordered tensor
hash, maximum returned minibatch allocation, wall duration and start/end
verified source hashes. No loss, ROC/AUC, recovery or improvement claim.
Publish only portable JSON and documentation, not raw tensors or weights.

## Next training gate

Generated fixtures must establish **exact** model/loss/BN update equivalence
against the unchanged legacy four-row numerical optimizer engine, plus
isolated fitting and adversarial input/deadline/source-mutation tests.
Passing that and full real streaming is necessary but insufficient for real
learning. Preregister adequate train-only exposure and a learning gate before
another real fit or validation inspection. The historical 1,580-update CPU
fit required 1,787s; extrapolating to 3,288 updates suggests ~62 minutes per
epoch, not measured throughput. Do not run a predictably incomplete 1800s
fit, extend its budget silently, provision paid compute, or upload restricted
datasets automatically. GPU/compute choice and subsequent bounded training
protocol remain separate decisions. No GPU implementation is claimed here.
