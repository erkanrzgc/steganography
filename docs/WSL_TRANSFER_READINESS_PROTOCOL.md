# Local WSL transfer and train-only readiness — frozen before checking

This gate verifies a private local copy and the exact train-only I/O schedule.
It authorizes no real fitting, validation inference, threshold/model selection,
model installation, cloud transfer or accuracy claim. The separate generated
CUDA preflight has completed; it is not real-data learning evidence.

## Fixed inputs and identity checks

- Complete prepared index SHA-256:
  `89f4c726922eb22e3c2edefeb24bf7d2a90473c4f9695a419be642df05fe76fd`.
- Independent preparation audit SHA-256:
  `00c2a19067efa78fd9d2b214a6452d9e606c8a169eb54f9ae2c6521f807b40b8`.
- Original generated WSL GPU report SHA-256:
  `27d89db7a1f8ab4aea8fff80c820f133cfc3029d5fb3134e14ff803f7c979d11`.
  Its 14 execution-source hashes must equal the current checkout.
- The private prepared tree contains 9,082 regular files. Transfer it into a
  fresh owner-only local WSL staging directory using pinned-host, key-only SSH;
  do not overwrite an existing destination or accept symlinks.
- After copying, require equal regular-file counts, no symlinks and an empty
  recursive `rsync --dry-run --checksum --itemize-changes` comparison. This is
  a transport checksum check, not a claim that rsync uses SHA-256 or independently
  replays preprocessing. Copying/comparing the complete tree reads validation
  bytes for transport integrity only; no validation predictions or tuning.

## Native train-only check

Use `steganography.research_srnet_stream --operation check`, a fresh output,
`device: cuda:0`, one epoch, seed 20261008, threads 2, max_seconds 1800,
learning_rate .001 and weight_decay .0001. Preserve NumPy 2.4.6/jpeglib 1.0.2
and the audited unrounded float-cache contract. Before publishing, verify:

- Every index/block/train-cache binding and all train tensors by SHA-256;
  required independent preparation audit and complete split/lineage metadata.
- Exactly 7,380 unique train rows and 1,638 original train lineages;
  6,576 scheduled pairs, 3,288 **planned** updates, 13,152 presentations.
- No validation cache/tensor/JPEG opened by this native check. Validation
  metadata is checked; transport-byte verification above is a separate operation.
- Ordered tensor SHA-256 identical to the historical CPU check:
  `38a8794d44f3edde43cd38e6b577b982f06b4a021e7d61ceccb0700a011ff91a`.
- All 13 execution-source hashes captured before native reads and unchanged at
  publication; identical recipe/accounting to the historical check. A fresh
  CUDA v2 plan binds execution metadata; never reuse an old CPU plan.
- Kernel cgroup-v2 RAM bound <=8 GiB, four-row/1 MiB returned tensor batches,
  propagated 1800s deadline, 1830s isolated parent wall timeout, 3630s CPU
  bound, 32 MiB output-file bound, no core dump or CPU fallback.

The native `check` validates CUDA availability/policy metadata but performs
its tensor I/O on CPU. It does **not** execute an optimizer or prove GPU
training throughput. Report actual duration without a speedup claim.
Failures/partial copies are retained and unusable; no unbounded retry.

Publish only portable evidence JSON and documentation, with no host paths,
addresses, SSH identities/keys, credentials, raw data or weights. Adequate
train-only learning and independent held-out evaluation remain separate gates.
