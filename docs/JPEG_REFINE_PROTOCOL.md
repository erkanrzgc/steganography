# Frozen JPEG refinement — second round

This opt-in intervention is separate from the completed October 14 controls.
Historical receipts and all 16 private snapshots retain their original bindings.
Implementation is available; physical execution and learning evidence must be
recorded separately. No model is installed, exported or deployed by this command.

```text
steganography research jpeg-refine --config PRIVATE_CONFIG --out FRESH_DIR
```

The private config has schema `jpeg-refine-config-v1`, `root`, `manifest_sha256`,
`audit`, `audit_sha256`, `source_sha256` from
`steganography.research_jpeg_refine.sources()`, `recipe` from
`core.jpeg_refine.recipe()`, `budget_seconds: 21600`, `budget_bytes: 10737418240`,
and `expected_gpu: {name, uuid_sha256}`. The UUID hash is SHA-256 of the string
representation of the physical Torch device UUID, verified against the owned
RTX 5060 over the existing host-key-verified SSH connection. Paths stay private;
public receipts contain hashes, numeric evidence and portable names only.
The data and original audit identities are frozen respectively to
`87df666c9896cec3e1b4897ead54321af4b8f79f4a0ec02e397582e725638fb5` and
`e9aae4d6c35199fcefa7e7752b790ec7143fb8a1ebd059bfc2f3867fc1de163b`.
Configurations cannot substitute data, override the recipe, reduce updates or
enable CPU fallback.

## Frozen data and arithmetic

Reuse the checksum-bound five-view BOSSbase-1.01/BOWS2 data: 256 fit originals
and 64 development originals per source, native grayscale Q75 JPEG, controlled
JUNIWARD .4 bpnzAC simulation. Unrounded float32 center/four-corner views and
D4 augmentation are unchanged. Views do not multiply independent original
counts; camera independence remains unknown. No WIFD/test/reserved source is
opened, and no model or dataset is downloaded. A fresh independent SciPy IDCT,
native coefficient and cache-hash audit precedes generated CUDA preflight.

Same-pooling SRNet-style architecture, BN epsilon/momentum, original Kaiming
initialization, author-rule Adamax, convolution-only L2 and constant LR .001
remain unchanged. Genuine batches have 16 images. AMP, TF32, accumulation and
CPU fallback are disabled. This is not a full upstream SRNet reproduction.

For seeds 20261012 and 20261013, train 2,048 paired updates from scratch once.
Branch the complete model/optimizer/BN state into paired and unpaired arms,
each continuing for 6,144 updates. Each final model has 8,192 total updates;
actual matrix training cost is 28,672 updates, with six evaluated endpoints.
Warm-up reproduces the historical genuine-16 schedule prefix exactly.

Each batch contains four covers and four stegos per source. Paired batches
contain eight originals; unpaired batches contain 16 different originals.
Per epoch and source, stego rows undergo a separate deterministic, nonzero
cyclic shift by a multiple of four original positions. Image/view/D4 tuples
move together; canonical per-epoch exposure hashes match between arms.
The shared warm-up and 512-update checkpoints fall at complete epoch boundaries.

The paired continuation versus warm-up measures added exposure; unpaired versus
paired measures the batch context intervention. The motivation is discussed in
[How to Pretrain for Steganalysis](https://ws.binghamton.edu/fridrich/Research/IN-pretraining-6.pdf),
which investigates removing pair constraints after initialization/pretraining.
Our smaller fixed-data intervention is not a reproduction of that paper.

## Ownership, limits and continuation

The controller alternates seeds at every warm-up checkpoint, then all four
arm/seed cells at every continuation checkpoint. It emits 56 numeric-only
`jpeg-refine-snapshot-v2` NPZ checkpoints, each <=96 MiB; pickle/torch.load,
symlinks and overwrite are forbidden. Exact array shapes/dtypes, finite values,
optimizer steps, BN counters, schedule position, parent checksum/binding and
shared-warm-up checksum are validated before accepting a receipt. Legacy v1
snapshots cannot be imported or relabeled. Every receipt records loss/objective,
time, CUDA peak allocated/reserved memory, updates and exit status.

CUDA policy requires a kernel cgroup-v2 resident limit <=8 GiB and allocator
limit <=4 GiB, two compute threads, IEEE FP32 and deterministic execution.
Each worker has 1,800-second CPU/internal-wall limits; the parent permits at
most 1,830 seconds. Fresh audit, generated parity, checkpoint overhead,
evaluation and final read-only verification consume the 21,600-second budget.
Conservative timing uses 1.25 times the maximum generated update time and
per-worker startup/checkpoint cost, plus 900 seconds for endpoint evaluation.
If the whole matrix does not fit, fitting never starts. No retry or limit increase.

Run on the owned GPU host under a bounded service so SSH loss cannot restart
or kill the controller prematurely. A host-local watchdog must enforce the
overall budget with at most 60 seconds for shutdown, killing only the owned
service cgroup. For example, with fresh absolute private config/output paths:

```text
systemd-run --user --collect --unit=FRESH_OWNED_UNIT \
  --property=MemoryMax=8G --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=21660 --property=TimeoutStopSec=5 \
  --property=KillMode=control-group --property=Restart=no \
  --setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --setenv=OMP_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=TMPDIR=FRESH_DISK_BACKED_TMP \
  --working-directory=REPOSITORY PRIVATE_VENV/bin/python \
  -m steganography.research_jpeg_refine --config PRIVATE_CONFIG --out FRESH_DIR
```

The controller includes a final host-local read-only verifier; completion
requires replay of all 56 receipts, numeric snapshots, branch ancestry,
initialization/exposure/schedule hashes and all six immutable endpoint evaluations.
For resumes, reduce the service watchdog to the declared remaining budget plus
60 seconds. An owned `CANCEL` marker interrupts only the current worker group;
the last verified snapshot survives. Uncheckpointed updates are never counted.
Integrity failure stops the entire matrix. OOM, signals and timeout remain failures.

Explicit resume uses a new private config and fresh output with:
`resume: {parent, report_sha256, lost_seconds, remaining_seconds}`.
The source/data/recipe/physical-device/library/runtime bindings must match,
the parent must be cancelled/unavailable, and remaining time must equal
21,600 minus cumulative spent and declared lost time. The verified round-robin
prefix is copied byte-for-byte, never retrained. Endpoint evaluation may be
replayed. Storage includes prior attempt trees and copied inherited snapshots;
the cumulative 10-GiB budget does not reset. Missing final parent evidence after
an external hard kill requires an explicit accounting repair before resuming.

## Measurements and decisions

Development evaluation occurs only after each shared warm-up and each final
model; no epoch selection or threshold optimization. Stored-BN singleton
inference averages five view probabilities and checks batch parity. Finals also
evaluate fit using the same method. Parameter and buffer hashes must match
before and after evaluation; no optimizer is created for evaluation.

Per source/seed: ROC-AUC, balanced accuracy, recall, FPR, ECE, confusion matrix,
and 1,000 original-pair bootstrap draws with 95% intervals. All other metric
intervals reuse the same pair draws. Report paired AUC changes for long paired
versus warm-up, unpaired versus warm-up, and unpaired versus long paired.
Fit/development gaps and every checkpoint loss/objective remain visible even
when the research rule fails. The optional 16-original normalization contrast
remains a separately invoked diagnosis, never a selection or single-file score.

An arm passes only if both sources and both seeds have AUC >=.70, AUC CI lower
bound >.50 and paired AUC-change CI lower bound >0 against their shared warm-up.
Choose the passing arm with the highest minimum of four AUCs; prefer paired
on ties. Claim unpaired superiority only if all four unpaired-minus-paired
intervals have positive lower bounds. Incomplete matrices cannot pass.
Negative results do not trigger a third training round.

Production qualification remains unavailable: the separate goal requires an
independent source, balanced accuracy >=.85, AUC >=.90, recall >=.80, deployed
FPR <=.03, ECE <=.05, >=1,000 covers and >=1,000 stegos per cell and two
independent source groups. This previously inspected development set cannot
support a real-world 85–90% claim or automatic promotion.

## Verification

Run schedule/multiset/zero-partner tests, generated historical arithmetic and
uninterrupted/save-resume state comparisons, malicious NPZ/provenance/limit
tests, CLI adapters and incomplete-matrix gates. Actual CUDA checks skip as
unavailable when no bounded owned GPU exists. Then run the unchanged full
coverage >=90% gate, new-code coverage >=95%, lint, mypy, identity and diff checks.
CI retains Python 3.11–3.14 and the fully provisioned CPU research job.
Use fresh disk-backed test/worker temporary directories; preserve failed logs.
Clean only owned generated test temporaries, never historical data/checkpoints.
