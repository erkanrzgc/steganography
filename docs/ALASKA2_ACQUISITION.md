# ALASKA2 evaluation subset — 2026-10-04

This is corpus acquisition, not a detector result, training run or passed support
gate. The earlier BOSSbase and Kodak results remain frozen and unchanged.

Acquisition completed and an independent audit re-read all 4,000 files, checked
their sizes, CRC32 and SHA-256, decoded every JPEG and verified the four-way
lineage membership. The portable summary is
`benchmarks/alaska2-acquisition-20261004.json`.

## Selection fixed before scoring

The authorized competition archive contains 305,001 members and 75,000 complete
cover/JMiPOD/JUNIWARD/UERD groups. Sort the complete groups by source basename,
sample 1,000 with `random.Random(20261004).sample`, then retain all four files
for each selected original. Do not select by detector score or recovery success.
The unlabeled `Test/` files and submission CSV are excluded.

- Selected files: 4,000, comprising 1,000 covers and 1,000 per embedding method.
- Advertised uncompressed selected size: 398,254,243 bytes (about 380 MiB).
- Full remote ZIP size: 32,203,919,104 bytes; it is **not** downloaded in full.
- Selection SHA-256:
  `a83b1924e5cadeb64ec169f87ba80cebb4592f7cf28cfcd675f4c369e8998244`.

The local selection document is
`.benchmark/alaska2-holdout-20261004/selection.json`. It was written before
downloading the selected JPEG bodies. All selected samples are reserved for
evaluation, including failures. They must not become training, threshold-tuning
or calibration data. Camera/device/scene independence is unknown; the three
method folders do **not** constitute three independent sources.

## Explicit, bounded acquisition

Use the provisioned environment and a privately stored Kaggle credential for an
account with authorized competition access. The script neither accepts terms
nor downloads/creates credentials. Do not paste secrets into chat or commit them.

```sh
venv/bin/python scripts/fetch-alaska2-pilot.py \
  --out .benchmark/alaska2-holdout-20261004 \
  --count 1000 \
  --reserved-manifest .benchmark/boss-pilot-1000/source.json \
  --reserved-manifest .benchmark/pilot-v1/manifest.json \
  --reserved-manifest .benchmark/kodak-20261001/source.json \
  --reserved-manifest .benchmark/kodak-pilot-20261001-serial/manifest.json
```

For an incomplete attempt with an existing `selection.json`, repeat the same
command with `--resume`. Completed outputs and mismatched provenance are refused;
corrupt or symlink files are never overwritten. An attempt that failed before
writing its selection is not resumable; use a fresh output directory.

The standalone research script does not add dependencies to the base wheel:

- Read the bounded ZIP/ZIP64 directory and only selected local-member ranges.
  Allow only the fixed Kaggle authentication endpoint and its HTTPS Google
  Storage download origin; never forward the API credential to storage.
- Require HTTP 206, exact Content-Range/Content-Length and matching ETag for
  every range. Do not fall back to fetching the complete archive.
- Limit each invocation to 1 GiB requested range bytes, including retries,
  1 GiB selected output, 64 MiB per metadata read, 310,000 directory members,
  1,000 groups, 2 MiB per compressed/expanded member and 4 million decoded pixels.
- Use four workers, at most three attempts per range, at most 12,100 range
  requests and a cooperative 1,800-second deadline with socket timeouts at most
  30 seconds. These are acquisition limits, not CTF service limits or a hard
  operating-system sandbox. A partial attempt's network counter is per invocation.
- Check local/central headers, bounded DEFLATE completion and ZIP CRC32; decode
  JPEGs, check matched dimensions and reject exact overlap with reserved hashes.
  Keep all variants grouped by the original cover's SHA-256. Record unavailable
  camera, device, app, payload rate and quality-factor metadata as null.
- Re-read written files against their locally computed SHA-256 before publishing
  `source.json`. No source-supplied per-file SHA-256 was available, so this is
  local integrity plus upstream ZIP CRC checking, not independent authentication
  of the dataset's labels or embedding process.

Signed storage URLs, credentials and absolute local paths are excluded from
selection/source reports and CLI errors. Downloaded content is never executed.
The corpus and manifests remain in ignored `.benchmark/`; source images are not
redistributed. Kaggle marks the data as subject to competition rules. Successful
access does not establish unrestricted dataset or derived-model licensing.

## Acquisition-stage limitations

There are 1,000 distinct original-cover hashes and 3,997 distinct hashes across
all 4,000 files. Three UERD files (`36603.jpg`, `47895.jpg`, `59799.jpg`) are
byte-for-byte identical to their respective covers. Separate, authenticated
single-file downloads of all six files match the range-acquired files; this
is an upstream data property, not a local extraction mix-up. Original ZIP CRCs
also agree. Do not silently drop or replace these samples, relabel them, or
claim their source labels prove an actual hidden payload.

Keep the fixed selection and source labels, flag the three ambiguous pairs,
and declare any sensitivity analysis before scoring. A content-only detector
cannot assign different predictions to byte-identical inputs. This small label
ambiguity does not explain or excuse the earlier chance-level detection results.

The completed source manifest SHA-256 is
`a75c1237d9d67194374062e62baba69b9c84e9283b2e2680cceaf04869bc6844`.
The run requested 440,192,893 range bytes across 4,007 requests; the six additional
single-file checks read 107,614 image bytes. No whole archive was downloaded.
There was no exact hash overlap with the four reserved source/pilot manifests.

Before scoring, freeze the detector revision, preprocessing, threshold and
unavailable-coverage treatment. Publish per-method failures as well as successes.
Do not claim a supported JPEG cell, calibration quality or blind CTF recovery
from having downloaded this single-source corpus. Separate development data and
a second sufficiently sized independent source remain necessary.

Subsequent evaluation is now complete under the pre-scoring
`ALASKA2_PROTOCOL.md`. See `ALASKA2_RESULTS.md`: all files completed, but all
three methods failed the detection targets. Acquisition evidence above remains
unchanged and must not be confused with an accuracy improvement.
