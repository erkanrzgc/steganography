# Preparing the next detector experiment

The published pilot remains frozen. Its images, covers and derivatives are not
development or validation data. This workflow prepares a new experiment; it
does not train a detector or establish an accuracy claim.

## Required inputs

Supply a verified research manifest (`schema_version: "1.0"`) for new, locally
available images. Each sample needs a relative path, SHA-256, size, cover/stego
label, explicit source group and cover lineage. Prefer the original cover's
SHA-256 as lineage, retained unchanged in every transformation/embedding.
Provide camera and device IDs where known; leave unknown values null rather
than inventing them. Review usage conditions before acquiring any new corpus.

Filename-based `research import` inference is only a starting point: review
and correct ancestry/source metadata before partitioning. Hashes cannot discover
undocumented transformations or perceptual duplicates. Metadata declarations
and independence still require human/source documentation.

## Command

Activate the provisioned Python environment first (`source venv/bin/activate`).
The following paths are placeholders except the existing frozen pilot manifest:

```sh
steganography research partition \
  --manifest new-corpus.json --source /path/to/new-corpus \
  --reserved-manifest .benchmark/pilot-v1/manifest.json \
  --test-source untouched-source \
  --out next-experiment.json
```

Repeat `--reserved-manifest` for other published/frozen corpora and
`--test-source` for additional untouched sources. No network access, training,
calibration or embedding occurs. Existing output files are never overwritten.

The command:

1. Verifies input file sizes/hashes and rejects symlinks.
2. Rejects any sample identity or original-cover hash overlapping reserved data.
3. Groups all connected lineages, cameras and devices within each source. A
   content-hash lineage is global, even if source/camera metadata changes.
4. Assigns entire selected sources to test; rejects related samples bridging
   the test/development source boundary. Remaining groups receive a deterministic
   80/20 train/validation assignment (seed 20260918), not a guaranteed count ratio.
5. Requires both labels in every split. Too few independent groups fail; the
   command does not silently split a camera or rebalance related images.
6. Records input/reserved-manifest hashes, grouping policy, counts and metadata
   completeness. Subsequent research manifest verification checks the recorded
   held-out-source and camera/device separation policy again.

Source names, not source paths, identify a source group. Human-readable lineage
IDs are scoped to their source; only SHA-256 lineage IDs link across renamed
sources. Renaming arbitrary lineage IDs can defeat ancestry checks and must
never be used to make data appear independent.

## Remaining work before a new accuracy measurement

- Acquire/document a second source; none has yet been added by this change.
- Extract versioned features with per-row sample/lineage provenance and bind
  training inputs to the verified split manifest. The existing raw-NPZ training
  command does not enforce that binding yet; partitioning alone cannot make an
  arbitrary training run leakage-free.
- Train on train only; choose thresholds/calibration on validation only.
- Freeze preprocessing/model/threshold before accessing the new test source.
- Publish failures and per-cell results using the release benchmark protocol.

Cloud AI cannot alter the analysis service's aggregate score or the pipeline's
primary findings/verdict. Its original output remains available as triage only.
The pilot had AI disabled, so this isolation fix does not improve or invalidate
its failed detector baseline.
