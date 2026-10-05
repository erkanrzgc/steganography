# Separate spatial development covers — 2026-10-05

`scripts/fetch-boss-development.py` explicitly acquires up to 1,000 new
BOSSbase 1.01 PGM covers. It excludes reserved archive-member names before
fixed-seed selection, then rejects exact file/lineage hash overlap with all
provided manifests. No source or generated images enter Git.

The purpose is development only. The BOSSbase and Kodak published tests remain
reserved; their inspected outcomes must not choose models or thresholds.
Another BOSSbase subset is not another source. Unknown camera/device ancestry
remains null, and exact-hash exclusion cannot identify undocumented related
scenes or transforms. This acquisition alone supplies no stego examples or
improved detection score.

Selection uses Random(seed=20261005) over sorted eligible PGM member identities.
Before downloading image bodies, save member sizes/CRC, archive size/ETag and
all reserved-manifest hashes in `selection.json`. Generated local filenames
are `0000.pgm` etc.; archive paths are never extraction paths. Validate grayscale
512×512 PGM data through Pillow, SHA-256 and upstream CRC. Split entire original
lineages by first 64 SHA-256 bits / 2^64, below 0.8 train, otherwise validation.
Future embeddings inherit each original's split and lineage unchanged.

The script explicitly reuses the ALASKA2 standalone bounded ZIP/range helper,
with a separate validator allowing only the fixed official BOSSbase HTTPS URL.
It does not read credentials or call Kaggle. Redirects stay disabled; exact
Content-Range/ETag/size checks, retry/network/time/member/directory limits and
bounded decompression are inherited. The default ALASKA2 validator remains
unchanged. This keeps one tested range implementation instead of another parser.

Limits are 1,000 originals, 2 MiB/member, 512 MiB expanded selection, 1 GiB total
requested ranges, four workers and 1,800 seconds. Existing complete outputs
are refused. `--resume` requires the exact same selection/archive/reserved
provenance and verifies existing files; it never overwrites them. A success
manifest appears only after all members and duplicate-content checks complete.
Interrupted attempts and partial files remain available for inspection.

```sh
timeout --kill-after=10s 1800s python scripts/fetch-boss-development.py \
  --out .benchmark/boss-development-20261005 \
  --reserved-manifest .benchmark/boss-pilot-1000/source.json \
  --reserved-manifest .benchmark/boss-preview/source.json \
  --reserved-manifest .benchmark/pilot-v1/manifest.json \
  --reserved-manifest .benchmark/pilot-preview/manifest.json \
  --reserved-manifest .benchmark/kodak-20261001/source.json \
  --reserved-manifest .benchmark/kodak-pilot-20261001-serial/manifest.json \
  --reserved-manifest .benchmark/alaska2-holdout-20261004/source.json \
  --reserved-manifest .benchmark/alaska2-development-20261004/source.json
```

Source: [official Binghamton DDE download section](https://dde.binghamton.edu/download/).
No explicit redistribution license was verified on that page; data stays local
for the user's research. The repository publishes acquisition instructions,
source links, hashes and eventual measurements, not originals or derivatives.
