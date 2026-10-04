# Next real-image evaluation: access status — 2026-10-03

Existing BOSSbase/Kodak measurements remain frozen in `PILOT_RESULTS.md` and
`KODAK_RESULTS.md`. They use real photographs with controlled embeddings, not
unknown field evidence. The latest CTF reruns do not improve the failed native
detection baseline. The subsequent ALASKA2 baseline is recorded below;
adequately sized cross-source qualification and blind CTF evaluation have not run.

## Read-only checks from this environment

| Source | Observation | What this establishes |
| --- | --- | --- |
| BOSSbase official ZIP | HEAD 200; advertised size 1,671,626,159 bytes | Archive endpoint responds; no new image download or license grant verified |
| ALASKA official site | GET 200 | Project information is reachable; not proof of dataset access |
| ALASKA2 legacy Kaggle data page | GET 200 | Public landing page is reachable |
| ALASKA2 Kaggle file-list API | Unauthenticated GET 401 | Dataset listing requires authorized access from this environment |
| StegoAppDB database endpoint | HEAD 404 | That endpoint was not accessible here; no claim that the dataset has ceased to exist |

Checked endpoints:

- [BOSSbase archive](https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip)
- [ALASKA project](https://alaska.utt.fr/)
- [ALASKA2 data page](https://www.kaggle.com/c/alaska2-image-steganalysis/data)
- [ALASKA2 file-list API](https://www.kaggle.com/api/v1/competitions/data/list/alaska2-image-steganalysis)
- [StegoAppDB database](https://data.csafe.iastate.edu/StegoDatabase/)

Only headers and bounded public metadata were requested. No corpus, model or
account credential was downloaded/read. The local source manifests still contain
1,000 BOSSbase pilot originals, a four-image BOSSbase preview and 24 Kodak
originals; the preview is not another independent source. Kaggle CLI is absent.
Approximately 92 GiB of free disk was observed; this is not a guarantee that a
complete download plus extracted corpus will fit.

## Authorized access follow-up — 2026-10-04

At the user's request, their locally downloaded Kaggle credential was moved to
the standard credential location outside the repository, with directory mode
0700 and file mode 0600. Credential contents were not printed or recorded.
Authenticated file listing returned HTTP 200. A single-file download permission
check returned HTTP 403: Kaggle requires the account holder to accept the
competition rules. No image was downloaded and no terms were accepted on the
user's behalf.

The requested initial acquisition is 1,000 original covers and the three matching
stego variants (4,000 files), not the entire 32.27 GB competition download.
Exact subset size and membership remain unverified. Keep this intended evaluation
subset separate from future training/development data; acquisition alone is not
a passed benchmark.

### Account and acquisition decision follow-up

The browser session was subsequently switched to a different, phone-verified
account. After navigating via the competition's `Late Submission` button, the
ALASKA2 rules page displayed `You have accepted the rules for this competition`.
The locally installed API credential still belongs to the earlier account, and
no replacement credential was found in the user's download directory. Browser
access does not authorize the terminal as the new account; acquisition remains
blocked on matching API credentials. No account was deleted and no image was
downloaded.

Select 1,000 complete cover lineages without inspecting detection scores, using
a fixed seed (20261004) over sorted, verified source file identities, then retain
all three matching JMiPOD/JUNIWARD/UERD variants. Exclude the unlabeled `Test/`
directory. Record exact membership, file sizes and hashes before evaluation;
keep these 4,000 files out of training and threshold/calibration selection.
This is one ALASKA2 source, not three independent sources. Exact membership and
total bytes are still pending an authorized listing/download workflow.

### Completed acquisition and independent integrity audit

The user supplied a replacement credential matching the verified account.
It was installed with private permissions outside the repository; the previous
credential was retained in a private backup. Authenticated listing returned
HTTP 200 and authorized archive requests returned a signed storage redirect.
No credential contents or signed storage URLs were printed or recorded.

The explicitly requested subset is now local in
`.benchmark/alaska2-holdout-20261004`: 1,000 covers and 1,000 images from each of
JMiPOD, JUNIWARD and UERD, totaling 398,254,243 image bytes. Only bounded ZIP64
metadata and member ranges were fetched, not the complete 32.2 GB archive.
All 4,000 files passed a separate size/CRC/SHA-256 audit and JPEG decode; all
are 512 by 512 pixels, with matching four-way lineages and no exact hash overlap
with the reserved BOSSbase/Kodak source and pilot manifests.

Three UERD samples are identical to their covers, also confirmed by separate
single-file downloads. They remain in the frozen selection and are explicitly
flagged, not silently removed. See `ALASKA2_ACQUISITION.md` and the portable
`benchmarks/alaska2-acquisition-20261004.json` for selection/provenance hashes,
counts, limitations and the quality exception. The earlier account/access
blockers above were resolved; at acquisition completion no ALASKA2 detector
evaluation had yet run.

### Completed baseline evaluation — 2026-10-04

The preregistered `ALASKA2_PROTOCOL.md` was committed before first scoring.
All 4,000 files completed, but native detection failed: JMiPOD 4/1,000,
JUNIWARD 1/1,000 and UERD 1/1,000 true positives at threshold 70, with two
false positives among the shared 1,000 covers. Independent recomputation
confirmed the metrics and paired bootstrap intervals. See `ALASKA2_RESULTS.md`.
There was no training, threshold tuning or model deployment. This subset is
now inspected and must remain excluded from development/validation.

### Authorized development and audio follow-up — 2026-10-04

The user explicitly authorized separate development acquisition and additional
useful sources. Another 1,000 ALASKA2 lineages (4,000 JPEGs, 421,459,421 bytes)
were acquired with reserved-basename/hash exclusion and a train/validation
partition. Six contradictory-label lineages were quarantined before training.
The first 3,156-train / 820-validation CPU JPEG experiment completed; its poor
validation scores and 41.46% FPR do not justify deployment. See
`JPEG_DEVELOPMENT_RESULTS.md`. The earlier native-test evidence remains frozen.

Pinned FSDD v1.0.10 was downloaded and independently audited: 3,000 unique
mono PCM16/8 kHz recordings from six speakers, 16,624,960 compressed bytes.
Its pinned README declares CC-BY-SA-4.0; the license/source/hash record is in
`DATASET_CATALOG.md`. These are candidate covers only, not a passed WAV test.
No original corpus or trained model was added to Git or uploaded elsewhere.

BOSSbase's archive still responded to HEAD with the same advertised size;
no additional spatial-development covers were downloaded. BOWS2 remained
unreachable here and the checked StegoAppDB endpoint returned 404. No mirror
was substituted, and no new terms were accepted on the user's behalf.

## Next required input and evaluation sequence

ALASKA2 test/development and FSDD cover acquisition are complete. New
development/training data still requires explicit user-authorized acquisition,
provenance and a leakage-safe partition.
Do not request passwords or API tokens in chat, accept terms on the user's
behalf, automatically fetch another corpus, or silently substitute an unverified
mirror. Review use/model-distribution conditions separately from URL access.

For subsequent development and a newly untouched evaluation:

1. Inventory size, complete cover/stego pairs, hashes, licenses and available
   camera/device/scene metadata. Exclude the frozen pilots and their derivatives.
2. Declare the selected methods, lineage grouping, holdout membership, detector
   revision, fixed threshold and unavailable-coverage policy before examining
   scores. Method folders are not independent source groups. Filename-based
   generic import alone does not establish correct ALASKA2 labels/lineage.
3. Run the shared analysis service and publish method-specific failures as well
   as successes. A new baseline can run before model training; measuring an
   improved trained detector requires separate development/validation data and
   a still-untouched final test source.

No date is promised for completing the detector accuracy gates. Do not represent
successful acquisition as a passed dataset benchmark. One experimental JPEG
training run completed, but it did not meet the quality gates. Audio and new
spatial development/evaluation remain work; supplying more data alone does not
establish better detection accuracy.
