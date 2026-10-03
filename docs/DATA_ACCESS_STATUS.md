# Next real-image evaluation: access status — 2026-10-03

Existing BOSSbase/Kodak measurements remain frozen in `PILOT_RESULTS.md` and
`KODAK_RESULTS.md`. They use real photographs with controlled embeddings, not
unknown field evidence. The latest CTF reruns do not improve the failed native
detection baseline. A new independent-source or blind benchmark has not run.

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

## Next required input and evaluation sequence

The user can download an authorized ALASKA2 subset through their own account and
provide the local directory, or provide another licensed cover/stego corpus with
method/ancestry metadata. Do not request passwords or API tokens in chat, accept
terms on the user's behalf, automatically fetch a corpus, or silently substitute
an unverified mirror. Review use/model-distribution conditions separately from
whether a URL responds.

Before running the next evaluation:

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

No completion date is established while access, size and metadata are unknown.
Do not represent this access check as a passed dataset benchmark. Training is
not complete, the JPEG training/preprocessing workflow is not yet validated,
and supplying more data alone does not establish better detection accuracy.
