# Complete matched JPEG controls — 2026-10-14

**Execution complete; learning failed; production qualification unavailable.**
All four arms and both seeds completed their full 32,768 image-view exposures. The own-host
read-only verifier checked all 16 numeric snapshots, matching initialization /
schedules, source closures and metric/bootstrap replay. No candidate passes
both source groups and both seeds; no model was promoted or deployed.

The two first-seed legacy/reference controls were inherited from the verified
user-cancelled parent, not fitted again. Six other controls completed in this
attempt. Historical cancelled/limit-exceeded jobs retain their original status.
Both genuine-batch controls reproduce **every** saved development probability
and standalone metric of the earlier incomplete matrix exactly, under distinct
bindings. Execution repair did not create their algorithmic accuracy gain.

## Scope and fixed-threshold development results

Real native-gray BOSSbase-1.01 and BOWS2 cover originals; positives are
**controlled embedding simulations**, not upstream app-generated stegos or
an in-the-wild challenge corpus. Simulation contract is
`conseal-2025.11-juniward-bounded-entropy-refinement-v1`, Q75/JUNIWARD at .4 bpnzac;
512 fitting originals and 128 development originals (64 pairs per source).
Five 256-pixel views come from each native 512-pixel JPEG. Views are not new
independent originals. There is no reserved or unseen-source evaluation and
independent camera/processing groups are not established.

Batch 16 also has fewer optimizer updates at matched image exposure (2,048
versus 8,192); do not attribute between-arm outcomes to batch size alone.

Threshold is fixed at stego probability .5. These balanced development sets
have equal accuracy and balanced accuracy; **ROC-AUC is not accuracy**.
Full confusion matrices, FPR, ECE, original-pair bootstrap 95% intervals and
paired baseline changes are retained in the portable record, including failures.

| Arm | Seed | BOSS accuracy / AUC | BOWS accuracy / AUC |
| --- | --- | --- | --- |
| Legacy, 4 images | 20261012 | 53.91% / .547241 | 50.78% / .538818 |
| Reference Adamax, 4 images | 20261012 | 60.94% / .594482 | 51.56% / .536377 |
| Convolution L2, 4 images | 20261012 | 51.56% / .528564 | 50.00% / .518311 |
| Genuine batch, 16 images | 20261012 | 67.97% / .770508 | 63.28% / .679199 |
| Legacy, 4 images | 20261013 | 56.25% / .611816 | 55.47% / .565918 |
| Reference Adamax, 4 images | 20261013 | 57.81% / .611816 | 55.47% / .581055 |
| Convolution L2, 4 images | 20261013 | 50.78% / .546387 | 52.34% / .552002 |
| Genuine batch, 16 images | 20261013 | 57.81% / .613037 | 57.03% / .574951 |

The first genuine-batch BOSS cell alone passes the small research learning
rule. BOWS and the second seed fail, so **0/3 candidate arms pass the complete
research rule**. The .770508 AUC cannot be advertised as 77% detection accuracy
or stable context-aware performance. Production sample-size, held-out-source,
FPR/calibration and accuracy gates remain unchanged and unavailable.

## Read-only fit versus development

All eight final models were evaluated on the existing fit cache using exactly
the stored-statistic, singleton-parity/five-view development inference method.
No optimizer was created; before/after parameter and buffer digests match.
Each source's fit metrics use 256 originals /512 cover-stego rows, not a new
held-out evaluation. Threshold recommendations are disabled.

| Arm | Seed | BOSS fit / development AUC | BOWS fit / development AUC |
| --- | --- | --- | --- |
| Legacy | 20261012 | .542862 / .547241 | .534256 / .538818 |
| Reference Adamax | 20261012 | .560165 / .594482 | .529938 / .536377 |
| Convolution L2 | 20261012 | .519989 / .528564 | .511597 / .518311 |
| Genuine batch | 20261012 | .793625 / .770508 | .730225 / .679199 |
| Legacy | 20261013 | .614182 / .611816 | .552216 / .565918 |
| Reference Adamax | 20261013 | .639343 / .611816 | .590179 / .581055 |
| Convolution L2 | 20261013 | .547386 / .546387 | .538109 / .552002 |
| Genuine batch | 20261013 | .596191 / .613037 | .573349 / .574951 |

Fit-minus-development AUC ranges from -.034317 to +.051026. Small-batch
controls are generally weak on fit as well as development, rather than strongly
memorizing fit while only failing new originals. This weakens a pure-overfitting
explanation; it does not establish that data diversity or sample size is sufficient.
The large-batch second seed is weak on fit too. Normalization, optimization /
exposure, processing and seed sensitivity still require separate interventions.

## Controlled train-only normalization contrast

The separately fixed [diagnostic protocol](JPEG_NORMALIZATION_DIAGNOSTIC.md)
used the same 16 fitting originals for every model, one center view and genuine
4/16-image batches. Only BatchNorm computation mode changes; running-stat
tracking is disabled in paired-batch mode, all flags are restored and every
parameter/buffer digest matches the corresponding fit-diagnostic digest.
**This is a post-hoc mode diagnosis, not an accuracy or deployment test.**

| Arm | Seed | Stored-statistic cross entropy | Paired-batch cross entropy |
| --- | --- | --- | --- |
| Legacy | 20261012 | 6.621002 | .539682 |
| Reference Adamax | 20261012 | 3.377569 | .561066 |
| Convolution L2 | 20261012 | 8.896896 | .743635 |
| Genuine batch | 20261012 | .666002 | .668651 |
| Legacy | 20261013 | 2.327998 | .607497 |
| Reference Adamax | 20261013 | 1.239831 | .562538 |
| Convolution L2 | 20261013 | 3.558512 | .565597 |
| Genuine batch | 20261013 | .695247 | .668793 |

All six small-batch controls have substantial loss/margin sensitivity to this
mode change on the selected fit blocks. The two genuine-batch controls have
much smaller loss changes; importantly, the weak second-seed genuine model
still has only modest fit separation. Thus stored-statistic behavior explains
a measured mode contrast, **not all poor learning or a proven production fix**.
Known-cover batch statistics must never become single-file detector scores.
Center-view subset loss is not directly comparable to full five-view fit loss.

Prior [epoch diagnosis](JPEG_EPOCH_DIAGNOSTIC_RESULTS.md) and the failed
[population-BN control](JPEG_POPULATION_BN_RESULTS.md) remain distinct evidence.
Do not rerun the same failed population-statistics idea and assume a fix.
The next intervention needs a separately frozen matched protocol addressing
training/inference statistics and weak second-seed learning, not another blind
restart, development-threshold tuning or a promise that a larger dataset suffices.
See [evidence-led follow-up](JPEG_LEARNING_FOLLOWUP.md).

## Execution, safety and verification

Own RTX 5060 Laptop, Torch 2.14.0+cu130, deterministic FP32/IEEE, no TF32/AMP
or CPU fallback; 4 GiB allocator /8 GiB kernel RAM caps. Batch 8/16 capacity
passes; 32 remains allocator-OOM, not successful capacity. Each fitting half
retains its original 1,800-second CPU/wall limits. Parent cancelled evidence,
including discarded work, consumes 5,403.52 seconds. This attempt adds 9,404.69
seconds, for 14,808.21 cumulative controls seconds. Fit diagnosis adds 835.60
seconds and normalization contrast 23.07 seconds: **15,666.88 seconds total**,
below the unchanged 21,600-second envelope. No paid compute.

All fitting, observer and diagnostic services are stopped after completion;
snapshots are retained privately, not packaged or published. The completed
code slice passed 2,861 tests /one actual-CUDA skip, total coverage 96.48% and
new diagnostic coverage 100%; all eight CI jobs at `7d8e6dd` passed. The final
publication gate, including ten new evidence checks, passed **2,871 tests** /
one actual-CUDA skip, 96.48% coverage in 852.69 s, plus lint/type/identity/diff
checks. Its first attempt failed from a full 7.8 GiB `/tmp` tmpfs; that failed
log is preserved and the complete retry used fresh disk-backed pytest and
worker temporary directories without reducing coverage or limits. Subsequent
portable-record verification independently replays recorded metrics and checks
receipt/snapshot ancestry, identical state digests, selection and privacy.
This replay uses the existing metric implementation, not a new mathematical
oracle. The CPU test skip is not an actual-CUDA pytest success.
Post-diagnostic checks also rehash all 16 snapshots and the original parent:
unchanged, with 52 fitting /three controller files intact, all four services
successfully stopped and no CUDA compute process remaining.

## Portable evidence

- [All eight controls and 16 source-specific cells](../benchmarks/jpeg-owned-controls-results-20261014.json)
- [Full fixed-statistic fit/development comparison](../benchmarks/jpeg-fit-development-20261014.json)
- [All eight descriptive known-cover score audits](../benchmarks/jpeg-all-paired-diagnosis-20261014.json)
- [Train-only normalization-mode contrast and pre-execution boundary](../benchmarks/jpeg-normalization-contrast-20261014.json)

Native complete report SHA-256:
`09e8862627ea337c48d9591c16df128c9e638229d91dd356f769c61ba50e7883`.
Public records contain hashes and portable metadata, not host paths, raw
corpora, secrets, model weights or private SSH details.
