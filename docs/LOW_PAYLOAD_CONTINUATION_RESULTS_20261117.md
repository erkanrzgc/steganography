# Low-payload continuation: WAV diagnosis and CC-JRM (2026-11-17)

**Update 2026-11-18: the approved ESC-50 resume completed and passed the full
WAV audit. All six ESC cells pass on the historical (adaptive) test:
replacement 5/10% 95.27/96.77%, matching 5/10/20/40% 82.59/83.83/87.31/88.68%.**
See [Approved ESC-50 test resume](#approved-esc-50-test-resume-2026-11-18).
AudioMNIST matching 5/10%, all eight FSDD cells and nsF5 .05 remain misses.

The original 2026-11-17 closeout text follows unchanged below.

**JPEG closed at its calibration gate; WAV stopped during test-feature
construction. No new test accuracy is available.** The continuation recovered
interrupted private work rather than rerunning closed studies or changing
previously reported test results. Both supervised attempts now have resource
closeouts. JPEG passed its full numeric audit; WAV passed a calibration-only
audit, not a completed test audit. The
[numeric receipt](../benchmarks/low-payload-continuation-20261117.json) records
all 53 candidate calibration records, selections, exclusions and resources.

## Recovered work and boundaries

- The prior private nsF5 CC-JRM log reaches 1,000 training rows, but has no
  reusable checkpoint binding or elapsed-time closeout. It remains preserved;
  no cumulative runtime is invented.
- The subsequently frozen, chunked JPEG attempt retained seven verified
  checkpoints: **224 of 3,000 training/calibration source pairs**. Its recovery
  reuses these checkpoints, verifies their source/code bindings and does not
  repeat preparation or change extractors, candidates, thresholds or seeds.
  Recovery refuses an already started partial model phase instead of refitting.
- The WAV continuation had generated tests but no runner or launched attempt.
  Its missing private runner now implements the existing plan: 16 compact
  features, three fixed candidates for each of the sixteen failing conditions.
- No application source changed. `core/` and `steganography/` remain frozen.
  No downloads, GPU, model deployment, locked ALASKA2 data, data deletion or
  modification of prior evidence. All models/features remain private.
- Preserve `red-proof.json`: a generated regression check caught the inherited
  CC-JRM runner opening test workers at calibration BA .79. No native media
  was decoded by that check. The separate corrected runner gates test workers
  and images; the original `.benchmark/nsf5-ccjrm-20261117/run.py` remains intact.
- Preserve `jpeg-precheck-failure.json` and the subsequent passing precheck:
  the inherited 1,152,120,502-byte input archive exceeded the new 512-MiB input
  check before training/native decoding. The explicit legacy-input bound is
  2 GiB with a frozen digest and 256-KiB expanded change-member limit; new-output
  and per-file limits did not change. This is an input-contract repair, not an
  accuracy gain or a reset of the interrupted attempt's unknown runtime.

## WAV detectability: measured fit/calibration diagnostics

Cover statistics use a fixed sample of 200 fit/calibration clips per dataset.
Weighted-stego and spectral-floor probes use 240 fit clips per dataset. These
are exploratory diagnostics, **not held-out balanced accuracy**, and do not
open any historical test decoder. FSDD keeps the existing matched-depth
overlay: fit speakers george/jackson, calibration lucas; nicolas is excluded.

| Cover statistic, median | ESC-50 | AudioMNIST | FSDD |
|---|---:|---:|---:|
| PCM sample count | 220,500 | 29,017.5 | 3,997 |
| Sample rate, Hz | 44,100 | 48,000 | 8,000 |
| Local three-sample prediction residual RMS, PCM units | 144.570 | 4.841 | 597.782 |
| Fraction of 16-sample windows with residual power <4 | 0 | .53593 | 0 |
| Sampled covers with unused low bits | 0/200 | 0/200 | 0/200 |
| Spectral floor, fifth-percentile power, separate 240-clip probe | .17449 | 1.92285 | 643.30534 |

The spectral-floor normalization makes a white-noise level comparable to
per-sample variance. It is not the same statistic as local prediction RMS;
their dataset ordering need not agree.

| Fit-only probe AUC | ESC-50 | AudioMNIST | FSDD |
|---|---:|---:|---:|
| Weighted-stego, replacement 5% | .6260 | .8922 | .5049 |
| Weighted-stego, replacement 10% | .6949 | .9862 | .5020 |
| Weighted-stego, replacement 20% | .7363 | 1.0000 | .5076 |
| Weighted-stego, replacement 40% | .8107 | 1.0000 | .4912 |
| Spectral floor, matching 5/10% | .6059/.6292 | .5390/.5649 | .5000/.5005 |
| Spectral floor, matching 20/40% | .6560/.6698 | .6140/.6749 | .4999/.5002 |

At 5% embedding approximately 2.5% of samples change by one PCM unit, giving
about .025 per-sample squared perturbation (.158 RMS). This is a scale
comparison, not an assertion that replacement perturbations are independent
white noise. FSDD has both very short clips and much larger natural residuals;
its measured floor is over 25,000 times .025. Its weighted-stego cover
interquartile range is **[-4.70267, 5.08984]**, far wider than a .05 payload
shift. These two simple probes remain at chance even at higher payloads.

AudioMNIST's replacement estimator has a much narrower cover range
**[-.01876, .01454]** and a .05-stego median of .04888. Symmetric matching does
not have replacement's same signed parity bias; low-rate matching must rely
on weaker residual/distribution changes. ESC-50 mixes low-floor and noisy
recordings: 52.5% of the fit probe has floor below .2, but its upper floor
quartile is 134.686. A pooled scalar floor alone does not separate that mix.

The observations support **estimator variance, clip length, cover heterogeneity
and feature-family mismatch** as explanations for the plateau. They do not
prove an information-theoretic impossibility or attribute a classifier gain
to any single cause. No payload is increased to turn a miss into a pass.

## Fixed comparison outcomes

### WAV

The compact descriptor uses symmetric per-clip LPC residuals (orders 2/8),
weighted-stego estimates, residual energy, spectral-floor summaries and small
sample differences. Existing mono-first-channel PCM16 clips remain unchanged.
The plan's Fisher ridge .03/.3 and 300-iteration, 15-leaf gradient-boosted tree
candidates fit only on existing fit roles. Thresholds and candidate selection
use calibration BA, then AUC, then declared order. Original embedding seeds
remain unchanged. No additional fit-only CV search is launched.

All sixteen choices were globally frozen before eligible test features were
built. The table reports **calibration BA, not held-out test BA**. Each row used
three candidates; the numeric receipt retains all candidates, including losses.

| Dataset | Method | Payload | Selected candidate | Calibration BA, % | Test status |
|---|---|---:|---|---:|---|
| ESC-50 | replacement | 5% | hgb-300-15 | 95.23 | interrupted; no scores |
| ESC-50 | replacement | 10% | hgb-300-15 | 96.86 | interrupted; no scores |
| ESC-50 | matching | 5% | hgb-300-15 | 83.42 | interrupted; no scores |
| ESC-50 | matching | 10% | hgb-300-15 | 84.30 | interrupted; no scores |
| ESC-50 | matching | 20% | hgb-300-15 | 85.80 | interrupted; no scores |
| ESC-50 | matching | 40% | hgb-300-15 | 88.07 | interrupted; no scores |
| AudioMNIST | matching | 5% | hgb-300-15 | 64.00 | below calibration gate |
| AudioMNIST | matching | 10% | hgb-300-15 | 73.00 | below calibration gate |
| FSDD | replacement | 5% | fisher-0.03 | 52.60 | below calibration gate |
| FSDD | replacement | 10% | hgb-300-15 | 52.30 | below calibration gate |
| FSDD | replacement | 20% | hgb-300-15 | 56.70 | below calibration gate |
| FSDD | replacement | 40% | hgb-300-15 | 64.00 | below calibration gate |
| FSDD | matching | 5% | hgb-300-15 | 51.40 | below calibration gate |
| FSDD | matching | 10% | hgb-300-15 | 52.20 | below calibration gate |
| FSDD | matching | 20% | hgb-300-15 | 51.40 | below calibration gate |
| FSDD | matching | 40% | fisher-0.3 | 53.00 | below calibration gate |

The supervisor stopped WAV at **1,800.507 seconds**, with `wall_cap` and exit
code -9. All fit/calibration checkpoints and 48 candidate records exist. Only
one ESC-50 test checkpoint receipt exists: 32 of 402 source rows, with all six
eligible conditions. There is no completed test binding, test-score file or
per-cell result receipt. This closeout inspected that partial checkpoint's JSON
metadata only; it did not verify or load its binary features. AudioMNIST and
FSDD have no test checkpoints. Their ten selected conditions are below the .80
gate; these exclusions are reported here without manufacturing missing runner
result files or relabeling the stopped attempt as completed.

The ESC calibration results are promising, but they do not establish that any
previous test miss is fixed. Native historical ESC test decoding had started in
the stopped attempt; the metadata flag in `choice-freeze.json` describes the
earlier freeze moment, not the entire attempt. This closeout performed no new
historical test decoding or scoring. Any approved completion must retain the
historical/adaptive label and the already frozen models, thresholds and seeds.

### JPEG

nsF5 remains at **.05 bpnzac**, using the original 2,400 fit/600 calibration
sources and embedding seeds. Installed sealwatch 2024.12 Cartesian calibration
adds 11,255 JRM features to the prior corrected DCTR/full-JRM cache. Fixed
51-learner candidates are CC-JRM subspaces 1,000/2,000 and joint DCTR+CC-JRM
subspaces 1,000/2,000/3,000. The model seed remains 2026111701. No additional
candidate or payload was introduced by recovery. All 3,000 source pairs were
completed in 94 checkpoints, reusing the original seven (224 pairs).

| Candidate | Calibration BA, % | Calibration AUC | Threshold |
|---|---:|---:|---:|
| ccjrm-1000 | 64.33 | .683047 | .137255 |
| ccjrm-2000 | 64.08 | .673772 | .098039 |
| all-1000 | 66.17 | .705617 | .137255 |
| all-2000 | 65.83 | .701063 | .098039 |
| all-3000 | 63.42 | .684664 | -.019608 |

The frozen winner is `all-1000`, with exact calibration BA
**.6616666666666666** (419/600 covers and 375/600 stegos correct). It remains
below .80, so the result is `not_evaluated_calibration_gate` with
`new_test_inference: false`. No JPEG test features or scores were constructed.
This is a negative CC-JRM comparison, not a new held-out accuracy estimate.

## Verification and resources

- JPEG's prepared `audit.py jpeg` passed: all five calibration records,
  selection and threshold/count/AUC replay, 30 direct scalar predictions with
  zero error, and two native calibration descriptor replays. Source bindings
  and absence of gated test features/scores were verified; audit time was
  17.960 seconds.
- A separate **WAV calibration-only** replay reused the same auditor helpers
  without changing hash-frozen code or bypassing the completed-lane check in
  `audit.py wav`. It verified all 48 candidate records, sixteen frozen choices,
  288 direct predictions (maximum error 4.44e-16) and three native calibration
  descriptors. It took 6.599 seconds, with 282,759,168 bytes peak self RSS.
  This is not a full WAV closeout audit. No test descriptors or test decoder
  were opened by this replay.
- Fresh checks: **38 passed** (15 cloud-removal/document checks and 23 private
  runner/recovery/audit checks, including nine audit-oracle cases). Repository
  Ruff and the focused private audit/audit-test/WAV-runner Ruff pass. An
  expanded strict private-script Ruff check exposes 13 existing style findings:
  two E402 delayed imports, one E501 line length and ten S101 pytest assertions.
  Frozen scripts were not rewritten for style; those findings remain visible.
  Public receipt replay and `git diff --check` pass. No full application gate
  was rerun because application/package/public test sources did not change.

| Measured resource | JPEG recovery | WAV stopped attempt |
|---|---:|---:|
| Supervisor status | completed, exit 0 | stopped, wall_cap, exit -9 |
| Elapsed seconds | 2174.9200611059996 | 1800.506770371 |
| Peak summed sampled RSS, bytes | 3319660544 | 717574144 |
| RSS observations | 4316 | 3569 |
| Output bytes | 225774640, including interrupted chunks | 6256691 |
| Largest output file, bytes | 2425056 | 784479 |
| Wall cap, seconds | 5400 | 1800 |
| Summed sampled RSS cap, bytes | 8589934592 | 2147483648 |
| New-output cap, bytes | 536870912 | 67108864 |
| Per-file cap, bytes | 8388608 | 8388608 |

Sampling is an observer/stop mechanism, not a cgroup memory guarantee. The
interrupted JPEG attempt and the older uncheckpointed attempt have no elapsed
closeout; their runtime and cumulative compute remain unknown. Output accounting
includes retained interrupted JPEG chunks; it is not recovery-only write volume.
No retry, training rerun, download, deployment or GPU work occurred in this
closeout. Existing failure evidence and the unrelated editor save remain intact.

## Remaining work

JPEG needs no restart: the frozen comparison failed its gate and is closed.
WAV remains **stopped, not running**. Completing ESC-50 test evaluation needs
explicit approval for a separately bounded resume; merely rerunning the current
runner would collide with exclusive checkpoint publication and could repeat
fitting. An approved resume must first verify the partial binary checkpoint,
reuse all frozen calibration models/choices, preserve the original stopped
receipt, process only missing eligible test rows, and run the full test audit
after a successful new closeout. No extra candidate, seed or payload is justified
by these results. AudioMNIST/FSDD exclusions and the unresolved JPEG miss remain.

**No new test accuracy or 80%-per-method success is claimed.**

## Documentation-only session handoff — 2026-11-17

This final closeout inspected existing Git state, source files, numeric
receipts and checkpoint metadata. It did not run fitting, feature extraction,
native media decoding, new experiments, full application QA or deployment.
The 38 targeted study passes above and the 2026-11-16 full gate (4,651 passed,
two CUDA skips, 97.42% coverage) are prior recorded verification, not new
results from this documentation step. No local study-script PID was found.

### Working tree and saved context

- Branch `main`, HEAD `809d9ed`, two commits ahead of the locally cached
  `origin/main`; no fetch or remote status verification was performed. Those
  existing commits condense `AGENTS.md` and archive the old handoff verbatim
  in `docs/HANDOFF_HISTORY.md`. No commit or push occurs in this closeout.
- Before these edits, `AGENTS.md` already had continuation notes, this results
  document and `benchmarks/low-payload-continuation-20261117.json` were untracked,
  and unrelated `nano.383070.save` was untracked. All are retained; the numeric
  receipt and editor save are unchanged. No application/research/public-test
  source changed. Private study files and historical documentation are retained.
- Persistent entry points now agree: `AGENTS.md`, [status](STATUS.md),
  [roadmap](ROADMAP.md), [method/payload progress](DETECTION_EXPANSION_20261111.md),
  [documentation index](README.md) and [changelog](../CHANGELOG.md). Earlier
  status/roadmap entries are explicitly historical, not live launch instructions.

### Evidence anchors for the next session

All following paths are relative to
`.benchmark/low-payload-continuation-20261117/` and remain private:

| Scope | Required retained evidence |
|---|---|
| JPEG closed comparison | `jpeg/freeze.json`, `jpeg/train-binding.json`, `jpeg/selection.json`, `jpeg/result.json`, `jpeg/independent-audit.json`, `jpeg/recovery-closing.json`; all 94 checkpoint parts and earlier logs/attempt receipts |
| WAV frozen choices | `wav/freeze.json`, `wav/choice-freeze.json`, dataset `selection.json` files, model/calibration exports and fit/calibration checkpoint bindings |
| WAV stopped boundary | `wav/closing.json`, `wav/attempt.json`, `wav/run.log`, `wav/partial-calibration-audit.json`, `wav/esc50/test-part-0000.json` and its `.npz` |
| Preparatory failures | `red-proof.json`, `jpeg-precheck-failure.json` and `jpeg-precheck.json`; original inherited runner and interrupted JPEG evidence remain separate |

Read receipts and verify code/source bindings first. Hash-preservation checks
are not numeric/descriptor validation of the partial test binary. Do not invoke
the fresh-only WAV runner or full WAV auditor as a substitute for an approved
resume: the current lane is incomplete. No resume implementation or added
compute allowance is authorized by this documentation request.

### Final documentation verification

- Fresh `tests/test_cloud_removal.py`: **15 passed in 0.47 s**, with
  `venv/bin/python -m pytest -q -p no:cacheprovider --no-cov --basetemp
  <fresh disk-backed scratch> tests/test_cloud_removal.py`. `TMPDIR` used that
  separate scratch and bytecode writes were disabled; prior coverage evidence
  was not replaced by a partial run.
- `venv/bin/ruff check . --no-cache` and `git diff --check` pass. No frozen
  private scripts were restyled and no expanded private lint pass is implied.
- Read-only JSON comparison checks all **53 candidate calibration records**,
  frozen selections/counts, four audit/resource receipts, 94 JPEG checkpoint
  receipts and the single 32-row ESC test checkpoint's metadata against the
  public receipt. All **168 local links** in the seven edited documents resolve.
  This uses no model/feature load, descriptor computation or native decoding;
  it does not replace the earlier numeric audits or verify the partial binary.
- A pre-edit digest inventory covers **1,938 existing file/symlink entries**,
  including **642 private continuation entries**. All remain present; only the
  seven intended Markdown documents changed. The public numeric receipt,
  editor save, application sources, private study files and archived handoff
  remain unchanged. Original status/roadmap/changelog bodies and all earlier
  numeric results in this document remain intact.
- No new experiment, development task, full-suite pass, commit or push was
  performed. WAV remains stopped and approval-gated; JPEG remains closed.

## Approved ESC-50 test resume (2026-11-18)

The user approved the bounded resume. **Cause of the original stop:** for each
test row, the frozen runner re-verified all 6,183 bound source digests about
twelve times (`require_choice` → `freeze_hash`). A training chunk took about 25
seconds, a test chunk about 5 minutes. The new private `wav/resume.py`
verifies the same bindings once before and once after the run. It leaves the
frozen runner, models, thresholds, embedding seeds and fit/calibration
checkpoints unchanged and refuses to repeat any published evaluation.

- The partial 32-row checkpoint was **recomputed and matched exactly** (all
  features and change rates) before reuse; only the 370 missing test rows were added.
- The supervisor closed `completed`, exit 0, after **160.53 seconds** under the
  same 1,800-second cap (peak sampled RSS 693,637,120 bytes; 297,074 new bytes).
  The original stopped receipt is preserved; the cumulative WAV time is 1,961.04 s.
- The unchanged `audit.py wav` passed through a one-read adapter (it reads
  `resume-closing.json` instead of the preserved stopped `closing.json`):
  48 calibration records, six test cells, 324 direct predictions (maximum error
  4.44e-16), three native calibration descriptor replays and all bindings.
- Fit, calibration and test share no ESC recording group (402 test clips,
  310 groups). The ten gated conditions are written as `not_evaluated`.

| ESC-50 condition | Test BA, % | AUC | Paired-group 95% interval, % | 80%? |
|---|---:|---:|---|---|
| replacement 5% | 95.27 | .9932 | 93.5–96.9 | yes |
| replacement 10% | 96.77 | .9968 | 95.5–97.9 | yes |
| matching 5% | 82.59 | .9355 | 79.7–85.4 | yes (interval crosses 80) |
| matching 10% | 83.83 | .9451 | 80.8–86.7 | yes |
| matching 20% | 87.31 | .9642 | 84.7–90.0 | yes |
| matching 40% | 88.68 | .9704 | 85.9–91.3 | yes |

Previous ESC test results on the same 402 clips were 73.13/78.36%
(replacement 5/10%) and 65.17–69.40% (matching). The historical test had been
used before, so these are **adaptive** estimates, although choices and thresholds were
frozen on calibration only before any test row was built. AudioMNIST matching
5/10% and FSDD stay excluded at the calibration gate.

## nsF5 .05 learning curve (2026-11-18)

Before extracting rich features for more BOSS sources, a fit/calibration-only
learning curve tested whether training size limits nsF5 .05. Subsets of the
2,400 fit pairs (three subset seeds below 2,400) train the frozen recipe
(51-learner FLD ensemble, d_sub 1000, model seed unchanged) on the cached
features; the 600 calibration pairs score them. No test image or feature was
opened; 121 seconds. Private evidence: `.benchmark/nsf5-learning-curve-20261118/`.

| Fit pairs | all-1000 calibration BA, % | AUC | ccjrm-1000 calibration BA, % | AUC |
|---:|---:|---:|---:|---:|
| 300 | 60.92 | .644 | 59.61 | .625 |
| 600 | 60.08 | .638 | 59.03 | .620 |
| 1,200 | 64.56 | .692 | 62.19 | .668 |
| 2,400 | 66.17 | .706 | 64.33 | .683 |

The 2,400-pair row reproduces the frozen `all-1000` calibration result
exactly. Doubling from 1,200 to 2,400 pairs adds only 1.6–2.1 BA points. At
that rate, the ~7,500 deduplicated BOSS fit sources would give roughly 69%,
and 80% would need hundreds of thousands of pairs. The rich-feature
extraction for more BOSS sources was therefore **not launched**. nsF5 .05
stays a documented miss. This is an extrapolation from calibration data, not
a proof of a ceiling.
