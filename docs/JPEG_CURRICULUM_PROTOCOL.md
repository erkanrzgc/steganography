# Frozen Q75 two-source curriculum — 2026-10-10

Freeze before preparation or fitting. The four-row learnability control passed;
the historical 50% probe failures remain unchanged. This new experiment asks
whether easier-payload initialization, paired D4 augmentation and substantially
more per-original exposure improve original-disjoint development detection.
These bundled changes do not isolate one cause; no held-out accuracy claim.

Fixed audited 480-row kit, unchanged manifest/audit identities. Use only first
24 sorted fit originals each from BOSS and BOWS, Q75/JUNIWARD: 48 originals,
96 rows. No ALASKA, Q95 or UERD fit. Generate .4 bpnzAC JUNIWARD from these same
Q75 covers using pinned conseal 2025.11 and fixed lineage seeds. Never filter
weak/no-signal examples or regenerate by prediction: fail the job instead.
Coefficients must change by at most one and tables stay unchanged; verify every new
cover/stego unrounded crop against independent SciPy IDCT math. This is not
an independent native parser or a natural in-the-wild stego corpus.
The pinned upstream JUNIWARD simulator may change DC coefficients: bpnzAC is
the payload denominator, not a guarantee of AC-only embedding. Do not impose
an invented DC exclusion or silently change the upstream simulation recipe.

Separately tagged SAME-pool/BN-eps model, fresh seed 20261011. Warm-up 40 epochs
at .4, Adamax lr .001; then carry all model weights/statistics into 40 .2 epochs,
fresh Adamax optimizer lr .0001. Other optimizer settings unchanged (weight
decay .0001, eps 1e-8, betas .9/.999, foreach false). Four complete exposures
per original each epoch, shuffle by seed+epoch, two pairs/four rows per update:
96 updates/epoch, 7,680 total. Each four-row batch uses the same deterministic
D4 transform for covers and stegos. No resizing/arbitrary rotations, no crop
or threshold tuning, no early stopping/best epoch. Report every epoch loss.

One final ordinary singleton evaluation on all 120 historical train-role probe
rows, 24 original-disjoint groups: all ten source/quality/method cells, including
out-of-fit Q95/UERD/ALASKA contexts. The probe has been inspected previously;
it is NOT blind validation. Preserve threshold zero and report raw logits,
confusion, ROC-AUC/FPR/recall/BA/loss; compare with published failed development controls
descriptively, not as a matched causal trial or deployment gate. Low support
(eight cover/eight stego per cell) prevents support qualification. No success
claim on a cherry-picked cell, and no cross-camera independence claim.

Own RTX 5060 only, strict deterministic FP32, no AMP/TF32/fallback, host 8 GiB,
GPU allocator 4 GiB, two CPU threads. Prep inner 600s/outer630s; training inner
2400s/outer2430s, CPU4800s. Bounded nonsymlink/exclusive outputs; cache<=24 MiB,
JPEG worker90s/2 GiB/256 KiB. Source and both cache identities verified before
and after. Report only, models discarded; no deployment/weights/cloud spending
or new downloads. Primary verdicts and legacy source/checkpoint contracts stay
unchanged. See `JPEG_RESEARCH_FINDINGS.md` for primary-source motivation.
