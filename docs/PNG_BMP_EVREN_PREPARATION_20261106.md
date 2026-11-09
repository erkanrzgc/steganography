# EVREN backend preparation — 2026-11-06

The user selects EVREN for future GPU work because local compute prevents
normal laptop use. The completed WSL cost profile remains historical evidence;
the local fit proposal is deferred. Do not schedule further local GPU work
unless the user explicitly changes that preference.

## Current observable state

The closing check finds both WSL profile/audit units inactive, PID zero, and
no CUDA compute process. The local controller is also stopped. Numeric states
and every consumed-attempt receipt remain private and preserved.

The existing EVREN browser session opens successfully. After normal session
initialization, navigating to `/workspace` redirects to `/forbidden` and shows
an explicit access-denied page. The account therefore still lacks the observed
custom-notebook entry point on this date. Earlier normal-training access and
reservation visibility do not establish access to custom Python/SRNet execution.
The reservation interface previously states it grants no SSH/direct server.
No restriction is bypassed and no application, upload, reservation or GPU job
is submitted by this preparation.

The user clarifies that AI Workspace currently belongs to the 29-person
early-access programme. This is a service-availability constraint. The
[unsent access-request draft](EVREN_WORKSPACE_ACCESS_REQUEST_DRAFT_20261106.md)
is deferred and is not the next step for enabling ordinary access. It invents
no institution, degree, location, identity or accuracy claim. Do not submit
an application or message without an explicit user instruction. The existing
normal-training registry lists built-in families and no observed SRNet option;
this does not establish a supported way to run our custom source code.

## Prepared CPU-only diagnostic

`notebooks/png_bmp_evren_readiness_20261106.ipynb` inventories Python, installed
package versions and bounded cgroup RAM/swap metadata. It creates no model,
optimizer, tensor, network request or GPU context. It does not install packages,
start a runtime or consume a GPU allocation. It can be used only after ordinary
notebook access is granted; a CPU runtime is sufficient. Its output intentionally
excludes account identifiers, tokens, environment dumps and absolute host paths.
This is a diagnostic notebook, not an executable training protocol.

## Dependency order after access

1. Inspect the permitted runtime/container and resource controls; run the CPU
   diagnostic. Confirm that custom source and dependencies are supported.
2. Bind a new environment and supervisor suited to the actual EVREN scheduler.
   The WSL SSH/systemd controller and its time/cost result do not transfer as
   EVREN execution authority. A new environment needs its own generated cost
   contract, resource evidence, interruption handling and credit tariff.
3. Make the source/license/known-ancestry package reviewable before external
   transfer. Preserve the 64/20/21 RGB proposal, all unassigned source records,
   the thirteen L sources, reserved media, holds and every historical ledger.
4. Only a separately frozen and explicitly approved native intervention may
   use real fitting inputs. The local 1,717-second projection is not an EVREN
   estimate, and the old unused WSL allowance is not a cloud budget.

There is no new fitting approval, data acquisition, model deployment or
qualification in this preparation. The prior 4,092-test software gate remains
historical for the unchanged application/test sources; this slice adds only
documents and a standalone diagnostic notebook.

## YOLO-classify augmentation check (2026-11-09)

EVREN's ordinary training exposes only built-in YOLO models with default
augmentations. A local CPU check (isolated venv: Ultralytics 8.4.208,
torchvision 0.29.2) applied the real `classify_augmentations` /
`classify_transforms` with cfg defaults (crop scale 0.5–1, hflip .5,
RandAugment, erasing .4) to the 64 train pairs only, six draws each:

| Input (64 train pairs) | No transform | Val, imgsz 256 | Val, 224 | Train aug, 256 | Train aug, 224 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw RGB, read by WS (ideal parity reader) | 1.000 | 1.000 | .815 | .623 | .603 |
| Parity-residual feature image, mean(ch0−ch1) | .980 | .980 | .980 | .824 | .761 |

Training augmentations destroy raw LSB parity, so raw images on EVREN repeat
the diagnosed failure. Precomputed antisymmetric parity-residual images keep
part of the signal (AUC proxy .76–.82), but such a model would only relearn
the Weighted Stego statistic, which scores 1.0 without training. EVREN YOLO
training is therefore not worth credits for this LSB-replacement task; no
upload or job was made. Script: private `evren-aug-check-20261109`.
