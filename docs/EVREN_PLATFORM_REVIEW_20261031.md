# EVREN suitability review — 2026-10-31

EVREN is a plausible additional research-compute option for this project's
custom SRNet, conditional on actual account runtime access and environment
verification. This is a public-information review, not a deployment or job.

## Verified public descriptions

The platform's [infrastructure page](https://evren.ssyz.org.tr/infrastructure)
identifies H200 SXM GPUs with 141 GB HBM3e per GPU. Its current public metadata
advertises eight nodes with eight GPUs each. The older
[operator announcement](https://sayzek.org.tr/duyurular/savunma-sanayiinin-yapay-zeka-evreni-faaliyete-gecti)
from June lists forty-eight GPUs; these descriptions do not establish a
particular user's allocation or present availability. Training is described
as Slurm jobs using Apptainer containers, with a contribution-credit economy.

The public notebook client implements editable Python notebooks and a runtime
picker offering CPU or one to eight H200 GPUs. It explicitly handles code
execution being disabled by an administrator. This is stronger evidence for
custom-code support than a catalogue of YOLO/ResNet training templates, but
an authenticated account's actual runtime options remain unverified. Notebook
import/export exists; successful SRNet execution has not been tested there.
See the platform's [workspace](https://evren.ssyz.org.tr/workspace).

The current [credit page](https://evren.ssyz.org.tr/pricing) advertises 1,000 CR
on e-Devlet registration and credits earned through contributions. Compute
rates are loaded dynamically. The public
[compute-tier registry](https://api.ssyz.org.tr/api/v1/configs/registries/compute-tiers),
read without authentication on this date, lists the active single-GPU training
tier at **10.00 CR/hour**. At that tariff alone, 1,000 CR divided by 10 CR/hour
is 100 GPU-hours. This arithmetic is not an account allocation: notebook rates,
other debits, credit expiry, queue access and actual balance remain unverified. Older university announcements ending promotional
credits on July 31 are not treated as the current tariff. Credit reservation,
settlement and runtime-idle billing need checking in the authenticated panel.

## Project-specific implications

The first useful account check is runtime availability, its GPU-hour credit
rate, maximum duration and persistent-storage allowance. A new environment
would then need read-only Python/Torch/CUDA/cuDNN/driver inventory and confirmation
that our dependencies and custom source package can run. The current local
pilot is frozen for WSL; its systemd/cgroup supervisor is not automatically
portable to a Slurm allocation. Preserve the FP32, no-AMP/no-TF32 recipe and
adapt resource/time evidence before proposing a separately bound remote job.

A larger advertised GPU cannot resolve the native L shortage, unknown ancestry
or the need for independent grouped development/testing. Existing private
numeric states, source licences, reserved partitions and all historical
allowances remain unchanged. The product does not gain a cloud AI provider.
Only public pages and static client code were read; no authentication,
account change, dataset/model upload, credit reservation or job was performed.

## Later user-authorized browser inspection

The user then opened the existing authenticated browser session for inspection.
The normal training panel is accessible; its new-job form offers built-in model
families and requires a project/dataset. The custom workspace route redirects
to an explicit access-denied page for this account. This confirms the earlier
public notebook interface is not presently available to the inspected account.
No attempt was made to bypass that restriction.

The GPU-reservation page explicitly states that reservations grant neither SSH
nor direct server access: they reserve capacity for jobs launched through the
training interface. It therefore does not establish a separate way to execute
our custom SRNet code. The Kaşif programme advertises early access to AI
Workspace; the review does not establish that an application guarantees access
or that it is the only route to permission.

Private screenshots and a minimal receipt retain the UI observations without
publishing account identifiers. Existing credit was visible. No credentials or
browser tokens were extracted; no project, notebook, reservation, application,
training job or data upload was submitted. Custom SRNet execution remains
unavailable through the inspected account pending workspace permission and
backend compatibility checks. All existing experiment contracts stay unchanged.

## User backend selection and recheck — 2026-11-06

The user now selects EVREN for future GPU compute to keep the laptop usable.
The completed local cost profile is retained, and its local fit proposal is
deferred. Normal session initialization succeeds in the browser, but the
custom Workspace route still explicitly denies access. Current account
screenshots/credit details remain private. See the new
[handoff and CPU diagnostic](PNG_BMP_EVREN_PREPARATION_20261106.md) and
[unsent permission request](EVREN_WORKSPACE_ACCESS_REQUEST_DRAFT_20261106.md).
No cloud runtime, reservation, application, upload or fitting is started.

## Subsequent user clarification

The user clarifies that AI Workspace is limited to a 29-person early-access
programme. Defer the unsent access request and avoid treating the observed
restriction as an ordinary account permission error. The normal training
registry offers built-in families; no custom SRNet route has been established.
No model recipe is changed to work around that limitation, and the user's
preference against further local GPU jobs remains in effect.
