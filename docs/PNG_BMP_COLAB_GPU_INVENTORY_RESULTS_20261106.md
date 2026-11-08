# Free Colab T4 inventory results — 2026-11-06

**GPU access ready; strict resource acceptance unavailable.** One ordinary
allocation in the existing user-selected account yields a Tesla T4 with
15,360 MiB VRAM and driver 580.82.07. No active Pro entitlement was visible
in the earlier subscription inspection. No payment, account switching, native
media/model upload or local GPU work occurs.

The selected Runtime Version is 2026.07. Package metadata matches Python
3.12.13 /NumPy 2.0.2 /Pillow 11.3.0 /Torch 2.11.0+cu128. ONNX and ONNX Runtime
are absent. Torch is not imported, and no tensor, model, forward/backward or
optimizer operation runs. This establishes visible hardware/package metadata,
not physical CUDA kernel compatibility or an optimizer/I/O cost estimate.

## Preserved failure and closing audit

The first inventory stops after 0.035111434 seconds with `ValueError` while
validating its cgroup identity. Colab exposes an absolute identity containing
parent (`..`) segments. The probe correctly refuses to join that identity to
host filesystem paths; it does not normalize or bypass it. Its status remains
`unavailable_inventory`, with SHA-256
`08017f769fdbb51ae41ef6475ea0f8ede8e4f254c6542ef7ee2bb40cb5cd5042`.
The NVIDIA query has not yet been called at this failure.

A separate read-only closing audit in the **same allocation** preserves and
hash-verifies that failed result, records safe identity flags and performs the
first and only bounded NVIDIA query. It does not relaunch the first inventory,
clear its marker or request a second GPU. This confirms the T4 metadata while
keeping resource enforcement `unavailable`; the initial gate is not relabelled
as passing. Closing SHA-256 is
`b47b562ef3941a2d9786bfc71f78c0b376ae5e50f768882f7b32681da5f9a6e6`.

The saved notebook is exported through the ordinary UI after disconnecting.
Both executed code cells match their reviewed source **byte for byte**, and
both JSON receipt digests replay independently from the exported output.
The exported notebook, account-bearing screenshots and exact first prelaunch
sources remain private. Only redacted results and account-free tool sources
are public.

The finished GPU runtime is disconnected/deleted. The independent UI controller
observes 294.536867 seconds from its one allocation request through the closing
action, below the 300-second observation ceiling. This includes allocation,
image startup and inspection; it is neither GPU kernel time nor billing data.
No new allocation or automatic inventory/profile rerun follows.

## Preparation and verification

The initial Ruff S603 suppression-placement failure is preserved and corrected
before allocation. The first source also used the kernel `/proc/self` symlink;
a prospective v2 prelaunch amendment reads numeric own-PID metadata and keeps
artifact-style symlink rejection. These prelaunch corrections consume no GPU
allocation. Generated metadata, exclusive-attempt and symlink checks pass.
Offline regression checks reject relative/parent-segment cgroup identities
and verify ordinary CPU metadata access; the unsafe Colab identity is not
accepted as a valid host path.

Ruff and diff checks pass. All 464 application/test input hashes still match
the historical 4,092-pass /two-CUDA-skip /97.18% full QA. No application or test
source changes, and that long suite is not claimed as a fresh execution.
All older Colab/WSL trials, cache acceptance, private checkpoints, closed pixels
and transfer/metadata/time ledgers remain unchanged.

| Component | Status |
| --- | --- |
| Free T4 allocation and hardware identity | ready |
| Pinned Python/package metadata | ready |
| Exact executed sources and both receipt hashes | ready |
| Cgroup path/strict host RAM/swap/peak acceptance | unavailable |
| CUDA model/kernel and new optimizer/I/O cost | unattempted /missing |
| Colab-native execution supervisor and fitting contract | missing |

The concrete [portability review](PNG_BMP_COLAB_RGB_PORTABILITY_20261106.md)
identifies the existing private WSL actor, strict cgroup-v2 envelope and
independent systemd timing as the components to replace in a separately
source-bound Colab implementation. Do not remove the WSL checks or upload its
machine-specific actor. Any later profile needs the new implementation and
software gate; native fitting additionally needs its measured, concrete
contract and approval. Keep all real source records unassigned and reserved
media closed.

See the [redacted receipts](../benchmarks/png-bmp-colab-gpu-inventory-results-20261106.json),
[executed source snapshot](../benchmarks/png-bmp-colab-gpu-inventory-tools-20261106.json)
and [prelaunch v2 binding](../benchmarks/png-bmp-colab-gpu-inventory-protocol-v2-20261106.json).
