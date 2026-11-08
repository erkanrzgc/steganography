# Colab T4 read-only inventory — 2026-11-06

The user explicitly continues with free Colab after the GPU-access discussion.
This permits one ordinary T4 allocation request and a separate hardware/package/
resource metadata inspection. It grants no profile retry, optimizer update,
native image/model upload, acquisition or local GPU execution.

The source-bound notebook is
`notebooks/png_bmp_colab_gpu_inventory_20261106.ipynb`; its code and complete
notebook hashes are frozen in
`benchmarks/png-bmp-colab-gpu-inventory-protocol-20261106.json` before allocation.
Select Runtime Version 2026.07, Python 3, T4. Do not purchase credits, switch
accounts to evade a limit or request another card after a failed allocation.
One allocation is observed for at most 300 seconds. A rejected/unavailable GPU
is recorded honestly and the session is closed without automatic retry.

The explicit inventory cell uses a separate exclusive output/attempt directory.
It reads package metadata, cgroup counters if exposed, process limits and one
fixed `nvidia-smi` query. It does not import Torch, create a model/tensor or
perform forward/backward/optimizer operations. The query has a 10-second
parent timeout, five CPU seconds and an 8-KiB kernel file-size limit. The cell
checks a 60-second elapsed deadline and bounds receipts to 64 KiB. This is a
read-only diagnostic, not a native training resource supervisor or memory-peak
acceptance. GUI allocation/initialization time is observed separately, not
reported as kernel or training time.

Only the notebook code is uploaded. Package installation, Drive mounting,
SSH/proxy configuration and private dataset/checkpoint transfer are absent.
Save the output before disconnecting/deleting the GPU runtime. Missing cgroup
visibility, delegated controls or packages stays unavailable/unverified; do not
interpret a metadata pass as profile or fitting approval. Preserve all earlier
consumed Colab/WSL attempts, failed observers, closed reserved pixels and budgets.

Generated local syntax/metadata, exclusive-attempt and symlink checks pass
without importing Torch or calling a GPU. Ruff passes. Application/test source
remains unchanged; the earlier full software gate is not claimed as rerun.
Preserve the initial Ruff suppression-placement failure and pre-freeze source.

Before allocation, a real CPU metadata check identifies `/proc/self` as a
legitimate kernel symlink rejected by the artifact-style reader. The separate
prelaunch v2 binding uses the numeric own PID instead and retains symlink
rejection. The real CPU cgroup metadata check passes. The prior binding and
exact source are preserved; neither has consumed an allocation or worker
attempt. Only the v2-bound cell is used for the first actual inventory.
