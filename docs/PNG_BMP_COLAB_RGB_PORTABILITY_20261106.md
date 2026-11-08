# Colab RGB pilot portability review — 2026-11-06

The user's free-Colab continuation first checks one T4 allocation and read-only
metadata. This review describes concrete implementation dependencies; it is not
an executable training protocol or a new GPU/profile allowance.

The reusable model/data/optimizer/checkpoint services already exist and remain
unchanged: `core/spatial_native_data.py`, `core/spatial_native_training.py` and
the existing SRNet/numeric checkpoint modules. The proposal keeps 64 train,
20 development and 21 reserved RGB originals, with 13 L originals unassigned.
All source records remain unassigned; roles are an access overlay. No new
acquisition, native cache regeneration or historical budget reset is needed or
permitted by this review.

The current executor cannot be transplanted as-is:

- `spatial_native_pilot.source_names()` includes the private WSL actor/deployment
  and requires the existing RGB proposal and WSL protocol document. Those actor
  files are not a Colab source bundle and must not be uploaded blindly.
- `spatial_native_supervisor.kernel()` requires cgroup v2, finite <=8-GiB RAM
  and exactly zero swap; its live observer must retain the kernel peak. A GPU
  name or package-version match cannot satisfy this contract.
- The outer controller obtains independent systemd start/exit times, keeps an
  exclusive attempt marker, enforces a deadline, inventories output, stops the
  unit, transports evidence and reserves time for closing. Colab needs its own
  parent/worker/evidence implementation and generated interruption tests.
- The WSL source/environment/profile digest and cost projection stay historical.
  A distinct Colab profile must bind the actual source/environment, resource
  controls, complete command time, zero native updates and generated-only
  optimizer/checkpoint QA before a Colab fit can be proposed for approval.

Keep actual native/cache/model pixels off Colab during this metadata inventory.
Build and review any later package with source/license attribution and per-file
hashes, excluding closed reserved media and unrelated private machine configuration.
A missing strict resource control remains unavailable rather than silently
changing the WSL acceptance rule. A new Colab contract may define its own
explicit limits prospectively, with measured cost and separate approval.
