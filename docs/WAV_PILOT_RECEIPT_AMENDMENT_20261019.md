# AudioMNIST receipt amendment — 2026-10-19

This amendment versions the AudioMNIST WAV pilot after its first generated CUDA
preflight. It changes receipt assembly only. The data, rows, model arms, seeds,
exposure, optimizer, threshold, metrics, compute caps and storage caps of
[the 2026-10-17 protocol](WAV_PILOT_TRAINING_PROTOCOL_20261017.md) are
unchanged. The original protocol, preparation receipt and failed attempt stay
immutable.

The [first attempt](../benchmarks/wav-audiomnist-cuda-preflight-20261019.json)
verified all 16 bound sources and ran both generated pairs without reaching
their 3,600/600-second ceilings. It then raised `KeyError: 'seed'` because
`generated_cnn_preflight` read a seed key from `core.nonjpeg_linear.fit`
output, which stores the seed only on the enclosing per-seed row. Per-arm
timings, update counts and the CUDA peak were not recorded, so that gate is
`unavailable`.

The only source change takes each linear receipt's seed from its per-seed row.
CNN receipts, the 2,280-update CNN and linear gates and the four-GiB allocator
check are unchanged. A regression test runs the real linear pair through the
receipt path; it fails with the original source and passes with the amendment.

Run exactly one new generated preflight under the original limits: 3,600
seconds shared by both CNN seeds, 600 seconds for both linear seeds, a four-GiB
CUDA allocator fraction, eight-GiB host memory and zero swap. Generated PCM
files use the 400 train /200 development frame lengths recorded in the
preparation receipt. No real or reserved media is opened, no checkpoint is
written and no retry follows a failure. The gate asks whether one complete
schedule fits the per-run ceilings, so the first attempt's 3,081 seconds are
recorded separately as spent preflight time, not deducted from the new run.

A pass establishes generated execution capacity only. It does not authorize an
AudioMNIST fit, threshold selection or any accuracy claim; a real fit requires
separate authorization. The machine-readable binding is
[`wav-audiomnist-receipt-amendment-20261019.json`](../benchmarks/wav-audiomnist-receipt-amendment-20261019.json).
