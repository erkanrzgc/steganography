# AudioMNIST WAV comparison protocol — 2026-10-17

This is a separate source/data/exposure design for the new mono WAV pilot. It
does not alter the historical WAV or JPEG experiments. Training was not
started by the preparation task. The protocol is bound to the data and source
hashes in
[`wav-audiomnist-preparation-20261017.json`](../benchmarks/wav-audiomnist-preparation-20261017.json).

The source is the fixed AudioMNIST commit recorded in that receipt. Use only its
hash-sorted 20 train speakers and ten development speakers: ten digits, takes
zero and one per speaker, or 400 train and 200 development originals. Each
original and every controlled derivative stays in its speaker role. The other
30 speakers are reserved and remain inaccessible. Parent recording IDs are
unknown; speaker is the independent group. The raw acquisition used a separate
384-MiB ceiling; it does not reset or reconcile the historical cumulative
transfer ledger. No AudioMNIST file is added to a JPEG experiment.

Create rows in memory from each native 48-kHz mono PCM16 original: one cover and
eight deterministic variants (sequential/scattered, 5%/40%, bit plane 0/1).
Do not resample, normalize, pad, truncate, or save derivatives. The train and
development sets contain 3,600 and 1,800 rows. Verify exact opaque payload
recovery and the sample-plane-only distortion for every row before fitting.
The model receives only PCM-derived inputs; source names, speaker IDs, split,
lineage and payload/oracle data remain metadata.

Compare the 760-value `audio-pcm16-parity-residual-v1` linear reference with
the existing three-channel FP32 PCM residual CNN (raw PCM, first difference,
second difference). Use seeds `20261016` and `20261017`, 20 epochs, the same
seed-specific row permutations in both arms, and no OOF refit. Each arm has
72,000 presentations per seed and 1,140 updates per seed. Use virtual batches
of 64, including the final 16-row batch; SGD learning rate `0.01`, L2 `0.0001`
on weights only, and balanced weights that give each original equal total mass
with half the mass assigned to each class. Linear normalization is fitted on
training rows only. CNN inputs stay unpadded and FP32; each waveform is a
separate microbatch. Do not use AMP or TF32.

The threshold is fixed at `0.5` for all cells. Report ROC-AUC, balanced
accuracy, recall, false-positive rate and ECE. Pair each source file's cover
with its eight stegos and bootstrap uncertainty over the ten development
speaker groups. Do not select a threshold, open reserved data or claim that
fixture/oracle success demonstrates detection accuracy.

The design caps are 600 seconds for both linear seeds and 3,600 seconds shared
by both CNN seeds, including development scoring. They are proposed native
pilot limits and do not draw from JPEG's six-hour allowance. Before any fit,
run a source-bound generated preflight for the **complete current trainer**.
Exercise all 144,000 CNN presentations and 2,280 updates on generated native
PCM files using the observed frame-length distribution, both initializations,
development scoring and metrics; the shared hour must include all of them.
Exercise the generated linear feature path, both seeds and 2,280 updates within
the separate 600-second CPU cap. The CUDA phase must fit the existing four-GiB
allocator/eight-GiB resident limits and zero swap. The prior generated checks
validate only the model skeleton and do not satisfy this gate. If either phase
is unavailable or exceeds its bound, record `unavailable`; do not fit, reduce
exposure, restart or use CPU as a CNN substitute.

The acquisition, feature, training and metric source hashes are recorded in
the preparation receipt. Any code or data change invalidates that binding and
requires a new protocol version. This protocol has no optimizer updates or
checkpoint result; accuracy is unavailable until a separately authorized,
fully preflighted execution completes.
