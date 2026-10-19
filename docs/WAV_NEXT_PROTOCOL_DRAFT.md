# WAV follow-up design — draft, not executable

This proposal follows the completed linear pilot and its separate
[read-only diagnosis](WAV_READONLY_PROTOCOL.md). It authorizes no fitting or
download, changes no historical result and consumes none of JPEG's allowance.
The completed [diagnostic results](WAV_READONLY_RESULTS.md) distinguish observed
train/development gaps from unproven causes. Neither stronger training AUC nor
a different threshold demonstrates a successful intervention.

## Source and independence

Candidate: AudioMNIST at commit
`630d7dab4c040882834de6fa21baf9a60372accd`. Its
[official README](https://github.com/soerenab/AudioMNIST/blob/630d7dab4c040882834de6fa21baf9a60372accd/README.md)
describes 30,000 spoken-digit samples from 60 speakers; the pinned metadata
contains 60 speaker entries. The repository has a
[root MIT license](https://github.com/soerenab/AudioMNIST/blob/630d7dab4c040882834de6fa21baf9a60372accd/LICENSE).
Record the pinned license linkage and any file-level exceptions before accepting
media. Metadata review is not a completed licensed carrier pilot.

The [capture settings](https://github.com/soerenab/AudioMNIST/blob/630d7dab4c040882834de6fa21baf9a60372accd/recording_scripts/main.py)
and [recorder](https://github.com/soerenab/AudioMNIST/blob/630d7dab4c040882834de6fa21baf9a60372accd/recording_scripts/recording_gui.py)
specify mono 48-kHz int16 capture. That does not verify individual WAV headers.
The [cutting routine](https://github.com/soerenab/AudioMNIST/blob/630d7dab4c040882834de6fa21baf9a60372accd/recording_scripts/util.py)
splits multi-digit recordings; samples are shuffled before capture. A filename's
repetition index is not evidence of a parent recording ID. Count independent
**speaker groups**, never individual cuts as independent original recordings.
Keep every utterance/cut/cover/stego from one speaker in one role. Exact parent
recording ancestry remains unavailable until supported by provenance.

Hash-sort speaker IDs using the fixed `wav-audiomnist-pilot-20261016:` namespace:
20 train, 10 development and 30 reserved. Proposed selection is digits 0..9,
takes 0/1 per train/development speaker: 400 train and 200 development carrier
files, representing 20/10 speaker groups. Proposed paths and roles are listed in
[the draft manifest](../benchmarks/wav-next-experiment-draft-20261016.json).
Path existence, file hashes and native headers are explicitly unavailable.
Reserved speaker media is neither downloaded nor inspected. Do not substitute
files or move roles after seeing development scores.

Accept only native mono PCM16 within the existing 160..220,500-frame contract,
with no resampling, normalization, channel duplication or lossy conversion.
Record original file/decoded-PCM hashes, license, source path, speaker group and
unknown parent-recording ancestry. Detect exact duplicates before role use.
Preserve the historical FSDD speakers, ESC recording groups and all test roles.

## Matched comparison proposed before new media access

Compare the current 760-feature linear reference with the existing FP32
raw/first-/second-difference PCM residual skeleton. Bind the skeleton source,
initialization and complete trainer source closure before fitting. Both arms use
the same controlled sequential/scattered, 5/40%, plane-0/1 matrix and independent
opaque payload/extraction oracle. No marker/oracle/source IDs enter inputs.

Use seeds 20261016/20261017, exactly 20 epochs and identical row permutations
over 400 x9 =3,600 training rows (one cover +eight stegos per source file).
Each original file contributes equal total mass, half cover and half stego;
speaker counts are equal by construction. Record 72,000 row presentations per
seed/arm and every optimizer step, including partial or discarded work. Speaker
grouping governs uncertainty; 400 files are not 400 independent recordings.

Proposed optimizer for both arms: plain SGD, LR .01, weight-only L2 .0001,
virtual batch 64, with the final partial batch normalized by its actual size.
The CNN processes one whole native waveform per microbatch and accumulates
gradients to that virtual batch; no truncation, padding-dependent pooling, AMP
or hidden window replication. FP32 scaling must preserve one LSB. Keep the
existing architecture and record BatchNorm buffers without post-hoc refresh.
Feature normalization uses fitting data only; CNN input scaling is fixed. This
is a comparison of representation/model candidates, not an isolated causal
architecture ablation.

This new pilot proposes fixed probability threshold .5 for both arms, frozen
before media access, and no OOF refits or threshold search. Its primary outcome
is source/seed/cell ROC-AUC with paired speaker-cluster intervals; also retain
balanced accuracy, recall/FPR, confusion and calibration limitations. Do not
compare its operating-point numbers directly with the earlier pilot's selected
OOF thresholds or claim production qualification. No old validation resweep.

## Gates before freezing or execution

The proposed raw-source ceiling is 384 MiB, with each file <=512 KiB; 600 files
have a 300-MiB aggregate upper bound before bounded metadata. This budget is
**not granted or spent**: the earlier cumulative one-GiB transfer ledger remains
incomplete and must not be reset. No Commons retry or AudioMNIST media request
was made. Reconcile that allowance or explicitly establish a separate acquisition
scope before any media transfer; all outputs still share the four-GiB native
allocation and overall ten-GiB ceiling.

Propose a separate one-hour GPU ceiling across both CNN seeds, with <=4-GiB
allocator /8-GiB resident memory and zero swap; the CPU reference has its own
ten-minute ceiling. These are planning ceilings, not an extension of JPEG's
completed six-hour allowance. Owned GPU connectivity, identities, native inputs,
licenses, hashes, actual code and whole-update capacity remain unverified.
Before fitting, a bounded generated-data preflight must conservatively account
for every microbatch/update, evaluation and checkpoint within these ceilings.
If capacity fails, record unavailable; do not silently reduce exposure or retry.

Stream controlled stegos from immutable native sources rather than retain
all waveform derivatives. Keep feature outputs within the unchanged 64-MiB
per-artifact limit and the existing row bound. Audit exact payload recovery and
maximum bit-plane distortion before fitting. Require scalar/security fixtures,
actual kernel/CUDA preflights, the unchanged full coverage gate and a separately
frozen source/data/exposure/budget protocol. No trainer or acquisition is started
by this draft.

Other native pilots remain incomplete: cached Kodak metadata lacks a verified
formal license grant; WIFD's 200 cached files are reserved JPEG/MPO test data. Neither
fills the licensed native RGB target. AudioMNIST capture is mono in its source
code and does not fill stereo/MP3 targets. See
[the source-gap audit](../benchmarks/native-source-gaps-20261016.json).
