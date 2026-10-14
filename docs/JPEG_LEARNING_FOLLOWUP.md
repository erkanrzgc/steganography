# JPEG learning follow-up: evidence before another training run

This note does not change live fitting/controller sources, thresholds or
qualification gates, and does not authorize another dataset download or
training stage. The frozen four-arm/two-seed control matrix and its independent
snapshot/metric verification must finish first. Paired diagnostics remain
descriptive; every stego outranking its own known cover is not standalone
classification accuracy.

## What differs from the original SRNet experiment

The authors' main setup trained on 14,000 original cover/stego pairs, with
1,000 validation and 5,000 test originals. Batches contained 16 pairs (32
images). Training used 400,000 optimizer iterations at learning rate 0.001,
then 100,000 at 0.0001. Their BOSS/BOWS originals were resized to 256 pixels
before JPEG encoding; JPEG inference used unrounded decompression.
[Original SRNet paper, Sections III.A–B](https://ws2.binghamton.edu/fridrich/research/SRNet.pdf).

Our small native-gray Q75/JUNIWARD study uses 512 fitting and 128 development
originals, 4 or 16 images per batch, and 8,192 or 2,048 updates at constant
learning rate. Five native 256-pixel views come from 512-pixel JPEGs, not
14,000 independent originals. Native cropping and pre-embedding resizing are
different processing contracts. These facts mean our controls are not a
replication of the original training experiment; they do **not** establish that
500,000 iterations are necessary or sufficient for our target distribution.

The current implementation already initializes convolutions with fan-in He,
biases 0.2, and the classifier with Gaussian standard deviation 0.01; do not
claim an unverified initialization defect. Architecture/optimizer equivalence,
processing, exposure and deployment generalization are separate questions.

## Read-only measurement before choosing an intervention

After a complete matrix, a bounded own-GPU diagnostic may evaluate already
fitted models on their existing **fit** cache using the same singleton/five-view
evaluation as development. Record fixed-threshold fit and development metrics,
fit eval-mode cross entropy, model/snapshot/source hashes, and equal parameter
and buffer digests before/after evaluation. No optimizer, score adjustment,
threshold recommendation, reserved pixels or new training. Deduct all added
diagnostic time from the existing remaining controls envelope; if no safe time
remains, record the diagnostic unavailable rather than extending the budget.

- Strong fit separation but weak development supports investigating data
  diversity, lineage/camera splits and generalization; it is not causal proof
  of overfitting.
- Weak fit and weak development supports investigating representation,
  optimization/exposure and normalization, rather than assuming a bigger
  dataset alone solves the problem.
- Compare both source groups and both seeds; do not select an intervention
  using one favorable development result or change thresholds to pass gates.

## What the literature supports, and what it does not

Processing choices can strongly affect cover-source mismatch. Do not equate a
dataset name or camera count with a matching image-processing distribution.
Our midpoint dispersion is not itself a measurement or proof of this mismatch.
[Giboulot et al., Effects and solutions of Cover-Source Mismatch (2020)](https://www.sciencedirect.com/science/article/pii/S0923596520300941).

Paired learning and normalization statistics can complicate generalization;
the shared-normalization paper studies this issue. It motivates testing a
normalization hypothesis, not silently replacing normalization in a running
study or promising its proposed method fixes our models.
[Wu et al., Shared Normalization (2017)](https://arxiv.org/abs/1711.07306).

Any subsequent candidate needs a new preregistered source/data-bound protocol,
matched exposure, train-only selection/calibration, original-family grouping
and untouched independent-source evaluation. Keep the production sample-size,
ROC-AUC/FPR/calibration acceptance criteria: a small development learning gate
or known-cover ranking fraction cannot promote an experimental model.
