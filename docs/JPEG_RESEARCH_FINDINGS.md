# JPEG learning research — 2026-10-10

This is a hypothesis register, not proof of our current failure's sole cause.
The historical failed development results remain public. No reported upstream
score qualifies this detector, and no third-party weights were downloaded.

## Primary references and implications

- [Original SRNet paper](https://ws.binghamton.edu/fridrich/research/SRNet.pdf)
  uses substantially larger paired batches and training exposure than our small
  pilots, paired rotations/reflections, and easier-task initialization for hard
  JPEG settings. Our four-row short mixed-source pilot is not a reproduction.
  Next: test optimization on fixed real pairs before expanding exposure; then
  preregister easier Q75/high-payload warm-up and harder-payload adaptation.
- [Author SRNet implementation](https://dde.binghamton.edu/download/feature_extractors/)
  and [TensorFlow 1.15 BatchNorm defaults](https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/contrib/layers/python/layers/layers.py)
  expose two verified numerical differences: epsilon .001 versus .00001, and
  SAME versus symmetric average-pool alignment. The new isolated variant tests
  both together; it cannot identify their separate effects or claim full upstream
  parity. Bias, regularization, augmentation, batch size and schedule still differ.
- [JPEG pretraining study](https://ws2.binghamton.edu/fridrich/Research/IN-pretraining-6.pdf)
  motivates high-payload steganographic pretraining and cover-based crop policy.
  Crop choice must not depend on stego labels or prediction outcomes. Larger
  payload is a curriculum hypothesis, not evidence that .2 detection improved.
- [ALASKA2 research report](https://ws.binghamton.edu/fridrich/Research/Alaska-2-Revised.pdf)
  and [third-place implementation](https://github.com/pfnet-research/kaggle-alaska2-3rd-place-solution)
  motivate non-rounded inputs, paired augmentation, quality-aware evaluation and
  complementary RGB/DCT models. Multi-GPU competition training and ensembles
  are not assumed feasible or equally effective on an 8 GB laptop.
- [Aletheia](https://github.com/daniellerch/aletheia) separates method-specific
  CNN training/evaluation and dataset checks. It is an independent comparison,
  not a source of labels or a substitute for cross-source evidence. Its detector
  confidence is not automatically our calibrated probability.
- [StegExpose](https://github.com/b3dk7/StegExpose) primarily targets spatial
  LSB statistics; it does not solve adaptive JPEG learning. Spatial architectures
  such as [GBRAS](https://github.com/BioAITeam/Steganalysis) likewise need their
  own preprocessing, datasets and evaluation rather than JPEG score transfer.

## Ordered decision gates

1. Run the frozen four-real-row legacy/reference learnability comparison and
   independently replay final singleton logits. Even 4/4 is memorization only.
2. If learnability passes, freeze a larger train-only Q75 curriculum: stronger
   payload first, paired D4 augmentation, then .2 adaptation, with sufficient
   per-original exposure and explicit laptop memory/time limits. Preserve all
   failed arms. Do not select an epoch/threshold using a final test set.
3. Evaluate untouched original groups and independent sources; report source,
   method, quality, payload, FPR and uncertainty separately. Keep camera/device
   independence unverified where metadata is missing. A positive training result
   cannot unlock deployment or the published support/accuracy gates.

Do not replace the legacy checkpoint contract, increase cloud spend, repeatedly
inspect blind data, or mask unsupported cells to make an aggregate look better.
