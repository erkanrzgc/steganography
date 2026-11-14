# Storage cleanup (2026-11-14)

The user's explicit request to remove unnecessary project files permits this
cleanup of completed generated QA scratch data and an unused CI environment.
Scientific datasets, feature caches, learned research states, source bindings,
reports, successful/failed logs and failed/interrupted QA scratch are retained.

## Cause and measured cleanup

Before cleanup, the 197-GiB guest filesystem has about 2.2 GiB available
(99% used). The repository occupies about 85 GiB: approximately 83 GiB is
`.benchmark`, while application/source code is small. Home-cache QA trees
add about 31 GiB. Earlier pytest runs retained all per-test generated files;
several repeated `basetemp` trees are over eight GiB each. They contain many
copies of generated numeric states and other unit-test fixtures.

The following successfully closed QA scratch trees are removed after their
passing logs/status are verified and per-file inventories are saved:

| Disposable tree | Passing record |
|---|---|
| `.benchmark/png-bmp-rgb-parity-probe-20261108/qa/basetemp` | 4,425 passes, exit 0 |
| `~/.cache/steg-qa-20261111/full2/bt` | 4,496 passes, exit 0 |
| `~/.cache/steg-qa-20261112b/basetemp` | 4,578 passes, exit 0 |
| `.benchmark/academic-closeout-qa-20261114-q65ywhk3/basetemp` | 4,588 passes, status passed |
| `.benchmark/rich-model-full-qa-20261114/basetemp` | 4,624 passes, status passed |
| `.benchmark/rich-model-full-qa-final-20261114/basetemp` | 4,646 passes, status passed |

The unused one-off `.benchmark/ci-minimal-env-20261010` virtual environment
is also removed. Its configuration and package/version inventory are retained;
no running process or current project import path uses it. The active `venv`
and installed research dependencies remain intact. No package is downloaded.

**Net reclaimed: 27.859466552734375 GiB.** Available space becomes about
30 GiB and usage falls to 85%. Each removed tree has a compressed inventory
with file hashes, sparse-file extent signatures or explicit metadata-only
sparse-fixture records. Symlinks are never followed; external hardlinks are
protected. All seven approved roots are removed. Actual research states
outside these disposable pytest trees retain their hashes.

## Retained evidence and future use

The interrupted `~/.cache/steg-qa-20261111/full/bt` and failed
`~/.cache/steg-qa-20261112/basetemp` trees remain. So do all other failed
attempts, including the preexec deadlock and numeric replay diagnoses.
All passed/failed QA logs checked by this cleanup retain their hashes.
The 570 source bindings and six predecessor/reference model-state hashes
also verify after cleanup. No test result is recomputed or relabelled.

Receipts and inventories are private under
`.benchmark/qa-storage-cleanup-20261114-3k9p9lsx/`. Earlier records saying the
successful temporary trees were retained describe their pre-cleanup state;
those temporary bytes are now intentionally pruned under this user request.
This is not permission to remove real model checkpoints or training data.

Continue using fresh disk-backed TMPDIR/basetemp with
`-o tmp_path_retention_policy=failed`. Passed generated fixture directories
can be inventoried/pruned during explicit housekeeping; failed records and
scientific artifacts stay outside that disposal scope. There is no reason
to restart a consumed experiment to recreate temporary test copies.
