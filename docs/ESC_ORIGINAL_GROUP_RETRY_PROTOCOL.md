# ESC original-recording group retry — 2026-10-10

The original pinned ESC-50 acquisition failed the same-original-fold guard
after archive download and before writing any audio or success manifest.
Retain that failure. Contrary to its README's fold-separation statement,
four src_file IDs span upstream folds: 131943 (2/3), 134049 (2/3),
209698 (4/5), 234879 (4/5). The last two cross validation/test roles. No
method, model or accuracy score was inspected to discover this metadata issue.

Import only the retained exact archive SHA256
`661183a6f53ef04f12c9bd618fed0ddc1713280d6c94a5a5431e844ba6f6a21f`
(645,832,161 bytes), with native CSV SHA256
`ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246`.
No redownload or general checksum/count/fold override. Default strict import
still rejects cross-fold recording IDs. Original CSV/fold/category/take and
media bytes remain unchanged; raw metadata is never edited to invent provenance.

Separate explicit opt-in recipe `esc-original-group-max-fold-v1`: each original
src_file and every fragment inherits the highest upstream fold in that group.
Folds 1..3 train, 4 validation, 5 untouched test. This moves two formerly
validation clips to test, never test to validation/training. Retain all 2,000
originals /1,524 recording groups; resulting roles 1,200 train /398 validation
/402 test. Preserve assigned_group_fold separately from the native fold.
Upstream 250 class/fold cells still have eight clips; assigned-role class balance
is no longer exactly the original fold balance and must not be claimed as such.

Fresh output `.benchmark/esc50-grouped-diversity-20261010`, all ten reserved
manifest bindings, original bounds/usage evidence and independent complete
media/metadata/role audit still required. No extraction, fitting, GPU time,
qualified accuracy or unrestricted publication is established by acquisition.
