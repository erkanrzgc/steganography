# Local document preparation and complete format inventory — 2026-10-16

This slice prepares locally installed, declared-license PDF/TXT/MD/ZIP documents
and records the remaining dataset gaps for every explicitly handled extension.
It downloads no media, opens no reserved dataset, trains no model and changes
no detector or completed source binding. The older cumulative raw-transfer
ledger is incomplete and must not be reset. New native media acquisition is
unavailable within that ledger until it can be reconciled or a separate scope
is explicitly established.

## Input and output contract

`python -m steganography.research_documents --config FILE --out FRESH_DIR`
calls `core/document_corpus.py`. A private `local-document-pilot-config-v1`
configuration lists exact input/license paths and SHA-256 values, a portable
source ID, whole-package source group, declared file-scope license and an
explicit completed scope review. No scanning, downloader or installed-tool
execution is performed by the service. Installed-package provenance is locally
hashed; upstream archive authenticity is not established by installation paths.

Maximum 64 documents, 2 MiB per regular input/license, 16 MiB total outputs
including the manifest, and 120 seconds. Symlinks, FIFOs, wrong hashes, duplicate
original bytes, overwrite, malformed supported headers, unsafe/encrypted ZIP
members, over 512 ZIP members and over 16 MiB declared expanded ZIP bytes fail.
ZIP member bodies are neither extracted nor executed. PDF checks cover the
header/end only; text must be UTF-8. Inventory success is not full parser,
malware, hidden-payload absence or steganalysis certification.

The natural document is a **benign candidate**, with unchanged original bytes.
Each source produces one controlled positive and one equal-byte-length
whitespace-tail control. PDF/ZIP append 32 opaque bytes; TXT/MD append 256
U+200B/U+200C bits encoding 32 opaque bytes. The source-derived deterministic
payload contains no challenge flag or application marker. A separate scalar
decoder uses the known original byte extent and verifies the payload hash;
this is controlled ground truth, not autonomous recovery. A whitespace control
is a generated benign transformation, not another independent natural original.

Read every output back, recheck the source/license hashes and verify every
on-disk payload before exclusively publishing `manifest.json`. Failed partial
directories never contain a successful manifest and are preserved. Public
manifests contain relative artifact names, portable identities and hashes;
private input paths are not published. Licenses and originals stay local.

All outputs are `development_unassigned`, grouped by the entire upstream
package across files, formats, versions and every derivative. No file receives
a train/validation/test role or a certified clean label. Natural files do not
establish independent author/recording groups. Freeze future source roles,
whole-document grouping, realistic negatives and separate encoder/extraction
protocols before any training or detector accuracy claim. No synthetic carrier,
conversion, alias or duplicated document fills a missing native-source target.

## Verification and resources

Independent scalar/security fixtures must cover each encoder/oracle and every
input/output bound. Run the unchanged repository pytest/coverage gate, Ruff,
mypy and diff checks; new service/adapter coverage must be at least 95%.
Record optional unavailable formats honestly. Full verification uses fresh
disk-backed temporary storage, failed-only fixture retention and a two-GiB
task-output watchdog within the four-GiB native allocation, preserving JPEG's
six-GiB reserve. Keep initial failed setup/verification logs.

The preparation has a separate 120-second /16-MiB output cap, one CPU,
512-MiB resident-memory limit and zero swap. Its source, private config, license
records and worker are hashed before the owned one-shot run. No retry or
optimizer is authorized. Final receipts distinguish a prepared controlled
document pilot from a complete representative dataset for any extension.
