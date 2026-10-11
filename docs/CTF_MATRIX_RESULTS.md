# CTF development recovery matrix — 2026-10-11

Controlled recovery regression, **not real-world detection accuracy** or the
120-case blind gate. Seventeen generated fixtures share one source group. The
recipe uses Python encoders, Pillow/piexif and independent PCM bit construction,
never the project's Carrier embedding implementation. Inspected cases are
development evidence, not blind challenges.

## Paired native result

[Portable evidence](../benchmarks/ctf-matrix-20261011.json) retains both runs,
input/payload hashes, budgets, per-case outcomes and source identities. The final
frozen manifest is `58fb54cf1b2dc0933a45b0b64927dbe5efee1f89617718145dc19f87f234d8d8`.
The old evaluator ran from a separate unchanged `1b54f0c` checkout. Preliminary
runs are preserved separately; ZIP timestamps were subsequently fixed for
reproducible fixture generation before the final paired replay.

| Controlled scope | Before | After |
|---|---:|---:|
| Base32/Base85/URL/nested Base64+gzip | 4/4 | 4/4 |
| ZIP and gzip+tar | 2/2 | 2/2 |
| PNG/BMP appended bytes | 1/2 | 2/2 |
| JPEG EXIF and GIF comment | 2/2 | 2/2 |
| WAV mono PCM8/16/32 | 3/3 | 3/3 |
| WAV stereo left/right-channel PCM16 LSB | 0/2 | 2/2 |
| Trailing whitespace and zero-width text | 2/2 | 2/2 |
| Completed exact recovery | 14/17 | 17/17 |

Both runs use identical bytes and quick-mode budgets: depth 3, artifacts 256,
output 1 GiB, tool timeout 30s and job timeout 180s. All jobs completed in both
runs. No limits or detection thresholds increased. Reference environment:
Kali VM CPU, Python 3.11.14, NumPy 2.4.6, Pillow 12.2.0. Timings are controlled VM
fixture timings, not a representative hardware/latency qualification.

BMP recovery previously lacked root trailer carving. The fix requires coherent
uncompressed pixel extent; invalid size, compression or dimensions do not become
trailer payloads. WAV extraction retains interleaved inspection and adds bounded
per-channel 1-/2-bit PCM8/16/32 inspection for up to eight channels. Raw reads
are capped at 524,288 aggregate samples (at most 2 MiB). PCM24, arbitrary sample
ordering and non-flag archive exact-length trimming are not qualified here.

## Reproduce

Install this checkout as in `CTF_EVALUATION.md`; use fresh folders:

```bash
python scripts/generate-ctf-matrix.py --out .benchmark/matrix-inputs
sha256sum .benchmark/matrix-inputs/manifest.json
python scripts/evaluate-ctf.py \
  --manifest .benchmark/matrix-inputs/manifest.json \
  --manifest-sha256 HASH_FROM_ABOVE \
  --out .benchmark/matrix-recovery --hardware cpu-reference
```

Generation is explicit, exclusive and network-free. ZIP/gzip/tar metadata is
fixed, but encoder/interpreter versions can still change bytes; freeze the
produced manifest. Replay the old implementation using a separate checkout and
the same unchanged inputs. Preserve old reports. WAV source identities are
recorded separately because the evaluator fingerprint is not a full analyzer lock.

## External tools and remaining gates

`--external` requires actual Steghide (BMP/JPEG), OutGuess (JPEG), OpenStego (PNG).
Missing/failing generators never publish a successful manifest. These four
no-password fixtures are not cracking-performance evidence. Local hardened
Docker recovered 4/4 external and 17/17 native payloads; tool-level failures and
missing optional jsteg remain in the report, not relabeled as successful tools.
Its CI artifact `ctf-development-matrix` uses the existing non-root/read-only,
network-disabled, dropped-capability and bounded tmpfs settings.

An initial local complete-suite attempt hit a filesystem write failure and
ended with exit 120; it is not counted as a successful verification run.
Focused tests and Docker matrix passed; the fresh complete-suite recheck and
remote verification are recorded separately when completed.

This slice does not improve the learned JPEG detector or qualify source,
payload-rate, FPR, AUC or calibration. PNG/BMP cells cover appended data, not
arbitrary spatial LSB. JPEG EXIF is not DCT steganalysis. Independent real-carrier
datasets, blind custody, richer recommendations and broader method/payload
coverage remain required.
