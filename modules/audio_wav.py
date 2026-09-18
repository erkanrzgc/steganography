from __future__ import annotations

import re
import struct
import wave
from pathlib import Path

import numpy as np

from core.carrier import Carrier, InsufficientCapacityError
from core.context import AnalysisContext
from core.payload import MAGIC
from core.result import AnalysisResult, EmbedResult, Signal

_LEN_PREFIX = 4

_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "ZIP archive"),
    (b"\x1f\x8b\x08", "gzip archive"),
    (b"7z\xbc\xaf\x27\x1c", "7z archive"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"%PDF-", "PDF document"),
    (b"\x7fELF", "ELF binary"),
    (b"MZ", "PE binary"),
)

_FLAG_PATTERN = re.compile(rb"([a-zA-Z0-9_]{3,24}\{[ -~]{4,120}\})")


def _compute_audio_8bit_chi2(samples: np.ndarray) -> tuple[float, int]:
    counts = np.bincount(samples.astype(np.uint8), minlength=256)
    even = counts[0::2]
    odd = counts[1::2]
    tot = even + odd
    valid = tot >= 6
    if np.sum(valid) < 8:
        return 0.0, 0
    e = even[valid]
    o = odd[valid]
    t = tot[valid]
    exp = t / 2.0
    chi2 = float(np.sum(((e - exp) ** 2 + (o - exp) ** 2) / exp) / np.sum(valid))
    return chi2, int(np.sum(valid))


def extract_raw_lsb(samples: np.ndarray, max_bytes: int = 65536) -> tuple[bytes, bytes]:
    """Extract 1-bit and 2-bit LSB byte streams from audio samples."""
    flat = samples.reshape(-1)
    limit1 = min(len(flat), max_bytes * 8)
    limit1 -= limit1 % 8
    b1 = np.packbits(flat[:limit1].astype(np.uint8) & 1).tobytes() if limit1 > 0 else b""

    limit2 = min(len(flat), max_bytes * 4)
    limit2 -= limit2 % 4
    if limit2 > 0:
        s_2bit = flat[:limit2].astype(np.uint8) & 3
        b2_bits = np.zeros(len(s_2bit) * 2, dtype=np.uint8)
        b2_bits[0::2] = s_2bit & 1
        b2_bits[1::2] = (s_2bit >> 1) & 1
        b2 = np.packbits(b2_bits).tobytes()
    else:
        b2 = b""
    return b1, b2


def extract_wav_payload(src: Path) -> tuple[bytes, str] | None:
    """Attempt extraction of plaintext flags or standard archives from WAV LSBs."""
    try:
        with wave.open(str(src), "rb") as w:
            width = w.getsampwidth()
            raw = w.readframes(min(w.getnframes(), 65536 * 8))
        if width == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        elif width == 2:
            samples = np.frombuffer(raw, dtype="<i2")
        elif width == 4:
            samples = np.frombuffer(raw, dtype="<i4")
        else:
            return None
    except Exception:
        return None

    b1, b2 = extract_raw_lsb(samples, max_bytes=65536)
    for stream, label in ((b1, "1-bit LSB"), (b2, "2-bit LSB")):
        if not stream:
            continue
        m = _FLAG_PATTERN.search(stream)
        if m:
            return m.group(0), f"extracted flag from WAV {label}"
        for magic, magic_name in _MAGICS:
            if stream.startswith(magic):
                return stream, f"extracted {magic_name} from WAV {label}"
    return None



class AudioWav(Carrier):
    name = "audio_wav"
    extensions = (".wav",)

    def _read(self, src: Path) -> tuple[np.ndarray, wave._wave_params]:
        with wave.open(str(src), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        if params.sampwidth != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        samples = np.frombuffer(frames, dtype=np.int16).copy()
        return samples, params

    def capacity(self, src: Path) -> int:
        samples, _ = self._read(src)
        return max(0, samples.size // 8 - _LEN_PREFIX)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        if len(payload) > self.capacity(src):
            raise InsufficientCapacityError("payload exceeds WAV LSB capacity")
        samples, params = self._read(src)
        blob = struct.pack(">I", len(payload)) + payload
        bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
        samples[: bits.size] = (samples[: bits.size] & ~np.int16(1)) | bits.astype(np.int16)
        with wave.open(str(out), "wb") as w:
            w.setparams(params)
            w.writeframes(samples.tobytes())
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        samples, _ = self._read(src)
        length_bits = samples[: _LEN_PREFIX * 8].astype(np.uint8) & 1
        (length,) = struct.unpack(">I", np.packbits(length_bits).tobytes())
        start = _LEN_PREFIX * 8
        end = start + length * 8
        return np.packbits(samples[start:end].astype(np.uint8) & 1).tobytes()

    def analyze(self, src: Path) -> AnalysisResult:
        return self.analyze_context(AnalysisContext(src))

    def analyze_context(self, context: AnalysisContext) -> AnalysisResult:
        try:
            samples, _rate, sample_width = context.wav_samples
        except Exception as exc:
            return AnalysisResult(self.name, 0, (), None, status="unsupported", error=str(exc))
        if sample_width not in (1, 2, 4):
            return AnalysisResult(
                self.name,
                0,
                (),
                None,
                status="unsupported",
                error=f"unsupported sample width: {sample_width}",
            )
        shaped_samples = samples
        flat_samples = samples.reshape(-1)
        signals: list[Signal] = []

        std_val = float(np.std(flat_samples.astype(np.float64)))
        lsb_mean = float(np.mean(flat_samples & 1))
        dev = abs(lsb_mean - 0.5) * 200
        bias_score = int(min(100, dev)) if std_val > 10.0 else 0
        signals.append(
            Signal(
                name="wav_lsb_bias",
                score=bias_score,
                detail=f"LSB mean={lsb_mean:.3f}",
                category="audio_lsb",
                evidence="heuristic" if (bias_score >= 10 and std_val > 10.0) else "informational",
            )
        )

        data = context.data
        if data.startswith(b"RIFF") and len(data) >= 8:
            riff_size = int.from_bytes(data[4:8], "little")
            structural_len = 8 + riff_size
            if len(data) > structural_len:
                trailer_len = len(data) - structural_len
                trailer = data[structural_len:]
                matched_magic = [name for magic, name in _MAGICS if magic in trailer[:64]]
                flag_match = _FLAG_PATTERN.search(trailer[:2048])
                if flag_match:
                    flag_text = flag_match.group(0).decode("latin-1", errors="replace")
                    signals.append(
                        Signal(
                            name="riff_appended_flag",
                            score=98,
                            detail=f"found flag in RIFF trailer: {flag_text}",
                            category="known_marker",
                            evidence="verified",
                        )
                    )
                elif matched_magic:
                    signals.append(
                        Signal(
                            name="riff_appended_data",
                            score=90,
                            detail=f"{trailer_len} bytes after RIFF; contains {matched_magic[0]}",
                            category="appended_data",
                            evidence="strong",
                        )
                    )
                else:
                    signals.append(
                        Signal(
                            name="riff_appended_data",
                            score=min(85, 50 + trailer_len // 64),
                            detail=f"{trailer_len} bytes after RIFF chunk",
                            category="appended_data",
                            evidence="strong",
                        )
                    )

        chunks: list[tuple[str, int]] = []
        offset = 12
        while offset + 8 <= len(data):
            chunk_id = data[offset : offset + 4].decode("ascii", errors="replace")
            chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
            chunks.append((chunk_id, chunk_size))
            offset += 8 + chunk_size + (chunk_size & 1)
        unknown = [name for name, _size in chunks if name not in {"fmt ", "data", "LIST", "fact"}]
        if unknown:
            signals.append(
                Signal(
                    "unusual_riff_chunks",
                    min(60, 25 + len(unknown) * 5),
                    "unusual RIFF chunks: " + ", ".join(unknown[:8]),
                    category="wav_structure",
                    evidence="heuristic",
                )
            )

        if sample_width == 1:
            chi2, df = _compute_audio_8bit_chi2(flat_samples + 128)
            if df >= 8:
                if chi2 <= 1.2:
                    chi_score = min(85, round(60 + (1.2 - chi2) / 1.2 * 25))
                elif chi2 <= 2.5:
                    chi_score = min(60, round(35 + (2.5 - chi2) / 1.3 * 25))
                else:
                    chi_score = 0
                signals.append(
                    Signal(
                        name="audio_8bit_chi_square",
                        score=chi_score,
                        detail=f"8-bit audio pairs-of-values chi-square={chi2:.4f} (df={df})",
                        category="audio_lsb",
                        evidence="heuristic" if chi_score >= 40 else "informational",
                    )
                )

        quiet_mask = np.abs(flat_samples) <= 2
        if np.sum(quiet_mask) >= 400:
            quiet_lsb = flat_samples[quiet_mask] & 1
            quiet_one_ratio = float(np.mean(quiet_lsb))
            quiet_trans = float(np.mean(quiet_lsb[1:] != quiet_lsb[:-1]))
            if abs(quiet_one_ratio - 0.5) < 0.12 and quiet_trans > 0.38:
                n_quiet = int(np.sum(quiet_mask))
                silence_score = min(80, round(quiet_trans * 120))
                signals.append(
                    Signal(
                        name="audio_quiet_region_lsb_noise",
                        score=silence_score,
                        detail=(
                            f"quiet samples ({n_quiet}) have random LSB: "
                            f"ones={quiet_one_ratio:.3f}, transitions={quiet_trans:.3f}"
                        ),
                        category="audio_lsb",
                        evidence="heuristic" if silence_score >= 45 else "informational",
                    )
                )

        b1, b2 = extract_raw_lsb(flat_samples, max_bytes=8192)
        m1 = _FLAG_PATTERN.search(b1)
        m2 = _FLAG_PATTERN.search(b2)
        flag_found = m1 or m2
        if flag_found:
            mode_desc = "1-bit LSB" if m1 else "2-bit LSB"
            flag_str = flag_found.group(0).decode("latin-1", errors="replace")
            signals.append(
                Signal(
                    name="audio_lsb_plaintext_flag",
                    score=98,
                    detail=f"found flag in {mode_desc}: {flag_str}",
                    category="known_marker",
                    evidence="verified",
                )
            )

        if shaped_samples.ndim == 2 and shaped_samples.shape[1] >= 2:
            channel_difference = shaped_samples[:, 0].astype(np.int64) - shaped_samples[
                :, 1
            ].astype(np.int64)
            difference_lsb = float(np.mean(channel_difference & 1))
            signals.append(
                Signal(
                    "channel_difference_lsb",
                    0,
                    f"left/right difference LSB mean={difference_lsb:.4f}",
                    category="audio_channel_difference",
                    evidence="informational",
                )
            )

        window = flat_samples[: min(flat_samples.size, 65_536)].astype(np.float64)
        if window.size >= 64:
            spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size))) + 1e-12
            flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
            signals.append(
                Signal(
                    "spectral_flatness",
                    0,
                    f"bounded FFT spectral flatness={flatness:.5f}",
                    category="audio_spectrogram",
                    evidence="informational",
                )
            )

        if flat_samples.size >= (_LEN_PREFIX + len(MAGIC)) * 8:
            length_bits = flat_samples[: _LEN_PREFIX * 8].astype(np.uint8) & 1
            (length,) = struct.unpack(">I", np.packbits(length_bits).tobytes())
            start = _LEN_PREFIX * 8
            marker_end = start + len(MAGIC) * 8
            marker = np.packbits(flat_samples[start:marker_end].astype(np.uint8) & 1).tobytes()
            if marker == MAGIC and length <= max(0, flat_samples.size // 8 - _LEN_PREFIX):
                signals.append(
                    Signal(
                        "steg_payload_header",
                        98,
                        f"validated STEG envelope prefix; embedded length={length}",
                        category="known_marker",
                        evidence="verified",
                    )
                )

        suspicion = max(
            (signal.score for signal in signals if signal.evidence != "informational"),
            default=0,
        )
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
