"""WAV PCM LSB steganography."""
import struct
import wave
from pathlib import Path

import numpy as np

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_LEN_PREFIX = 4


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
        samples, _ = self._read(src)
        lsb_mean = float(np.mean(samples & 1))
        dev = abs(lsb_mean - 0.5) * 200
        sig = Signal(
            name="wav_lsb_bias",
            score=int(min(100, dev)),
            detail=f"LSB mean={lsb_mean:.3f}",
        )
        return AnalysisResult(self.name, sig.score, (sig,), None)
