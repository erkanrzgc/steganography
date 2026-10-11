"""Independent 1/2-bit channel encoders and malformed PCM read budgets."""

import wave

import numpy as np
import pytest

from modules.audio_wav import extract_wav_payload


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("bits_per_sample", [1, 2])
def test_independent_channel_specific_flags(width, channel, bits_per_sample, tmp_path):
    flag = b"flag{independent_channel_pcm}"
    bits = np.unpackbits(np.frombuffer(flag, dtype=np.uint8))
    samples = np.random.default_rng(4356).integers(-80, 80, (2048, 2), dtype=np.int32)
    values = bits if bits_per_sample == 1 else bits[::2] | (bits[1::2] << 1)
    mask = (1 << bits_per_sample) - 1
    samples[:len(values), channel] = (samples[:len(values), channel] & ~mask) | values
    raw = (samples + 128).astype(np.uint8).tobytes() if width == 1 else samples.astype(
        "<i2" if width == 2 else "<i4"
    ).tobytes()
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(width)
        output.setframerate(8000)
        output.writeframes(raw)
    result = extract_wav_payload(path)
    assert result and result[0] == flag and f"channel {channel}" in result[1]


@pytest.mark.parametrize("fault", ["channels", "width", "partial", "empty", "invalid", "budget"])
def test_raw_audio_recovery_rejects_unsupported_and_bounds_aggregate_samples(
    tmp_path, monkeypatch, fault
):
    calls = []

    class Audio:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getsampwidth(self):
            return 3 if fault == "width" else 4

        def getnchannels(self):
            return 9 if fault == "channels" else 8

        def getnframes(self):
            return 10**12

        def readframes(self, count):
            calls.append(count)
            return b"x" if fault == "partial" else b""

    if fault == "invalid":
        assert extract_wav_payload(tmp_path / "missing") is None
    else:
        monkeypatch.setattr("modules.audio_wav.wave.open", lambda *a: Audio())
        assert extract_wav_payload(tmp_path / "unused") is None
        assert calls == ([] if fault in {"width", "channels"} else [65536])


def test_existing_interleaved_archive_stream_remains_extractable(tmp_path):
    payload = b"PK\x03\x04archive fixture"
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    path = tmp_path / "archive.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(bits.astype("<i2").tobytes())
    result = extract_wav_payload(path)
    assert result and result[0] == payload
