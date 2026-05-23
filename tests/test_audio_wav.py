from pathlib import Path

import pytest

from core.carrier import InsufficientCapacityError
from modules.audio_wav import AudioWav


def test_roundtrip(wav_pcm16: Path, tmp_outdir: Path):
    c = AudioWav()
    payload = b"audio stego payload"
    out = tmp_outdir / "out.wav"
    c.embed(wav_pcm16, payload, out)
    assert c.extract(out) == payload


def test_capacity_matches_sample_count(wav_pcm16: Path):
    c = AudioWav()
    # 8000 samples → 8000 bits → 1000 bytes - 4 (length prefix) = 996
    assert c.capacity(wav_pcm16) == 8000 // 8 - 4


def test_oversized_rejected(wav_pcm16: Path, tmp_outdir: Path):
    c = AudioWav()
    with pytest.raises(InsufficientCapacityError):
        c.embed(wav_pcm16, b"x" * (c.capacity(wav_pcm16) + 1), tmp_outdir / "out.wav")


def test_analyze_returns_result(wav_pcm16: Path):
    c = AudioWav()
    r = c.analyze(wav_pcm16)
    assert 0 <= r.suspicion <= 100
