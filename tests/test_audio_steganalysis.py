"""Tests for advanced audio WAV steganalysis and extraction."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from core.ctf import CTFService
from core.service import AnalysisService
from modules.audio_wav import AudioWav, extract_wav_payload


def _create_wav(
    path: Path,
    samples: np.ndarray,
    rate: int = 44100,
    sampwidth: int = 2,
    channels: int = 1,
) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        if sampwidth == 1:
            w.writeframes(samples.astype(np.uint8).tobytes())
        elif sampwidth == 2:
            w.writeframes(samples.astype("<i2").tobytes())
        elif sampwidth == 4:
            w.writeframes(samples.astype("<i4").tobytes())
    return path


def test_clean_silence_wav_no_false_alarm(tmp_path: Path):
    silence = np.zeros(20000, dtype=np.int16)
    wav_path = _create_wav(tmp_path / "silence.wav", silence)

    service = AnalysisService(profile="balanced")
    analysis = service.analyze(wav_path)
    assert analysis.severity == "low"
    assert analysis.overall_score < 40


def test_wav_riff_trailer_detection_and_carving(tmp_path: Path):
    t = np.linspace(0, 0.2, 8820)
    audio = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    wav_path = _create_wav(tmp_path / "base.wav", audio)

    flag = b"flag{wav_trailer_appended_data}"
    raw_data = wav_path.read_bytes() + flag
    stego_path = tmp_path / "stego_trailer.wav"
    stego_path.write_bytes(raw_data)

    analyzer = AudioWav()
    result = analyzer.analyze(stego_path)
    signal_names = {s.name for s in result.signals}
    assert "riff_appended_flag" in signal_names or "riff_appended_data" in signal_names
    assert result.suspicion >= 85

    engine = CTFService()
    ctf_report = engine.solve(stego_path, output_dir=tmp_path / "ctf_trailer_out")
    assert ctf_report.verdict in ("confirmed", "likely")
    recovered = [a for a in ctf_report.artifacts if "trailer.bin" in a.name]
    assert len(recovered) >= 1
    content = (tmp_path / "ctf_trailer_out" / "artifacts" / recovered[0].name).read_bytes()
    assert flag in content


def test_wav_quiet_region_lsb_noise_detection(tmp_path: Path):
    # Mixed audio: silence + burst + silence
    s1 = np.zeros(4000, dtype=np.int16)
    t = np.linspace(0, 0.2, 8820)
    s2 = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    s3 = np.zeros(4000, dtype=np.int16)
    audio = np.concatenate([s1, s2, s3])

    # Invert/randomize LSBs across audio
    rng = np.random.default_rng(42)
    stego = audio.copy()
    stego = (stego & ~np.int16(1)) | rng.integers(0, 2, size=len(stego), dtype=np.int16)

    stego_path = _create_wav(tmp_path / "stego_quiet.wav", stego)
    analyzer = AudioWav()
    result = analyzer.analyze(stego_path)

    signal_names = {s.name for s in result.signals}
    assert "audio_quiet_region_lsb_noise" in signal_names
    assert result.suspicion >= 50


def test_wav_8bit_chi_square_detection(tmp_path: Path):
    t = np.linspace(0, 0.5, 22050)
    clean_8bit = ((np.sin(2 * np.pi * 440 * t) + 1.0) * 110.0 + 15.0).astype(np.uint8)

    rng = np.random.default_rng(42)
    random_bits = rng.integers(0, 2, size=len(clean_8bit), dtype=np.uint8)
    stego_8bit = (clean_8bit & np.uint8(0xFE)) | random_bits

    stego_path = _create_wav(tmp_path / "stego_8bit.wav", stego_8bit, sampwidth=1)
    analyzer = AudioWav()
    result = analyzer.analyze(stego_path)

    signal_names = {s.name for s in result.signals}
    assert "audio_8bit_chi_square" in signal_names
    assert result.suspicion >= 50


def test_wav_raw_lsb_flag_detection_and_extraction(tmp_path: Path):
    t = np.linspace(0, 0.5, 22050)
    audio = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)

    flag = b"flag{audio_lsb_secret_flag_extracted}"
    flag_bits = np.unpackbits(np.frombuffer(flag, dtype=np.uint8))
    audio[: len(flag_bits)] = (audio[: len(flag_bits)] & ~np.int16(1)) | flag_bits.astype(np.int16)

    stego_path = _create_wav(tmp_path / "stego_lsb_flag.wav", audio)

    analyzer = AudioWav()
    result = analyzer.analyze(stego_path)
    signal_names = {s.name for s in result.signals}
    assert "audio_lsb_plaintext_flag" in signal_names
    assert result.suspicion >= 90

    extracted = extract_wav_payload(stego_path)
    assert extracted is not None
    payload, desc = extracted
    assert payload == flag

    engine = CTFService()
    ctf_report = engine.solve(stego_path, output_dir=tmp_path / "ctf_wav_lsb_out")
    assert ctf_report.verdict == "confirmed"
    recovered = [a for a in ctf_report.artifacts if "recovered.bin" in a.name]
    assert len(recovered) >= 1
    content = (tmp_path / "ctf_wav_lsb_out" / "artifacts" / recovered[0].name).read_bytes()
    assert content == flag
