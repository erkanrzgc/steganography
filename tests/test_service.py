from pathlib import Path

import pytest

from core.filetype import detect_type, extension_mismatch
from core.payload import pack, pack_v1
from core.result import AnalysisResult, Signal
from core.service import (
    AmbiguousPayloadError,
    AnalysisService,
    ExtractionError,
    OutputExistsError,
    StegoService,
    aggregate_score,
)
from modules.image_lsb import ImageLsb
from registry import Registry


def _registry() -> Registry:
    registry = Registry()
    registry.autodiscover()
    return registry


def test_detect_type_by_content_and_mismatch(png_64x64: Path, tmp_path: Path):
    detected = detect_type(png_64x64)
    assert detected.name == "png"
    disguised = tmp_path / "picture.txt"
    disguised.write_bytes(png_64x64.read_bytes())
    detected = detect_type(disguised)
    assert detected.name == "png"
    assert extension_mismatch(disguised, detected)

    text = tmp_path / "readme.bin"
    text.write_text("ordinary UTF-8 text\n")
    assert detect_type(text).name == "text"
    binary = tmp_path / "unknown.bin"
    binary.write_bytes(b"\x00\x01\x02")
    assert detect_type(binary).name == "unknown"


def test_analysis_service_clean_and_stego_scores(png_64x64: Path, tmp_path: Path):
    service = AnalysisService(_registry())
    clean = service.analyze(png_64x64)
    assert clean.overall_score < 30
    assert clean.file.sha256
    assert clean.file.detected_type == "png"

    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"service payload")
    stego = tmp_path / "stego.png"
    StegoService(_registry()).embed(secret, png_64x64, stego)
    suspicious = service.analyze(stego)
    assert suspicious.overall_score >= 70
    assert suspicious.severity == "high"


def test_analysis_safe_records_missing_and_size_errors(tmp_path: Path):
    service = AnalysisService(_registry(), max_file_size=1)
    missing = service.analyze_safe(tmp_path / "missing.bin")
    assert missing.results[0].status == "error"
    large = tmp_path / "large.bin"
    large.write_bytes(b"xx")
    result = service.analyze_safe(large)
    assert result.results[0].status == "error"
    assert "maximum" in (result.results[0].error or "")


def test_aggregate_profiles_categories_and_ai():
    results = [
        AnalysisResult(
            "a",
            60,
            (
                Signal("one", 60, "", category="x"),
                Signal("duplicate", 40, "", category="x"),
            ),
            None,
        ),
        AnalysisResult(
            "b", 50, (Signal("two", 50, "", category="y"),), None
        ),
        AnalysisResult("ai_triage", 80, (), "external provider"),
    ]
    assert aggregate_score(results, "balanced") == 80
    assert aggregate_score(results[:-1], "balanced") == 80
    verified = AnalysisResult(
        "marker",
        5,
        (Signal("v", 5, "", evidence="verified"),),
        None,
    )
    assert aggregate_score([verified], "strict") == 95
    with pytest.raises(ValueError):
        aggregate_score([], "unknown")


def test_stego_service_encrypted_roundtrip_and_atomic_no_clobber(
    png_64x64: Path, tmp_path: Path
):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"encrypted service bytes")
    stego = tmp_path / "stego.png"
    service = StegoService(_registry())
    result = service.embed(
        secret,
        png_64x64,
        stego,
        password="correct horse",  # noqa: S106 - test credential
        no_clobber=True,
    )
    assert result.encrypted is True
    recovered = tmp_path / "recovered.bin"
    size, carrier, version = service.extract_to(
        stego,
        recovered,
        password="correct horse",  # noqa: S106 - test credential
        no_clobber=True,
    )
    assert recovered.read_bytes() == secret.read_bytes()
    assert size == len(secret.read_bytes())
    assert carrier == "image_lsb"
    assert version == 2
    with pytest.raises(OutputExistsError):
        service.embed(secret, png_64x64, stego, no_clobber=True)
    with pytest.raises(OutputExistsError):
        service.extract_to(
            stego,
            recovered,
            password="correct horse",  # noqa: S106 - test credential
            no_clobber=True,
        )


def test_extract_validation_failures(png_64x64: Path, tmp_path: Path):
    service = StegoService(_registry())
    with pytest.raises(ExtractionError, match="no valid payload"):
        service.extract(png_64x64)
    with pytest.raises(LookupError):
        service.extract(png_64x64, method="missing")


def test_service_extracts_legacy_v1_from_real_carrier(
    png_64x64: Path, tmp_path: Path
):
    blob = pack_v1(
        payload=b"legacy data",
        encrypted=False,
        salt=b"\x00" * 16,
        nonce=b"\x00" * 12,
    )
    stego = tmp_path / "legacy.png"
    ImageLsb().embed(png_64x64, blob, stego)
    data, method, version = StegoService(_registry()).extract(stego)
    assert (data, method, version) == (b"legacy data", "image_lsb", 1)


def test_extract_requires_method_when_two_payloads_validate(tmp_path: Path):
    from core.carrier import Carrier
    from core.result import EmbedResult

    class FakeCarrier(Carrier):
        extensions = (".fake",)

        def __init__(self, name: str, payload: bytes) -> None:
            self.name = name
            self.method_id = name
            self.payload = payload

        def embed(self, src, payload, out):
            return EmbedResult(self.name, out, len(payload), False)

        def extract(self, src):
            return self.payload

        def analyze(self, src):
            return AnalysisResult(self.name, 0, (), None)

        def capacity(self, src):
            return 100

    envelope = pack(
        payload=b"valid",
        encrypted=False,
        salt=b"\x00" * 16,
        nonce=b"\x00" * 12,
    )
    registry = Registry()
    registry.register(FakeCarrier("one", envelope))
    registry.register(FakeCarrier("two", envelope))
    path = tmp_path / "x.fake"
    path.write_bytes(b"carrier")
    with pytest.raises(AmbiguousPayloadError, match="multiple valid"):
        StegoService(registry).extract(path)


def test_content_dispatches_disguised_png(png_64x64: Path, tmp_path: Path):
    disguised = tmp_path / "cover.dat"
    disguised.write_bytes(png_64x64.read_bytes())
    carriers = _registry().select_carriers(disguised, detected_extension=".png")
    assert any(carrier.identifier == "image_lsb" for carrier in carriers)


def test_analysis_ai_error_is_structured(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("hello")

    def failing_provider(info, signals):
        raise RuntimeError("provider boom")

    analysis = AnalysisService(_registry(), ai_provider=failing_provider).analyze(path)
    ai = next(result for result in analysis.results if result.analyzer == "ai_triage")
    assert ai.status == "error"
    assert "provider boom" in (ai.error or "")
