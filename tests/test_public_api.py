import steganography
from core.service import AnalysisService, StegoService


def test_public_package_exports_services_and_version():
    assert steganography.AnalysisService is AnalysisService
    assert steganography.StegoService is StegoService
    assert steganography.__version__ == "0.6.0"
