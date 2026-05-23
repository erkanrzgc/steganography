from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult
from registry import Registry


class _FakePng(Carrier):
    name = "fake_png"
    extensions = (".png",)

    def embed(self, src, payload, out):
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src):
        return b""

    def analyze(self, src):
        return AnalysisResult(self.name, 0, (), None)

    def capacity(self, src):
        return 1


def test_register_and_select_by_extension():
    reg = Registry()
    reg.register(_FakePng())
    chosen = reg.select_carriers(Path("foo.png"))
    assert len(chosen) == 1
    assert chosen[0].name == "fake_png"


def test_select_returns_empty_for_unknown_extension():
    reg = Registry()
    reg.register(_FakePng())
    assert reg.select_carriers(Path("foo.xyz")) == []


def test_autodiscover_picks_up_modules():
    reg = Registry()
    reg.autodiscover()
    assert isinstance(reg.all_carriers(), list)
    assert isinstance(reg.all_analyzers(), list)


def test_select_carrier_for_embed_raises_when_no_match():
    import pytest
    reg = Registry()
    with pytest.raises(LookupError):
        reg.select_carrier_for_embed(Path("foo.xyz"))


def test_register_rejects_wrong_type():
    import pytest
    reg = Registry()
    with pytest.raises(TypeError):
        reg.register("not a carrier")
