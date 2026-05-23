from pathlib import Path

import pytest

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult


def test_cannot_instantiate_abstract_carrier():
    with pytest.raises(TypeError):
        Carrier()


def test_concrete_subclass_works():
    class Dummy(Carrier):
        name = "dummy"
        extensions = (".dum",)

        def embed(self, src, payload, out):
            out.write_bytes(payload)
            return EmbedResult(self.name, out, len(payload), encrypted=False)

        def extract(self, src):
            return src.read_bytes()

        def analyze(self, src):
            return AnalysisResult(self.name, 0, (), None)

        def capacity(self, src):
            return 1024

    d = Dummy()
    assert d.name == "dummy"
    assert ".dum" in d.extensions
    assert d.capacity(Path("/x")) == 1024
