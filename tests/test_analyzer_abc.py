from pathlib import Path

import pytest

from core.analyzer import Analyzer
from core.result import AnalysisResult


def test_cannot_instantiate_abstract_analyzer():
    with pytest.raises(TypeError):
        Analyzer()


def test_concrete_analyzer():
    class Dummy(Analyzer):
        name = "dummy"

        def analyze(self, src):
            return AnalysisResult(self.name, 42, (), None)

    assert Dummy().analyze(Path("/x")).suspicion == 42
