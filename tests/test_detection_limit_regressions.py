"""Score-correlation and PDF allocation-limit regressions, not accuracy claims."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from core.ensemble import HybridEnsemble
from core.service import aggregate_score
from modules.image_bitplane import ImageBitplaneAnalyzer


@pytest.mark.parametrize("seed", [0, 17, 42, 100])
def test_lsb_adjacency_aliases_never_double_count(tmp_path, seed):
    rng = np.random.default_rng(seed)
    rgb = np.broadcast_to(np.arange(128, dtype=np.uint8)[None, :, None], (128, 128, 3)).copy()
    rgb = (rgb & 254) | rng.integers(0, 2, rgb.shape, dtype=np.uint8)
    # Independent equivalence check: the two legacy names measure the same event.
    equality = np.mean((rgb[:, 1:, :] & 1) == (rgb[:, :-1, :] & 1))
    parity = np.mean((np.abs(np.diff(rgb.astype(np.int16), axis=1)) & 1) == 0)
    assert equality == parity
    path = tmp_path / "correlated.png"
    Image.fromarray(rgb).save(path)
    result = ImageBitplaneAnalyzer().analyze(path)
    proxies = tuple(
        s
        for s in result.signals
        if s.name in {"sample_pair_balance", "rs_regular_singular_balance"}
    )
    assert len(proxies) == 2
    assert proxies[0].score == proxies[1].score > 25
    assert proxies[0].category == proxies[1].category == "image_lsb_adjacency"
    pair = replace(result, signals=proxies)
    single = replace(result, signals=proxies[:1])
    without_alias = replace(
        result, signals=tuple(s for s in result.signals if s.name != "rs_regular_singular_balance")
    )
    for profile in ("sensitive", "balanced", "strict"):
        assert (
            aggregate_score([pair], profile)
            == aggregate_score([single], profile)
            == proxies[0].score
        )
        assert aggregate_score([result], profile) == aggregate_score([without_alias], profile)
    ensemble = HybridEnsemble()
    assert ensemble.combine((pair,)).score == ensemble.combine((single,)).score == proxies[0].score
    assert ensemble.combine((result,)).score == ensemble.combine((without_alias,)).score
