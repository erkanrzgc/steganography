"""Tests for the NVIDIA NIM AI provider. Network is fully mocked."""
import json
from pathlib import Path

import httpx
import pytest

import config
from core.result import Signal
from modules.ai_provider_nim import _parse_response, make_provider

# --- response parsing ------------------------------------------------------- #

def test_parse_response_plain_json():
    score, expl = _parse_response('{"suspicion": 75, "explanation": "LSB anomaly"}')
    assert score == 75
    assert expl == "LSB anomaly"


def test_parse_response_clamps_out_of_range():
    s, _ = _parse_response('{"suspicion": 500, "explanation": "x"}')
    assert s == 100
    s2, _ = _parse_response('{"suspicion": -10, "explanation": "x"}')
    assert s2 == 0


def test_parse_response_extracts_json_from_prose():
    score, expl = _parse_response('Sure! {"suspicion": 42, "explanation": "ok"} done.')
    assert score == 42
    assert expl == "ok"


def test_parse_response_raises_on_garbage():
    from modules.ai_provider_nim import NimError
    with pytest.raises(NimError):
        _parse_response("no json here at all")


# --- make_provider() guard ------------------------------------------------- #

def test_make_provider_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    monkeypatch.setattr(config, "_ENV_LOADED", True)  # skip loading the real .env
    assert make_provider() is None


# --- end-to-end via mocked httpx ------------------------------------------ #

def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_provider_calls_nim_with_image_payload(tmp_path: Path):
    # Build a fake 10-byte "image" file (existence + extension matter, not content).
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"suspicion": 88, "explanation": "stego likely"}'}}
                ]
            },
        )

    client = httpx.Client(transport=_mock_transport(handler))
    provider = make_provider(api_key="nvapi-test", model="my-model", client=client)
    assert provider is not None

    score, expl = provider({"path": str(img)}, [Signal("lsb_bias", 90, "high")])
    assert score == 88
    assert "stego" in expl.lower()
    assert "/chat/completions" in captured["url"]
    assert captured["auth"] == "Bearer nvapi-test"
    assert captured["body"]["model"] == "my-model"
    # Vision payload includes image_url part
    content = captured["body"]["messages"][1]["content"]
    assert any(p.get("type") == "image_url" for p in content)


def test_provider_text_only_for_non_image(tmp_path: Path):
    txt = tmp_path / "x.txt"
    txt.write_text("some text")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"suspicion": 12, "explanation": "clean"}'}}
                ]
            },
        )

    client = httpx.Client(transport=_mock_transport(handler))
    provider = make_provider(api_key="k", client=client)
    score, _ = provider({"path": str(txt)}, [])
    assert score == 12
    content = captured["body"]["messages"][1]["content"]
    # Text-only: no image_url part
    assert all(p.get("type") != "image_url" for p in content)


def test_provider_returns_zero_on_http_error(tmp_path: Path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=_mock_transport(handler))
    provider = make_provider(api_key="k", client=client)
    score, expl = provider({"path": str(img)}, [])
    assert score == 0
    assert "NIM provider error" in expl


def test_provider_returns_zero_on_unparseable_reply(tmp_path: Path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "no json in here"}}]},
        )

    client = httpx.Client(transport=_mock_transport(handler))
    provider = make_provider(api_key="k", client=client)
    score, expl = provider({"path": str(img)}, [])
    assert score == 0
    assert "NIM provider error" in expl
