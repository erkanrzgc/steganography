"""NVIDIA NIM AI provider for ai_triage.

Implements the AIProvider protocol from modules.ai_triage.

- Reads `NVIDIA_NIM_API_KEY` (required) and optional `NVIDIA_NIM_MODEL`,
  `NVIDIA_NIM_BASE_URL` via config.get_secret().
- For image files, sends the image as base64-encoded data URL alongside signal
  context (vision models).
- For non-image files, falls back to text-only prompting.
- Returns (suspicion: int 0-100, explanation: str).
- Network errors and malformed responses are non-fatal: caller gets (0, "<err>")
  so the toolkit never crashes mid-scan.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import httpx

import config
from core.result import Signal

_DEFAULT_MODEL = "meta/llama-3.2-90b-vision-instruct"
_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"})
_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MiB safety cap for upload payload
_REQUEST_TIMEOUT = 30.0

_SYSTEM_PROMPT = (
    "You are a steganalysis expert. Given heuristic signals (and an image when "
    "present), assess whether the file likely contains hidden data via LSB, "
    "EXIF, appended-data, or similar techniques. Reply with ONLY a JSON object: "
    '{"suspicion": <int 0-100>, "explanation": "<one sentence>"}. '
    "No markdown fences, no prose around the JSON."
)


class NimError(Exception):
    """Raised internally on transport or parse failures; caught by provider."""


def _mime_for(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")


def _build_messages(path: Path, signals: list[Signal]) -> list[dict]:
    sig_text = (
        "\n".join(f"- {s.name} (score {s.score}): {s.detail}" for s in signals)
        if signals
        else "(no prior signals)"
    )
    text_part = {
        "type": "text",
        "text": (
            f"File: {path.name}\n"
            f"Heuristic signals:\n{sig_text}\n\n"
            "Return the JSON now."
        ),
    }
    content: list[dict] = [text_part]

    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS and path.exists() and path.stat().st_size <= _MAX_IMAGE_BYTES:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{_mime_for(ext)};base64,{encoded}"},
            }
        )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(text: str) -> tuple[int, str]:
    """Extract {suspicion, explanation} from the model's reply."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise NimError(f"no JSON found in reply: {text[:200]!r}") from None
        data = json.loads(match.group(0))
    score = int(data.get("suspicion", 0))
    explanation = str(data.get("explanation", "")).strip()
    return max(0, min(100, score)), explanation or "(no explanation)"


def make_provider(
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    client: httpx.Client | None = None,
):
    """Return an AIProvider callable bound to the given config.

    Looks up missing values via config.get_secret(). Returns None if no API key
    is configured — caller can fall back to heuristic mode.
    """
    key = api_key or config.get_secret("NVIDIA_NIM_API_KEY")
    if not key:
        return None
    model_name = model or config.get_secret("NVIDIA_NIM_MODEL") or _DEFAULT_MODEL
    url = (base_url or config.get_secret("NVIDIA_NIM_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    endpoint = f"{url}/chat/completions"
    http = client or httpx.Client(timeout=_REQUEST_TIMEOUT)

    def provider(file_info: dict, signals: list[Signal]) -> tuple[int, str]:
        path = Path(file_info.get("path", ""))
        body = {
            "model": model_name,
            "messages": _build_messages(path, signals),
            "max_tokens": 256,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        try:
            r = http.post(endpoint, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return _parse_response(text)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, NimError) as e:
            return 0, f"NIM provider error: {type(e).__name__}: {e}"

    return provider
