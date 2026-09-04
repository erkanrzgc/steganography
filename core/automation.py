"""Offline NDJSON and signed-webhook payload helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Iterable
from typing import Any


def iter_ndjson(items: Iterable[dict[str, Any]]) -> Iterable[bytes]:
    for item in items:
        yield json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sign_webhook(body: bytes, secret: bytes, *, timestamp: int | None = None) -> dict[str, str]:
    if len(secret) < 16:
        raise ValueError("webhook secret must be at least 16 bytes")
    sent_at = int(time.time()) if timestamp is None else timestamp
    material = str(sent_at).encode() + b"." + body
    signature = hmac.new(secret, material, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Steganography-Timestamp": str(sent_at),
        "X-Steganography-Signature": f"sha256={signature}",
    }
