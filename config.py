"""Centralized secret loading. Single source of truth — no literal keys in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False


def _ensure_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _ENV_LOADED = True


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return the named secret from .env or environment.

    Empty or whitespace-only values are treated as unset and the default is returned.
    """
    _ensure_loaded()
    value = os.environ.get(name, default)
    if value is None or not str(value).strip():
        return default
    return value
