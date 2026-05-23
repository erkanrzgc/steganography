from config import get_secret


def test_get_secret_returns_env_value(monkeypatch):
    monkeypatch.setenv("STEGO_TEST_KEY", "abc123")
    assert get_secret("STEGO_TEST_KEY") == "abc123"


def test_get_secret_returns_default_when_missing(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST_AAA", raising=False)
    assert get_secret("DOES_NOT_EXIST_AAA") is None
    assert get_secret("DOES_NOT_EXIST_AAA", default="x") == "x"


def test_get_secret_treats_empty_as_unset(monkeypatch):
    monkeypatch.setenv("STEGO_EMPTY", "")
    assert get_secret("STEGO_EMPTY", default="fallback") == "fallback"


def test_get_secret_loads_dotenv_file(tmp_path, monkeypatch):
    """Verify _ensure_loaded reads an actual .env file from disk."""
    import config

    env_file = tmp_path / ".env"
    env_file.write_text("STEGO_FROM_FILE=loaded_from_disk\n")

    # Reset the load state and point config at our temp .env via a patched module path.
    monkeypatch.setattr(config, "_ENV_LOADED", False)
    monkeypatch.setattr(config, "__file__", str(tmp_path / "config.py"))
    monkeypatch.delenv("STEGO_FROM_FILE", raising=False)

    assert config.get_secret("STEGO_FROM_FILE") == "loaded_from_disk"
