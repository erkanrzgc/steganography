# steganography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python 3.11+ steganography toolkit (`steganography`) — dual-purpose embed/extract + steganalysis — with a flat repo-root package, scanner-style `modules/` plug-in layout, AES-GCM crypto, pluggable AI triage, and JSON/HTML reporting.

**Architecture:** Repo root = import root (`from core.carrier import ...`, `from modules.image_lsb import ...`). One file per capability under `modules/`. Auto-discovery via `registry.py`. All modules implement a shared `Carrier` or `Analyzer` ABC. Results are frozen dataclasses.

**Tech Stack:** Python 3.11+, `cryptography` (AES-256-GCM + scrypt), `pillow` + `numpy` (image LSB/stats), `piexif` (EXIF), `rich` (CLI + report), `pytest` + `pytest-cov` (TDD, ≥80%).

**Spec:** `docs/superpowers/specs/2026-05-23-steganography-design.md`

**Working directory:** `/home/erkanrzgc/steganography` (fresh, not yet a git repo).

---

## Task 0: Project scaffolding & git init

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `ruff.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config.py`
- Create: `README.md` (stub — full content in Task 20)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: `git init` and configure**

```bash
cd /home/erkanrzgc/steganography
git init
git config user.email "benerkanrzgc@gmail.com"
git config user.name "erkanrzgc"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.pytest_cache/
.coverage
.coverage.*
htmlcov/
coverage.xml
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
.env
.env.local
scans/
```

Note: `.env` is ignored; secrets stay local. Only `.env.example` is committed.

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "steganography"
version = "0.1.0"
description = "Dual-purpose steganography toolkit: embed/extract + steganalysis"
requires-python = ">=3.11"
authors = [{ name = "erkanrzgc" }]
license = { text = "MIT" }
readme = "README.md"

[project.scripts]
steganography = "cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.coverage.run]
source = ["core", "modules", "ui", "report", "registry"]
omit = ["tests/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

- [ ] **Step 4: Create `requirements.txt`**

```
cryptography>=41
pillow>=10
numpy>=1.26
rich>=13
piexif>=1.1
python-dotenv>=1.0
```

- [ ] **Step 5: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8
pytest-cov>=5
ruff
mypy
```

- [ ] **Step 6: Create `ruff.toml`**

```toml
line-length = 100
target-version = "py311"

[lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]
```

- [ ] **Step 7: Create `README.md` stub**

```markdown
# steganography

Dual-purpose steganography toolkit (embed/extract + steganalysis).

See [design spec](docs/superpowers/specs/2026-05-23-steganography-design.md).

Full README is generated in Task 20.
```

- [ ] **Step 7b: Create `.env.example` (committed) and `config.py`**

`.env.example`:

```
# Copy to .env (gitignored) and fill in real values.
# Any AI provider integration reads its key via config.get_secret().
NVIDIA_NIM_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

`config.py`:

```python
"""Centralized secret loading. Single source of truth — no literal keys in code."""
import os
from pathlib import Path
from typing import Optional

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


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return the named secret from .env or environment. Empty string is treated as unset."""
    _ensure_loaded()
    value = os.environ.get(name, default)
    if value == "":
        return default
    return value
```

`tests/test_config.py`:

```python
import os

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
```

Run: `./venv/bin/pytest tests/test_config.py -v`
Expected: 3 PASS.

- [ ] **Step 8: Create empty `tests/__init__.py` and `tests/conftest.py`**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""Shared pytest fixtures."""
from pathlib import Path

import pytest


@pytest.fixture
def tmp_outdir(tmp_path: Path) -> Path:
    """Throwaway output directory for embed tests."""
    out = tmp_path / "out"
    out.mkdir()
    return out
```

- [ ] **Step 9: Create venv & install dev deps**

```bash
cd /home/erkanrzgc/steganography
python3.11 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements-dev.txt
```

Expected: clean install, no errors.

- [ ] **Step 10: Verify pytest runs (no tests yet)**

Run: `./venv/bin/pytest`
Expected: `no tests ran` (exit 5) — confirms pytest is wired.

- [ ] **Step 11: Initial commit**

```bash
git add .
git commit -m "chore: scaffold steganography project (pyproject, deps, ruff, gitignore, config)"
```

---

## Task 1: Banner port (cyberm4fia gradient)

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/banner.py`
- Test: `tests/test_banner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_banner.py
from io import StringIO

from rich.console import Console

from ui.banner import print_gradient_banner


def test_banner_renders_cyberm4fia_wordmark():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, record=True)
    print_gradient_banner(console=console)
    text = console.export_text()
    # All six wordmark rows must be present.
    assert "██████╗██╗" in text
    assert "AI-destekli" in text or "steganografi" in text.lower()


def test_banner_respects_quiet():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, record=True)
    print_gradient_banner(console=console, quiet=True)
    assert console.export_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_banner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui'`.

- [ ] **Step 3: Create `ui/__init__.py`**

```python
# ui/__init__.py
```

- [ ] **Step 4: Create `ui/banner.py` (port + adapt cyberm4fia banner)**

```python
"""Gradient ASCII banner (ported from cyberm4fia-scanner/utils/colors.py)."""
from rich.console import Console

_BANNER = r"""
 ██████╗██╗   ██╗██████╗ ███████╗██████╗ ███╗   ███╗██╗  ██╗███████╗██╗ █████╗
██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██║  ██║██╔════╝██║██╔══██╗
██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██╔████╔██║███████║█████╗  ██║███████║
██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██║╚██╔╝██║╚════██║██╔══╝  ██║██╔══██║
╚██████╗   ██║   ██████╔╝███████╗██║  ██║██║ ╚═╝ ██║     ██║██║     ██║██║  ██║
 ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
"""

_DESCRIPTION = (
    "steganography — AI-destekli çift yönlü steganografi & steganaliz toolkit\n"
    "embed · extract · analyze · scan"
)

_default_console = Console(record=True)


def print_gradient_banner(*, console: Console | None = None, quiet: bool = False) -> None:
    """Print the CYBERM4FIA gradient banner with steganography subtitle."""
    if quiet:
        return
    out = console or _default_console
    lines = _BANNER.strip("\n").split("\n")
    start, end = (230, 230, 230), (40, 40, 40)
    for i, line in enumerate(lines):
        ratio = i / max(len(lines) - 1, 1)
        r, g, b = (int(start[j] + (end[j] - start[j]) * ratio) for j in range(3))
        out.print(f"[#{r:02x}{g:02x}{b:02x}]{line}[/]")
    out.print(f"[dim white]{_DESCRIPTION}[/]")
    out.print(f"[#646464]{'─' * 80}[/]\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_banner.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ui/ tests/test_banner.py
git commit -m "feat(ui): port cyberm4fia gradient banner with steganography subtitle"
```

---

## Task 2: Result types (frozen dataclasses)

**Files:**
- Create: `core/__init__.py`
- Create: `core/result.py`
- Test: `tests/test_result.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_result.py
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.result import AnalysisResult, EmbedResult, Signal


def test_embed_result_is_frozen():
    r = EmbedResult(carrier="image_lsb", out_path=Path("/tmp/x.png"), bytes_written=42, encrypted=True)
    with pytest.raises(FrozenInstanceError):
        r.bytes_written = 99


def test_signal_is_frozen_and_has_score():
    s = Signal(name="lsb_chi_square", score=87, detail="strong LSB anomaly")
    assert s.score == 87
    with pytest.raises(FrozenInstanceError):
        s.score = 0


def test_analysis_result_aggregates_signals():
    sigs = (Signal("a", 10, "x"), Signal("b", 90, "y"))
    r = AnalysisResult(analyzer="image_lsb", suspicion=50, signals=sigs, explanation=None)
    assert r.signals[1].name == "b"
    assert r.suspicion == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_result.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core'`.

- [ ] **Step 3: Create `core/__init__.py` and `core/result.py`**

```python
# core/__init__.py
```

```python
# core/result.py
"""Immutable result types returned by carriers and analyzers."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Signal:
    """A single heuristic finding from an analyzer."""
    name: str
    score: int          # 0-100, contribution to overall suspicion
    detail: str


@dataclass(frozen=True, slots=True)
class EmbedResult:
    carrier: str
    out_path: Path
    bytes_written: int
    encrypted: bool


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    analyzer: str
    suspicion: int                       # 0-100 aggregate
    signals: tuple[Signal, ...]
    explanation: str | None              # human-readable; may be AI-generated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_result.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/__init__.py core/result.py tests/test_result.py
git commit -m "feat(core): immutable Signal/EmbedResult/AnalysisResult types"
```

---

## Task 3: Crypto layer (AES-256-GCM + scrypt)

**Files:**
- Create: `core/crypto.py`
- Test: `tests/test_crypto.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crypto.py
import pytest

from core.crypto import (
    AES_GCM_NONCE_LEN,
    DecryptionError,
    KDF_SALT_LEN,
    derive_key,
    encrypt,
    decrypt,
)


def test_derive_key_is_deterministic():
    salt = b"\x01" * KDF_SALT_LEN
    k1 = derive_key("hunter2", salt)
    k2 = derive_key("hunter2", salt)
    assert k1 == k2
    assert len(k1) == 32


def test_derive_key_changes_with_password():
    salt = b"\x01" * KDF_SALT_LEN
    assert derive_key("a", salt) != derive_key("b", salt)


def test_roundtrip_encrypts_and_decrypts():
    salt = b"\x02" * KDF_SALT_LEN
    nonce, ct = encrypt(b"top secret", "pw", salt)
    assert len(nonce) == AES_GCM_NONCE_LEN
    pt = decrypt(ct, "pw", salt, nonce)
    assert pt == b"top secret"


def test_wrong_password_raises():
    salt = b"\x03" * KDF_SALT_LEN
    nonce, ct = encrypt(b"top secret", "pw", salt)
    with pytest.raises(DecryptionError):
        decrypt(ct, "WRONG", salt, nonce)


def test_tampered_ciphertext_raises():
    salt = b"\x04" * KDF_SALT_LEN
    nonce, ct = encrypt(b"top secret", "pw", salt)
    tampered = bytes([ct[0] ^ 0x01]) + ct[1:]
    with pytest.raises(DecryptionError):
        decrypt(tampered, "pw", salt, nonce)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_crypto.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `core/crypto.py`**

```python
# core/crypto.py
"""AES-256-GCM encryption with scrypt password KDF."""
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

AES_KEY_LEN = 32
AES_GCM_NONCE_LEN = 12
KDF_SALT_LEN = 16
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


class DecryptionError(Exception):
    """Raised when authentication fails (wrong password or tampered ciphertext)."""


def derive_key(password: str, salt: bytes) -> bytes:
    if len(salt) != KDF_SALT_LEN:
        raise ValueError(f"salt must be {KDF_SALT_LEN} bytes")
    kdf = Scrypt(salt=salt, length=AES_KEY_LEN, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def encrypt(plaintext: bytes, password: str, salt: bytes) -> tuple[bytes, bytes]:
    """Encrypt plaintext, returning (nonce, ciphertext_with_tag)."""
    key = derive_key(password, salt)
    nonce = os.urandom(AES_GCM_NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, plaintext, associated_data=None)
    return nonce, ct


def decrypt(ciphertext: bytes, password: str, salt: bytes, nonce: bytes) -> bytes:
    key = derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag as e:
        raise DecryptionError("authentication failed") from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_crypto.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/crypto.py tests/test_crypto.py
git commit -m "feat(core): AES-256-GCM crypto with scrypt KDF"
```

---

## Task 4: Payload header (encode/decode)

**Files:**
- Create: `core/payload.py`
- Test: `tests/test_payload.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_payload.py
import pytest

from core.payload import (
    HEADER_OVERHEAD,
    MAGIC,
    InvalidPayloadError,
    pack,
    unpack,
)


def test_pack_unpack_unencrypted():
    blob = pack(payload=b"hello", encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)
    assert blob.startswith(MAGIC)
    assert len(blob) == HEADER_OVERHEAD + len(b"hello")
    parsed = unpack(blob)
    assert parsed.payload == b"hello"
    assert parsed.encrypted is False


def test_pack_unpack_encrypted():
    salt = b"\xaa" * 16
    nonce = b"\xbb" * 12
    blob = pack(payload=b"\x01\x02\x03", encrypted=True, salt=salt, nonce=nonce)
    parsed = unpack(blob)
    assert parsed.encrypted is True
    assert parsed.salt == salt
    assert parsed.nonce == nonce
    assert parsed.payload == b"\x01\x02\x03"


def test_bad_magic_raises():
    with pytest.raises(InvalidPayloadError):
        unpack(b"XXXX" + b"\x00" * 50)


def test_truncated_raises():
    with pytest.raises(InvalidPayloadError):
        unpack(b"STEG")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_payload.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `core/payload.py`**

```python
# core/payload.py
"""Binary payload header: magic | version | flags | length | salt | nonce | payload."""
import struct
from dataclasses import dataclass

MAGIC = b"STEG"
VERSION = 1
FLAG_ENCRYPTED = 0x01

# magic(4) + version(1) + flags(1) + length(4 BE) + salt(16) + nonce(12) = 38
_HEADER_STRUCT = struct.Struct(">4sBBI16s12s")
HEADER_OVERHEAD = _HEADER_STRUCT.size  # 38


class InvalidPayloadError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPayload:
    payload: bytes
    encrypted: bool
    salt: bytes
    nonce: bytes


def pack(*, payload: bytes, encrypted: bool, salt: bytes, nonce: bytes) -> bytes:
    if len(salt) != 16 or len(nonce) != 12:
        raise ValueError("salt must be 16 bytes and nonce 12 bytes")
    flags = FLAG_ENCRYPTED if encrypted else 0
    header = _HEADER_STRUCT.pack(MAGIC, VERSION, flags, len(payload), salt, nonce)
    return header + payload


def unpack(blob: bytes) -> ParsedPayload:
    if len(blob) < HEADER_OVERHEAD:
        raise InvalidPayloadError("blob too short to contain header")
    magic, version, flags, length, salt, nonce = _HEADER_STRUCT.unpack(blob[:HEADER_OVERHEAD])
    if magic != MAGIC:
        raise InvalidPayloadError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise InvalidPayloadError(f"unsupported version: {version}")
    payload = blob[HEADER_OVERHEAD : HEADER_OVERHEAD + length]
    if len(payload) != length:
        raise InvalidPayloadError("truncated payload")
    return ParsedPayload(
        payload=payload,
        encrypted=bool(flags & FLAG_ENCRYPTED),
        salt=salt,
        nonce=nonce,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_payload.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/payload.py tests/test_payload.py
git commit -m "feat(core): payload header pack/unpack with STEG magic"
```

---

## Task 5: Carrier ABC

**Files:**
- Create: `core/carrier.py`
- Test: `tests/test_carrier_abc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_carrier_abc.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_carrier_abc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `core/carrier.py`**

```python
# core/carrier.py
"""Carrier ABC: every embed/extract/analyze module implements this."""
from abc import ABC, abstractmethod
from pathlib import Path

from core.result import AnalysisResult, EmbedResult


class CarrierError(Exception):
    """Base for all carrier-level errors."""


class InsufficientCapacityError(CarrierError):
    pass


class Carrier(ABC):
    name: str = ""
    extensions: tuple[str, ...] = ()
    can_embed: bool = True
    can_extract: bool = True

    @abstractmethod
    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult: ...

    @abstractmethod
    def extract(self, src: Path) -> bytes: ...

    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...

    @abstractmethod
    def capacity(self, src: Path) -> int: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_carrier_abc.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/carrier.py tests/test_carrier_abc.py
git commit -m "feat(core): Carrier ABC and CarrierError hierarchy"
```

---

## Task 6: Analyzer ABC

**Files:**
- Create: `core/analyzer.py`
- Test: `tests/test_analyzer_abc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzer_abc.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_analyzer_abc.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `core/analyzer.py`**

```python
# core/analyzer.py
"""Analyzer ABC: analysis-only modules (signatures, AI triage)."""
from abc import ABC, abstractmethod
from pathlib import Path

from core.result import AnalysisResult


class Analyzer(ABC):
    name: str = ""

    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_analyzer_abc.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/analyzer.py tests/test_analyzer_abc.py
git commit -m "feat(core): Analyzer ABC"
```

---

## Task 7: Registry (auto-discovery + dispatch)

**Files:**
- Create: `modules/__init__.py`
- Create: `registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
from pathlib import Path

import pytest

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


def test_autodiscover_picks_up_modules(tmp_path, monkeypatch):
    # autodiscover() loads every Carrier/Analyzer found under modules/.
    reg = Registry()
    reg.autodiscover()
    # Will be empty until Task 8 onward; this test just asserts the API exists.
    assert isinstance(reg.all_carriers(), list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'registry'`.

- [ ] **Step 3: Create `modules/__init__.py`**

```python
# modules/__init__.py
```

- [ ] **Step 4: Create `registry.py`**

```python
# registry.py
"""Auto-discovery and dispatch for Carrier / Analyzer plug-ins under modules/."""
import importlib
import inspect
import pkgutil
from pathlib import Path

import modules
from core.analyzer import Analyzer
from core.carrier import Carrier


class Registry:
    def __init__(self) -> None:
        self._carriers: list[Carrier] = []
        self._analyzers: list[Analyzer] = []

    # --- registration -------------------------------------------------------
    def register(self, obj: Carrier | Analyzer) -> None:
        if isinstance(obj, Carrier):
            self._carriers.append(obj)
        elif isinstance(obj, Analyzer):
            self._analyzers.append(obj)
        else:
            raise TypeError(f"unsupported plug-in type: {type(obj)!r}")

    def autodiscover(self) -> None:
        """Import every submodule of `modules` and register concrete plug-ins."""
        for mod_info in pkgutil.iter_modules(modules.__path__):
            mod = importlib.import_module(f"modules.{mod_info.name}")
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                if inspect.isabstract(cls):
                    continue
                if issubclass(cls, Carrier):
                    self._carriers.append(cls())
                elif issubclass(cls, Analyzer):
                    self._analyzers.append(cls())

    # --- query --------------------------------------------------------------
    def all_carriers(self) -> list[Carrier]:
        return list(self._carriers)

    def all_analyzers(self) -> list[Analyzer]:
        return list(self._analyzers)

    def select_carriers(self, src: Path) -> list[Carrier]:
        ext = src.suffix.lower()
        return [c for c in self._carriers if ext in c.extensions]

    def select_carrier_for_embed(self, dest: Path) -> Carrier:
        candidates = [c for c in self.select_carriers(dest) if c.can_embed]
        if not candidates:
            raise LookupError(f"no carrier for extension {dest.suffix!r}")
        return candidates[0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_registry.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/__init__.py registry.py tests/test_registry.py
git commit -m "feat: registry with auto-discovery of modules/ plug-ins"
```

---

## Task 8: Module — image_lsb (PNG/BMP)

**Files:**
- Create: `modules/image_lsb.py`
- Test: `tests/test_image_lsb.py`

- [ ] **Step 1: Add image fixture helper in `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
import numpy as np
from PIL import Image


@pytest.fixture
def png_64x64(tmp_path: Path) -> Path:
    """64x64 RGB PNG with deterministic noise."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p = tmp_path / "cover.png"
    Image.fromarray(arr, "RGB").save(p, format="PNG")
    return p
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_image_lsb.py
from pathlib import Path

import pytest

from core.carrier import InsufficientCapacityError
from modules.image_lsb import ImageLsb


def test_roundtrip_embed_extract(png_64x64: Path, tmp_outdir: Path):
    c = ImageLsb()
    payload = b"hello stego world"
    out = tmp_outdir / "out.png"
    res = c.embed(png_64x64, payload, out)
    assert res.bytes_written == len(payload)
    assert out.exists()
    assert c.extract(out) == payload


def test_capacity_is_pixels_times_3_bits_div_8(png_64x64: Path):
    c = ImageLsb()
    # 64*64*3 bits = 12288 bits = 1536 bytes; usable = 1536 - 4 (length prefix)
    assert c.capacity(png_64x64) == 64 * 64 * 3 // 8 - 4


def test_oversized_payload_rejected(png_64x64: Path, tmp_outdir: Path):
    c = ImageLsb()
    too_big = b"x" * (c.capacity(png_64x64) + 1)
    with pytest.raises(InsufficientCapacityError):
        c.embed(png_64x64, too_big, tmp_outdir / "out.png")


def test_analyze_clean_image_low_suspicion(png_64x64: Path):
    c = ImageLsb()
    r = c.analyze(png_64x64)
    assert 0 <= r.suspicion <= 100
    # Random noise should not score very high.
    assert r.suspicion < 70


def test_analyze_stego_image_higher_than_clean(png_64x64: Path, tmp_outdir: Path):
    c = ImageLsb()
    out = tmp_outdir / "stego.png"
    c.embed(png_64x64, b"A" * 800, out)
    clean = c.analyze(png_64x64).suspicion
    stego = c.analyze(out).suspicion
    assert stego >= clean
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_image_lsb.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Create `modules/image_lsb.py`**

```python
# modules/image_lsb.py
"""LSB steganography for lossless images (PNG, BMP)."""
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_LEN_PREFIX = 4  # bytes used to encode payload length in-image


class ImageLsb(Carrier):
    name = "image_lsb"
    extensions = (".png", ".bmp")

    # --- helpers ------------------------------------------------------------
    def _load_rgb(self, src: Path) -> np.ndarray:
        img = Image.open(src).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _bits_from_bytes(self, data: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def _bytes_from_bits(self, bits: np.ndarray) -> bytes:
        return np.packbits(bits).tobytes()

    # --- Carrier API --------------------------------------------------------
    def capacity(self, src: Path) -> int:
        arr = self._load_rgb(src)
        total_bytes = arr.size // 8
        return max(0, total_bytes - _LEN_PREFIX)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        if len(payload) > self.capacity(src):
            raise InsufficientCapacityError(
                f"payload {len(payload)} > capacity {self.capacity(src)}"
            )
        arr = self._load_rgb(src)
        flat = arr.reshape(-1)
        blob = struct.pack(">I", len(payload)) + payload
        bits = self._bits_from_bytes(blob)
        flat = flat.copy()
        flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
        Image.fromarray(flat.reshape(arr.shape), "RGB").save(out, format=src.suffix.lstrip(".").upper())
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        arr = self._load_rgb(src).reshape(-1)
        length_bits = arr[: _LEN_PREFIX * 8] & 1
        (length,) = struct.unpack(">I", self._bytes_from_bits(length_bits))
        start = _LEN_PREFIX * 8
        end = start + length * 8
        if end > arr.size:
            raise ValueError("declared length exceeds image LSB capacity")
        return self._bytes_from_bits(arr[start:end] & 1)

    def analyze(self, src: Path) -> AnalysisResult:
        arr = self._load_rgb(src).reshape(-1)
        # Chi-square style heuristic on pairs of adjacent values (0,1), (2,3), ...
        even = arr[::2].astype(np.int64)
        odd = arr[1::2].astype(np.int64)
        pairs = np.minimum(even, odd)
        denom = np.maximum(even + odd, 1)
        ratio = float(np.mean(pairs * 2 / denom))
        # Bit-1 fraction of LSBs (clean ≈ 0.5; messages often skew this).
        lsb_mean = float(np.mean(arr & 1))
        lsb_dev = abs(lsb_mean - 0.5) * 200  # 0..100
        chi_signal = Signal(
            name="lsb_pair_ratio",
            score=int(min(100, ratio * 100)),
            detail=f"adjacent-pair LSB ratio = {ratio:.3f} (≈1.0 suggests embedding)",
        )
        bias_signal = Signal(
            name="lsb_bit_bias",
            score=int(min(100, lsb_dev)),
            detail=f"LSB mean = {lsb_mean:.3f}",
        )
        # Aggregate: max of signals, capped at 100.
        suspicion = max(chi_signal.score, bias_signal.score)
        return AnalysisResult(
            analyzer=self.name,
            suspicion=suspicion,
            signals=(chi_signal, bias_signal),
            explanation=None,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_image_lsb.py -v`
Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/image_lsb.py tests/test_image_lsb.py tests/conftest.py
git commit -m "feat(modules): image_lsb — PNG/BMP LSB embed/extract/analyze"
```

---

## Task 9: Module — image_jpeg (appended-data path)

> Per spec §15, full DCT-LSB is deferred. This task ships JPEG support via the appended-data technique (embed after `FFD9` end marker) + an analyzer that flags it.

**Files:**
- Create: `modules/image_jpeg.py`
- Test: `tests/test_image_jpeg.py`

- [ ] **Step 1: Add JPEG fixture in `tests/conftest.py`**

Append:

```python
@pytest.fixture
def jpeg_64x64(tmp_path: Path) -> Path:
    rng = np.random.default_rng(seed=7)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p = tmp_path / "cover.jpg"
    Image.fromarray(arr, "RGB").save(p, format="JPEG", quality=85)
    return p
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_image_jpeg.py
from pathlib import Path

from modules.image_jpeg import ImageJpeg


def test_roundtrip_appended(jpeg_64x64: Path, tmp_outdir: Path):
    c = ImageJpeg()
    payload = b"appended jpeg secret"
    out = tmp_outdir / "out.jpg"
    c.embed(jpeg_64x64, payload, out)
    assert c.extract(out) == payload


def test_clean_jpeg_analyze_low(jpeg_64x64: Path):
    c = ImageJpeg()
    assert c.analyze(jpeg_64x64).suspicion < 30


def test_stego_jpeg_analyze_high(jpeg_64x64: Path, tmp_outdir: Path):
    c = ImageJpeg()
    out = tmp_outdir / "stego.jpg"
    c.embed(jpeg_64x64, b"S" * 200, out)
    assert c.analyze(out).suspicion >= 70
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_image_jpeg.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Create `modules/image_jpeg.py`**

```python
# modules/image_jpeg.py
"""JPEG carrier using the appended-data technique (after FFD9 EOI marker).

Full DCT-coefficient LSB is deferred (see spec §15).
"""
import struct
from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_JPEG_EOI = b"\xff\xd9"
_APP_MAGIC = b"STEGAPP"
_LEN_PREFIX = 4


class ImageJpeg(Carrier):
    name = "image_jpeg"
    extensions = (".jpg", ".jpeg")

    def capacity(self, src: Path) -> int:
        # Appended-data path: practically unbounded; expose a generous cap.
        return 10 * 1024 * 1024  # 10 MiB

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        cover = src.read_bytes()
        eoi = cover.rfind(_JPEG_EOI)
        if eoi == -1:
            raise ValueError("not a JPEG (no FFD9)")
        head = cover[: eoi + len(_JPEG_EOI)]
        trailer = _APP_MAGIC + struct.pack(">I", len(payload)) + payload
        out.write_bytes(head + trailer)
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        data = src.read_bytes()
        idx = data.find(_APP_MAGIC)
        if idx == -1:
            raise ValueError("no STEGAPP trailer present")
        cursor = idx + len(_APP_MAGIC)
        (length,) = struct.unpack(">I", data[cursor : cursor + _LEN_PREFIX])
        cursor += _LEN_PREFIX
        return data[cursor : cursor + length]

    def analyze(self, src: Path) -> AnalysisResult:
        data = src.read_bytes()
        eoi = data.rfind(_JPEG_EOI)
        signals: list[Signal] = []
        if eoi == -1:
            return AnalysisResult(self.name, 0, (), None)
        trailer_len = len(data) - (eoi + len(_JPEG_EOI))
        if trailer_len > 0:
            signals.append(
                Signal(
                    name="appended_data",
                    score=min(100, 50 + trailer_len // 32),
                    detail=f"{trailer_len} bytes after JPEG EOI",
                )
            )
        if _APP_MAGIC in data:
            signals.append(
                Signal(name="stegapp_marker", score=95, detail="STEGAPP marker found"),
            )
        suspicion = max((s.score for s in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_image_jpeg.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/image_jpeg.py tests/test_image_jpeg.py tests/conftest.py
git commit -m "feat(modules): image_jpeg appended-data carrier + analyzer"
```

---

## Task 10: Module — audio_wav (LSB)

**Files:**
- Create: `modules/audio_wav.py`
- Test: `tests/test_audio_wav.py`

- [ ] **Step 1: Add WAV fixture in `tests/conftest.py`**

Append:

```python
import wave


@pytest.fixture
def wav_pcm16(tmp_path: Path) -> Path:
    """1 second of 16-bit mono PCM @ 8 kHz, deterministic noise."""
    rng = np.random.default_rng(seed=11)
    samples = rng.integers(-1000, 1000, size=8000, dtype=np.int16)
    p = tmp_path / "cover.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.tobytes())
    return p
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_audio_wav.py
from pathlib import Path

import pytest

from core.carrier import InsufficientCapacityError
from modules.audio_wav import AudioWav


def test_roundtrip(wav_pcm16: Path, tmp_outdir: Path):
    c = AudioWav()
    payload = b"audio stego payload"
    out = tmp_outdir / "out.wav"
    c.embed(wav_pcm16, payload, out)
    assert c.extract(out) == payload


def test_capacity_matches_sample_count(wav_pcm16: Path):
    c = AudioWav()
    # 8000 samples → 8000 bits → 1000 bytes - 4 (length prefix) = 996
    assert c.capacity(wav_pcm16) == 8000 // 8 - 4


def test_oversized_rejected(wav_pcm16: Path, tmp_outdir: Path):
    c = AudioWav()
    with pytest.raises(InsufficientCapacityError):
        c.embed(wav_pcm16, b"x" * (c.capacity(wav_pcm16) + 1), tmp_outdir / "out.wav")


def test_analyze_returns_result(wav_pcm16: Path):
    c = AudioWav()
    r = c.analyze(wav_pcm16)
    assert 0 <= r.suspicion <= 100
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_audio_wav.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Create `modules/audio_wav.py`**

```python
# modules/audio_wav.py
"""WAV PCM LSB steganography."""
import struct
import wave
from pathlib import Path

import numpy as np

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_LEN_PREFIX = 4


class AudioWav(Carrier):
    name = "audio_wav"
    extensions = (".wav",)

    def _read(self, src: Path) -> tuple[np.ndarray, wave._wave_params]:
        with wave.open(str(src), "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
        if params.sampwidth != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        samples = np.frombuffer(frames, dtype=np.int16).copy()
        return samples, params

    def capacity(self, src: Path) -> int:
        samples, _ = self._read(src)
        return max(0, samples.size // 8 - _LEN_PREFIX)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        if len(payload) > self.capacity(src):
            raise InsufficientCapacityError("payload exceeds WAV LSB capacity")
        samples, params = self._read(src)
        blob = struct.pack(">I", len(payload)) + payload
        bits = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
        samples[: bits.size] = (samples[: bits.size] & ~np.int16(1)) | bits.astype(np.int16)
        with wave.open(str(out), "wb") as w:
            w.setparams(params)
            w.writeframes(samples.tobytes())
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        samples, _ = self._read(src)
        length_bits = samples[: _LEN_PREFIX * 8].astype(np.uint8) & 1
        (length,) = struct.unpack(">I", np.packbits(length_bits).tobytes())
        start = _LEN_PREFIX * 8
        end = start + length * 8
        return np.packbits(samples[start:end].astype(np.uint8) & 1).tobytes()

    def analyze(self, src: Path) -> AnalysisResult:
        samples, _ = self._read(src)
        lsb_mean = float(np.mean(samples & 1))
        dev = abs(lsb_mean - 0.5) * 200
        sig = Signal(name="wav_lsb_bias", score=int(min(100, dev)), detail=f"LSB mean={lsb_mean:.3f}")
        return AnalysisResult(self.name, sig.score, (sig,), None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_audio_wav.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add modules/audio_wav.py tests/test_audio_wav.py tests/conftest.py
git commit -m "feat(modules): audio_wav — 16-bit PCM WAV LSB"
```

---

## Task 11: Module — text_zerowidth

**Files:**
- Create: `modules/text_zerowidth.py`
- Test: `tests/test_text_zerowidth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_text_zerowidth.py
from pathlib import Path

from modules.text_zerowidth import TextZeroWidth


def test_roundtrip(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("Hello, world! This is cover text.\n", encoding="utf-8")
    c = TextZeroWidth()
    out = tmp_outdir / "out.txt"
    payload = b"hi"
    c.embed(src, payload, out)
    assert c.extract(out) == payload


def test_analyze_clean_low(tmp_path: Path):
    src = tmp_path / "clean.txt"
    src.write_text("plain ASCII only", encoding="utf-8")
    assert TextZeroWidth().analyze(src).suspicion < 20


def test_analyze_stego_high(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("some carrier text", encoding="utf-8")
    out = tmp_outdir / "out.txt"
    TextZeroWidth().embed(src, b"abc", out)
    assert TextZeroWidth().analyze(out).suspicion >= 70
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_text_zerowidth.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `modules/text_zerowidth.py`**

```python
# modules/text_zerowidth.py
"""Zero-width unicode steganography.

Encodes payload bits as ZWSP (​ = 0) and ZWNJ (‌ = 1).
Appended at the end of the cover text.
"""
from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_ZERO = "​"
_ONE = "‌"
_TERMINATOR = "‍"   # marks end of encoded run


class TextZeroWidth(Carrier):
    name = "text_zerowidth"
    extensions = (".txt", ".md")

    def capacity(self, src: Path) -> int:
        return 1_000_000  # effectively unbounded; appended

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        cover = src.read_text(encoding="utf-8")
        bits = "".join(f"{b:08b}" for b in payload)
        encoded = "".join(_ONE if c == "1" else _ZERO for c in bits) + _TERMINATOR
        out.write_text(cover + encoded, encoding="utf-8")
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        text = src.read_text(encoding="utf-8")
        run = []
        for ch in text:
            if ch == _TERMINATOR:
                break
            if ch in (_ZERO, _ONE):
                run.append("1" if ch == _ONE else "0")
        bits = "".join(run)
        out = bytearray()
        for i in range(0, len(bits) - 7, 8):
            out.append(int(bits[i : i + 8], 2))
        return bytes(out)

    def analyze(self, src: Path) -> AnalysisResult:
        text = src.read_text(encoding="utf-8", errors="ignore")
        count = sum(text.count(ch) for ch in (_ZERO, _ONE, _TERMINATOR))
        if count == 0:
            return AnalysisResult(self.name, 0, (), None)
        score = min(100, 50 + count // 4)
        sig = Signal(name="zero_width_chars", score=score, detail=f"{count} zero-width chars")
        return AnalysisResult(self.name, score, (sig,), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_text_zerowidth.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/text_zerowidth.py tests/test_text_zerowidth.py
git commit -m "feat(modules): text_zerowidth — ZWSP/ZWNJ unicode steganography"
```

---

## Task 12: Module — text_whitespace

**Files:**
- Create: `modules/text_whitespace.py`
- Test: `tests/test_text_whitespace.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_text_whitespace.py
from pathlib import Path

from modules.text_whitespace import TextWhitespace


def test_roundtrip(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("line one\nline two\nline three\nline four\n", encoding="utf-8")
    c = TextWhitespace()
    out = tmp_outdir / "out.txt"
    payload = b"x"
    c.embed(src, payload, out)
    assert c.extract(out) == payload


def test_analyze_detects_trailing_whitespace(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("a\nb\nc\nd\ne\nf\ng\nh\n", encoding="utf-8")
    out = tmp_outdir / "out.txt"
    TextWhitespace().embed(src, b"!", out)
    assert TextWhitespace().analyze(out).suspicion >= 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_text_whitespace.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `modules/text_whitespace.py`**

```python
# modules/text_whitespace.py
"""Trailing-whitespace steganography.

Each line carries one bit: trailing TAB = 1, trailing SPACE = 0.
Requires cover to have at least 8 * len(payload) lines.
"""
from pathlib import Path

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_ONE = "\t"
_ZERO = " "


class TextWhitespace(Carrier):
    name = "text_whitespace"
    extensions = (".txt", ".md")

    def capacity(self, src: Path) -> int:
        lines = src.read_text(encoding="utf-8").splitlines()
        return max(0, len(lines) // 8)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        cap = self.capacity(src)
        if len(payload) > cap:
            raise InsufficientCapacityError(f"payload {len(payload)} > capacity {cap}")
        lines = src.read_text(encoding="utf-8").splitlines()
        bits = [(b >> (7 - i)) & 1 for b in payload for i in range(8)]
        new_lines = []
        for idx, line in enumerate(lines):
            if idx < len(bits):
                marker = _ONE if bits[idx] else _ZERO
                new_lines.append(line.rstrip(" \t") + marker)
            else:
                new_lines.append(line)
        out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        lines = src.read_text(encoding="utf-8").splitlines()
        bits = []
        for line in lines:
            if line.endswith(_ONE):
                bits.append(1)
            elif line.endswith(_ZERO):
                bits.append(0)
            else:
                break
        out = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            out.append(byte)
        return bytes(out)

    def analyze(self, src: Path) -> AnalysisResult:
        lines = src.read_text(encoding="utf-8").splitlines()
        if not lines:
            return AnalysisResult(self.name, 0, (), None)
        flagged = sum(1 for ln in lines if ln.endswith((_ONE, _ZERO)) and ln.strip())
        ratio = flagged / len(lines)
        score = int(min(100, ratio * 100))
        sig = Signal(name="trailing_whitespace_ratio", score=score, detail=f"{flagged}/{len(lines)}")
        return AnalysisResult(self.name, score, (sig,), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_text_whitespace.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/text_whitespace.py tests/test_text_whitespace.py
git commit -m "feat(modules): text_whitespace — trailing whitespace bit encoding"
```

---

## Task 13: Module — filestruct_exif

**Files:**
- Create: `modules/filestruct_exif.py`
- Test: `tests/test_filestruct_exif.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_filestruct_exif.py
from pathlib import Path

from modules.filestruct_exif import FilestructExif


def test_roundtrip_jpeg_exif_usercomment(jpeg_64x64: Path, tmp_outdir: Path):
    c = FilestructExif()
    out = tmp_outdir / "out.jpg"
    payload = b"exif secret"
    c.embed(jpeg_64x64, payload, out)
    assert c.extract(out) == payload


def test_analyze_flags_present_usercomment(jpeg_64x64: Path, tmp_outdir: Path):
    c = FilestructExif()
    out = tmp_outdir / "out.jpg"
    c.embed(jpeg_64x64, b"x" * 30, out)
    assert c.analyze(out).suspicion >= 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_filestruct_exif.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `modules/filestruct_exif.py`**

```python
# modules/filestruct_exif.py
"""EXIF UserComment-based steganography (JPEG/TIFF)."""
from pathlib import Path

import piexif

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_MAGIC = b"STEGEXIF"


class FilestructExif(Carrier):
    name = "filestruct_exif"
    extensions = (".jpg", ".jpeg", ".tiff")

    def capacity(self, src: Path) -> int:
        return 60_000  # EXIF UserComment practical limit

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        try:
            exif_dict = piexif.load(str(src))
        except piexif.InvalidImageDataError:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = _MAGIC + payload
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(src), str(out))
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        exif_dict = piexif.load(str(src))
        comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        if not comment.startswith(_MAGIC):
            raise ValueError("no STEGEXIF marker in EXIF UserComment")
        return comment[len(_MAGIC) :]

    def analyze(self, src: Path) -> AnalysisResult:
        try:
            exif_dict = piexif.load(str(src))
        except piexif.InvalidImageDataError:
            return AnalysisResult(self.name, 0, (), None)
        comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        signals = []
        if comment.startswith(_MAGIC):
            signals.append(Signal("stegexif_marker", 95, "STEGEXIF marker in UserComment"))
        elif len(comment) > 64:
            signals.append(Signal("large_usercomment", 60, f"UserComment len={len(comment)}"))
        suspicion = max((s.score for s in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_filestruct_exif.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/filestruct_exif.py tests/test_filestruct_exif.py
git commit -m "feat(modules): filestruct_exif — EXIF UserComment carrier"
```

---

## Task 14: Module — filestruct_appended (polyglot detection)

**Files:**
- Create: `modules/filestruct_appended.py`
- Test: `tests/test_filestruct_appended.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_filestruct_appended.py
from pathlib import Path

from modules.filestruct_appended import FilestructAppended


def test_detects_zip_after_png(png_64x64: Path, tmp_path: Path):
    poly = tmp_path / "poly.png"
    poly.write_bytes(png_64x64.read_bytes() + b"PK\x03\x04" + b"\x00" * 30)
    r = FilestructAppended().analyze(poly)
    assert r.suspicion >= 80


def test_clean_png_no_appended(png_64x64: Path):
    r = FilestructAppended().analyze(png_64x64)
    assert r.suspicion < 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_filestruct_appended.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `modules/filestruct_appended.py`**

```python
# modules/filestruct_appended.py
"""Detect data appended after a file's structural end marker.

Analyzer-only (uses Carrier base for registry uniformity but disables embed/extract).
"""
from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_END_MARKERS = {
    ".png": b"IEND\xaeB`\x82",
    ".jpg": b"\xff\xd9",
    ".jpeg": b"\xff\xd9",
    ".gif": b";",
    ".pdf": b"%%EOF",
}

_APPENDED_SIGNATURES = {
    b"PK\x03\x04": "ZIP archive",
    b"Rar!\x1a\x07": "RAR archive",
    b"7z\xbc\xaf\x27\x1c": "7z archive",
    b"MZ": "PE executable",
    b"\x7fELF": "ELF executable",
}


class FilestructAppended(Carrier):
    name = "filestruct_appended"
    extensions = tuple(_END_MARKERS.keys())
    can_embed = False
    can_extract = False

    def capacity(self, src: Path) -> int:
        return 0

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        raise NotImplementedError("filestruct_appended is analysis-only")

    def extract(self, src: Path) -> bytes:
        raise NotImplementedError("filestruct_appended is analysis-only")

    def analyze(self, src: Path) -> AnalysisResult:
        ext = src.suffix.lower()
        marker = _END_MARKERS.get(ext)
        if marker is None:
            return AnalysisResult(self.name, 0, (), None)
        data = src.read_bytes()
        idx = data.rfind(marker)
        signals: list[Signal] = []
        if idx == -1:
            return AnalysisResult(self.name, 0, (), None)
        trailer = data[idx + len(marker) :]
        if not trailer:
            return AnalysisResult(self.name, 0, (), None)
        signals.append(
            Signal(name="appended_data", score=70, detail=f"{len(trailer)} bytes after EOF marker")
        )
        for sig, label in _APPENDED_SIGNATURES.items():
            if trailer.startswith(sig):
                signals.append(Signal(name=f"embedded_{label.replace(' ', '_')}", score=95, detail=label))
                break
        suspicion = max(s.score for s in signals)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_filestruct_appended.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/filestruct_appended.py tests/test_filestruct_appended.py
git commit -m "feat(modules): filestruct_appended polyglot detector"
```

---

## Task 15: Module — signatures (known stego tools)

**Files:**
- Create: `modules/signatures.py`
- Test: `tests/test_signatures.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signatures.py
from pathlib import Path

from modules.signatures import Signatures


def test_detects_openstego_marker(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100 + b"OPENSTEGO" + b"\x00" * 50)
    assert Signatures().analyze(p).suspicion >= 80


def test_clean_returns_zero(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    assert Signatures().analyze(p).suspicion == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_signatures.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `modules/signatures.py`**

```python
# modules/signatures.py
"""Known-tool signature detection. Declarative — add a row, get detection."""
from dataclasses import dataclass
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal


@dataclass(frozen=True, slots=True)
class _Sig:
    name: str
    needle: bytes
    score: int
    detail: str


_SIGNATURES: tuple[_Sig, ...] = (
    _Sig("openstego", b"OPENSTEGO", 90, "OpenStego marker"),
    _Sig("steghide", b"shsh", 70, "possible steghide artifact"),
    _Sig("outguess", b"OutGuess", 85, "OutGuess marker"),
    _Sig("stegapp", b"STEGAPP", 95, "STEGAPP (this tool's JPEG trailer)"),
    _Sig("stegexif", b"STEGEXIF", 95, "STEGEXIF (this tool's EXIF marker)"),
)


class Signatures(Analyzer):
    name = "signatures"

    def analyze(self, src: Path) -> AnalysisResult:
        data = src.read_bytes()
        hits = [
            Signal(name=s.name, score=s.score, detail=s.detail)
            for s in _SIGNATURES
            if s.needle in data
        ]
        suspicion = max((h.score for h in hits), default=0)
        return AnalysisResult(self.name, suspicion, tuple(hits), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_signatures.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/signatures.py tests/test_signatures.py
git commit -m "feat(modules): signatures — known stego-tool marker detection"
```

---

## Task 16: Module — ai_triage (pluggable, heuristic fallback)

**Files:**
- Create: `modules/ai_triage.py`
- Test: `tests/test_ai_triage.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ai_triage.py
from pathlib import Path

from core.result import Signal
from modules.ai_triage import AITriage, set_provider


def test_heuristic_fallback_uses_signals(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"x")
    t = AITriage()
    # No provider registered → heuristic-only.
    signals = (Signal("a", 30, ""), Signal("b", 80, ""))
    r = t.score_from_signals(signals)
    assert r.suspicion == 80
    assert r.explanation is not None


def test_pluggable_provider_overrides_score(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"x")

    def fake_provider(file_info: dict, signals: list[Signal]) -> tuple[int, str]:
        return 42, "ai says 42"

    set_provider(fake_provider)
    try:
        r = AITriage().analyze(p)
        assert r.suspicion == 42
        assert r.explanation == "ai says 42"
    finally:
        set_provider(None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_ai_triage.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `modules/ai_triage.py`**

```python
# modules/ai_triage.py
"""Pluggable AI triage analyzer. Heuristic-only if no provider registered.

Providers are user-supplied callables. Any provider that needs an API key MUST
read it via `config.get_secret(...)` — no literal keys in this file.
"""
from pathlib import Path
from typing import Callable, Optional

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

AIProvider = Callable[[dict, list[Signal]], tuple[int, str]]
_provider: Optional[AIProvider] = None


def set_provider(provider: Optional[AIProvider]) -> None:
    """Register or clear the AI provider used by AITriage.analyze()."""
    global _provider
    _provider = provider


def get_provider() -> Optional[AIProvider]:
    return _provider


class AITriage(Analyzer):
    name = "ai_triage"

    def score_from_signals(self, signals: tuple[Signal, ...]) -> AnalysisResult:
        if not signals:
            return AnalysisResult(self.name, 0, (), "no signals")
        suspicion = max(s.score for s in signals)
        explanation = "; ".join(f"{s.name}:{s.score}" for s in signals)
        return AnalysisResult(self.name, suspicion, signals, explanation)

    def analyze(self, src: Path) -> AnalysisResult:
        info = {"path": str(src), "size": src.stat().st_size if src.exists() else 0}
        signals: list[Signal] = []
        if _provider is not None:
            score, explanation = _provider(info, signals)
            return AnalysisResult(self.name, score, tuple(signals), explanation)
        return self.score_from_signals(tuple(signals))
```

**Note for future provider implementations:** A concrete provider (e.g., NVIDIA NIM, Anthropic, OpenAI) lives in its own file (`modules/ai_provider_nim.py`, etc.) and is wired up by the caller via `set_provider(...)`. The provider reads its key via `config.get_secret("NVIDIA_NIM_API_KEY")` — never hardcode keys.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_ai_triage.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/ai_triage.py tests/test_ai_triage.py
git commit -m "feat(modules): ai_triage — pluggable AI provider with heuristic fallback"
```

---

## Task 17: Report writer (JSON + HTML)

**Files:**
- Create: `report/__init__.py`
- Create: `report/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
import json
from pathlib import Path

from core.result import AnalysisResult, Signal
from report.report import write_html, write_json


def _sample():
    return [
        (Path("/tmp/a.png"), AnalysisResult("image_lsb", 80, (Signal("x", 80, "d"),), "high")),
        (Path("/tmp/b.png"), AnalysisResult("image_lsb", 10, (), None)),
    ]


def test_write_json_writes_valid_file(tmp_path: Path):
    out = tmp_path / "r.json"
    write_json(_sample(), out)
    data = json.loads(out.read_text())
    assert len(data["files"]) == 2
    assert data["files"][0]["suspicion"] == 80


def test_write_html_writes_renderable(tmp_path: Path):
    out = tmp_path / "r.html"
    write_html(_sample(), out)
    text = out.read_text()
    assert "<html" in text.lower()
    assert "image_lsb" in text
    assert "/tmp/a.png" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `report/__init__.py` and `report/report.py`**

```python
# report/__init__.py
```

```python
# report/report.py
"""JSON and HTML report writers for batch scans."""
import json
from pathlib import Path

from core.result import AnalysisResult


def _serialize(item: tuple[Path, AnalysisResult]) -> dict:
    path, r = item
    return {
        "path": str(path),
        "analyzer": r.analyzer,
        "suspicion": r.suspicion,
        "signals": [{"name": s.name, "score": s.score, "detail": s.detail} for s in r.signals],
        "explanation": r.explanation,
    }


def write_json(results: list[tuple[Path, AnalysisResult]], out: Path) -> None:
    out.write_text(json.dumps({"files": [_serialize(r) for r in results]}, indent=2), encoding="utf-8")


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>steganography report</title>
<style>
 body{{font-family:system-ui,monospace;background:#111;color:#eee;padding:24px}}
 h1{{font-weight:300}}
 table{{width:100%;border-collapse:collapse}}
 th,td{{padding:8px;border-bottom:1px solid #333;text-align:left;font-size:14px}}
 .high{{color:#f55}} .mid{{color:#fb3}} .low{{color:#6f6}}
 details{{margin:4px 0}}
</style></head><body>
<h1>steganography — scan report</h1>
<table><thead><tr><th>file</th><th>analyzer</th><th>suspicion</th><th>signals</th></tr></thead><tbody>
{rows}
</tbody></table></body></html>"""


def _row(item: tuple[Path, AnalysisResult]) -> str:
    path, r = item
    cls = "high" if r.suspicion >= 70 else "mid" if r.suspicion >= 30 else "low"
    sigs = "<br>".join(f"{s.name}({s.score}): {s.detail}" for s in r.signals) or "—"
    return (
        f"<tr><td>{path}</td><td>{r.analyzer}</td>"
        f"<td class='{cls}'>{r.suspicion}</td><td>{sigs}</td></tr>"
    )


def write_html(results: list[tuple[Path, AnalysisResult]], out: Path) -> None:
    rows = "\n".join(_row(r) for r in results)
    out.write_text(_HTML_TEMPLATE.format(rows=rows), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_report.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add report/ tests/test_report.py
git commit -m "feat(report): JSON and HTML scan report writers"
```

---

## Task 18: CLI (embed / extract / analyze / scan / list-modules)

**Files:**
- Create: `cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
    )


def test_list_modules_exits_zero():
    r = _run(["list-modules"])
    assert r.returncode == 0
    assert "image_lsb" in r.stdout


def test_embed_extract_roundtrip(png_64x64: Path, tmp_outdir: Path, tmp_path: Path):
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"cli round trip")
    out = tmp_outdir / "out.png"
    extracted = tmp_path / "got.bin"
    r1 = _run(["embed", "--in", str(secret), "--carrier", str(png_64x64), "--out", str(out)])
    assert r1.returncode == 0, r1.stderr
    r2 = _run(["extract", "--in", str(out), "--out", str(extracted)])
    assert r2.returncode == 0, r2.stderr
    assert extracted.read_bytes() == b"cli round trip"


def test_analyze_returns_json_when_flag(png_64x64: Path):
    r = _run(["analyze", "--in", str(png_64x64), "--json"])
    assert r.returncode == 0
    assert r.stdout.strip().startswith("{")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `cli.py` not found / no commands.

- [ ] **Step 3: Create `cli.py`**

```python
# cli.py
"""steganography CLI: embed | extract | analyze | scan | list-modules."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from core.crypto import KDF_SALT_LEN, decrypt, encrypt
from core.payload import pack, unpack
from registry import Registry
from report.report import write_html, write_json
from ui.banner import print_gradient_banner


def _build_registry() -> Registry:
    reg = Registry()
    reg.autodiscover()
    return reg


def _maybe_password() -> str | None:
    return os.environ.get("STEGANO_PASSWORD") or None


def cmd_embed(args: argparse.Namespace) -> int:
    reg = _build_registry()
    carrier = reg.select_carrier_for_embed(Path(args.carrier))
    raw = Path(args.input).read_bytes()
    password = args.password or _maybe_password()
    if password:
        salt = os.urandom(KDF_SALT_LEN)
        nonce, ct = encrypt(raw, password, salt)
        blob = pack(payload=ct, encrypted=True, salt=salt, nonce=nonce)
    else:
        blob = pack(payload=raw, encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)
    res = carrier.embed(Path(args.carrier), blob, Path(args.out))
    print(f"embedded {res.bytes_written} bytes via {res.carrier} → {res.out_path}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    reg = _build_registry()
    carriers = reg.select_carriers(Path(args.input))
    if not carriers:
        print(f"no carrier for {args.input}", file=sys.stderr)
        return 2
    blob = carriers[0].extract(Path(args.input))
    parsed = unpack(blob)
    password = args.password or _maybe_password()
    if parsed.encrypted:
        if not password:
            print("payload encrypted; set --password or STEGANO_PASSWORD", file=sys.stderr)
            return 2
        data = decrypt(parsed.payload, password, parsed.salt, parsed.nonce)
    else:
        data = parsed.payload
    Path(args.out).write_bytes(data)
    print(f"extracted {len(data)} bytes → {args.out}")
    return 0


def _analyze_file(reg: Registry, path: Path) -> list:
    results = []
    for c in reg.select_carriers(path):
        results.append(c.analyze(path))
    for a in reg.all_analyzers():
        results.append(a.analyze(path))
    return results


def cmd_analyze(args: argparse.Namespace) -> int:
    reg = _build_registry()
    results = _analyze_file(reg, Path(args.input))
    if args.json:
        print(json.dumps([
            {"analyzer": r.analyzer, "suspicion": r.suspicion,
             "signals": [s.__dict__ for s in r.signals],
             "explanation": r.explanation}
            for r in results
        ], indent=2))
    else:
        for r in results:
            print(f"  [{r.suspicion:3d}] {r.analyzer}")
            for s in r.signals:
                print(f"        - {s.name}({s.score}): {s.detail}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    reg = _build_registry()
    root = Path(args.dir)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        for r in _analyze_file(reg, p):
            rows.append((p, r))
    out = Path(args.out)
    if args.report == "json":
        write_json(rows, out)
    else:
        write_html(rows, out)
    print(f"scanned {len(rows)} results → {out}")
    return 0


def cmd_list_modules(_: argparse.Namespace) -> int:
    reg = _build_registry()
    print("carriers:")
    for c in reg.all_carriers():
        print(f"  {c.name}  {c.extensions}")
    print("analyzers:")
    for a in reg.all_analyzers():
        print(f"  {a.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="steganography")
    p.add_argument("--quiet", action="store_true", help="suppress banner")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--in", dest="input", required=True)
    e.add_argument("--carrier", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--password")
    e.set_defaults(fn=cmd_embed)

    x = sub.add_parser("extract")
    x.add_argument("--in", dest="input", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--password")
    x.set_defaults(fn=cmd_extract)

    a = sub.add_parser("analyze")
    a.add_argument("--in", dest="input", required=True)
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("scan")
    s.add_argument("--dir", required=True)
    s.add_argument("--report", choices=("json", "html"), default="html")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_scan)

    lm = sub.add_parser("list-modules")
    lm.set_defaults(fn=cmd_list_modules)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.quiet and args.cmd != "list-modules":
        print_gradient_banner()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_cli.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat(cli): embed/extract/analyze/scan/list-modules subcommands"
```

---

## Task 19: Integration round-trip suite

**Files:**
- Create: `tests/test_integration_roundtrip.py`

- [ ] **Step 1: Write the integration tests**

```python
# tests/test_integration_roundtrip.py
"""End-to-end: encrypted payload through the registry-selected carrier."""
import os
from pathlib import Path

import pytest

from core.crypto import KDF_SALT_LEN, decrypt, encrypt
from core.payload import pack, unpack
from registry import Registry


@pytest.fixture
def registry() -> Registry:
    r = Registry()
    r.autodiscover()
    return r


def _wrap(payload: bytes, password: str | None) -> bytes:
    if password:
        salt = os.urandom(KDF_SALT_LEN)
        nonce, ct = encrypt(payload, password, salt)
        return pack(payload=ct, encrypted=True, salt=salt, nonce=nonce)
    return pack(payload=payload, encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)


def _unwrap(blob: bytes, password: str | None) -> bytes:
    parsed = unpack(blob)
    if parsed.encrypted:
        return decrypt(parsed.payload, password or "", parsed.salt, parsed.nonce)
    return parsed.payload


def test_png_lsb_encrypted_roundtrip(registry, png_64x64: Path, tmp_outdir: Path):
    carrier = registry.select_carrier_for_embed(Path("x.png"))
    blob = _wrap(b"top secret bytes", "hunter2")
    out = tmp_outdir / "stego.png"
    carrier.embed(png_64x64, blob, out)
    recovered = _unwrap(carrier.extract(out), "hunter2")
    assert recovered == b"top secret bytes"


def test_wav_encrypted_roundtrip(registry, wav_pcm16: Path, tmp_outdir: Path):
    carrier = registry.select_carrier_for_embed(Path("x.wav"))
    blob = _wrap(b"audio secret", "p@ss")
    out = tmp_outdir / "stego.wav"
    carrier.embed(wav_pcm16, blob, out)
    recovered = _unwrap(carrier.extract(out), "p@ss")
    assert recovered == b"audio secret"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `./venv/bin/pytest tests/test_integration_roundtrip.py -v`
Expected: 2 PASS.

- [ ] **Step 3: Full suite + coverage gate**

Run: `./venv/bin/pytest --cov`
Expected: all tests PASS, coverage ≥ 80% (fail_under enforced via `pyproject.toml`).
If coverage falls short, add tests for the lowest-covered modules until the gate passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_roundtrip.py
git commit -m "test: end-to-end encrypted roundtrips through registry"
```

---

## Task 20: README + ethical-use note

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write final `README.md`**

```markdown
# steganography

Dual-purpose steganography toolkit — embed/extract & steganalysis.

A standalone Python 3.11+ companion to [cyberm4fia-scanner](https://github.com/erkanrzgc/cyberm4fia-scanner). Shares the modular `modules/` plug-in style, `rich` CLI, and the cyberm4fia gradient banner.

## Features

- **Embed / extract** across image (PNG/BMP LSB, JPEG appended), audio (WAV LSB), text (zero-width unicode, trailing whitespace), and file-structure (EXIF UserComment) carriers
- **AES-256-GCM** encryption layer with scrypt password KDF
- **Steganalysis**: per-carrier statistical heuristics + known-tool signature detection + polyglot/appended-data detection
- **Pluggable AI triage** (any callable matching the `AIProvider` protocol)
- **Batch directory scan** with JSON and HTML reports
- Auto-discovered `modules/` — drop a new file in to add a technique

## Install

```bash
git clone <repo> steganography && cd steganography
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Embed
python cli.py embed --in secret.bin --carrier cover.png --out stego.png [--password pw]

# Extract
python cli.py extract --in stego.png --out recovered.bin [--password pw]

# Analyze a single file
python cli.py analyze --in suspect.png [--json]

# Batch scan a directory
python cli.py scan --dir ./samples --report html --out report.html

# List discovered modules
python cli.py list-modules
```

Set `STEGANO_PASSWORD` env var to avoid passing `--password` on the command line.

## Architecture

See [design spec](docs/superpowers/specs/2026-05-23-steganography-design.md). Repo root = import root; `modules/` is auto-discovered by `registry.py`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest --cov
```

Coverage gate: 80% (enforced via `pyproject.toml`).

## Ethical use

This toolkit is for **authorized** red-team operations, blue-team / DFIR analysis, CTF play, security research, and education. Do not use it against systems or data you are not authorized to test. The author assumes no responsibility for misuse.

## License

MIT.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: full README with usage, architecture pointer, ethical use"
```

---

## Self-review (writing-plans checklist)

**Spec coverage:**
- Scope §2 carriers → Tasks 8–14 (every carrier covered).
- Scope §2 cross-cutting → crypto Task 3, AI triage Task 16, signatures Task 15, batch+report Tasks 17 & 18 `scan`.
- §3 architecture → flat layout established in Tasks 0, 1, 5–7; modules added Tasks 8–16.
- §4 interfaces → Tasks 2 (results), 5 (Carrier), 6 (Analyzer).
- §5 data flow → embed/extract wiring in CLI (Task 18); analyze in Task 18 `cmd_analyze`; scan in `cmd_scan`.
- §6 header → Task 4.
- §7 crypto → Task 3.
- §8 AI triage → Task 16 (protocol + heuristic fallback).
- §9 signatures → Task 15.
- §10 reporting → Task 17.
- §11 CLI → Task 18.
- §12 error handling → typed exceptions in crypto/payload/carrier; CLI returns non-zero on failures.
- §13 testing → per-task TDD + integration Task 19 + coverage gate enforced.
- §14 dependencies → Task 0.
- §15 deferred → JPEG DCT explicitly out of v1 (Task 9 uses appended-data only); reference AI client out (Task 16 protocol only). Both choices documented in plan.
- §16 success criteria → round-trip tests (Tasks 8/10/19), encrypted roundtrip (Task 19), suspicion bands (per-carrier analyze tests), new-module dropping (registry autodiscover test Task 7), banner (Task 1), coverage gate (Task 19 Step 3).

**Placeholder scan:** none — every code block is concrete; no TBD/TODO; no "similar to Task N".

**Type consistency:**
- `Signal(name, score, detail)` — used identically in Tasks 2, 8–16, 17.
- `EmbedResult(carrier, out_path, bytes_written, encrypted)` — used identically across carriers.
- `AnalysisResult(analyzer, suspicion, signals, explanation)` — consistent.
- `pack(*, payload, encrypted, salt, nonce)` and `unpack(blob) → ParsedPayload(payload, encrypted, salt, nonce)` — match across Tasks 4, 18, 19.
- `encrypt(plaintext, password, salt) → (nonce, ct)` and `decrypt(ct, password, salt, nonce) → bytes` — match Tasks 3, 18, 19.
- `Registry.select_carrier_for_embed`, `select_carriers`, `all_carriers`, `all_analyzers` — defined in Task 7, used Tasks 18, 19.

No drift detected.
