"""Password-unlocked, content-addressed local evidence vault."""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

from core.database import Database, utc_now

try:
    from nacl.bindings import (
        crypto_secretstream_xchacha20poly1305_ABYTES,
        crypto_secretstream_xchacha20poly1305_HEADERBYTES,
        crypto_secretstream_xchacha20poly1305_init_pull,
        crypto_secretstream_xchacha20poly1305_init_push,
        crypto_secretstream_xchacha20poly1305_KEYBYTES,
        crypto_secretstream_xchacha20poly1305_pull,
        crypto_secretstream_xchacha20poly1305_push,
        crypto_secretstream_xchacha20poly1305_state,
        crypto_secretstream_xchacha20poly1305_TAG_FINAL,
        crypto_secretstream_xchacha20poly1305_TAG_MESSAGE,
    )
except ImportError:  # pragma: no cover - exercised when optional runtime is absent
    crypto_secretstream_xchacha20poly1305_KEYBYTES = 32
    crypto_secretstream_xchacha20poly1305_HEADERBYTES = 24
    crypto_secretstream_xchacha20poly1305_ABYTES = 17
    crypto_secretstream_xchacha20poly1305_TAG_MESSAGE = 0
    crypto_secretstream_xchacha20poly1305_TAG_FINAL = 3
    crypto_secretstream_xchacha20poly1305_init_pull = None  # type: ignore[assignment]
    crypto_secretstream_xchacha20poly1305_init_push = None  # type: ignore[assignment]
    crypto_secretstream_xchacha20poly1305_pull = None  # type: ignore[assignment]
    crypto_secretstream_xchacha20poly1305_push = None  # type: ignore[assignment]
    crypto_secretstream_xchacha20poly1305_state = None  # type: ignore[misc,assignment]


_MAGIC = b"SVLT1"
_CHECK = b"steganography-vault-check-v1"
_CHUNK_SIZE = 1024 * 1024
_DEFAULT_MEMORY_KIB = 64 * 1024
_DEFAULT_ITERATIONS = 3
_DEFAULT_LANES = 1


class VaultError(Exception):
    """Base class for local vault failures."""


class VaultLockedError(VaultError):
    """The requested operation needs an unlocked in-memory key."""


class VaultAuthenticationError(VaultError):
    """The supplied password is wrong or the check record was modified."""


class VaultIntegrityError(VaultError):
    """Encrypted evidence was truncated or modified."""


class VaultUnavailableError(VaultError):
    """The required libsodium bindings are not installed."""


class VaultService:
    """Encrypt files independently with secretstream and deduplicate by SHA-256."""

    def __init__(self, database: Database, root: Path) -> None:
        self.database = database
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._key: bytearray | None = None

    @property
    def initialized(self) -> bool:
        with self.database.connect() as connection:
            return (
                connection.execute("SELECT 1 FROM vault_config WHERE singleton=1").fetchone()
                is not None
            )

    @property
    def unlocked(self) -> bool:
        return self._key is not None

    def initialize(self, password: str) -> None:
        self._require_secretstream()
        if not password:
            raise ValueError("vault password must not be empty")
        salt = os.urandom(16)
        key = self._derive(
            password,
            salt,
            memory_cost_kib=_DEFAULT_MEMORY_KIB,
            iterations=_DEFAULT_ITERATIONS,
            lanes=_DEFAULT_LANES,
        )
        check_blob = self._encrypt_bytes(_CHECK, key)
        with self.database.transaction() as connection:
            if connection.execute("SELECT 1 FROM vault_config WHERE singleton=1").fetchone():
                raise VaultError("vault is already initialized")
            connection.execute(
                """INSERT INTO vault_config
                   (singleton, salt, memory_cost_kib, iterations, lanes,
                    check_blob, created_at) VALUES (1, ?, ?, ?, ?, ?, ?)""",
                (
                    salt,
                    _DEFAULT_MEMORY_KIB,
                    _DEFAULT_ITERATIONS,
                    _DEFAULT_LANES,
                    check_blob,
                    utc_now(),
                ),
            )
            self.database.audit(
                "vault.initialized",
                object_type="vault",
                object_id="local",
                connection=connection,
            )
        self._set_key(key)

    def unlock(self, password: str) -> None:
        self._require_secretstream()
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT salt, memory_cost_kib, iterations, lanes, check_blob
                   FROM vault_config WHERE singleton=1"""
            ).fetchone()
        if row is None:
            raise VaultError("vault is not initialized")
        key = self._derive(
            password,
            row["salt"],
            memory_cost_kib=row["memory_cost_kib"],
            iterations=row["iterations"],
            lanes=row["lanes"],
        )
        try:
            check = self._decrypt_bytes(row["check_blob"], key)
        except VaultIntegrityError as exc:
            mutable = bytearray(key)
            _zero(mutable)
            raise VaultAuthenticationError("invalid password or modified vault") from exc
        if check != _CHECK:
            mutable = bytearray(key)
            _zero(mutable)
            raise VaultAuthenticationError("invalid password or modified vault")
        self._set_key(key)
        self.database.audit("vault.unlocked", object_type="vault", object_id="local")

    def lock(self) -> None:
        if self._key is not None:
            _zero(self._key)
            self._key = None

    def store(self, source: Path) -> tuple[str, int, bool]:
        key = self._require_key()
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        digest, size = _hash_file(source)
        storage_name = f"{digest[:2]}/{digest[2:]}.svlt"
        destination = self.objects_dir / storage_name
        with self.database.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM vault_objects WHERE sha256=?", (digest,)
            ).fetchone()
        if exists:
            if not destination.is_file():
                raise VaultIntegrityError("vault database references a missing object")
            return digest, size, True

        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as plain, temporary.open("wb") as encrypted:
                self._encrypt_stream(plain, encrypted, key)
                encrypted.flush()
                os.fsync(encrypted.fileno())
            os.replace(temporary, destination)
            with self.database.transaction() as connection:
                connection.execute(
                    """INSERT OR IGNORE INTO vault_objects
                       (sha256, storage_name, size, ref_count, created_at)
                       VALUES (?, ?, ?, 0, ?)""",
                    (digest, storage_name, size, utc_now()),
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return digest, size, False

    def read(self, sha256: str) -> bytes:
        key = self._require_key()
        path = self._object_path(sha256)
        with path.open("rb") as encrypted:
            return self._decrypt_stream(encrypted, key)

    @contextmanager
    def materialize(self, sha256: str, *, suffix: str = "") -> Iterator[Path]:
        data = self.read(sha256)
        descriptor, name = tempfile.mkstemp(prefix="steg-evidence-", suffix=suffix)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            yield Path(name)
        finally:
            Path(name).unlink(missing_ok=True)

    def delete_unreferenced(self) -> int:
        deleted = 0
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT sha256, storage_name FROM vault_objects WHERE ref_count=0"
            ).fetchall()
            for row in rows:
                (self.objects_dir / row["storage_name"]).unlink(missing_ok=True)
                connection.execute("DELETE FROM vault_objects WHERE sha256=?", (row["sha256"],))
                deleted += 1
        return deleted

    def _object_path(self, sha256: str) -> Path:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT storage_name FROM vault_objects WHERE sha256=?", (sha256,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"vault object {sha256}")
        path = self.objects_dir / row["storage_name"]
        if not path.is_file():
            raise VaultIntegrityError("encrypted vault object is missing")
        return path

    @staticmethod
    def _derive(
        password: str,
        salt: bytes,
        *,
        memory_cost_kib: int,
        iterations: int,
        lanes: int,
    ) -> bytes:
        return Argon2id(
            salt=salt,
            length=crypto_secretstream_xchacha20poly1305_KEYBYTES,
            iterations=iterations,
            lanes=lanes,
            memory_cost=memory_cost_kib,
        ).derive(password.encode("utf-8"))

    def _set_key(self, key: bytes) -> None:
        self.lock()
        self._key = bytearray(key)

    def _require_key(self) -> bytes:
        if self._key is None:
            raise VaultLockedError("vault is locked")
        return bytes(self._key)

    @staticmethod
    def _require_secretstream() -> None:
        if crypto_secretstream_xchacha20poly1305_init_push is None:
            raise VaultUnavailableError("PyNaCl is required for the libsodium secretstream vault")

    @classmethod
    def _encrypt_bytes(cls, data: bytes, key: bytes) -> bytes:
        from io import BytesIO

        source = BytesIO(data)
        destination = BytesIO()
        cls._encrypt_stream(source, destination, key)
        return destination.getvalue()

    @classmethod
    def _decrypt_bytes(cls, data: bytes, key: bytes) -> bytes:
        from io import BytesIO

        return cls._decrypt_stream(BytesIO(data), key)

    @staticmethod
    def _encrypt_stream(source: BinaryIO, destination: BinaryIO, key: bytes) -> None:
        if (
            crypto_secretstream_xchacha20poly1305_init_push is None
            or crypto_secretstream_xchacha20poly1305_push is None
        ):
            raise VaultUnavailableError("PyNaCl secretstream is unavailable")
        if crypto_secretstream_xchacha20poly1305_state is None:
            raise VaultUnavailableError("PyNaCl secretstream state is unavailable")
        state = crypto_secretstream_xchacha20poly1305_state()
        header = crypto_secretstream_xchacha20poly1305_init_push(state, key)
        destination.write(_MAGIC)
        destination.write(header)
        chunk = source.read(_CHUNK_SIZE)
        if not chunk:
            encrypted = crypto_secretstream_xchacha20poly1305_push(
                state, b"", tag=crypto_secretstream_xchacha20poly1305_TAG_FINAL
            )
            destination.write(struct.pack(">I", len(encrypted)))
            destination.write(encrypted)
            return
        while chunk:
            following = source.read(_CHUNK_SIZE)
            tag = (
                crypto_secretstream_xchacha20poly1305_TAG_FINAL
                if not following
                else crypto_secretstream_xchacha20poly1305_TAG_MESSAGE
            )
            encrypted = crypto_secretstream_xchacha20poly1305_push(state, chunk, tag=tag)
            destination.write(struct.pack(">I", len(encrypted)))
            destination.write(encrypted)
            chunk = following

    @staticmethod
    def _decrypt_stream(source: BinaryIO, key: bytes) -> bytes:
        if (
            crypto_secretstream_xchacha20poly1305_init_pull is None
            or crypto_secretstream_xchacha20poly1305_pull is None
        ):
            raise VaultUnavailableError("PyNaCl secretstream is unavailable")
        magic = source.read(len(_MAGIC))
        header = source.read(crypto_secretstream_xchacha20poly1305_HEADERBYTES)
        if magic != _MAGIC or len(header) != crypto_secretstream_xchacha20poly1305_HEADERBYTES:
            raise VaultIntegrityError("invalid or truncated vault header")
        try:
            if crypto_secretstream_xchacha20poly1305_state is None:
                raise VaultUnavailableError("PyNaCl secretstream state is unavailable")
            state = crypto_secretstream_xchacha20poly1305_state()
            crypto_secretstream_xchacha20poly1305_init_pull(state, header, key)
            output = bytearray()
            final_seen = False
            while True:
                size_blob = source.read(4)
                if not size_blob:
                    break
                if len(size_blob) != 4:
                    raise VaultIntegrityError("truncated vault frame length")
                size = struct.unpack(">I", size_blob)[0]
                maximum = _CHUNK_SIZE + crypto_secretstream_xchacha20poly1305_ABYTES
                if size < crypto_secretstream_xchacha20poly1305_ABYTES or size > maximum:
                    raise VaultIntegrityError("invalid vault frame size")
                encrypted = source.read(size)
                if len(encrypted) != size:
                    raise VaultIntegrityError("truncated vault frame")
                plain, tag = crypto_secretstream_xchacha20poly1305_pull(state, encrypted)
                if final_seen:
                    raise VaultIntegrityError("data follows final vault frame")
                output.extend(plain)
                final_seen = tag == crypto_secretstream_xchacha20poly1305_TAG_FINAL
            if not final_seen:
                raise VaultIntegrityError("vault stream has no authenticated final frame")
            return bytes(output)
        except VaultIntegrityError:
            raise
        except Exception as exc:
            raise VaultIntegrityError("vault authentication failed") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _zero(value: bytearray) -> None:
    for index in range(len(value)):
        value[index] = 0
