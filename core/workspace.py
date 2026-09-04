"""Shared construction of local workspace services.

The CLI, terminal UI, and web API use the same database, encrypted vault,
case, scan, model, and Studio services.  Keeping that wiring here prevents
front ends from developing subtly different security defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from core.cases import CaseService
from core.database import Database
from core.models import ModelRegistry
from core.pipeline import AnalysisPipeline
from core.scans import ScanService
from core.service import AnalysisService
from core.studio import StudioService
from core.vault import VaultService


def default_state_dir() -> Path:
    """Return the configured state directory without creating it."""
    configured = os.environ.get("STEGANO_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state" / "steganography"


@dataclass(slots=True)
class Workspace:
    """Services owned by one local workbench session."""

    state_dir: Path
    database: Database
    vault: VaultService
    cases: CaseService
    scans: ScanService
    models: ModelRegistry
    studio: StudioService

    def lock(self) -> None:
        """Forget the in-memory vault key."""
        self.vault.lock()


def create_workspace(state_dir: Path | str | None = None) -> Workspace:
    """Create a local workspace and perform safe housekeeping."""
    root = Path(state_dir).expanduser() if state_dir else default_state_dir()
    root.mkdir(parents=True, exist_ok=True)
    database = Database(root / "steganography.sqlite3")
    vault = VaultService(database, root / "vault")
    database.purge_expired_cases()
    vault.delete_unreferenced()
    cases = CaseService(database, vault)
    scans = ScanService(
        database,
        vault,
        cases,
        lambda profile: AnalysisPipeline(AnalysisService(profile=profile, ai_provider=None)),
    )
    return Workspace(
        state_dir=root,
        database=database,
        vault=vault,
        cases=cases,
        scans=scans,
        models=ModelRegistry(database, root / "models"),
        studio=StudioService(),
    )
