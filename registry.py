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
