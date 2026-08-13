"""Auto-discovery and deterministic dispatch for carrier/analyzer plug-ins."""
import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import modules
from core.analyzer import Analyzer
from core.carrier import Carrier


@dataclass(frozen=True, slots=True)
class PluginLoadError:
    plugin: str
    error: str


class Registry:
    def __init__(self) -> None:
        self._carriers: list[Carrier] = []
        self._analyzers: list[Analyzer] = []
        self._load_errors: list[PluginLoadError] = []

    def register(self, obj: Carrier | Analyzer) -> None:
        if isinstance(obj, Carrier):
            self._carriers.append(obj)
        elif isinstance(obj, Analyzer):
            self._analyzers.append(obj)
        else:
            raise TypeError(f"unsupported plug-in type: {type(obj)!r}")
        self._validate_unique_names()

    def autodiscover(self) -> None:
        """Import built-ins and installed ``steganography.*`` entry points."""
        for mod_info in pkgutil.iter_modules(modules.__path__):
            module_name = f"modules.{mod_info.name}"
            try:
                mod = importlib.import_module(module_name)
            except Exception as exc:
                self._load_errors.append(
                    PluginLoadError(module_name, f"{type(exc).__name__}: {exc}")
                )
                continue
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                if inspect.isabstract(cls):
                    continue
                if issubclass(cls, Carrier):
                    self._carriers.append(cls())
                elif issubclass(cls, Analyzer):
                    self._analyzers.append(cls())
        self._load_entry_points("steganography.carriers")
        self._load_entry_points("steganography.analyzers")
        self._validate_unique_names()

    def _load_entry_points(self, group: str) -> None:
        entry_points = metadata.entry_points().select(group=group)
        for entry_point in entry_points:
            try:
                loaded = entry_point.load()
                obj = loaded() if inspect.isclass(loaded) else loaded
                if not isinstance(obj, (Carrier, Analyzer)):
                    raise TypeError(
                        f"unsupported plug-in type: {type(obj)!r}"
                    )
                self.register(obj)
            except Exception as exc:
                self._load_errors.append(
                    PluginLoadError(
                        f"{group}:{entry_point.name}",
                        f"{type(exc).__name__}: {exc}",
                    )
                )

    def _validate_unique_names(self) -> None:
        names: set[str] = set()
        identifiers = [carrier.identifier for carrier in self._carriers]
        identifiers.extend(analyzer.name for analyzer in self._analyzers)
        for identifier in identifiers:
            if not identifier:
                raise ValueError("plug-in has no name")
            if identifier in names:
                raise ValueError(f"duplicate plug-in name: {identifier}")
            names.add(identifier)

    def all_carriers(self) -> list[Carrier]:
        return sorted(self._carriers, key=lambda item: (item.priority, item.identifier))

    def all_analyzers(self) -> list[Analyzer]:
        return sorted(self._analyzers, key=lambda item: item.name)

    def load_errors(self) -> list[PluginLoadError]:
        return list(self._load_errors)

    def select_carriers(
        self, src: Path, *, detected_extension: str | None = None
    ) -> list[Carrier]:
        return [
            carrier
            for carrier in self.all_carriers()
            if carrier.supports(src, detected_extension=detected_extension)
        ]

    def get_carrier(self, method: str) -> Carrier:
        for carrier in self._carriers:
            if carrier.identifier == method:
                return carrier
        raise LookupError(f"unknown carrier method {method!r}")

    def select_carrier_for_embed(self, dest: Path, method: str | None = None) -> Carrier:
        if method is not None:
            carrier = self.get_carrier(method)
            if not carrier.can_embed:
                raise LookupError(f"carrier {method!r} cannot embed")
            if not carrier.supports(dest):
                raise LookupError(
                    f"carrier {method!r} does not support extension {dest.suffix!r}"
                )
            return carrier
        candidates = [
            carrier
            for carrier in self.select_carriers(dest)
            if carrier.can_embed and not carrier.requires_explicit
        ]
        if not candidates:
            raise LookupError(f"no carrier for extension {dest.suffix!r}")
        return candidates[0]
