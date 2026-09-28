"""Audio presets: named, file-defined effect chains.

A preset is a YAML file::

    name: radio
    description: AM/FM radio transmission
    chain:
      - effect: bandpass
        low_hz: 300
        high_hz: 3400
      - effect: normalize
        peak_db: -1

Presets are looked up in the configured directories first, then among the
presets that ship with VoxLab, so a user file can override a built-in one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from voxlab.audio.effects import apply_effect, validate_params
from voxlab.errors import PresetError

log = logging.getLogger(__name__)

PRESET_SUFFIX = ".yaml"


@dataclass(frozen=True)
class EffectStep:
    effect: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Preset:
    name: str
    description: str = ""
    chain: tuple[EffectStep, ...] = ()
    path: Path | None = None

    def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        out = audio
        for step in self.chain:
            try:
                out = apply_effect(step.effect, out, sample_rate, **step.params)
            except ValueError as exc:
                raise PresetError(f"preset {self.name!r}: {exc}") from exc
        return out


def builtin_preset_dirs() -> list[Path]:
    """Presets bundled in the wheel, or the repository folder in a source checkout."""
    package_dir = Path(__file__).resolve().parents[1]
    candidates = [package_dir / "_presets", package_dir.parents[1] / "presets"]
    return [d for d in candidates if d.is_dir()]


def parse_preset(data: Any, name: str, path: Path | None = None) -> Preset:
    """Build and validate a preset from already-loaded YAML data."""
    if not isinstance(data, dict):
        raise PresetError(f"preset {name!r} must be a mapping")
    unknown = set(data) - {"name", "description", "chain"}
    if unknown:
        raise PresetError(f"preset {name!r}: unknown key(s) {', '.join(sorted(unknown))}")
    chain_data = data.get("chain") or []
    if not isinstance(chain_data, list):
        raise PresetError(f"preset {name!r}: 'chain' must be a list of effects")

    steps: list[EffectStep] = []
    for index, raw in enumerate(chain_data, start=1):
        if not isinstance(raw, dict) or "effect" not in raw:
            raise PresetError(f"preset {name!r}: step {index} must be a mapping with 'effect'")
        params = {k: v for k, v in raw.items() if k not in ("effect", "enabled")}
        if raw.get("enabled", True) is False:
            continue
        try:
            params = validate_params(str(raw["effect"]), params)
        except ValueError as exc:
            raise PresetError(f"preset {name!r}, step {index}: {exc}") from exc
        steps.append(EffectStep(effect=str(raw["effect"]), params=params))

    return Preset(
        name=str(data.get("name", name)),
        description=str(data.get("description", "")),
        chain=tuple(steps),
        path=path,
    )


def load_preset_file(path: str | Path) -> Preset:
    path = Path(path)
    if not path.is_file():
        raise PresetError(f"Preset file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PresetError(f"Invalid YAML in preset {path}: {exc}") from exc
    return parse_preset(data, name=path.stem, path=path)


class PresetLibrary:
    """Finds presets by name across user and built-in directories."""

    def __init__(self, search_dirs: list[Path] | None = None, include_builtin: bool = True):
        dirs = list(search_dirs or [])
        if include_builtin:
            dirs += builtin_preset_dirs()
        # Keep order, drop duplicates (the repo folder can be both user and built-in).
        self.search_dirs = list(dict.fromkeys(d.resolve() for d in dirs))

    def _files(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for directory in self.search_dirs:
            if not directory.is_dir():
                continue
            for file in sorted(directory.glob(f"*{PRESET_SUFFIX}")):
                found.setdefault(file.stem, file)
        return found

    def names(self) -> list[str]:
        return sorted(self._files())

    def get(self, name_or_path: str) -> Preset:
        """Load a preset by name, or directly from a ``.yaml`` path."""
        candidate = Path(name_or_path)
        if candidate.suffix == PRESET_SUFFIX and candidate.is_file():
            return load_preset_file(candidate)
        files = self._files()
        if name_or_path not in files:
            available = ", ".join(sorted(files)) or "none"
            raise PresetError(f"Unknown preset {name_or_path!r}. Available: {available}")
        log.debug("Loading preset %s from %s", name_or_path, files[name_or_path])
        return load_preset_file(files[name_or_path])

    def all(self) -> list[Preset]:
        return [self.get(name) for name in self.names()]
