"""Centralized configuration loaded from ``config.yaml``.

Lookup order: explicit ``--config`` path, ``$VOXLAB_CONFIG``, ``./config.yaml``,
then built-in defaults. Relative paths inside the file are resolved against the
directory that contains it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from voxlab.audio.export import SUPPORTED_BIT_DEPTHS, SUPPORTED_FORMATS, SUPPORTED_SAMPLE_RATES
from voxlab.errors import ConfigError

CONFIG_FILENAME = "config.yaml"
CONFIG_ENV_VAR = "VOXLAB_CONFIG"


@dataclass
class TTSConfig:
    backend: str = "auto"
    device: str = "auto"
    language: str = "es"
    # Backend-specific options, passed through untouched (e.g. Kokoro model variant).
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    bit_depth: int = 24
    format: str = "wav"
    gap_ms: int = 350
    padding_ms: int = 250


@dataclass
class VoiceConfig:
    default: str = "default"
    dir: str = "voices"
    # When true, speakers without an assigned voice are an error instead of
    # falling back to the default voice.
    strict: bool = False


@dataclass
class PresetConfig:
    default: str = "clean"
    dirs: list[str] = field(default_factory=lambda: ["presets"])


@dataclass
class StorageConfig:
    model_dir: str = "~/.cache/voxlab"
    auto_download: bool = True


@dataclass
class PrivacyConfig:
    # Once the model is cached, forbid any Hugging Face Hub network access.
    offline: bool = True


@dataclass
class OutputConfig:
    dir: str = "output"


@dataclass
class Config:
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    preset: PresetConfig = field(default_factory=PresetConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Speaker name (as written in dialogues) -> voice name, or a mapping
    # {voice: <name>, language: <code>} to also give the character a language.
    cast: dict[str, Any] = field(default_factory=dict)
    base_dir: Path = field(default_factory=Path.cwd, repr=False)
    source: Path | None = field(default=None, repr=False)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.base_dir / path

    @property
    def model_dir(self) -> Path:
        return self.resolve_path(self.storage.model_dir)

    @property
    def voices_dir(self) -> Path:
        return self.resolve_path(self.voice.dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.output.dir)

    @property
    def preset_dirs(self) -> list[Path]:
        return [self.resolve_path(d) for d in self.preset.dirs]


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration, falling back to defaults when no file is found."""
    config_path = _find_config(path)
    if config_path is None:
        return Config()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")
    config = config_from_dict(data, base_dir=config_path.parent.resolve())
    config.source = config_path
    return config


def config_from_dict(data: dict[str, Any], base_dir: Path | None = None) -> Config:
    config = _build(Config, data, section="")
    if base_dir is not None:
        config.base_dir = base_dir
    config.cast = {str(k): _cast_entry(str(k), v) for k, v in config.cast.items()}
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    audio = config.audio
    if audio.sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ConfigError(
            f"audio.sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {audio.sample_rate}"
        )
    if audio.bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ConfigError(
            f"audio.bit_depth must be one of {SUPPORTED_BIT_DEPTHS}, got {audio.bit_depth}"
        )
    if audio.format.lower() not in SUPPORTED_FORMATS:
        raise ConfigError(f"audio.format must be one of {SUPPORTED_FORMATS}, got {audio.format!r}")
    if audio.gap_ms < 0 or audio.padding_ms < 0:
        raise ConfigError("audio.gap_ms and audio.padding_ms must be >= 0")
    if not config.tts.language:
        raise ConfigError("tts.language must not be empty")


def _cast_entry(speaker: str, value: Any) -> str | dict[str, str]:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        unknown = set(value) - {"voice", "language"}
        if unknown:
            raise ConfigError(f"cast.{speaker}: unknown key(s) {', '.join(sorted(unknown))}")
        if not value:
            raise ConfigError(f"cast.{speaker}: set 'voice' and/or 'language'")
        return {k: str(v) for k, v in value.items() if v}
    raise ConfigError(f"cast.{speaker} must be a voice name or {{voice: ..., language: ...}}")


def _find_config(path: str | Path | None) -> Path | None:
    if path is not None:
        explicit = Path(path).expanduser()
        if not explicit.is_file():
            raise ConfigError(f"Config file not found: {explicit}")
        return explicit
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return _find_config(env_path)
    local = Path.cwd() / CONFIG_FILENAME
    return local if local.is_file() else None


_INTERNAL_FIELDS = {"base_dir", "source"}


def _build(cls: type, data: Any, section: str) -> Any:
    """Instantiate a config dataclass from a dict, rejecting unknown keys and bad types."""
    label = section or "config"
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{label}' must be a mapping")

    defaults = cls()
    known = {f.name: f for f in fields(cls) if f.name not in _INTERNAL_FIELDS}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{label}': {', '.join(sorted(map(str, unknown)))}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        key = f"{section}.{name}" if section else name
        default = getattr(defaults, name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, key)
        else:
            kwargs[name] = _check_type(value, default, key)
    return cls(**kwargs)


def _check_type(value: Any, default: Any, key: str) -> Any:
    expected = type(default)
    if value is None:
        raise ConfigError(f"'{key}' must not be empty")
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer")
    if not isinstance(value, expected):
        raise ConfigError(f"'{key}' must be of type {expected.__name__}, got {value!r}")
    return value
