"""Backend registry and construction.

Backends are registered by import path so that their heavy dependencies are
only imported when the backend is actually used. Adding a backend means
writing a :class:`~voxlab.tts.base.TTSBackend` subclass and registering it
here (or calling :func:`register_backend` from a plugin).
"""

from __future__ import annotations

import importlib

from voxlab.config import Config
from voxlab.errors import BackendError
from voxlab.tts.base import TTSBackend

_REGISTRY: dict[str, str | type[TTSBackend]] = {
    "kokoro": "voxlab.tts.kokoro:KokoroBackend",
    "qwen": "voxlab.tts.qwen:QwenBackend",
}

# Preference order for ``tts.backend: auto``.
AUTO_ORDER: list[str] = ["kokoro"]


def register_backend(name: str, backend: str | type[TTSBackend], prefer: bool = False) -> None:
    """Register a backend class (or ``"module:Class"`` path) under ``name``."""
    _REGISTRY[name] = backend
    if prefer:
        AUTO_ORDER.insert(0, name)
    elif name not in AUTO_ORDER:
        AUTO_ORDER.append(name)


def backend_names() -> list[str]:
    return list(_REGISTRY)


def get_backend_class(name: str) -> type[TTSBackend]:
    if name not in _REGISTRY:
        raise BackendError(f"Unknown TTS backend {name!r}. Available: {', '.join(_REGISTRY)}")
    target = _REGISTRY[name]
    if isinstance(target, str):
        module_name, _, class_name = target.partition(":")
        try:
            target = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as exc:
            raise BackendError(f"Could not load backend {name!r}: {exc}") from exc
    return target


def resolve_backend_name(requested: str) -> str:
    if requested != "auto":
        return requested
    for name in AUTO_ORDER:
        if get_backend_class(name).is_installed():
            return name
    raise BackendError("No TTS backend is installed. Reinstall VoxLab: pip install voxlab")


def resolve_clone_backend_name(requested: str) -> str | None:
    """Backend used for voices with a reference clip (``tts.clone_backend``)."""
    if requested == "none":
        return None
    if requested != "auto":
        return requested
    for name in _REGISTRY:
        cls = get_backend_class(name)
        if cls.capabilities.voice_cloning and cls.is_installed():
            return name
    return None


def create_clone_backend(config: Config) -> TTSBackend | None:
    name = resolve_clone_backend_name(config.tts.clone_backend)
    return create_backend(config, name) if name else None


def create_backend(config: Config, backend: str | None = None) -> TTSBackend:
    name = resolve_backend_name(backend or config.tts.backend)
    cls = get_backend_class(name)
    if not cls.is_installed():
        raise BackendError(f"Backend {name!r} is registered but its dependencies are missing")
    return cls(
        model_dir=config.model_dir,
        device=config.tts.device,
        options=config.tts.options,
        allow_download=config.storage.auto_download,
        offline=config.privacy.offline,
    )
