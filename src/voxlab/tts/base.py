"""Backend-agnostic TTS interface.

Nothing outside ``voxlab.tts`` should import a concrete backend: the pipeline
talks to :class:`TTSBackend` and inspects :class:`BackendCapabilities` to
decide what the model can do natively and what VoxLab must emulate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    voice_cloning: bool = False
    # ISO 639-1 codes (optionally with region, e.g. "en-gb").
    languages: tuple[str, ...] = ()
    # Line/voice parameters the model understands directly (e.g. "speed").
    native_params: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    model_id: str
    variant: str
    license: str
    download_mb: int
    url: str


@dataclass(frozen=True)
class SpeakerInfo:
    """A voice built into the model (not a VoxLab voice profile)."""

    id: str
    language: str = ""
    gender: str = ""


@dataclass
class SynthesisRequest:
    text: str
    language: str
    # Built-in speaker id; backend-specific syntax. None -> backend default.
    speaker: str | None = None
    # Reference clip for voice cloning (only if capabilities.voice_cloning).
    reference_audio: Path | None = None
    # Transcript of the reference clip, when known.
    reference_text: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.audio.size / self.sample_rate if self.sample_rate else 0.0


class TTSBackend(ABC):
    """Base class for TTS engines."""

    name: ClassVar[str]
    capabilities: ClassVar[BackendCapabilities]

    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        options: dict[str, Any] | None = None,
        allow_download: bool = True,
        offline: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.device = device
        self.options = dict(options or {})
        self.allow_download = allow_download
        self.offline = offline

    @classmethod
    def is_installed(cls) -> bool:
        """True when the backend's Python dependencies are importable."""
        return True

    @abstractmethod
    def model_info(self) -> ModelInfo: ...

    @abstractmethod
    def is_downloaded(self) -> bool: ...

    @abstractmethod
    def download(self) -> None:
        """Fetch model files into ``model_dir``. The only step that uses the network."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (downloading first if allowed). Idempotent."""

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...

    @abstractmethod
    def speakers(self) -> list[SpeakerInfo]:
        """Built-in speakers. Requires the model to be downloaded."""

    def default_speaker(self, language: str) -> str | None:
        """Speaker used when a voice profile does not name one."""
        return None

    def supports_language(self, language: str) -> bool:
        language = language.lower()
        return any(
            language == code or language.split("-")[0] == code
            for code in self.capabilities.languages
        )

    def runtime_info(self) -> dict[str, Any]:
        """Details for benchmarks and bug reports (device, library versions...)."""
        return {"backend": self.name, "device": self.device}
