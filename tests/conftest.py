from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxlab.config import Config, config_from_dict
from voxlab.tts.base import (
    BackendCapabilities,
    ModelInfo,
    SpeakerInfo,
    SynthesisRequest,
    SynthesisResult,
    TTSBackend,
)


class FakeBackend(TTSBackend):
    """Deterministic stand-in for a TTS model: a tone whose length follows the text."""

    name = "fake"
    capabilities = BackendCapabilities(
        voice_cloning=False,
        languages=("es", "en"),
        native_params=frozenset({"speed", "emotion"}),
    )
    sample_rate = 24000

    def __init__(self, model_dir: Path | None = None, **kwargs):
        super().__init__(model_dir or Path("/nonexistent"), **kwargs)
        self.requests: list[SynthesisRequest] = []
        self.loaded = False

    def model_info(self) -> ModelInfo:
        return ModelInfo("fake", "fake/model", "default", "MIT", 0, "")

    def is_downloaded(self) -> bool:
        return True

    def download(self) -> None:
        pass

    def load(self) -> None:
        self.loaded = True

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        seconds = 0.05 * len(request.text) / float(request.params.get("speed", 1.0))
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        return SynthesisResult(audio, self.sample_rate)

    def speakers(self) -> list[SpeakerInfo]:
        return [SpeakerInfo("fake_a", "es", "female"), SpeakerInfo("fake_b", "en", "male")]

    def default_speaker(self, language: str) -> str | None:
        return "fake_a"


class CloningBackend(FakeBackend):
    name = "fake_clone"
    capabilities = BackendCapabilities(
        voice_cloning=True, languages=("es",), native_params=frozenset()
    )


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = config_from_dict(
        {
            "tts": {"clone_backend": "none"},  # independent of installed extras
            "voice": {"dir": "voices"},
            "output": {"dir": "output"},
            "storage": {"model_dir": "models"},
            "preset": {"dirs": ["presets"]},
        },
        base_dir=tmp_path,
    )
    return cfg


@pytest.fixture
def sine() -> tuple[np.ndarray, int]:
    sr = 48000
    t = np.arange(sr) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


@pytest.fixture
def dialogue_file(tmp_path: Path) -> Path:
    path = tmp_path / "scene.txt"
    path.write_text(
        "[OPERATOR]\n¿Me recibes?\n\n[COMPUTER|speed=0.9]\nAFIRMATIVO.\n", encoding="utf-8"
    )
    return path
