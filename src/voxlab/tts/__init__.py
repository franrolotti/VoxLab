"""Text-to-speech backends behind a common interface."""

from voxlab.tts.base import (
    BackendCapabilities,
    ModelInfo,
    SpeakerInfo,
    SynthesisRequest,
    SynthesisResult,
    TTSBackend,
)
from voxlab.tts.factory import create_backend, register_backend

__all__ = [
    "BackendCapabilities",
    "ModelInfo",
    "SpeakerInfo",
    "SynthesisRequest",
    "SynthesisResult",
    "TTSBackend",
    "create_backend",
    "register_backend",
]
