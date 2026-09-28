"""Qwen3-TTS 0.6B Base: zero-shot voice cloning on the Apple Silicon GPU (MLX).

Installed with ``pip install "voxlab[clone]"``. The Base model has no built-in
speakers: every line needs a reference clip, ideally with its transcript
(in-context mode). The accent of the reference carries over to the output.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from voxlab.errors import BackendError
from voxlab.tts.base import (
    BackendCapabilities,
    ModelInfo,
    SpeakerInfo,
    SynthesisRequest,
    SynthesisResult,
    TTSBackend,
)

log = logging.getLogger(__name__)

REPO_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
# Pinned for reproducibility; bump deliberately.
REVISION = "1eccf1cb2519b5a4e8a95b5f0544f3303568164f"
DOWNLOAD_MB = 2400

# VoxLab language (prefix before '-') -> Qwen3-TTS language name.
LANGUAGES: dict[str, str] = {
    "es": "spanish",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}

# The codec runs at 12 tokens per second of audio. Cap generation relative to
# the text length so a sampling glitch cannot run on for minutes.
TOKENS_PER_CHAR = 2.0
MIN_TOKENS = 60


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class QwenBackend(TTSBackend):
    name = "qwen"
    capabilities = BackendCapabilities(
        voice_cloning=True,
        languages=tuple(LANGUAGES),
        native_params=frozenset({"temperature"}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seed = int(self.options.get("seed", 0))
        self._model: Any = None
        self._warned_no_transcript: set[Path] = set()

    @classmethod
    def is_installed(cls) -> bool:
        return _is_apple_silicon() and importlib.util.find_spec("mlx_audio") is not None

    @property
    def hub_dir(self) -> Path:
        return self.model_dir / "hub"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            backend=self.name,
            model_id=REPO_ID,
            variant="0.6B-bf16",
            license="Apache-2.0",
            download_mb=DOWNLOAD_MB,
            url=f"https://huggingface.co/{REPO_ID}",
        )

    def _snapshot(self, local_only: bool) -> Path:
        from huggingface_hub import snapshot_download

        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        return Path(
            snapshot_download(
                repo_id=REPO_ID,
                revision=REVISION,
                cache_dir=self.hub_dir,
                local_files_only=local_only,
            )
        )

    def _model_path(self) -> Path | None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot = self._snapshot(local_only=True)
        except (LocalEntryNotFoundError, FileNotFoundError, OSError):
            return None
        required = ("model.safetensors", "speech_tokenizer/model.safetensors")
        return snapshot if all((snapshot / f).is_file() for f in required) else None

    def is_downloaded(self) -> bool:
        return self._model_path() is not None

    def download(self) -> None:
        log.info("Downloading %s (~%d MB) to %s", REPO_ID, DOWNLOAD_MB, self.hub_dir)
        try:
            self._snapshot(local_only=False)
        except Exception as exc:  # network errors come in many types
            raise BackendError(f"Model download failed: {exc}") from exc

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_installed():
            raise BackendError(
                'Qwen3-TTS needs Apple Silicon and the clone extra: pip install "voxlab[clone]"'
            )
        path = self._model_path()
        if path is None:
            if not self.allow_download:
                raise BackendError(
                    "Qwen3-TTS model not found and auto-download is disabled. "
                    "Run `voxlab models --download --backend qwen` first."
                )
            self.download()
            path = self._model_path()
            if path is None:
                raise BackendError("Model files missing after download")
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"

        import mlx.core as mx

        if self.device.lower() == "cpu":
            mx.set_default_device(mx.cpu)
        elif self.device.lower() not in ("auto", "gpu", "mps"):
            raise BackendError(f"Unsupported device {self.device!r} for Qwen3-TTS; use auto or cpu")
        log.info("Loading Qwen3-TTS 0.6B on %s", mx.default_device())
        with _quiet():
            from mlx_audio.tts.utils import load_model

            self._model = load_model(path)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.reference_audio is None:
            raise BackendError(
                "Qwen3-TTS (Base) only clones voices: give the voice a reference clip"
            )
        language = request.language.lower().split("-")[0]
        if language not in LANGUAGES:
            raise BackendError(
                f"Qwen3-TTS does not support language {request.language!r}. "
                f"Supported: {', '.join(LANGUAGES)}"
            )
        if not request.reference_text and request.reference_audio not in self._warned_no_transcript:
            self._warned_no_transcript.add(request.reference_audio)
            log.warning(
                "No transcript for %s; cloning will be less accurate. Add a reference.txt "
                "with exactly what is said in the clip.",
                request.reference_audio,
            )
        self.load()

        import mlx.core as mx

        mx.random.seed(self.seed)
        kwargs: dict[str, Any] = {}
        if "temperature" in request.params:
            kwargs["temperature"] = float(request.params["temperature"])
        try:
            with _quiet():
                results = list(
                    self._model.generate(
                        text=request.text,
                        ref_audio=str(request.reference_audio),
                        ref_text=request.reference_text,
                        lang_code=LANGUAGES[language],
                        max_tokens=int(len(request.text) * TOKENS_PER_CHAR) + MIN_TOKENS,
                        **kwargs,
                    )
                )
        except (ValueError, RuntimeError) as exc:
            raise BackendError(f"Qwen3-TTS failed on {request.text[:40]!r}: {exc}") from exc
        if not results:
            raise BackendError(f"Qwen3-TTS produced no audio for {request.text[:40]!r}")
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
        return SynthesisResult(audio, int(self._model.sample_rate))

    def speakers(self) -> list[SpeakerInfo]:
        return []  # the Base model only clones

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "model": REPO_ID, "device": self.device}
        if importlib.util.find_spec("mlx"):
            import mlx.core as mx

            info["mlx"] = mx.__version__
            info["mlx_device"] = str(mx.default_device())
        return info


@contextlib.contextmanager
def _quiet():
    """Silence mlx-audio/transformers chatter (prints and warnings) unless debugging."""
    if log.isEnabledFor(logging.DEBUG):
        yield
        return
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        previous = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            yield
        finally:
            logging.disable(previous)
