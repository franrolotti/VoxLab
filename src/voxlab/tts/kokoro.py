"""Kokoro-82M backend (ONNX Runtime, no PyTorch). See MODEL_SELECTION.md."""

from __future__ import annotations

import importlib.util
import logging
import os
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

REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
# Pinned for reproducibility; bump deliberately.
REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
SAMPLE_RATE = 24000

VARIANTS: dict[str, tuple[str, int]] = {
    # name: (file in repo, approx. MB)
    "fp32": ("onnx/model.onnx", 326),
    "fp16": ("onnx/model_fp16.onnx", 163),
    "q8": ("onnx/model_quantized.onnx", 92),
    "q8f16": ("onnx/model_q8f16.onnx", 86),
}
DEFAULT_VARIANT = "fp32"
VOICES_MB = 28

# VoxLab language code -> espeak-ng language used by kokoro-onnx.
LANGUAGES: dict[str, str] = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "es-es": "es",  # Spain: distinción (c/z = /θ/), lateral ll
    "es-419": "es-419",  # Latin America: seseo, yeísmo
    "es-mx": "es-419",
    "es-ar": "es-419",  # Río de la Plata: es-419 + sheísmo (see DIALECT_PHONEMES)
    "es-uy": "es-419",
    "fr": "fr-fr",
    "it": "it",
    "pt": "pt-br",
    "pt-br": "pt-br",
    "hi": "hi",
}

# Phoneme substitutions applied after espeak-ng for dialects it lacks.
DIALECT_PHONEMES: dict[str, tuple[tuple[str, str], ...]] = {
    # Rioplatense sheísmo: "ll" and "y" are pronounced /ʃ/ (calle -> "cashe").
    "es-ar": (("jj", "ʃ"), ("ʝ", "ʃ"), ("ʎ", "ʃ")),
    "es-uy": (("jj", "ʃ"), ("ʝ", "ʃ"), ("ʎ", "ʃ")),
}

DEFAULT_SPEAKERS: dict[str, str] = {
    "en": "af_heart",
    "en-gb": "bf_emma",
    "es": "ef_dora",
    "fr": "ff_siwis",
    "it": "if_sara",
    "pt": "pf_dora",
    "hi": "hf_alpha",
}

# First letter of a Kokoro voice id -> language.
_VOICE_LANG_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "zh",
}
_GENDER = {"f": "female", "m": "male"}

_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


class KokoroBackend(TTSBackend):
    name = "kokoro"
    capabilities = BackendCapabilities(
        voice_cloning=False,
        languages=tuple(LANGUAGES),
        native_params=frozenset({"speed"}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variant = str(self.options.get("variant", DEFAULT_VARIANT))
        if self.variant not in VARIANTS:
            raise BackendError(
                f"Unknown Kokoro variant {self.variant!r}; choose one of {', '.join(VARIANTS)}"
            )
        self._engine: Any = None
        self._voices: dict[str, np.ndarray] | None = None
        self._providers: list[str] = []

    @classmethod
    def is_installed(cls) -> bool:
        return all(importlib.util.find_spec(m) for m in ("kokoro_onnx", "onnxruntime"))

    # --- files --------------------------------------------------------------

    @property
    def hub_dir(self) -> Path:
        return self.model_dir / "hub"

    @property
    def voices_pack(self) -> Path:
        return self.model_dir / "kokoro" / f"voices-{REVISION[:12]}.npz"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            backend=self.name,
            model_id=REPO_ID,
            variant=self.variant,
            license="Apache-2.0",
            download_mb=VARIANTS[self.variant][1] + VOICES_MB,
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
                allow_patterns=[VARIANTS[self.variant][0], "voices/*.bin"],
                local_files_only=local_only,
            )
        )

    def _model_path(self) -> Path | None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot = self._snapshot(local_only=True)
        except (LocalEntryNotFoundError, FileNotFoundError, OSError):
            return None
        model = snapshot / VARIANTS[self.variant][0]
        voices = list((snapshot / "voices").glob("*.bin"))
        return model if model.is_file() and voices else None

    def is_downloaded(self) -> bool:
        return self._model_path() is not None and self.voices_pack.is_file()

    def download(self) -> None:
        info = self.model_info()
        log.info(
            "Downloading %s (%s, ~%d MB) to %s",
            info.model_id,
            self.variant,
            info.download_mb,
            self.hub_dir,
        )
        try:
            snapshot = self._snapshot(local_only=False)
        except Exception as exc:  # network errors come in many types
            raise BackendError(f"Model download failed: {exc}") from exc
        self._pack_voices(snapshot / "voices")

    def _pack_voices(self, voices_dir: Path) -> None:
        """kokoro-onnx expects all voice styles in one .npz; build it from the .bin files."""
        styles = {
            f.stem: np.fromfile(f, dtype=np.float32).reshape(-1, 1, 256)
            for f in sorted(voices_dir.glob("*.bin"))
        }
        if not styles:
            raise BackendError(f"No voice files found in {voices_dir}")
        self.voices_pack.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.voices_pack.with_suffix(".tmp.npz")
        np.savez(tmp, **styles)
        tmp.replace(self.voices_pack)

    # --- inference ----------------------------------------------------------

    def load(self) -> None:
        if self._engine is not None:
            return
        model_path = self._model_path()
        if model_path is None or not self.voices_pack.is_file():
            if model_path is None and not self.allow_download:
                raise BackendError(
                    "Kokoro model not found and auto-download is disabled. "
                    "Run `voxlab models --download` first."
                )
            self.download()
            model_path = self._model_path()
            if model_path is None:
                raise BackendError("Model files missing after download")
        if self.offline:
            # From here on nothing may reach the network.
            os.environ["HF_HUB_OFFLINE"] = "1"

        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        self._providers = self._select_providers(rt.get_available_providers())
        log.info("Loading Kokoro %s on %s", self.variant, ", ".join(self._providers))
        session = rt.InferenceSession(str(model_path), providers=self._providers)
        self._engine = Kokoro.from_session(session, str(self.voices_pack))
        self._voices = dict(np.load(self.voices_pack))

    def _select_providers(self, available: list[str]) -> list[str]:
        device = self.device.lower()
        if device == "auto":
            preferred = [p for p in ("CUDAExecutionProvider",) if p in available]
            return [*preferred, "CPUExecutionProvider"]
        if device not in _PROVIDERS:
            raise BackendError(
                f"Unsupported device {self.device!r} for Kokoro; use auto, {', '.join(_PROVIDERS)}"
            )
        wanted = _PROVIDERS[device][0]
        if wanted not in available:
            raise BackendError(
                f"Device {device!r} needs {wanted}, which this onnxruntime build lacks "
                f"(available: {', '.join(available)})"
            )
        return _PROVIDERS[device]

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if request.reference_audio is not None:
            raise BackendError("Kokoro does not support voice cloning (reference audio)")
        language = request.language.lower()
        if language not in LANGUAGES and language.split("-")[0] not in LANGUAGES:
            raise BackendError(
                f"Kokoro does not support language {request.language!r}. "
                f"Supported: {', '.join(LANGUAGES)}"
            )
        self.load()
        speaker = request.speaker or self.default_speaker(language)
        style = self._style(speaker)
        speed = float(request.params.get("speed", 1.0))
        if not 0.5 <= speed <= 2.0:
            raise BackendError(f"Kokoro speed must be between 0.5 and 2.0, got {speed:.2f}")
        espeak_lang = LANGUAGES.get(language) or LANGUAGES[language.split("-")[0]]
        try:
            phonemes = self.phonemize(request.text, language)
            audio, sample_rate = self._engine.create(
                phonemes, voice=style, speed=speed, lang=espeak_lang, is_phonemes=True
            )
        except (ValueError, RuntimeError) as exc:
            raise BackendError(f"Kokoro failed on {request.text[:40]!r}: {exc}") from exc
        return SynthesisResult(np.asarray(audio, dtype=np.float32), int(sample_rate))

    def phonemize(self, text: str, language: str) -> str:
        """Text -> phonemes with espeak-ng, plus VoxLab's dialect adjustments."""
        self.load()
        language = language.lower()
        espeak_lang = LANGUAGES.get(language) or LANGUAGES[language.split("-")[0]]
        phonemes = self._engine.tokenizer.phonemize(text, espeak_lang)
        for old, new in DIALECT_PHONEMES.get(language, ()):
            phonemes = phonemes.replace(old, new)
        return phonemes

    def _style(self, speaker: str | None) -> np.ndarray:
        """Resolve ``id`` or a blend ``id1:0.6,id2:0.4`` into a style tensor."""
        assert self._voices is not None
        if not speaker:
            raise BackendError("No Kokoro speaker selected")
        weights: list[tuple[str, float]] = []
        for part in speaker.split(","):
            voice_id, _, weight = part.strip().partition(":")
            try:
                weights.append((voice_id.strip(), float(weight) if weight else 1.0))
            except ValueError as exc:
                raise BackendError(f"Invalid voice blend {speaker!r}") from exc
        missing = [v for v, _ in weights if v not in self._voices]
        if missing:
            raise BackendError(
                f"Unknown Kokoro speaker(s) {', '.join(missing)}. "
                "Run `voxlab voices --speakers` to list them."
            )
        total = sum(w for _, w in weights)
        if total <= 0:
            raise BackendError(f"Voice blend weights must be positive: {speaker!r}")
        return sum(self._voices[v] * (w / total) for v, w in weights)

    def speakers(self) -> list[SpeakerInfo]:
        if self._voices is None:
            if not self.voices_pack.is_file():
                return []
            self._voices = dict(np.load(self.voices_pack))
        return [
            SpeakerInfo(
                id=voice_id,
                language=_VOICE_LANG_PREFIX.get(voice_id[:1], ""),
                gender=_GENDER.get(voice_id[1:2], ""),
            )
            for voice_id in sorted(self._voices)
        ]

    def default_speaker(self, language: str) -> str | None:
        language = language.lower()
        return DEFAULT_SPEAKERS.get(language) or DEFAULT_SPEAKERS.get(language.split("-")[0])

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": self.name,
            "model": REPO_ID,
            "variant": self.variant,
            "device": self.device,
            "providers": self._providers,
        }
        if importlib.util.find_spec("onnxruntime"):
            import onnxruntime as rt

            info["onnxruntime"] = rt.__version__
        return info
