"""Dialogue -> voices -> TTS -> per-line processing -> mix -> preset -> WAV.

The pipeline is split in two phases:

* :meth:`Pipeline.plan` resolves voices, languages and parameters for every
  line and validates everything *before* the model is loaded, so mistakes in a
  dialogue fail fast.
* :meth:`Pipeline.render` runs the TTS backend and the audio chain.

Voices with a reference clip are routed to the clone backend
(``tts.clone_backend``) when the main backend cannot clone, so one dialogue can
mix built-in and cloned voices.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voxlab.audio.dsp import db_to_gain, resample
from voxlab.audio.export import ExportSettings, export_audio
from voxlab.audio.mixer import Segment, mix_sequence, trim_silence
from voxlab.audio.presets import Preset, PresetLibrary
from voxlab.config import Config
from voxlab.dialogue.models import Dialogue, DialogueLine
from voxlab.errors import DialogueError, VoiceError
from voxlab.tts.base import SynthesisRequest, TTSBackend
from voxlab.tts.factory import create_backend, create_clone_backend
from voxlab.voices.manager import VoiceManager, VoiceProfile

log = logging.getLogger(__name__)

# Parameters VoxLab implements itself (in post-processing), with valid ranges.
POST_PARAMS: dict[str, tuple[float, float]] = {
    "pitch": (-12.0, 12.0),  # semitones
    "volume": (-30.0, 12.0),  # dB
    "pause": (0.0, 10000.0),  # ms of silence after the line
}
SPEED_RANGE = (0.5, 2.0)
LANGUAGE_KEYS = ("lang", "language")


@dataclass(frozen=True)
class LineJob:
    index: int
    line: DialogueLine
    voice: VoiceProfile
    request: SynthesisRequest
    backend: TTSBackend = field(compare=False, repr=False)
    pitch: float = 0.0
    volume_db: float = 0.0
    pause_ms: float | None = None


@dataclass
class GenerateOptions:
    preset: str | None = None
    output: Path | None = None
    cast: dict[str, str] = field(default_factory=dict)
    language: str | None = None
    sample_rate: int | None = None
    bit_depth: int | None = None
    strict: bool | None = None


@dataclass
class GenerateResult:
    output: Path
    duration: float
    lines: int
    cast: dict[str, str]
    preset: str
    elapsed: float


@dataclass
class Plan:
    jobs: list[LineJob]
    preset: Preset
    voice_presets: dict[str, Preset]
    cast: dict[str, VoiceProfile]


class Pipeline:
    def __init__(
        self,
        config: Config,
        backend: TTSBackend | None = None,
        voices: VoiceManager | None = None,
        presets: PresetLibrary | None = None,
        clone_backend: TTSBackend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend
        self._clone_backend = clone_backend
        self._clone_resolved = clone_backend is not None
        self.voices = voices or VoiceManager(config.voices_dir)
        self.presets = presets or PresetLibrary(config.preset_dirs)

    @property
    def backend(self) -> TTSBackend:
        if self._backend is None:
            self._backend = create_backend(self.config)
        return self._backend

    @property
    def clone_backend(self) -> TTSBackend | None:
        if not self._clone_resolved:
            self._clone_backend = create_clone_backend(self.config)
            self._clone_resolved = True
        return self._clone_backend

    def backend_for(self, voice: VoiceProfile) -> TTSBackend:
        """Main backend, or the clone backend for reference voices it cannot handle."""
        if voice.uses_cloning and not self.backend.capabilities.voice_cloning:
            clone = self.clone_backend
            if clone is not None:
                return clone
        return self.backend

    # --- planning -----------------------------------------------------------

    def plan(self, dialogue: Dialogue, options: GenerateOptions | None = None) -> Plan:
        options = options or GenerateOptions()
        strict = self.config.voice.strict if options.strict is None else options.strict
        cast = self.voices.resolve_cast(
            dialogue.speakers,
            cast={**self.config.cast, **options.cast},
            default=self.config.voice.default,
            strict=strict,
        )
        preset = self.presets.get(options.preset or self.config.preset.default)
        voice_presets = {v.name: self.presets.get(v.preset) for v in cast.values() if v.preset}
        warned: set[str] = set()
        jobs = [
            self._plan_line(i, line, cast[line.speaker], options, warned)
            for i, line in enumerate(dialogue.lines)
        ]
        return Plan(jobs=jobs, preset=preset, voice_presets=voice_presets, cast=cast)

    def _plan_line(
        self,
        index: int,
        line: DialogueLine,
        voice: VoiceProfile,
        options: GenerateOptions,
        warned: set[str],
    ) -> LineJob:
        backend = self.backend_for(voice)
        caps = backend.capabilities
        where = f"line {line.line_number} [{line.speaker}]"
        params: dict[str, Any] = {**voice.params, **line.params}

        language = self._language(params, voice, options)
        if not backend.supports_language(language):
            raise DialogueError(
                f"{where}: backend {backend.name!r} does not support language {language!r} "
                f"(supported: {', '.join(caps.languages)})"
            )

        post = {
            key: _number(params.pop(key), key, where, *POST_PARAMS[key])
            for key in list(params)
            if key in POST_PARAMS
        }
        pitch = post.get("pitch", 0.0)
        native: dict[str, Any] = {}

        if "speed" in params:
            speed = _number(params.pop("speed"), "speed", where, *SPEED_RANGE)
            if "speed" in caps.native_params:
                native["speed"] = speed
            elif speed != 1.0:
                _warn_once(
                    warned,
                    "speed",
                    f"Backend {backend.name!r} has no speed control; "
                    "the 'speed' parameter is ignored",
                )
        if pitch and "speed" in caps.native_params:
            # Pitch is applied by resampling, which also changes tempo; ask the
            # model for the inverse tempo change so the final speed is unchanged.
            native["speed"] = native.get("speed", 1.0) / _pitch_ratio(pitch)
            if not SPEED_RANGE[0] <= native["speed"] <= SPEED_RANGE[1]:
                raise DialogueError(
                    f"{where}: pitch {pitch:+g} combined with this speed is "
                    "outside the range the backend can render"
                )

        for key in list(params):
            value = params.pop(key)
            if key in caps.native_params:
                native[key] = value
            else:
                _warn_once(
                    warned,
                    key,
                    f"Parameter {key!r} is not supported by backend {backend.name!r}; ignored",
                )

        request = SynthesisRequest(
            text=line.text,
            language=language,
            speaker=voice.speaker_for(backend.name, language),
            params=native,
        )
        if voice.uses_cloning:
            request.reference_audio = self._reference_for(voice, request, backend, warned)
            if request.reference_audio is not None:
                request.reference_text = voice.reference_text

        return LineJob(
            index=index,
            line=line,
            voice=voice,
            request=request,
            backend=backend,
            pitch=pitch,
            volume_db=post.get("volume", 0.0),
            pause_ms=post.get("pause"),
        )

    def _language(
        self, params: dict[str, Any], voice: VoiceProfile, options: GenerateOptions
    ) -> str:
        language = None
        for key in LANGUAGE_KEYS:
            if key in params:
                language = str(params.pop(key))
        return (language or voice.language or options.language or self.config.tts.language).lower()

    @staticmethod
    def _reference_for(
        voice: VoiceProfile, request: SynthesisRequest, backend: TTSBackend, warned: set[str]
    ) -> Path | None:
        if backend.capabilities.voice_cloning:
            return voice.reference
        if request.speaker:
            _warn_once(
                warned,
                f"clone:{voice.name}",
                f"Voice {voice.name!r} has a reference clip but backend {backend.name!r} "
                f"cannot clone voices; using speaker {request.speaker!r} instead",
            )
            return None
        raise VoiceError(
            f"Voice {voice.name!r} relies on voice cloning (reference audio), which backend "
            f"{backend.name!r} does not support and no cloning backend is installed. "
            'Install one with: pip install "voxlab[clone]" (Apple Silicon), '
            f"or add a 'speaker' to voices/{voice.name}/voice.yaml."
        )

    # --- rendering ----------------------------------------------------------

    def render(self, plan: Plan, sample_rate: int) -> np.ndarray:
        """Synthesize every line and return the processed mix at ``sample_rate``."""
        for backend in {id(job.backend): job.backend for job in plan.jobs}.values():
            backend.load()
        segments: list[Segment] = []
        total = len(plan.jobs)
        for job in plan.jobs:
            preview = job.line.text if len(job.line.text) <= 50 else job.line.text[:47] + "..."
            log.info(
                "[%d/%d] %s (%s): %s",
                job.index + 1,
                total,
                job.line.speaker,
                job.voice.name,
                preview,
            )
            segments.append(Segment(self._render_line(job, plan, sample_rate), job.pause_ms))

        mix = mix_sequence(
            segments,
            sample_rate,
            gap_ms=self.config.audio.gap_ms,
            padding_ms=self.config.audio.padding_ms,
        )
        return plan.preset.apply(mix, sample_rate)

    def _render_line(self, job: LineJob, plan: Plan, sample_rate: int) -> np.ndarray:
        result = job.backend.synthesize(job.request)
        audio = trim_silence(result.audio, result.sample_rate)
        if job.pitch:
            audio = shift_pitch(audio, result.sample_rate, job.pitch)
        if job.volume_db:
            audio = audio * db_to_gain(job.volume_db)
        audio = resample(audio, result.sample_rate, sample_rate)
        voice_preset = plan.voice_presets.get(job.voice.name)
        if voice_preset is not None:
            audio = voice_preset.apply(audio, sample_rate)
        return audio

    # --- end to end ---------------------------------------------------------

    def export_settings(self, options: GenerateOptions) -> ExportSettings:
        audio = self.config.audio
        settings = ExportSettings(
            sample_rate=options.sample_rate or audio.sample_rate,
            bit_depth=options.bit_depth or audio.bit_depth,
            format=audio.format,
        )
        settings.validate()
        return settings

    def output_path(self, dialogue: Dialogue, options: GenerateOptions) -> Path:
        if options.output is not None:
            return Path(options.output)
        stem = dialogue.source.stem if dialogue.source else "voxlab"
        return self.config.output_dir / f"{stem}.{self.config.audio.format}"

    def generate(
        self, dialogue: Dialogue, options: GenerateOptions | None = None
    ) -> GenerateResult:
        options = options or GenerateOptions()
        started = time.perf_counter()
        settings = self.export_settings(options)
        plan = self.plan(dialogue, options)
        audio = self.render(plan, settings.sample_rate)
        path = export_audio(
            audio, settings.sample_rate, self.output_path(dialogue, options), settings
        )
        return GenerateResult(
            output=path,
            duration=audio.size / settings.sample_rate,
            lines=len(plan.jobs),
            cast={speaker: voice.name for speaker, voice in plan.cast.items()},
            preset=plan.preset.name,
            elapsed=time.perf_counter() - started,
        )


def shift_pitch(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Varispeed pitch shift: resample so playback is ``2**(st/12)`` faster/higher."""
    target = round(sample_rate / _pitch_ratio(semitones))
    return resample(audio, sample_rate, target)


def _pitch_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


def _number(value: Any, key: str, where: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DialogueError(f"{where}: parameter {key!r} must be a number, got {value!r}")
    if not low <= float(value) <= high:
        raise DialogueError(f"{where}: parameter {key!r} must be between {low:g} and {high:g}")
    return float(value)


def _warn_once(warned: set[str], key: str, message: str) -> None:
    if key not in warned:
        warned.add(key)
        log.warning(message)
