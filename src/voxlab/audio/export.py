"""Write audio to disk (WAV, PCM 16/24-bit)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voxlab.audio.dsp import peak, resample
from voxlab.errors import ExportError

log = logging.getLogger(__name__)

SUPPORTED_SAMPLE_RATES = (22050, 24000, 44100, 48000)
SUPPORTED_BIT_DEPTHS = (16, 24)
SUPPORTED_FORMATS = ("wav",)

_SUBTYPES = {16: "PCM_16", 24: "PCM_24"}


@dataclass(frozen=True)
class ExportSettings:
    sample_rate: int = 48000
    bit_depth: int = 24
    format: str = "wav"

    def validate(self) -> None:
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ExportError(
                f"Unsupported sample rate {self.sample_rate}; use one of {SUPPORTED_SAMPLE_RATES}"
            )
        if self.bit_depth not in SUPPORTED_BIT_DEPTHS:
            raise ExportError(
                f"Unsupported bit depth {self.bit_depth}; use one of {SUPPORTED_BIT_DEPTHS}"
            )
        if self.format.lower() not in SUPPORTED_FORMATS:
            raise ExportError(f"Unsupported format {self.format!r}; use one of {SUPPORTED_FORMATS}")


def prepare_for_export(
    audio: np.ndarray, sample_rate: int, settings: ExportSettings, seed: int = 0
) -> np.ndarray:
    """Resample, prevent clipping and dither (16-bit) ready for integer PCM."""
    settings.validate()
    out = resample(audio, sample_rate, settings.sample_rate)
    level = peak(out)
    if level > 1.0:
        log.warning("Audio peaks at %.2f (above 0 dBFS); scaling down to avoid clipping", level)
        out = out / level * 0.999
    if settings.bit_depth == 16:
        # TPDF dither: +/- 1 LSB triangular noise masks quantisation distortion.
        rng = np.random.default_rng(seed)
        lsb = 1.0 / 32768
        out = out + (rng.random(out.size) - rng.random(out.size)) * lsb
        out = np.clip(out, -1.0, 1.0 - lsb)
    return out.astype(np.float32)


def export_audio(
    audio: np.ndarray, sample_rate: int, path: str | Path, settings: ExportSettings
) -> Path:
    """Write ``audio`` (mono float, any rate) to ``path`` using ``settings``."""
    settings.validate()
    path = Path(path)
    if path.suffix.lower() != f".{settings.format.lower()}":
        raise ExportError(f"Output path {path} must end in .{settings.format.lower()}")
    data = prepare_for_export(audio, sample_rate, settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, settings.sample_rate, subtype=_SUBTYPES[settings.bit_depth])
    except (OSError, sf.LibsndfileError) as exc:
        raise ExportError(f"Could not write {path}: {exc}") from exc
    log.info(
        "Wrote %s (%.1f s, %d Hz, %d-bit)",
        path,
        data.size / settings.sample_rate,
        settings.sample_rate,
        settings.bit_depth,
    )
    return path
