"""Assemble synthesized lines into a single track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxlab.audio.dsp import db_to_gain


@dataclass(frozen=True)
class Segment:
    audio: np.ndarray
    # Silence after this segment; ``None`` uses the mixer default gap.
    pause_ms: float | None = None


def silence(sample_rate: int, ms: float) -> np.ndarray:
    return np.zeros(max(0, int(sample_rate * ms / 1000)), dtype=np.float32)


def apply_fades(audio: np.ndarray, sample_rate: int, ms: float = 5.0) -> np.ndarray:
    """Short linear fade-in/out to avoid clicks at segment edges."""
    n = min(int(sample_rate * ms / 1000), audio.size // 2)
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def trim_silence(
    audio: np.ndarray, sample_rate: int, threshold_db: float = -50.0, keep_ms: float = 30.0
) -> np.ndarray:
    """Remove leading/trailing silence, keeping a small margin."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > db_to_gain(threshold_db))
    if loud.size == 0:
        return audio[:0]
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, loud[0] - keep)
    end = min(audio.size, loud[-1] + keep + 1)
    return audio[start:end]


def mix_sequence(
    segments: list[Segment], sample_rate: int, gap_ms: float = 350.0, padding_ms: float = 250.0
) -> np.ndarray:
    """Concatenate segments with gaps between them and padding around the track."""
    parts: list[np.ndarray] = [silence(sample_rate, padding_ms)]
    for index, segment in enumerate(segments):
        parts.append(apply_fades(segment.audio.astype(np.float32), sample_rate))
        if index < len(segments) - 1:
            pause = gap_ms if segment.pause_ms is None else segment.pause_ms
            parts.append(silence(sample_rate, pause))
    parts.append(silence(sample_rate, padding_ms))
    return np.concatenate(parts)
