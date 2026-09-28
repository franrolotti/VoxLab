"""Small DSP helpers shared by the audio modules."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Return a 1-D float32 array (averaging channels if needed)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0)
    return np.ascontiguousarray(audio.reshape(-1), dtype=np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resampling between arbitrary rates."""
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    ratio = Fraction(target_sr, orig_sr).limit_denominator(1000)
    out = resample_poly(audio, ratio.numerator, ratio.denominator)
    return out.astype(np.float32)


def recursive_delay(u: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Compute ``y[n] = u[n] + gain * y[n - delay]`` efficiently.

    Long feedback delays make direct IIR filtering expensive, so the signal is
    processed in blocks of ``delay`` samples where the recurrence becomes a
    simple vector operation.
    """
    n = u.size
    blocks = -(-n // delay)
    padded = np.zeros(blocks * delay, dtype=np.float64)
    padded[:n] = u
    y = padded.reshape(blocks, delay)
    for k in range(1, blocks):
        y[k] += gain * y[k - 1]
    return y.reshape(-1)[:n].astype(np.float32)


def delayed(x: np.ndarray, delay: int) -> np.ndarray:
    """Shift ``x`` right by ``delay`` samples, keeping its length."""
    if delay <= 0:
        return x.copy()
    out = np.zeros_like(x)
    if delay < x.size:
        out[delay:] = x[:-delay]
    return out
