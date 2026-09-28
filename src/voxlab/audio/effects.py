"""Audio effects used by presets.

Every effect is a function ``(audio, sample_rate, **params) -> audio`` working
on mono float32 arrays, registered by name with :func:`effect`. Presets refer
to effects by that name, and their parameters are validated against the
function signature.
"""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from voxlab.audio.dsp import db_to_gain, delayed, peak, recursive_delay

EffectFn = Callable[..., np.ndarray]
EFFECTS: dict[str, EffectFn] = {}

MAX_TAIL_SECONDS = 4.0


def effect(name: str) -> Callable[[EffectFn], EffectFn]:
    def register(fn: EffectFn) -> EffectFn:
        EFFECTS[name] = fn
        return fn

    return register


def validate_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Check and coerce preset parameters for effect ``name``.

    Raises ``ValueError`` with a readable message when something is wrong.
    """
    if name not in EFFECTS:
        raise ValueError(f"unknown effect {name!r}; available: {', '.join(sorted(EFFECTS))}")
    fn = EFFECTS[name]
    signature = inspect.signature(fn)
    hints = get_type_hints(fn)
    accepted = list(signature.parameters)[2:]  # skip audio, sample_rate

    unknown = set(params) - set(accepted)
    if unknown:
        raise ValueError(
            f"effect {name!r} got unknown parameter(s) {', '.join(sorted(unknown))}; "
            f"accepted: {', '.join(accepted)}"
        )
    coerced: dict[str, Any] = {}
    for key in accepted:
        parameter = signature.parameters[key]
        if key not in params:
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(f"effect {name!r} requires parameter {key!r}")
            continue
        coerced[key] = _coerce(name, key, params[key], hints.get(key))
    return coerced


def apply_effect(name: str, audio: np.ndarray, sample_rate: int, **params: Any) -> np.ndarray:
    params = validate_params(name, params)
    out = EFFECTS[name](audio, sample_rate, **params)
    return np.asarray(out, dtype=np.float32)


def _coerce(effect_name: str, key: str, value: Any, annotation: Any) -> Any:
    label = f"effect {effect_name!r} parameter {key!r}"
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number, got {value!r}")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            raise ValueError(f"{label} must be an integer, got {value!r}")
        return int(value)
    if annotation is str and not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {value!r}")
    return value


def _mix(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    _check_range("mix", mix, 0.0, 1.0)
    return (1.0 - mix) * dry + mix * wet


def _check_range(key: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}, got {value}")


def _check_frequency(key: str, hz: float, sample_rate: int) -> None:
    nyquist = sample_rate / 2
    if not 0 < hz < nyquist:
        raise ValueError(f"{key} must be between 0 and {nyquist:g} Hz, got {hz:g}")


def _with_tail(audio: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    tail = int(min(seconds, MAX_TAIL_SECONDS) * sample_rate)
    return np.concatenate([audio, np.zeros(tail, dtype=audio.dtype)]) if tail > 0 else audio


# --- Level -----------------------------------------------------------------


@effect("gain")
def gain(audio: np.ndarray, sample_rate: int, db: float) -> np.ndarray:
    return audio * db_to_gain(db)


@effect("normalize")
def normalize(audio: np.ndarray, sample_rate: int, peak_db: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits at ``peak_db`` dBFS."""
    current = peak(audio)
    if current < 1e-9:
        return audio
    return audio * (db_to_gain(peak_db) / current)


@effect("compressor")
def compressor(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """Feed-forward peak compressor evaluated on 1 ms blocks."""
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    hop = max(1, sample_rate // 1000)
    blocks = -(-audio.size // hop)
    padded = np.zeros(blocks * hop, dtype=np.float32)
    padded[: audio.size] = np.abs(audio)
    level_db = 20 * np.log10(np.maximum(padded.reshape(blocks, hop).max(axis=1), 1e-9))

    attack = np.exp(-1.0 / max(attack_ms, 0.01))  # coefficients per 1 ms block
    release = np.exp(-1.0 / max(release_ms, 0.01))
    envelope = np.empty_like(level_db)
    state = level_db[0] if blocks else 0.0
    for i, level in enumerate(level_db):
        coeff = attack if level > state else release
        state = coeff * state + (1 - coeff) * level
        envelope[i] = state

    over = np.maximum(envelope - threshold_db, 0.0)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    centers = np.arange(blocks) * hop + hop / 2
    sample_gain = 10 ** (np.interp(np.arange(audio.size), centers, gain_db) / 20)
    return audio * sample_gain.astype(np.float32)


# --- Filters ---------------------------------------------------------------


@effect("highpass")
def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    _check_frequency("cutoff_hz", cutoff_hz, sample_rate)
    sos = butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfilt(sos, audio)


@effect("lowpass")
def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    _check_frequency("cutoff_hz", cutoff_hz, sample_rate)
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return sosfilt(sos, audio)


@effect("bandpass")
def bandpass(
    audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    _check_frequency("low_hz", low_hz, sample_rate)
    _check_frequency("high_hz", high_hz, sample_rate)
    if low_hz >= high_hz:
        raise ValueError(f"low_hz ({low_hz:g}) must be below high_hz ({high_hz:g})")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return sosfilt(sos, audio)


@effect("peaking_eq")
def peaking_eq(
    audio: np.ndarray, sample_rate: int, freq_hz: float, gain_db: float, q: float = 1.0
) -> np.ndarray:
    """Bell EQ (RBJ cookbook biquad)."""
    _check_frequency("freq_hz", freq_hz, sample_rate)
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")
    a_gain = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sample_rate
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain]
    a = [1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain]
    return lfilter(b, a, audio)


# --- Colour and distortion -------------------------------------------------


@effect("saturation")
def saturation(
    audio: np.ndarray, sample_rate: int, drive: float = 2.0, mix: float = 1.0
) -> np.ndarray:
    """Soft tanh clipping; ``drive`` > 1 adds harmonics."""
    if drive <= 0:
        raise ValueError(f"drive must be > 0, got {drive}")
    wet = np.tanh(drive * audio) / np.tanh(drive)
    return _mix(audio, wet, mix)


@effect("bitcrush")
def bitcrush(
    audio: np.ndarray,
    sample_rate: int,
    bits: int = 8,
    downsample_hz: float = 0.0,
    mix: float = 1.0,
) -> np.ndarray:
    """Reduce bit depth and (optionally) sample rate with sample-and-hold."""
    _check_range("bits", bits, 2, 24)
    wet = audio
    if downsample_hz:
        _check_frequency("downsample_hz", downsample_hz / 2, sample_rate)
        step = sample_rate / downsample_hz
        held = (np.floor(np.arange(audio.size) / step) * step).astype(np.int64)
        wet = audio[np.minimum(held, audio.size - 1)]
    levels = 2 ** (bits - 1)
    wet = np.round(wet * levels) / levels
    return _mix(audio, wet, mix)


@effect("ring_mod")
def ring_mod(
    audio: np.ndarray, sample_rate: int, freq_hz: float = 40.0, mix: float = 0.5
) -> np.ndarray:
    """Multiply by a sine carrier: the classic robot/alien voice."""
    _check_frequency("freq_hz", freq_hz, sample_rate)
    t = np.arange(audio.size) / sample_rate
    return _mix(audio, audio * np.sin(2 * np.pi * freq_hz * t), mix)


# --- Added signals ---------------------------------------------------------


@effect("noise")
def noise(
    audio: np.ndarray,
    sample_rate: int,
    level_db: float = -50.0,
    color: str = "pink",
    seed: int = 0,
) -> np.ndarray:
    """Add white or pink noise at ``level_db`` dBFS RMS."""
    if color not in ("white", "pink"):
        raise ValueError(f"color must be 'white' or 'pink', got {color!r}")
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(audio.size)
    if color == "pink" and audio.size > 1:
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(audio.size)
        spectrum[1:] /= np.sqrt(freqs[1:])
        spectrum[0] = 0
        signal = np.fft.irfft(spectrum, n=audio.size)
    rms = np.sqrt(np.mean(signal**2)) or 1.0
    return audio + signal / rms * db_to_gain(level_db)


@effect("tone")
def tone(
    audio: np.ndarray, sample_rate: int, freq_hz: float, level_db: float = -50.0
) -> np.ndarray:
    """Add a constant sine tone (e.g. CRT flyback whine or mains hum)."""
    _check_frequency("freq_hz", freq_hz, sample_rate)
    t = np.arange(audio.size) / sample_rate
    return audio + np.sin(2 * np.pi * freq_hz * t) * db_to_gain(level_db)


# --- Time-based ------------------------------------------------------------


@effect("chorus")
def chorus(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float = 0.8,
    depth_ms: float = 2.0,
    delay_ms: float = 15.0,
    mix: float = 0.3,
) -> np.ndarray:
    """Single-voice chorus: a copy with a slowly modulated delay."""
    if depth_ms >= delay_ms:
        raise ValueError("depth_ms must be smaller than delay_ms")
    n = np.arange(audio.size)
    lag = (delay_ms + depth_ms * np.sin(2 * np.pi * rate_hz * n / sample_rate)) * sample_rate / 1000
    wet = np.interp(n - lag, n, audio, left=0.0)
    return _mix(audio, wet, mix)


@effect("delay")
def delay(
    audio: np.ndarray,
    sample_rate: int,
    time_ms: float = 250.0,
    feedback: float = 0.3,
    mix: float = 0.2,
) -> np.ndarray:
    """Feedback echo. Extends the audio so the repeats are not cut off."""
    _check_range("feedback", feedback, 0.0, 0.95)
    samples = max(1, int(time_ms * sample_rate / 1000))
    repeats = np.log(1e-3) / np.log(feedback) if feedback > 0 else 1
    audio = _with_tail(audio, sample_rate, samples * (repeats + 1) / sample_rate)
    wet = recursive_delay(delayed(audio, samples), samples, feedback)
    return audio + mix * wet


_COMB_MS = (29.7, 37.1, 41.1, 43.7)
_ALLPASS_MS = (5.0, 1.7)


@effect("reverb")
def reverb(
    audio: np.ndarray,
    sample_rate: int,
    room_size: float = 0.5,
    damping: float = 0.4,
    mix: float = 0.2,
    predelay_ms: float = 0.0,
) -> np.ndarray:
    """Compact Schroeder reverb: parallel combs, series all-passes, damped tail."""
    _check_range("room_size", room_size, 0.0, 1.0)
    _check_range("damping", damping, 0.0, 1.0)
    feedback = 0.7 + 0.28 * room_size
    longest = max(_COMB_MS) / 1000
    audio = _with_tail(audio, sample_rate, longest * np.log(1e-3) / np.log(feedback))

    source = delayed(audio, int(predelay_ms * sample_rate / 1000))
    wet = np.zeros_like(audio)
    for ms in _COMB_MS:
        d = max(1, int(ms * sample_rate / 1000))
        wet += recursive_delay(delayed(source, d), d, feedback)
    wet /= len(_COMB_MS)
    for ms in _ALLPASS_MS:
        d = max(1, int(ms * sample_rate / 1000))
        wet = recursive_delay(delayed(wet, d) - 0.5 * wet, d, 0.5)
    if damping > 0:
        cutoff = 12000 - 10000 * damping
        wet = lowpass(wet, sample_rate, min(cutoff, sample_rate * 0.45), order=1)
    return _mix(audio, wet, mix)
