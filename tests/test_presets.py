import numpy as np
import pytest
from scipy.signal import welch

from voxlab.audio.effects import EFFECTS, apply_effect, validate_params
from voxlab.audio.presets import PresetLibrary, builtin_preset_dirs, parse_preset
from voxlab.errors import PresetError

BUILTIN = ["arcade_80s", "clean", "crt_terminal", "cyberpunk", "radio", "sci_fi"]


def band_energy(audio, sr, low, high):
    freqs, power = welch(audio, sr, nperseg=4096)
    return power[(freqs >= low) & (freqs < high)].sum()


def test_builtin_presets_are_found():
    assert builtin_preset_dirs()
    assert set(BUILTIN) <= set(PresetLibrary().names())


@pytest.mark.parametrize("name", BUILTIN)
def test_builtin_preset_output_is_sane(name, sine):
    audio, sr = sine
    out = PresetLibrary().get(name).apply(audio, sr)
    assert out.dtype == np.float32
    assert out.size >= audio.size  # time effects may add a tail
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 10 ** (-0.9 / 20)  # all presets end normalized to -1 dBFS


def test_clean_preset_barely_changes_audio(sine):
    audio, sr = sine
    out = PresetLibrary().get("clean").apply(audio, sr)
    # Compare magnitude spectra: filters may shift phase, but the content must not change.
    spectrum_in = np.abs(np.fft.rfft(audio[sr // 10 :]))
    spectrum_out = np.abs(np.fft.rfft(out[sr // 10 : audio.size]))
    assert np.corrcoef(spectrum_in, spectrum_out)[0, 1] > 0.999


def test_arcade_preset_reduces_bandwidth():
    sr = 48000
    rng = np.random.default_rng(0)
    noise = (0.1 * rng.standard_normal(sr)).astype(np.float32)
    out = PresetLibrary().get("arcade_80s").apply(noise, sr)
    ratio_in = band_energy(noise, sr, 12000, 20000) / band_energy(noise, sr, 500, 4000)
    ratio_out = band_energy(out, sr, 12000, 20000) / band_energy(out, sr, 500, 4000)
    assert ratio_out < ratio_in / 10


def test_user_preset_overrides_builtin(tmp_path, sine):
    (tmp_path / "radio.yaml").write_text("name: radio\ndescription: mine\nchain: []\n")
    preset = PresetLibrary([tmp_path]).get("radio")
    assert preset.description == "mine"
    audio, sr = sine
    assert np.array_equal(preset.apply(audio, sr), audio)


def test_preset_by_path(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("chain:\n  - effect: gain\n    db: -6\n")
    preset = PresetLibrary().get(str(path))
    assert preset.name == "custom"
    assert preset.chain[0].params == {"db": -6.0}


def test_disabled_steps_are_skipped():
    preset = parse_preset(
        {"chain": [{"effect": "gain", "db": 3, "enabled": False}, {"effect": "normalize"}]},
        name="p",
    )
    assert [s.effect for s in preset.chain] == ["normalize"]


def test_unknown_preset():
    with pytest.raises(PresetError, match="Unknown preset 'nope'"):
        PresetLibrary().get("nope")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"chain": [{"effect": "flanger"}]}, "unknown effect"),
        ({"chain": [{"effect": "gain"}]}, "requires parameter 'db'"),
        ({"chain": [{"effect": "gain", "db": "loud"}]}, "must be a number"),
        ({"chain": [{"effect": "highpass", "cutoff_hz": 100, "slope": 2}]}, "unknown parameter"),
        ({"chain": [{"effect": "bitcrush", "bits": 7.5}]}, "must be an integer"),
        ({"chain": [{"db": 3}]}, "must be a mapping with 'effect'"),
        ({"chain": {"effect": "gain"}}, "must be a list"),
        ({"chian": []}, "unknown key"),
        (["gain"], "must be a mapping"),
    ],
)
def test_invalid_presets(data, message):
    with pytest.raises(PresetError, match=message):
        parse_preset(data, name="bad")


def test_invalid_yaml_file(tmp_path):
    (tmp_path / "broken.yaml").write_text("chain: [\n")
    with pytest.raises(PresetError, match="Invalid YAML"):
        PresetLibrary([tmp_path]).get("broken")


def test_runtime_range_errors_mention_preset(sine):
    preset = parse_preset({"chain": [{"effect": "lowpass", "cutoff_hz": 30000}]}, name="hi")
    audio, sr = sine
    with pytest.raises(PresetError, match=r"preset 'hi'.*cutoff_hz"):
        preset.apply(audio, sr)


# --- individual effects -------------------------------------------------------


def test_every_effect_runs_with_defaults(sine):
    audio, sr = sine
    required = {
        "gain": {"db": -3},
        "highpass": {"cutoff_hz": 200},
        "lowpass": {"cutoff_hz": 3000},
        "bandpass": {"low_hz": 300, "high_hz": 3000},
        "peaking_eq": {"freq_hz": 1000, "gain_db": 3},
        "tone": {"freq_hz": 1000},
    }
    for name in EFFECTS:
        out = apply_effect(name, audio, sr, **required.get(name, {}))
        assert np.isfinite(out).all(), name


def test_lowpass_attenuates_high_frequencies():
    sr = 48000
    t = np.arange(sr) / sr
    high = np.sin(2 * np.pi * 8000 * t).astype(np.float32)
    out = apply_effect("lowpass", high, sr, cutoff_hz=1000)
    assert np.abs(out[sr // 2 :]).max() < 0.01


def test_bitcrush_quantizes():
    audio = np.linspace(-1, 1, 1000, dtype=np.float32)
    out = apply_effect("bitcrush", audio, 48000, bits=3)
    assert len(np.unique(out)) <= 2**3 + 1


def test_normalize_sets_peak(sine):
    audio, sr = sine
    out = apply_effect("normalize", audio, sr, peak_db=-6)
    assert np.isclose(np.abs(out).max(), 10 ** (-6 / 20), atol=1e-4)


def test_normalize_handles_silence():
    silent = np.zeros(100, dtype=np.float32)
    assert np.array_equal(apply_effect("normalize", silent, 48000), silent)


def test_compressor_reduces_dynamic_range():
    sr = 48000
    t = np.arange(sr) / sr
    loud_then_quiet = np.sin(2 * np.pi * 200 * t) * np.where(t < 0.5, 1.0, 0.1)
    out = apply_effect(
        "compressor", loud_then_quiet.astype(np.float32), sr, threshold_db=-30, ratio=8
    )
    ratio_in = np.abs(loud_then_quiet[: sr // 2]).max() / np.abs(loud_then_quiet[-sr // 4 :]).max()
    ratio_out = np.abs(out[sr // 4 : sr // 2]).max() / np.abs(out[-sr // 4 :]).max()
    assert ratio_out < ratio_in / 2


def test_delay_and_reverb_add_tail(sine):
    audio, sr = sine
    assert apply_effect("delay", audio, sr).size > audio.size
    assert apply_effect("reverb", audio, sr).size > audio.size


def test_noise_is_deterministic():
    silent = np.zeros(4800, dtype=np.float32)
    a = apply_effect("noise", silent, 48000, level_db=-40)
    b = apply_effect("noise", silent, 48000, level_db=-40)
    assert np.array_equal(a, b)
    rms_db = 20 * np.log10(np.sqrt(np.mean(a**2)))
    assert abs(rms_db + 40) < 0.5


def test_validate_params_returns_coerced_values():
    assert validate_params("bitcrush", {"bits": 8.0, "mix": 1}) == {"bits": 8, "mix": 1.0}
