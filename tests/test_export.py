import numpy as np
import pytest
import soundfile as sf

from voxlab.audio.dsp import resample
from voxlab.audio.export import ExportSettings, export_audio
from voxlab.audio.mixer import Segment, mix_sequence, trim_silence
from voxlab.errors import ExportError


@pytest.mark.parametrize(
    ("sample_rate", "bit_depth", "subtype"),
    [(48000, 24, "PCM_24"), (44100, 16, "PCM_16"), (48000, 16, "PCM_16"), (44100, 24, "PCM_24")],
)
def test_export_formats(tmp_path, sample_rate, bit_depth, subtype):
    sr = 24000
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    path = export_audio(audio, sr, tmp_path / "out.wav", ExportSettings(sample_rate, bit_depth))
    info = sf.info(path)
    assert info.samplerate == sample_rate
    assert info.subtype == subtype
    assert info.channels == 1
    assert abs(info.duration - 1.0) < 0.01


def test_export_default_is_48k_24bit(tmp_path):
    path = export_audio(np.zeros(100, np.float32), 48000, tmp_path / "a.wav", ExportSettings())
    info = sf.info(path)
    assert (info.samplerate, info.subtype) == (48000, "PCM_24")


def test_export_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "er" / "out.wav"
    export_audio(np.zeros(10, np.float32), 48000, path, ExportSettings())
    assert path.is_file()


def test_export_prevents_clipping(tmp_path):
    loud = np.full(1000, 2.0, dtype=np.float32)
    path = export_audio(loud, 48000, tmp_path / "loud.wav", ExportSettings())
    data, _ = sf.read(path)
    assert np.abs(data).max() <= 1.0


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (ExportSettings(sample_rate=12345), "sample rate"),
        (ExportSettings(bit_depth=8), "bit depth"),
        (ExportSettings(format="mp3"), "format"),
    ],
)
def test_invalid_settings(tmp_path, settings, message):
    with pytest.raises(ExportError, match=message):
        export_audio(np.zeros(10, np.float32), 48000, tmp_path / "x.wav", settings)


def test_wrong_extension(tmp_path):
    with pytest.raises(ExportError, match=r"\.wav"):
        export_audio(np.zeros(10, np.float32), 48000, tmp_path / "x.mp3", ExportSettings())


def test_resample_changes_length():
    audio = np.zeros(24000, dtype=np.float32)
    assert resample(audio, 24000, 48000).size == 48000
    assert resample(audio, 24000, 44100).size == 44100
    assert resample(audio, 24000, 24000) is audio


def test_mix_sequence_gaps_and_padding():
    sr = 1000
    a = Segment(np.ones(100, np.float32))
    b = Segment(np.ones(50, np.float32), pause_ms=999)  # last pause is not added
    mix = mix_sequence([a, b], sr, gap_ms=200, padding_ms=100)
    assert mix.size == 100 + 100 + 200 + 50 + 100


def test_mix_sequence_uses_custom_pause():
    sr = 1000
    segments = [Segment(np.ones(10, np.float32), pause_ms=500), Segment(np.ones(10, np.float32))]
    assert mix_sequence(segments, sr, gap_ms=100, padding_ms=0).size == 10 + 500 + 10


def test_trim_silence():
    sr = 1000
    audio = np.concatenate([np.zeros(500), np.ones(100), np.zeros(500)]).astype(np.float32)
    trimmed = trim_silence(audio, sr, keep_ms=10)
    assert trimmed.size == 100 + 2 * 10
    assert trim_silence(np.zeros(100, np.float32), sr).size == 0
