import numpy as np
import pytest
import soundfile as sf

from conftest import CloningBackend, FakeBackend
from voxlab.benchmark import run_benchmark
from voxlab.dialogue.parser import parse_dialogue
from voxlab.errors import DialogueError, PresetError, UnknownSpeakerError, VoiceError
from voxlab.pipeline import GenerateOptions, Pipeline, shift_pitch


def make_pipeline(config, backend=None):
    return Pipeline(config, backend=backend or FakeBackend())


def test_generate_end_to_end(config, dialogue_file):
    from voxlab.dialogue.parser import load_dialogue

    backend = FakeBackend()
    result = make_pipeline(config, backend).generate(
        load_dialogue(dialogue_file), GenerateOptions(preset="arcade_80s")
    )
    assert result.output == config.output_dir / "scene.wav"
    assert result.cast == {"OPERATOR": "operator", "COMPUTER": "computer"}
    assert result.preset == "arcade_80s"
    info = sf.info(result.output)
    assert (info.samplerate, info.subtype) == (48000, "PCM_24")
    assert backend.loaded
    assert len(backend.requests) == 2


def test_same_engine_different_presets(config, dialogue_file, tmp_path):
    from voxlab.dialogue.parser import load_dialogue

    dialogue = load_dialogue(dialogue_file)
    pipeline = make_pipeline(config)
    clean = pipeline.generate(dialogue, GenerateOptions("clean", tmp_path / "c.wav"))
    arcade = pipeline.generate(dialogue, GenerateOptions("arcade_80s", tmp_path / "a.wav"))
    a, _ = sf.read(clean.output)
    b, _ = sf.read(arcade.output)
    assert not np.allclose(a[: min(a.size, b.size)], b[: min(a.size, b.size)])


def test_output_options(config, tmp_path):
    dialogue = parse_dialogue("[A]\nhola\n")
    result = make_pipeline(config).generate(
        dialogue, GenerateOptions(output=tmp_path / "x.wav", sample_rate=44100, bit_depth=16)
    )
    info = sf.info(result.output)
    assert (info.samplerate, info.subtype) == (44100, "PCM_16")


def test_params_are_routed(config):
    dialogue = parse_dialogue("[OPERATOR|speed=1.2|emotion=0.7|volume=-3|pause=800]\nhola\n")
    job = make_pipeline(config).plan(dialogue).jobs[0]
    assert job.request.params == {"speed": 1.2, "emotion": 0.7}
    assert job.volume_db == -3
    assert job.pause_ms == 800


def test_voice_params_are_defaults_for_lines(config):
    plan = make_pipeline(config).plan(parse_dialogue("[COMPUTER]\nuno\n[COMPUTER|pitch=0]\ndos\n"))
    # computer voice: speed 0.92, pitch -3 -> the model is asked to go slower/faster to compensate
    first, second = plan.jobs
    assert first.pitch == -3
    assert first.request.params["speed"] == pytest.approx(0.92 / 2 ** (-3 / 12))
    assert second.pitch == 0
    assert second.request.params["speed"] == pytest.approx(0.92)


def test_unsupported_params_are_ignored_with_warning(config, caplog):
    job = make_pipeline(config).plan(parse_dialogue("[A|whisper=1|style=calm]\nhola\n")).jobs[0]
    assert job.request.params == {}
    assert "'whisper' is not supported" in caplog.text


def test_speed_ignored_when_backend_lacks_it(config, caplog):
    job = make_pipeline(config, CloningBackend()).plan(parse_dialogue("[A|speed=1.5]\nhola\n"))
    assert job.jobs[0].request.params == {}
    assert "no speed control" in caplog.text


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("[A|speed=5]", "'speed' must be between"),
        ("[A|pitch=24]", "'pitch' must be between"),
        ("[A|volume=loud]", "must be a number"),
        ("[A|lang=xx]", "does not support language 'xx'"),
        ("[A|pitch=12|speed=0.6]", "outside the range"),
    ],
)
def test_invalid_line_params(config, header, message):
    with pytest.raises(DialogueError, match=message):
        make_pipeline(config).plan(parse_dialogue(f"{header}\nhola\n"))


def test_language_precedence(config):
    pipeline = make_pipeline(config)
    dialogue = parse_dialogue("[A]\nhola\n[A|lang=EN]\nhello\n")
    jobs = pipeline.plan(dialogue, GenerateOptions(language="en")).jobs
    assert [j.request.language for j in jobs] == ["en", "en"]
    jobs = pipeline.plan(dialogue).jobs
    assert [j.request.language for j in jobs] == ["es", "en"]


def test_unknown_speaker_strict(config):
    with pytest.raises(UnknownSpeakerError):
        make_pipeline(config).plan(parse_dialogue("[GHOST]\nboo\n"), GenerateOptions(strict=True))


def test_unknown_preset_fails_before_synthesis(config):
    backend = FakeBackend()
    with pytest.raises(PresetError):
        make_pipeline(config, backend).generate(
            parse_dialogue("[A]\nhola\n"), GenerateOptions(preset="nope")
        )
    assert not backend.loaded


def test_cast_override(config):
    plan = make_pipeline(config).plan(
        parse_dialogue("[OPERATOR]\nhola\n"), GenerateOptions(cast={"OPERATOR": "deep"})
    )
    assert plan.cast["OPERATOR"].name == "deep"


def _cloning_voice(config, with_speaker):
    folder = config.voices_dir / "clone"
    folder.mkdir(parents=True)
    (folder / "ref.wav").write_bytes(b"RIFF")
    extra = "speaker: fake_b\n" if with_speaker else ""
    (folder / "voice.yaml").write_text(f"reference: ref.wav\n{extra}")


def test_cloning_voice_on_backend_without_cloning_raises(config):
    _cloning_voice(config, with_speaker=False)
    with pytest.raises(VoiceError, match="voice cloning"):
        make_pipeline(config).plan(parse_dialogue("[CLONE]\nhola\n"))


def test_cloning_voice_falls_back_to_speaker(config, caplog):
    _cloning_voice(config, with_speaker=True)
    job = make_pipeline(config).plan(parse_dialogue("[CLONE]\nhola\n")).jobs[0]
    assert job.request.reference_audio is None
    assert job.request.speaker == "fake_b"
    assert "cannot clone" in caplog.text


def test_cloning_backend_receives_reference(config):
    _cloning_voice(config, with_speaker=False)
    job = make_pipeline(config, CloningBackend()).plan(parse_dialogue("[CLONE]\nhola\n")).jobs[0]
    assert job.request.reference_audio == config.voices_dir / "clone" / "ref.wav"


def test_per_voice_preset_is_applied(config):
    folder = config.voices_dir / "robot"
    folder.mkdir(parents=True)
    (folder / "voice.yaml").write_text("speaker: fake_a\npreset: sci_fi\n")
    plan = make_pipeline(config).plan(parse_dialogue("[ROBOT]\nhola\n"))
    assert plan.voice_presets["robot"].name == "sci_fi"


def test_shift_pitch_changes_length():
    audio = np.zeros(24000, dtype=np.float32)
    assert shift_pitch(audio, 24000, 12).size == pytest.approx(12000, abs=2)
    assert shift_pitch(audio, 24000, -12).size == pytest.approx(48000, abs=2)


def test_benchmark_with_fake_backend():
    report = run_benchmark(FakeBackend(), languages=["es", "fr"])
    assert {r.language for r in report.results} == {"es"}
    assert report.rtf > 0
    assert report.to_dict()["results"][0]["rtf"] >= 0
