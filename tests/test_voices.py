import pytest

from voxlab.errors import UnknownSpeakerError, VoiceError
from voxlab.voices.manager import VoiceManager, parse_profile


@pytest.fixture
def manager(tmp_path):
    return VoiceManager(tmp_path / "voices")


def test_builtin_voices_exist(manager):
    assert {"default", "operator", "computer", "narrator"} <= set(manager.names())


def test_speaker_resolution_by_backend_and_language():
    voice = parse_profile(
        {"speaker": {"kokoro": {"es": "em_alex", "en": "am_adam"}, "other": "x"}}, "v", "user"
    )
    assert voice.speaker_for("kokoro", "es") == "em_alex"
    assert voice.speaker_for("kokoro", "en-gb") == "am_adam"
    assert voice.speaker_for("kokoro", "fr") is None
    assert voice.speaker_for("other", "es") == "x"
    assert voice.speaker_for("unknown", "es") is None
    assert parse_profile({"speaker": "id"}, "v", "user").speaker_for("any", "es") == "id"


def test_cast_matches_speaker_names_case_insensitively(manager):
    cast = manager.resolve_cast(["OPERATOR", "Computer"])
    assert cast["OPERATOR"].name == "operator"
    assert cast["Computer"].name == "computer"


def test_explicit_cast_wins(manager):
    cast = manager.resolve_cast(["OPERATOR"], cast={"operator": "deep"})
    assert cast["OPERATOR"].name == "deep"


def test_unknown_speaker_falls_back_to_default(manager, caplog):
    cast = manager.resolve_cast(["STRANGER"], default="narrator")
    assert cast["STRANGER"].name == "narrator"
    assert "STRANGER" in caplog.text


def test_unknown_speaker_strict_raises(manager):
    with pytest.raises(UnknownSpeakerError, match="STRANGER"):
        manager.resolve_cast(["STRANGER"], strict=True)


def test_cast_to_unknown_voice_raises(manager):
    with pytest.raises(VoiceError, match="Unknown voice 'ghost'"):
        manager.resolve_cast(["A"], cast={"A": "ghost"})


def test_user_profile_overrides_builtin(tmp_path):
    folder = tmp_path / "voices" / "operator"
    folder.mkdir(parents=True)
    (folder / "voice.yaml").write_text("description: mine\nspeaker: am_adam\nlanguage: EN\n")
    voice = VoiceManager(tmp_path / "voices").get("operator")
    assert voice.source == "user"
    assert voice.description == "mine"
    assert voice.language == "en"


def test_folder_with_only_reference_is_cloning_voice(tmp_path):
    folder = tmp_path / "voices" / "captain"
    folder.mkdir(parents=True)
    (folder / "sample.wav").write_bytes(b"RIFF")
    voice = VoiceManager(tmp_path / "voices").get("captain")
    assert voice.uses_cloning
    assert voice.reference == folder / "sample.wav"


def test_add_voice_with_reference(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    manager = VoiceManager(tmp_path / "voices")
    voice = manager.add("captain", reference=clip, speaker="em_alex", language="es")
    assert voice.reference == tmp_path / "voices" / "captain" / "reference.wav"
    assert voice.speaker == "em_alex"
    with pytest.raises(VoiceError, match="already exists"):
        manager.add("captain", speaker="em_alex")
    assert manager.add("captain", speaker="am_adam", overwrite=True).speaker == "am_adam"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "Bad Name", "speaker": "x"}, "Invalid voice name"),
        ({"name": "ok"}, "needs a reference"),
        ({"name": "ok", "reference": "missing.wav"}, "not found"),
    ],
)
def test_add_voice_errors(tmp_path, kwargs, message):
    if "reference" in kwargs:
        kwargs["reference"] = tmp_path / kwargs["reference"]
    with pytest.raises(VoiceError, match=message):
        VoiceManager(tmp_path / "voices").add(**kwargs)
    assert not (tmp_path / "voices" / kwargs["name"]).exists()


def test_add_voice_rejects_non_audio(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    with pytest.raises(VoiceError, match="Reference must be"):
        VoiceManager(tmp_path / "voices").add("x", reference=notes)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"colour": "red"}, "unknown key"),
        ({"params": [1, 2]}, "'params' must be a mapping"),
        ({"speaker": 3}, "'speaker' must be"),
        ({"reference": "nope.wav"}, "reference audio not found"),
        ("text", "must be a mapping"),
    ],
)
def test_invalid_profiles(tmp_path, data, message):
    with pytest.raises(VoiceError, match=message):
        parse_profile(data, "v", "user", base_dir=tmp_path)
