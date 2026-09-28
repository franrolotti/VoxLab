from pathlib import Path

import pytest

from voxlab.config import Config, config_from_dict, load_config
from voxlab.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_defaults():
    config = Config()
    assert config.audio.sample_rate == 48000
    assert config.audio.bit_depth == 24
    assert config.audio.format == "wav"
    assert config.tts.backend == "auto"
    assert config.preset.default == "clean"


def test_repository_config_is_valid():
    config = load_config(REPO_CONFIG)
    assert config.source == REPO_CONFIG
    assert config.base_dir == REPO_CONFIG.parent


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_no_config_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOXLAB_CONFIG", raising=False)
    config = load_config()
    assert config.source is None
    assert config.audio.sample_rate == 48000


def test_env_var_selects_config(tmp_path, monkeypatch):
    path = tmp_path / "custom.yaml"
    path.write_text("audio:\n  bit_depth: 16\n")
    monkeypatch.setenv("VOXLAB_CONFIG", str(path))
    assert load_config().audio.bit_depth == 16


def test_partial_config_merges_with_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("audio:\n  sample_rate: 44100\ncast:\n  CAPTAIN: deep\n")
    config = load_config(path)
    assert config.audio.sample_rate == 44100
    assert config.audio.bit_depth == 24
    assert config.cast == {"CAPTAIN": "deep"}


def test_relative_paths_resolve_against_config_dir(tmp_path):
    config = config_from_dict({"voice": {"dir": "my_voices"}}, base_dir=tmp_path)
    assert config.voices_dir == tmp_path / "my_voices"
    assert config_from_dict({"storage": {"model_dir": "~/x"}}).model_dir == Path.home() / "x"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"audio": {"sample_rate": 12345}}, "sample_rate"),
        ({"audio": {"bit_depth": 32}}, "bit_depth"),
        ({"audio": {"format": "mp3"}}, "format"),
        ({"audio": {"gap_ms": -1}}, "gap_ms"),
        ({"audio": {"sample_rate": "48000"}}, "must be of type int"),
        ({"audio": {"sampel_rate": 48000}}, "Unknown key"),
        ({"unknown_section": {}}, "Unknown key"),
        ({"tts": "kokoro"}, "must be a mapping"),
        ({"voice": {"strict": None}}, "must not be empty"),
        ({"tts": {"language": ""}}, "language"),
    ],
)
def test_invalid_values(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_invalid_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("audio: [unclosed\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_top_level_must_be_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)
