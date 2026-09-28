import numpy as np
import pytest

from conftest import FakeBackend
from voxlab.config import config_from_dict
from voxlab.errors import BackendError, VoxLabError
from voxlab.storage import clean_model_dir, dir_size, human_size
from voxlab.tts import factory
from voxlab.tts.base import SynthesisRequest
from voxlab.tts.kokoro import KokoroBackend


@pytest.fixture
def restore_registry():
    registry, order = dict(factory._REGISTRY), list(factory.AUTO_ORDER)
    yield
    factory._REGISTRY.clear()
    factory._REGISTRY.update(registry)
    factory.AUTO_ORDER[:] = order


def test_auto_resolves_to_kokoro():
    assert factory.resolve_backend_name("auto") == "kokoro"


def test_unknown_backend():
    with pytest.raises(BackendError, match="Unknown TTS backend"):
        factory.get_backend_class("nope")


def test_register_custom_backend(restore_registry, tmp_path):
    factory.register_backend("fake", FakeBackend, prefer=True)
    config = config_from_dict({"storage": {"model_dir": str(tmp_path)}})
    backend = factory.create_backend(config)
    assert isinstance(backend, FakeBackend)
    assert backend.model_dir == tmp_path


def test_broken_backend_path(restore_registry):
    factory.register_backend("broken", "voxlab.does_not_exist:Nope")
    with pytest.raises(BackendError, match="Could not load backend"):
        factory.get_backend_class("broken")


def test_supports_language_with_region():
    backend = FakeBackend()
    assert backend.supports_language("es")
    assert backend.supports_language("EN-GB")
    assert not backend.supports_language("fr")


# --- Kokoro backend without the model -----------------------------------------


@pytest.fixture
def kokoro(tmp_path):
    backend = KokoroBackend(model_dir=tmp_path)
    style = np.ones((510, 1, 256), dtype=np.float32)
    backend._voices = {"ef_dora": style, "em_alex": 3 * style}
    return backend


def test_kokoro_capabilities():
    caps = KokoroBackend.capabilities
    assert not caps.voice_cloning
    assert "speed" in caps.native_params
    assert "es" in caps.languages and "en" in caps.languages


def test_kokoro_voice_blend(kokoro):
    blended = kokoro._style("ef_dora:0.5,em_alex:0.5")
    assert np.allclose(blended, 2.0)
    assert np.allclose(kokoro._style("em_alex"), 3.0)


@pytest.mark.parametrize("speaker", ["ghost", "ef_dora:x", "ef_dora:0,em_alex:0", None])
def test_kokoro_invalid_speakers(kokoro, speaker):
    with pytest.raises(BackendError):
        kokoro._style(speaker)


def test_kokoro_rejects_cloning_and_bad_language(kokoro, tmp_path):
    with pytest.raises(BackendError, match="voice cloning"):
        kokoro.synthesize(SynthesisRequest("hola", "es", reference_audio=tmp_path / "a.wav"))
    with pytest.raises(BackendError, match="language 'de'"):
        kokoro.synthesize(SynthesisRequest("hallo", "de"))


def test_kokoro_default_speakers(kokoro):
    assert kokoro.default_speaker("es") == "ef_dora"
    assert kokoro.default_speaker("en-us") == "af_heart"
    assert kokoro.default_speaker("xx") is None


def test_kokoro_speaker_listing(kokoro):
    ids = {(s.id, s.language, s.gender) for s in kokoro.speakers()}
    assert ("em_alex", "es", "male") in ids


def test_kokoro_invalid_variant(tmp_path):
    with pytest.raises(BackendError, match="variant"):
        KokoroBackend(model_dir=tmp_path, options={"variant": "fp64"})


def test_kokoro_without_model_and_no_download(tmp_path):
    backend = KokoroBackend(model_dir=tmp_path, allow_download=False)
    assert not backend.is_downloaded()
    with pytest.raises(BackendError, match="auto-download is disabled"):
        backend.load()


def test_kokoro_device_selection(tmp_path):
    backend = KokoroBackend(model_dir=tmp_path, device="cuda")
    with pytest.raises(BackendError, match="CUDAExecutionProvider"):
        backend._select_providers(["CPUExecutionProvider"])
    backend.device = "auto"
    assert backend._select_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


# --- storage ------------------------------------------------------------------


def test_human_size():
    assert human_size(512) == "512 B"
    assert human_size(2048) == "2.0 KB"
    assert human_size(5 * 1024**3) == "5.0 GB"


def test_clean_model_dir(tmp_path):
    target = tmp_path / "cache" / "voxlab"
    target.mkdir(parents=True)
    (target / "model.onnx").write_bytes(b"0" * 100)
    assert dir_size(target) == 100
    assert clean_model_dir(target, dry_run=True) == 100
    assert target.exists()
    assert clean_model_dir(target) == 100
    assert not target.exists()
    assert clean_model_dir(target) == 0


def test_clean_refuses_dangerous_paths(tmp_path, monkeypatch):
    from pathlib import Path

    with pytest.raises(VoxLabError, match="unsafe"):
        clean_model_dir(Path.home())
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text("")
    with pytest.raises(VoxLabError, match="project folder"):
        clean_model_dir(project)


def test_kokoro_spanish_dialects(kokoro):
    class FakeTokenizer:
        def phonemize(self, text, lang):
            return {"es": "ʎˈamo θeθˈilja", "es-419": "ʝˈamo sesˈilja kˈajje"}[lang]

    kokoro._engine = type("Engine", (), {"tokenizer": FakeTokenizer()})()
    assert kokoro.phonemize("x", "es-es") == "ʎˈamo θeθˈilja"
    assert kokoro.phonemize("x", "es-419") == "ʝˈamo sesˈilja kˈajje"
    assert kokoro.phonemize("x", "es-AR") == "ʃˈamo sesˈilja kˈaʃe"
    assert kokoro.supports_language("es-ar")
    assert kokoro.default_speaker("es-ar") == "ef_dora"
