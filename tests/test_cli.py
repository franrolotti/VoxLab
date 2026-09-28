import pytest
import soundfile as sf

from conftest import FakeBackend
from voxlab import cli


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Run the CLI inside an empty project that uses the fake backend."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOXLAB_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(f"storage:\n  model_dir: {tmp_path / 'models'}\n")
    backend = FakeBackend()
    monkeypatch.setattr("voxlab.pipeline.create_backend", lambda config: backend)
    monkeypatch.setattr(cli, "create_backend", lambda config: backend)
    return tmp_path


def run(capsys, *args):
    code = cli.main(list(args))
    return code, capsys.readouterr()


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "generate" in capsys.readouterr().out


def test_generate_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["generate", "--help"])
    out = capsys.readouterr().out
    assert "--preset" in out and "--output" in out


def test_no_command_prints_help(capsys):
    code, captured = run(capsys)
    assert code == 0
    assert "usage" in captured.out


def test_generate(project, dialogue_file, capsys):
    code, captured = run(capsys, "generate", str(dialogue_file), "--preset", "radio")
    assert code == 0
    output = project / "output" / "scene.wav"
    assert sf.info(output).samplerate == 48000
    assert "OPERATOR=operator" in captured.out


def test_generate_with_voice_override_and_output(project, dialogue_file, capsys):
    out = project / "custom" / "x.wav"
    code, captured = run(
        capsys,
        "generate",
        str(dialogue_file),
        "-o",
        str(out),
        "--voice",
        "OPERATOR=deep",
        "--bit-depth",
        "16",
    )
    assert code == 0
    assert sf.info(out).subtype == "PCM_16"
    assert "OPERATOR=deep" in captured.out


def test_generate_dry_run_writes_nothing(project, dialogue_file, capsys):
    code, captured = run(capsys, "generate", str(dialogue_file), "--dry-run")
    assert code == 0
    assert "COMPUTER" in captured.out
    assert not (project / "output").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["generate", "missing.txt"], "not found"),
        (["generate", "{dialogue}", "--preset", "nope"], "Unknown preset"),
        (["generate", "{dialogue}", "--voice", "OPERATOR"], "SPEAKER=VOICE"),
        (["generate", "{dialogue}", "--strict", "--voice", "OPERATOR=ghost"], "Unknown voice"),
    ],
)
def test_generate_errors(project, dialogue_file, capsys, caplog, args, message):
    args = [a.replace("{dialogue}", str(dialogue_file)) for a in args]
    code, _ = run(capsys, *args)
    assert code == 1
    assert message in caplog.text


def test_generate_empty_dialogue(project, capsys, caplog):
    (project / "empty.txt").write_text("\n# nothing\n")
    code, _ = run(capsys, "generate", "empty.txt")
    assert code == 1
    assert "empty" in caplog.text


def test_voices_list_and_add(project, capsys):
    code, captured = run(capsys, "voices")
    assert code == 0
    assert "operator" in captured.out
    code, _ = run(capsys, "voices", "add", "captain", "--speaker", "fake_b")
    assert code == 0
    assert (project / "voices" / "captain" / "voice.yaml").is_file()
    _, captured = run(capsys, "voices")
    assert "captain" in captured.out


def test_voices_speakers(project, capsys):
    code, captured = run(capsys, "voices", "--speakers")
    assert code == 0
    assert "fake_a" in captured.out


def test_presets(project, capsys):
    code, captured = run(capsys, "presets")
    assert code == 0
    for name in ("clean", "radio", "arcade_80s", "crt_terminal", "cyberpunk", "sci_fi"):
        assert name in captured.out


def test_models(project, capsys, monkeypatch):
    code, captured = run(capsys, "models")
    assert code == 0
    assert "kokoro" in captured.out


def test_benchmark(project, capsys):
    code, captured = run(capsys, "benchmark", "--json", str(project / "b.json"))
    assert code == 0
    assert "RTF" in captured.out
    assert (project / "b.json").is_file()


def test_clean(project, capsys):
    models = project / "models"
    (models / "hub").mkdir(parents=True)
    (models / "hub" / "weights.onnx").write_bytes(b"0" * 2048)
    code, captured = run(capsys, "clean", "--dry-run")
    assert code == 0 and models.exists()
    assert "2.0 KB" in captured.out
    code, _ = run(capsys, "clean", "--yes")
    assert code == 0
    assert not models.exists()
    code, captured = run(capsys, "clean", "--yes")
    assert "Nothing to clean" in captured.out


def test_clean_asks_for_confirmation(project, capsys, monkeypatch):
    (project / "models").mkdir()
    monkeypatch.setattr("builtins.input", lambda _: "n")
    code, _ = run(capsys, "clean")
    assert code == 1
    assert (project / "models").exists()
