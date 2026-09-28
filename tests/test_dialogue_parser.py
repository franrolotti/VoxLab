from pathlib import Path

import pytest

from voxlab.dialogue.parser import PLAIN_TEXT_SPEAKER, load_dialogue, parse_dialogue
from voxlab.errors import DialogueError, EmptyDialogueError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "dialogue.txt"


def test_parses_speakers_and_text():
    dialogue = parse_dialogue("[OPERATOR]\n¿Me recibes?\n\n[COMPUTER]\nAFIRMATIVO.\n")
    assert [(line.speaker, line.text) for line in dialogue.lines] == [
        ("OPERATOR", "¿Me recibes?"),
        ("COMPUTER", "AFIRMATIVO."),
    ]
    assert dialogue.speakers == ["OPERATOR", "COMPUTER"]


def test_parses_numeric_and_text_params():
    dialogue = parse_dialogue("[OPERATOR|speed=0.95|lang=en| Pitch = -2 ]\nHello\n")
    assert dialogue.lines[0].params == {"speed": 0.95, "lang": "en", "pitch": -2.0}


def test_multiline_block_is_joined_and_comments_skipped():
    text = "# intro\n[NARRATOR]\nLine one\n# note\nline two\n\nline three\n"
    assert parse_dialogue(text).lines[0].text == "Line one line two line three"


def test_line_numbers_point_to_headers():
    dialogue = parse_dialogue("\n\n[A]\nhola\n[B]\nadiós\n")
    assert [line.line_number for line in dialogue.lines] == [3, 5]


def test_plain_text_without_headers_is_narrated_by_paragraph():
    dialogue = parse_dialogue("First paragraph\ncontinues.\n\nSecond paragraph.\n")
    assert [line.speaker for line in dialogue.lines] == [PLAIN_TEXT_SPEAKER] * 2
    assert dialogue.lines[0].text == "First paragraph continues."


@pytest.mark.parametrize("text", ["", "   \n\n", "# only a comment\n"])
def test_empty_dialogue_raises(text):
    with pytest.raises(EmptyDialogueError):
        parse_dialogue(text)


def test_text_before_first_header_raises():
    with pytest.raises(DialogueError, match="line 1: text found before"):
        parse_dialogue("hola\n[A]\nadiós\n")


def test_header_without_text_raises():
    with pytest.raises(DialogueError, match=r"\[B\] has no text"):
        parse_dialogue("[A]\nhola\n[B]\n")


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("[]", "empty speaker"),
        ("[A|speed]", "expected 'key=value'"),
        ("[A|speed=]", "expected 'key=value'"),
        ("[A|speed=1|speed=2]", "duplicate parameter"),
        ("[A|1x=2]", "invalid parameter name"),
        ("[A/B]", "invalid speaker name"),
    ],
)
def test_invalid_headers(header, message):
    with pytest.raises(DialogueError, match=message):
        parse_dialogue(f"{header}\nhola\n")


def test_load_missing_file(tmp_path):
    with pytest.raises(DialogueError, match="not found"):
        load_dialogue(tmp_path / "missing.txt")


def test_load_directory_is_rejected(tmp_path):
    with pytest.raises(DialogueError, match="Not a file"):
        load_dialogue(tmp_path)


def test_load_non_utf8_file(tmp_path):
    path = tmp_path / "latin1.txt"
    path.write_bytes("[A]\nacción\n".encode("utf-16"))
    with pytest.raises(DialogueError, match="UTF-8"):
        load_dialogue(path)


def test_load_handles_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_text("[A]\nhola\n", encoding="utf-8-sig")
    assert load_dialogue(path).lines[0].speaker == "A"


def test_example_dialogue_parses():
    dialogue = load_dialogue(EXAMPLE)
    assert dialogue.speakers == ["OPERATOR", "COMPUTER"]
    assert len(dialogue) == 6
    assert dialogue.source == EXAMPLE
