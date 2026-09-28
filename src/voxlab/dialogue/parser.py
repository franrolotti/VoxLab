"""Parser for VoxLab dialogue files.

Format::

    # comments start with '#'
    [OPERATOR]
    ¿Me recibes?

    [COMPUTER|speed=0.9|pitch=-2]
    AFIRMATIVO.

A block starts with a ``[SPEAKER]`` header, optionally followed by
``|key=value`` parameters, and contains every non-empty line until the next
header. A file without any header is treated as plain text read by
``PLAIN_TEXT_SPEAKER``, one line per paragraph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from voxlab.dialogue.models import Dialogue, DialogueLine, ParamValue
from voxlab.errors import DialogueError, EmptyDialogueError

PLAIN_TEXT_SPEAKER = "NARRATOR"

_HEADER_RE = re.compile(r"^\[(?P<body>[^\[\]]*)\]$")
_SPEAKER_RE = re.compile(r"^[\w][\w .\-]*$")
_PARAM_KEY_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass
class _Block:
    speaker: str
    params: dict[str, ParamValue]
    line_number: int
    text_lines: list[str] = field(default_factory=list)


def load_dialogue(path: str | Path) -> Dialogue:
    """Read and parse a dialogue file."""
    path = Path(path)
    if not path.exists():
        raise DialogueError(f"Dialogue file not found: {path}")
    if not path.is_file():
        raise DialogueError(f"Not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DialogueError(f"{path} is not valid UTF-8 text") from exc
    return parse_dialogue(text, source=path)


def parse_dialogue(text: str, source: Path | None = None) -> Dialogue:
    """Parse dialogue text into a :class:`Dialogue`."""
    where = f"{source}: " if source else ""
    raw_lines = text.splitlines()

    if not any(_is_header(line.strip()) for line in raw_lines):
        lines = _parse_plain_text(raw_lines)
    else:
        lines = [_finish_block(block, where) for block in _parse_blocks(raw_lines, where)]

    if not lines:
        raise EmptyDialogueError(f"{where}the dialogue is empty; nothing to synthesize")
    return Dialogue(lines=tuple(lines), source=source)


def parse_header(body: str, line_number: int = 0, where: str = "") -> tuple[str, dict]:
    """Parse the inside of a ``[SPEAKER|key=value]`` header."""
    parts = [part.strip() for part in body.split("|")]
    speaker = parts[0]
    if not speaker:
        raise DialogueError(f"{where}line {line_number}: empty speaker name in header")
    if not _SPEAKER_RE.match(speaker):
        raise DialogueError(f"{where}line {line_number}: invalid speaker name {speaker!r}")

    params: dict[str, ParamValue] = {}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        key, value = key.strip().lower(), value.strip()
        if not sep or not key or not value:
            raise DialogueError(
                f"{where}line {line_number}: expected 'key=value' in header, got {part!r}"
            )
        if not _PARAM_KEY_RE.match(key):
            raise DialogueError(f"{where}line {line_number}: invalid parameter name {key!r}")
        if key in params:
            raise DialogueError(f"{where}line {line_number}: duplicate parameter {key!r}")
        params[key] = _parse_value(value)
    return speaker, params


def _parse_blocks(raw_lines: list[str], where: str) -> list[_Block]:
    blocks: list[_Block] = []
    for number, raw in enumerate(raw_lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _HEADER_RE.match(line)
        if match:
            speaker, params = parse_header(match.group("body"), number, where)
            blocks.append(_Block(speaker=speaker, params=params, line_number=number))
        elif not blocks:
            raise DialogueError(
                f"{where}line {number}: text found before the first [SPEAKER] header"
            )
        else:
            blocks[-1].text_lines.append(line)
    return blocks


def _finish_block(block: _Block, where: str) -> DialogueLine:
    if not block.text_lines:
        raise DialogueError(f"{where}line {block.line_number}: [{block.speaker}] has no text")
    return DialogueLine(
        speaker=block.speaker,
        text=" ".join(block.text_lines),
        params=block.params,
        line_number=block.line_number,
    )


def _parse_plain_text(raw_lines: list[str]) -> list[DialogueLine]:
    lines: list[DialogueLine] = []
    paragraph: list[str] = []
    start = 0
    for number, raw in enumerate([*raw_lines, ""], start=1):
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line:
            if not paragraph:
                start = number
            paragraph.append(line)
        elif paragraph:
            lines.append(
                DialogueLine(PLAIN_TEXT_SPEAKER, " ".join(paragraph), {}, line_number=start)
            )
            paragraph = []
    return lines


def _is_header(line: str) -> bool:
    return bool(_HEADER_RE.match(line))


def _parse_value(value: str) -> ParamValue:
    try:
        return float(value)
    except ValueError:
        return value
