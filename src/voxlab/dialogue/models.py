"""Data structures produced by the dialogue parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ParamValue = float | str


@dataclass(frozen=True)
class DialogueLine:
    speaker: str
    text: str
    params: dict[str, ParamValue] = field(default_factory=dict)
    line_number: int = 0


@dataclass(frozen=True)
class Dialogue:
    lines: tuple[DialogueLine, ...]
    source: Path | None = None

    @property
    def speakers(self) -> list[str]:
        """Unique speakers in order of first appearance."""
        return list(dict.fromkeys(line.speaker for line in self.lines))

    def __len__(self) -> int:
        return len(self.lines)
