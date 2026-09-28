"""Dialogue parsing."""

from voxlab.dialogue.models import Dialogue, DialogueLine
from voxlab.dialogue.parser import load_dialogue, parse_dialogue

__all__ = ["Dialogue", "DialogueLine", "load_dialogue", "parse_dialogue"]
