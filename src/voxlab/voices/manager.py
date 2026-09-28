"""Voice profiles and speaker-to-voice assignment.

A voice profile describes *how a character sounds*: which built-in speaker of
the backend to use (or a reference clip for cloning), default language,
default parameters and optionally a per-voice preset. Profiles come from
``builtin.yaml`` and from ``voices/<name>/voice.yaml`` (user profiles win).
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from voxlab.errors import UnknownSpeakerError, VoiceError

log = logging.getLogger(__name__)

BUILTIN_FILE = Path(__file__).with_name("builtin.yaml")
PROFILE_FILENAME = "voice.yaml"
REFERENCE_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
VOICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_PROFILE_KEYS = {"name", "description", "speaker", "reference", "language", "params", "preset"}

SpeakerSpec = str | Mapping[str, Any] | None


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    description: str = ""
    speaker: SpeakerSpec = None
    reference: Path | None = None
    language: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    preset: str | None = None
    source: str = "builtin"

    @property
    def uses_cloning(self) -> bool:
        return self.reference is not None

    def speaker_for(self, backend: str, language: str) -> str | None:
        """Pick the backend speaker id for ``language`` (None -> backend default)."""
        spec = self.speaker
        if isinstance(spec, Mapping):
            spec = spec.get(backend)
        if isinstance(spec, Mapping):
            language = language.lower()
            spec = spec.get(language) or spec.get(language.split("-")[0])
        return str(spec) if spec else None


def validate_voice_name(name: str) -> str:
    if not VOICE_NAME_RE.match(name):
        raise VoiceError(f"Invalid voice name {name!r}: use lowercase letters, digits, '-' and '_'")
    return name


def parse_profile(data: Any, name: str, source: str, base_dir: Path | None = None) -> VoiceProfile:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise VoiceError(f"Voice {name!r}: profile must be a mapping")
    unknown = set(data) - _PROFILE_KEYS
    if unknown:
        raise VoiceError(f"Voice {name!r}: unknown key(s) {', '.join(sorted(unknown))}")

    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise VoiceError(f"Voice {name!r}: 'params' must be a mapping")
    speaker = data.get("speaker")
    if speaker is not None and not isinstance(speaker, (str, dict)):
        raise VoiceError(f"Voice {name!r}: 'speaker' must be a string or mapping")

    reference = None
    if data.get("reference"):
        reference = Path(str(data["reference"])).expanduser()
        if not reference.is_absolute() and base_dir is not None:
            reference = base_dir / reference
        if not reference.is_file():
            raise VoiceError(f"Voice {name!r}: reference audio not found: {reference}")

    return VoiceProfile(
        name=name,
        description=str(data.get("description", "")),
        speaker=speaker,
        reference=reference,
        language=str(data["language"]).lower() if data.get("language") else None,
        params=dict(params),
        preset=str(data["preset"]) if data.get("preset") else None,
        source=source,
    )


class VoiceManager:
    def __init__(self, user_dir: Path | None = None, builtin_file: Path | None = BUILTIN_FILE):
        self.user_dir = Path(user_dir) if user_dir else None
        self.builtin_file = builtin_file
        self._cache: dict[str, VoiceProfile] | None = None

    def _load_builtin(self) -> dict[str, VoiceProfile]:
        if not self.builtin_file or not self.builtin_file.is_file():
            return {}
        data = yaml.safe_load(self.builtin_file.read_text(encoding="utf-8")) or {}
        return {name: parse_profile(body, name, "builtin") for name, body in data.items()}

    def _load_user(self) -> dict[str, VoiceProfile]:
        profiles: dict[str, VoiceProfile] = {}
        if not self.user_dir or not self.user_dir.is_dir():
            return profiles
        for directory in sorted(p for p in self.user_dir.iterdir() if p.is_dir()):
            profile = self._load_user_dir(directory)
            if profile is not None:
                profiles[profile.name] = profile
        return profiles

    def _load_user_dir(self, directory: Path) -> VoiceProfile | None:
        name = directory.name.lower()
        profile_file = directory / PROFILE_FILENAME
        if profile_file.is_file():
            try:
                data = yaml.safe_load(profile_file.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise VoiceError(f"Invalid YAML in {profile_file}: {exc}") from exc
            return parse_profile(data, name, "user", base_dir=directory)
        # A folder with just a reference clip is a valid cloning voice.
        clips = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in REFERENCE_SUFFIXES]
        if clips:
            return VoiceProfile(name=name, reference=clips[0], source="user")
        log.debug("Ignoring %s: no %s or reference audio", directory, PROFILE_FILENAME)
        return None

    def _profiles(self) -> dict[str, VoiceProfile]:
        if self._cache is None:
            self._cache = {**self._load_builtin(), **self._load_user()}
        return self._cache

    def all(self) -> list[VoiceProfile]:
        return sorted(self._profiles().values(), key=lambda v: v.name)

    def names(self) -> list[str]:
        return sorted(self._profiles())

    def get(self, name: str) -> VoiceProfile:
        profiles = self._profiles()
        key = name.lower()
        if key not in profiles:
            raise VoiceError(f"Unknown voice {name!r}. Available: {', '.join(sorted(profiles))}")
        return profiles[key]

    def has(self, name: str) -> bool:
        return name.lower() in self._profiles()

    def resolve_cast(
        self,
        speakers: Iterable[str],
        cast: Mapping[str, str | Mapping[str, str]] | None = None,
        default: str = "default",
        strict: bool = False,
    ) -> dict[str, VoiceProfile]:
        """Assign a voice profile to every speaker.

        Priority: explicit ``cast`` entry (case-insensitive) > voice with the
        speaker's name > ``default`` (or an error when ``strict``). A cast entry
        may be a voice name or ``{voice: ..., language: ...}``; its language
        becomes the character's default language.
        """
        lookup = {k.lower(): v for k, v in (cast or {}).items()}
        assignment: dict[str, VoiceProfile] = {}
        for speaker in speakers:
            key = speaker.lower()
            entry = lookup.get(key)
            if isinstance(entry, Mapping):
                voice_name = entry.get("voice") or (key if self.has(key) else default)
                voice = self.get(voice_name)
                if entry.get("language"):
                    voice = replace(voice, language=str(entry["language"]).lower())
                assignment[speaker] = voice
            elif entry:
                assignment[speaker] = self.get(entry)
            elif self.has(key):
                assignment[speaker] = self.get(key)
            elif strict:
                raise UnknownSpeakerError(
                    f"Speaker {speaker!r} has no voice. Add it to 'cast' in config.yaml, "
                    f"pass --voice {speaker}=<voice>, or create voices/{key}/"
                )
            else:
                log.warning("Speaker %r has no assigned voice; using %r", speaker, default)
                assignment[speaker] = self.get(default)
        return assignment

    def add(
        self,
        name: str,
        reference: Path | None = None,
        speaker: str | None = None,
        language: str | None = None,
        description: str = "",
        overwrite: bool = False,
    ) -> VoiceProfile:
        """Create ``voices/<name>/voice.yaml`` (copying a reference clip if given)."""
        if self.user_dir is None:
            raise VoiceError("No user voices directory configured")
        validate_voice_name(name)
        if reference is None and speaker is None:
            raise VoiceError("A voice needs a reference clip (--reference) or a --speaker")
        target = self.user_dir / name
        if (target / PROFILE_FILENAME).exists() and not overwrite:
            raise VoiceError(f"Voice {name!r} already exists in {target} (use --force)")

        if reference is not None:
            reference = Path(reference)
            if not reference.is_file():
                raise VoiceError(f"Reference audio not found: {reference}")
            if reference.suffix.lower() not in REFERENCE_SUFFIXES:
                raise VoiceError(f"Reference must be one of {', '.join(REFERENCE_SUFFIXES)}")

        data: dict[str, Any] = {"description": description} if description else {}
        target.mkdir(parents=True, exist_ok=True)
        if reference is not None:
            copied = target / f"reference{reference.suffix.lower()}"
            shutil.copyfile(reference, copied)
            data["reference"] = copied.name
        if speaker:
            data["speaker"] = speaker
        if language:
            data["language"] = language.lower()
        (target / PROFILE_FILENAME).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        self._cache = None
        return self.get(name)
