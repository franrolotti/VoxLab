"""Exception hierarchy. Every error meant for the user derives from VoxLabError."""


class VoxLabError(Exception):
    """Base class for errors reported to the user without a traceback."""


class ConfigError(VoxLabError):
    """Invalid or unreadable configuration."""


class DialogueError(VoxLabError):
    """Invalid dialogue file."""


class EmptyDialogueError(DialogueError):
    """The dialogue contains no lines to synthesize."""


class VoiceError(VoxLabError):
    """Invalid voice profile or voice assignment."""


class UnknownSpeakerError(VoiceError):
    """A speaker could not be mapped to a voice."""


class PresetError(VoxLabError):
    """Invalid or missing audio preset."""


class BackendError(VoxLabError):
    """TTS backend failure (missing model, unsupported feature, ...)."""


class ExportError(VoxLabError):
    """Audio export failure."""
