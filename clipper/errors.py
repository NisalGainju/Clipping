class ClipperError(Exception):
    """Base error; message is safe to show to end users."""


class VideoUnavailableError(ClipperError):
    """Private, removed, age-restricted, geo-blocked or otherwise inaccessible video."""


class NoSpeechError(ClipperError):
    """Transcription produced no usable speech."""


class DependencyError(ClipperError):
    """A required external tool or package is missing."""
